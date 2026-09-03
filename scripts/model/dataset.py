"""Paired NMR and molecular-fingerprint input for foundation-model training."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as parquet
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info

from data.schema import CanonicalRecord, ensure_record
from data.validation import IncompatibleRecordError, validate_canonical_record
from data.postprocess.calculate_mol_properties import (
    MOLECULAR_PROPERTIES_SCHEMA_VERSION,
    MORGAN_FIELD,
    MORGAN_FP_BYTES,
    MORGAN_FP_SIZE,
    MORGAN_RADIUS,
    MACCS_OUTPUT_BITS,
    MACCS_FIELD,
)


@dataclass(frozen=True)
class PairedFoundationRecord:
    """One validated NMR record and its molecular fingerprints."""

    record: CanonicalRecord
    morgan_fingerprint: bytes
    maccs_fingerprint: str
    shift_only: bool = False


def _checked_record(value: object) -> CanonicalRecord:
    record = ensure_record(value)
    validation = validate_canonical_record(record)
    if not validation.is_valid:
        raise IncompatibleRecordError(record.record_id, validation.issues)
    return record


def _buffered_shuffle(
    records: Iterable[PairedFoundationRecord],
    *,
    random_generator: random.Random,
    buffer_size: int,
) -> Iterator[PairedFoundationRecord]:
    """Shuffle a stream while keeping only a bounded record buffer in memory."""

    buffer = []
    for record in records:
        if len(buffer) < buffer_size:
            buffer.append(record)
            continue

        selected_index = random_generator.randrange(len(buffer))
        yield buffer[selected_index]
        buffer[selected_index] = record

    random_generator.shuffle(buffer)
    yield from buffer


def _distributed_info():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()

    # useful also inside DataLoader worker processes
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return rank, world_size


class PairedFoundationDataset(IterableDataset):
    """Stream aligned NMR records and Morgan fingerprints from two Parquets.

    The two files must have identical row counts, row-group boundaries, and
    record IDs. Workers receive disjoint row groups by default; small datasets
    can explicitly replicate row groups, and rank sharding can be disabled.
    Every worker reads both files independently without loading the full
    dataset. Training shuffle uses a bounded record buffer.
    """

    def __init__(
        self,
        nmr_path: str | Path,
        molecular_properties_path: str | Path,
        *,
        arrow_batch_size: int = 2048,
        shuffle: bool = True,
        shuffle_buffer_size: int = 8192,
        seed: int = 0,
        shift_only: bool = False,
        source_name: str | None = None,
        include_unimol: bool = False,
        shard_across_ranks: bool = True,
        replicate_when_too_small: bool = False,
    ):
        if include_unimol:
            raise NotImplementedError(
                "UniMol inputs are reserved for a future PairedFoundationDataset "
                "implementation"
            )
        if arrow_batch_size < 1:
            raise ValueError("arrow_batch_size must be at least 1")
        if shuffle_buffer_size < 1:
            raise ValueError("shuffle_buffer_size must be at least 1")

        self.nmr_path = Path(nmr_path)
        self.molecular_properties_path = Path(molecular_properties_path)
        self.arrow_batch_size = arrow_batch_size
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.shift_only = shift_only
        self.source_name = source_name or self.nmr_path.stem
        self.shard_across_ranks = shard_across_ranks
        self.replicate_when_too_small = replicate_when_too_small
        self._shuffle_random = None
        self._shuffle_random_context = None

        for label, path in (
            ("NMR Parquet", self.nmr_path),
            ("molecular-property Parquet", self.molecular_properties_path),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{label} not found: {path}")

        nmr_file = parquet.ParquetFile(self.nmr_path)
        molecular_file = parquet.ParquetFile(self.molecular_properties_path)
        self._validate_layout(nmr_file, molecular_file)
        self.num_rows = nmr_file.metadata.num_rows
        self.num_row_groups = nmr_file.num_row_groups

    def _validate_layout(self, nmr_file, molecular_file) -> None:
        if "record_id" not in nmr_file.schema_arrow.names:
            raise ValueError(f"{self.nmr_path} is missing required field: record_id")

        required_fields = {"record_id", "rdkit_status", MORGAN_FIELD}
        missing_fields = required_fields.difference(molecular_file.schema_arrow.names)
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(
                f"{self.molecular_properties_path} is missing required fields: {missing}"
            )

        fingerprint_type = molecular_file.schema_arrow.field(MORGAN_FIELD).type
        expected_type = pa.binary(MORGAN_FP_BYTES)
        if fingerprint_type != expected_type:
            raise ValueError(
                f"{self.molecular_properties_path} field {MORGAN_FIELD} must be "
                f"{expected_type}, found {fingerprint_type}"
            )

        metadata = molecular_file.schema_arrow.metadata or {}
        expected_metadata = {
            b"molecular_properties_schema_version": (
                MOLECULAR_PROPERTIES_SCHEMA_VERSION.encode("utf-8")
            ),
            b"morgan_radius": str(MORGAN_RADIUS).encode("ascii"),
            b"morgan_num_bits": str(MORGAN_FP_SIZE).encode("ascii"),
        }
        for key, expected in expected_metadata.items():
            actual = metadata.get(key)
            if actual != expected:
                name = key.decode("utf-8")
                raise ValueError(
                    f"{self.molecular_properties_path} has {name}={actual!r}; "
                    f"expected {expected!r}"
                )

        if nmr_file.metadata.num_rows != molecular_file.metadata.num_rows:
            raise ValueError(
                "NMR and molecular-property Parquets have different row counts"
            )
        if nmr_file.num_row_groups != molecular_file.num_row_groups:
            raise ValueError(
                "NMR and molecular-property Parquets have different row-group counts"
            )
        for index in range(nmr_file.num_row_groups):
            nmr_rows = nmr_file.metadata.row_group(index).num_rows
            molecular_rows = molecular_file.metadata.row_group(index).num_rows
            if nmr_rows != molecular_rows:
                raise ValueError(
                    f"row group {index} has {nmr_rows} NMR rows and "
                    f"{molecular_rows} molecular-property rows"
                )

    def __len__(self) -> int:
        return self.num_rows

    def _iter_row_groups(
        self,
        nmr_file,
        molecular_file,
        row_group_indices,
    ) -> Iterator[PairedFoundationRecord]:
        stored_columns = nmr_file.schema_arrow.names
        nmr_columns = [
            field.name
            for field in fields(CanonicalRecord)
            if field.name in stored_columns
        ]

        for row_group_index in row_group_indices:
            molecular_table = molecular_file.read_row_group(
                row_group_index,
                columns=["record_id", "rdkit_status", MORGAN_FIELD, MACCS_FIELD],
                use_threads=True,
            )
            molecular_ids = molecular_table.column("record_id").to_pylist()
            statuses = molecular_table.column("rdkit_status").to_pylist()
            fingerprints = molecular_table.column(MORGAN_FIELD).to_pylist()
            maccs_vectors = molecular_table.column(MACCS_FIELD).to_pylist()

            offset = 0
            for nmr_batch in nmr_file.iter_batches(
                batch_size=self.arrow_batch_size,
                row_groups=[row_group_index],
                columns=nmr_columns,
                use_threads=True,
            ):
                values = nmr_batch.to_pylist()
                for value in values:
                    nmr_record_id = value.get("record_id")
                    molecular_record_id = molecular_ids[offset]
                    if nmr_record_id != molecular_record_id:
                        raise ValueError(
                            f"record_id mismatch in row group {row_group_index}, "
                            f"row {offset}: NMR={nmr_record_id!r}, "
                            f"molecular={molecular_record_id!r}"
                        )

                    status = statuses[offset]
                    if status != "ok":
                        raise ValueError(
                            f"record {nmr_record_id} has rdkit_status={status!r}; "
                            "training requires a valid Morgan fingerprint"
                        )

                    fingerprint = fingerprints[offset]
                    if fingerprint is None or len(fingerprint) != MORGAN_FP_BYTES:
                        length = None if fingerprint is None else len(fingerprint)
                        raise ValueError(
                            f"record {nmr_record_id} has Morgan byte length {length}; "
                            f"expected {MORGAN_FP_BYTES}"
                        )

                    maccs_vector = maccs_vectors[offset]
                    if maccs_vector is None or len(maccs_vector) != MACCS_OUTPUT_BITS:
                        length = None if maccs_vector is None else len(maccs_vector)
                        raise ValueError(
                            f"record {nmr_record_id} has MACCS byte length {length}; "
                            f"expected {MACCS_OUTPUT_BITS}"
                        )

                    yield PairedFoundationRecord(
                        record=_checked_record(value),
                        morgan_fingerprint=fingerprint,
                        maccs_fingerprint=maccs_vector,
                        shift_only=self.shift_only,
                    )
                    offset += 1

            if offset != molecular_table.num_rows:
                raise RuntimeError(
                    f"row group {row_group_index} yielded {offset} NMR rows but "
                    f"{molecular_table.num_rows} molecular rows"
                )

    def _get_shuffle_random(self, worker) -> random.Random:
        if worker is None:
            context = ("main",)
            initial_seed = self.seed
        else:
            context = ("worker", worker.id, worker.seed)
            initial_seed = self.seed + worker.seed

        if self._shuffle_random_context != context:
            self._shuffle_random = random.Random(initial_seed)
            self._shuffle_random_context = context

        return self._shuffle_random

    def _assigned_row_group_indices(
        self,
        *,
        rank: int,
        world_size: int,
        worker_id: int,
        num_workers: int,
    ) -> list[int]:
        """Return this process-worker pair's row groups."""

        shard_rank = rank if self.shard_across_ranks else 0
        shard_world_size = world_size if self.shard_across_ranks else 1
        global_worker_id = shard_rank * num_workers + worker_id
        global_num_workers = shard_world_size * num_workers

        if self.num_row_groups < global_num_workers:
            if self.replicate_when_too_small:
                return list(range(self.num_row_groups))
            raise RuntimeError(
                f"dataset {self.source_name!r} has {self.num_row_groups} Parquet "
                f"row groups but must be sharded across {global_num_workers} "
                f"workers ({shard_world_size} ranks x {num_workers} DataLoader "
                "workers); reduce the worker count or explicitly enable "
                "small-dataset replication"
            )

        return list(
            range(
                global_worker_id,
                self.num_row_groups,
                global_num_workers,
            )
        )

    def __iter__(self) -> Iterator[PairedFoundationRecord]:
        nmr_file = parquet.ParquetFile(self.nmr_path)
        molecular_file = parquet.ParquetFile(self.molecular_properties_path)

        rank, world_size = _distributed_info()
        worker = get_worker_info()
        if worker is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker.id
            num_workers = worker.num_workers

        row_group_indices = self._assigned_row_group_indices(
            rank=rank,
            world_size=world_size,
            worker_id=worker_id,
            num_workers=num_workers,
        )

        if not self.shuffle:
            yield from self._iter_row_groups(
                nmr_file,
                molecular_file,
                row_group_indices,
            )
            return

        random_generator = self._get_shuffle_random(worker)
        random_generator.shuffle(row_group_indices)
        records = self._iter_row_groups(
            nmr_file,
            molecular_file,
            row_group_indices,
        )
        yield from _buffered_shuffle(
            records,
            random_generator=random_generator,
            buffer_size=self.shuffle_buffer_size,
        )


class MixedFoundationDataset(IterableDataset):
    """Mix paired datasets indefinitely, cycling each source as needed."""

    def __init__(self, datasets, proportions, shift_only, seed=0):
        if not (len(datasets) == len(proportions) == len(shift_only)):
            raise ValueError("datasets, proportions, and shift_only must match")
        if abs(sum(proportions) - 1.0) > 1e-6:
            raise ValueError("proportions must sum to 1")

        self.datasets = list(datasets)
        self.proportions = list(proportions)
        self.shift_only = list(shift_only)
        self.seed = seed

    def __iter__(self):
        iterators = [iter(dataset) for dataset in self.datasets]
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        random_generator = random.Random(self.seed + worker_id)

        while True:
            source = random_generator.choices(
                range(len(self.datasets)),
                weights=self.proportions,
            )[0]

            try:
                sample = next(iterators[source])
            except StopIteration:
                iterators[source] = iter(self.datasets[source])
                try:
                    sample = next(iterators[source])
                except StopIteration as error:
                    source_name = getattr(
                        self.datasets[source],
                        "source_name",
                        f"source {source}",
                    )
                    raise RuntimeError(
                        f"mixed-dataset source {source_name!r} yielded no samples "
                        "after its iterator was restarted"
                    ) from error

            yield PairedFoundationRecord(
                record=sample.record,
                morgan_fingerprint=sample.morgan_fingerprint,
                maccs_fingerprint=sample.maccs_fingerprint,
                shift_only=self.shift_only[source],
            )


__all__ = [
    "MixedFoundationDataset",
    "PairedFoundationDataset",
    "PairedFoundationRecord",
]

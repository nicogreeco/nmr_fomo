"""Datasets for records that already follow the canonical NMR schema."""

import json
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

from .schema import CanonicalRecord, ensure_record
from .validation import IncompatibleRecordError, validate_canonical_record

try:
    from torch.utils.data import IterableDataset, get_worker_info
except ModuleNotFoundError:
    # Conversion and schema tools should remain importable without PyTorch.
    class IterableDataset:  # type: ignore[no-redef]
        pass

    def get_worker_info():  # type: ignore[no-redef]
        return None


def _checked_record(value: object) -> CanonicalRecord:
    """Convert and validate one record before returning it from a dataset."""

    record = ensure_record(value)
    validation = validate_canonical_record(record)
    if not validation.is_valid:
        raise IncompatibleRecordError(validation.record_id, validation.issues)
    return record


class CanonicalNMRDataset(Sequence[CanonicalRecord]):
    """Small in-memory dataset that always returns canonical records."""

    def __init__(self, records: Iterable[object]):
        self.records = [_checked_record(record) for record in records]

        record_ids = [record.record_id for record in self.records]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("record_id values must be unique within a dataset")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


class JsonlCanonicalReader:
    """Stream canonical records from a JSONL fixture or small input file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def __iter__(self) -> Iterator[CanonicalRecord]:
        with self.path.open("r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue

                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid JSON on line {line_number} of {self.path}"
                    ) from error

                if not isinstance(value, dict):
                    raise ValueError(
                        f"line {line_number} of {self.path} must contain an object"
                    )
                yield _checked_record(value)


class CanonicalParquetDataset(IterableDataset):
    """Stream canonical records from one or more Parquet files.

    The file is read in Arrow batches, so a large Parquet file does not need to
    be loaded fully into memory. The dataset yields model-independent
    ``CanonicalRecord`` objects and leaves all model preparation to a processor.

    With no DataLoader workers, records keep file and row order. With multiple
    workers, whole files are divided among workers. For one large file, start
    with ``num_workers=0``.
    """

    def __init__(
        self,
        paths: str | Path | Iterable[str | Path],
        arrow_batch_size: int = 2048,
    ):
        if isinstance(paths, (str, Path)):
            paths = [paths]

        self.paths = [Path(path) for path in paths]
        if not self.paths:
            raise ValueError("at least one Parquet path is required")

        missing_paths = [str(path) for path in self.paths if not path.is_file()]
        if missing_paths:
            missing_text = ", ".join(missing_paths)
            raise FileNotFoundError(
                f"canonical Parquet input not found: {missing_text}"
            )

        if arrow_batch_size < 1:
            raise ValueError("arrow_batch_size must be at least 1")
        self.arrow_batch_size = arrow_batch_size

    def __iter__(self) -> Iterator[CanonicalRecord]:
        try:
            import pyarrow.parquet as parquet
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "CanonicalParquetDataset requires pyarrow; install it in the "
                "environment used for extraction"
            ) from error

        worker = get_worker_info()
        if worker is None:
            worker_paths = self.paths
        else:
            worker_paths = self.paths[worker.id :: worker.num_workers]

        for path in worker_paths:
            parquet_file = parquet.ParquetFile(path)
            for arrow_batch in parquet_file.iter_batches(
                batch_size=self.arrow_batch_size
            ):
                for value in arrow_batch.to_pylist():
                    yield _checked_record(value)

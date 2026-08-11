"""Convert SimNMR-PubChem from NMR-Solver into canonical Parquet.

The source LMDB stores one predicted shift per atom. Hydrogen and carbon atoms
are grouped by the supplied ``equi_class`` value, never by shift proximity.
Each equivalence group becomes one canonical resonance at the mean member
shift. Proton groups also retain their original shifts and group size.

The database has no published train, validation, or test split, so one input
produces one physical Parquet file without inventing split labels.

Example:
    PYTHONPATH=scripts python scripts/data/canonicalize/convert_nmrsolver.py \
        datasets/raw/simnmr_pubchem/metadata/PubChem_merged_id.lmdb \
        datasets/canonical/simnmr/all.parquet
"""

import argparse
import json
from collections.abc import Iterator, Mapping
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any, TextIO

from data.canonicalize.common import (
    ConversionError,
    chemical_metadata_from_smiles,
    checked_canonical_record,
    finite_float,
    iter_lmdb_records,
    lmdb_key_text,
    require_mapping,
    write_canonical_parquet,
)
from data.schema import CanonicalRecord
from data.reporting import (
    default_report_path,
    prepare_report_output,
    write_processing_report,
)


SOURCE_NAME = "SimNMR-PubChem"
HYDROGEN_ATOMIC_NUMBER = 1
CARBON_ATOMIC_NUMBER = 6
_WORKER_RECORD_FIELDS = (
    "smiles",
    "canonical_smiles",
    "nmr_predict",
    "atom_index",
    "equi_class",
)


class ChemicalMetadataConversionError(ConversionError):
    """A source structure cannot produce coherent RDKit-derived metadata."""


class ChemicalMetadataRejectionReport:
    """Write an auditable record for each skipped source structure."""

    def __init__(self, output_file: TextIO) -> None:
        self.output_file = output_file
        self.count = 0

    def write(
        self,
        record_id: str,
        raw_record: Mapping[str, Any],
        reason: str,
    ) -> None:
        source_smiles = raw_record.get("smiles") or raw_record.get(
            "canonical_smiles"
        )
        if not isinstance(source_smiles, str):
            source_smiles = repr(source_smiles)

        json.dump(
            {
                "record_id": record_id,
                "source_smiles": source_smiles,
                "reason": reason,
            },
            self.output_file,
            sort_keys=True,
        )
        self.output_file.write("\n")
        self.count += 1


def _source_array(
    raw_record: Mapping[str, Any], field_name: str, location: str
) -> list[object]:
    """Read one required NumPy-style source array as a Python list."""

    if field_name not in raw_record:
        raise ConversionError(f"{location} is missing {field_name!r}")
    value = raw_record[field_name]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ConversionError(f"{location}.{field_name} must be an array")
    return list(value)


def _source_integer(value: object, location: str) -> int:
    """Read an integer stored either by Python or NumPy."""

    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConversionError(f"{location} must be an integer")
    return int(value)


def _group_nuclei(
    shifts: list[object],
    atom_numbers: list[object],
    equivalence_classes: list[object],
    location: str,
) -> tuple[list[tuple[int, list[float]]], list[tuple[int, list[float]]]]:
    """Group H and C atoms by source equivalence class in one pass."""

    if not (len(shifts) == len(atom_numbers) == len(equivalence_classes)):
        raise ConversionError(
            f"{location}: nmr_predict, atom_index, and equi_class must have "
            "the same length"
        )

    hydrogen_groups: dict[int, list[float]] = {}
    carbon_groups: dict[int, list[float]] = {}
    hydrogen_order: list[int] = []
    carbon_order: list[int] = []
    for atom_position in range(len(shifts)):
        atom_number = _source_integer(
            atom_numbers[atom_position],
            f"{location}.atom_index[{atom_position}]",
        )
        if (
            atom_number != HYDROGEN_ATOMIC_NUMBER
            and atom_number != CARBON_ATOMIC_NUMBER
        ):
            continue

        equivalence_class = _source_integer(
            equivalence_classes[atom_position],
            f"{location}.equi_class[{atom_position}]",
        )
        shift = finite_float(
            shifts[atom_position],
            f"{location}.nmr_predict[{atom_position}]",
        )
        if atom_number == HYDROGEN_ATOMIC_NUMBER:
            groups = hydrogen_groups
            group_order = hydrogen_order
        else:
            groups = carbon_groups
            group_order = carbon_order

        if equivalence_class not in groups:
            groups[equivalence_class] = []
            group_order.append(equivalence_class)
        groups[equivalence_class].append(shift)

    grouped_hydrogens = [
        (group, hydrogen_groups[group]) for group in hydrogen_order
    ]
    grouped_carbons = [(group, carbon_groups[group]) for group in carbon_order]
    return grouped_hydrogens, grouped_carbons


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def convert_record(
    raw_record: object,
    record_id: str,
    location: str,
) -> CanonicalRecord:
    """Map one atom-level SimNMR-PubChem record to resonance-level peaks."""

    record = require_mapping(raw_record, location)
    shifts = _source_array(record, "nmr_predict", location)
    atom_numbers = _source_array(record, "atom_index", location)
    equivalence_classes = _source_array(record, "equi_class", location)

    hydrogen_groups, carbon_groups = _group_nuclei(
        shifts,
        atom_numbers,
        equivalence_classes,
        location,
    )

    h_peaks = [
        {
            "shift": _mean(member_shifts),
            "integration": len(member_shifts),
            "multiplicity_raw": None,
            "multiplicity": None,
            "j_values": None,
            "range_min": None,
            "range_max": None,
            "range_half_span": None,
            "equivalence_class": equivalence_class,
            "member_shifts": member_shifts,
        }
        for equivalence_class, member_shifts in hydrogen_groups
    ]
    c_peaks = [
        {
            "shift": _mean(member_shifts),
            "integral": None,
            "intensity": None,
            "width": None,
        }
        for _, member_shifts in carbon_groups
    ]

    try:
        chemical_metadata = chemical_metadata_from_smiles(
            record.get("smiles") or record.get("canonical_smiles"),
            f"{location}.smiles",
        )
    except ConversionError as error:
        raise ChemicalMetadataConversionError(str(error)) from error

    canonical_data: dict[str, object] = {
        "record_id": record_id,
        "source": SOURCE_NAME,
        **chemical_metadata,
        "nmr_frequency": None,
        "nmr_solvent": None,
        "h_nmr_peaks": h_peaks,
        "c_nmr_peaks": c_peaks,
    }
    return checked_canonical_record(canonical_data, location)


def resolve_input(input_path: str | Path) -> Path:
    """Accept the metadata LMDB itself or a nearby repository directory."""

    path = Path(input_path)
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"NMR-Solver input not found: {path}")

    candidates = [
        path / "PubChem_merged_id.lmdb",
        path / "metadata" / "PubChem_merged_id.lmdb",
        path / "database" / "metadata" / "PubChem_merged_id.lmdb",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"could not find PubChem_merged_id.lmdb under {path}"
    )


def _record_for_worker(raw_record: Mapping[str, Any]) -> dict[str, object]:
    """Keep only source fields needed to convert one record in a worker."""

    return {
        field_name: raw_record[field_name]
        for field_name in _WORKER_RECORD_FIELDS
        if field_name in raw_record
    }


def _record_for_rejection_report(
    raw_record: Mapping[str, Any],
) -> dict[str, object]:
    """Keep only the source SMILES fields needed by the parent-side report."""

    return {
        field_name: raw_record[field_name]
        for field_name in ("smiles", "canonical_smiles")
        if field_name in raw_record
    }


def _convert_record_in_worker(
    raw_record: Mapping[str, Any],
    record_id: str,
    location: str,
) -> CanonicalRecord:
    """Run one independent record conversion in a spawned worker process."""

    return convert_record(raw_record, record_id, location)


def _convert_record_batch_in_worker(
    records: list[tuple[Mapping[str, Any], str, str]],
) -> list[tuple[CanonicalRecord | None, str | None]]:
    """Convert one bounded batch and keep metadata failures in-band.

    A metadata failure is an expected NMR-Solver exception: the parent needs
    to write it to the ordered rejection report while allowing the other rows
    in the same task to reach Parquet. Any other conversion error still fails
    the task and stops the run with its source location.
    """

    results: list[tuple[CanonicalRecord | None, str | None]] = []
    for raw_record, record_id, location in records:
        try:
            results.append((
                _convert_record_in_worker(raw_record, record_id, location),
                None,
            ))
        except ChemicalMetadataConversionError as error:
            results.append((None, str(error)))
    return results


def _iter_parallel_converted_records(
    source_records: Iterator[tuple[bytes, Mapping[str, Any]]],
    lmdb_path: Path,
    rejection_report: ChemicalMetadataRejectionReport | None,
    workers: int,
    max_in_flight: int,
    records_per_task: int,
) -> Iterator[CanonicalRecord]:
    """Convert records concurrently while yielding source-key order.

    LMDB reading, Parquet writing, and rejection-report writing stay in the
    parent process. Independent record conversion, including RDKit work, runs
    in child processes in bounded batches to avoid one process-pool task per
    record. Results are retired in submission order so the generated Parquet
    and JSONL files remain deterministic.
    """

    pending: list[
        tuple[
            Future[list[tuple[CanonicalRecord | None, str | None]]],
            list[tuple[str, Mapping[str, Any]]],
        ]
    ] = []
    pending_record_count = 0
    source_exhausted = False

    # Spawn prevents child processes from inheriting the parent's LMDB state.
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("spawn"),
    ) as executor:
        while pending or not source_exhausted:
            while (
                not source_exhausted
                and pending_record_count < max_in_flight
            ):
                available_slots = max_in_flight - pending_record_count
                task_size = min(records_per_task, available_slots)
                worker_records: list[tuple[Mapping[str, Any], str, str]] = []
                report_records: list[tuple[str, Mapping[str, Any]]] = []
                while len(worker_records) < task_size:
                    try:
                        key, raw_record = next(source_records)
                    except StopIteration:
                        source_exhausted = True
                        break

                    key_name = lmdb_key_text(key)
                    record_id = f"nmrsolver-simnmr-pubchem:{key_name}"
                    location = f"{lmdb_path} key {key_name}"
                    worker_records.append(
                        (_record_for_worker(raw_record), record_id, location)
                    )
                    report_records.append(
                        (record_id, _record_for_rejection_report(raw_record))
                    )

                if not worker_records:
                    continue
                future = executor.submit(
                    _convert_record_batch_in_worker,
                    worker_records,
                )
                pending.append((future, report_records))
                pending_record_count += len(worker_records)

            if not pending:
                continue

            future, report_records = pending.pop(0)
            results = future.result()
            pending_record_count -= len(report_records)
            if len(results) != len(report_records):
                raise RuntimeError(
                    "NMR-Solver worker returned a different number of results "
                    "than submitted records"
                )
            for (record, rejection_reason), (record_id, report_record) in zip(
                results, report_records
            ):
                if rejection_reason is not None:
                    if rejection_report is None:
                        raise ChemicalMetadataConversionError(rejection_reason)
                    rejection_report.write(
                        record_id,
                        report_record,
                        rejection_reason,
                    )
                    continue
                if record is None:
                    raise RuntimeError(
                        "NMR-Solver worker returned neither a record nor a "
                        "rejection reason"
                    )
                yield record


def iter_converted_records(
    input_path: str | Path,
    rejection_report: ChemicalMetadataRejectionReport | None = None,
    *,
    workers: int = 1,
    max_in_flight: int | None = None,
    records_per_task: int = 1,
) -> Iterator[CanonicalRecord]:
    """Yield valid records and report source structures RDKit cannot process."""

    if workers < 1:
        raise ValueError("workers must be at least 1")
    if records_per_task < 1:
        raise ValueError("records_per_task must be at least 1")
    if max_in_flight is not None and max_in_flight < workers:
        raise ValueError("max_in_flight must be at least workers")

    lmdb_path = resolve_input(input_path)
    source_records = iter(iter_lmdb_records(lmdb_path, readahead=True))
    if workers > 1:
        yield from _iter_parallel_converted_records(
            source_records,
            lmdb_path,
            rejection_report,
            workers,
            max_in_flight or workers * records_per_task * 2,
            records_per_task,
        )
        return

    for key, raw_record in source_records:
        key_name = lmdb_key_text(key)
        record_id = f"nmrsolver-simnmr-pubchem:{key_name}"
        location = f"{lmdb_path} key {key_name}"
        try:
            yield convert_record(raw_record, record_id, location)
        except ChemicalMetadataConversionError as error:
            if rejection_report is None:
                raise
            rejection_report.write(record_id, raw_record, str(error))


def _records_with_progress(
    records: Iterator[CanonicalRecord], progress_every: int
) -> Iterator[CanonicalRecord]:
    """Print a small progress message while preserving streaming behavior."""

    for record_count, record in enumerate(records, start=1):
        yield record
        if record_count % progress_every == 0:
            print(f"converted {record_count:,} records", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="PubChem_merged_id.lmdb file or NMR-Solver directory",
    )
    parser.add_argument("output", type=Path, help="single canonical .parquet output")
    parser.add_argument(
        "--row-group-size",
        type=int,
        default=50000,
        help="rows per internal Parquet row group (default: 50000)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000000,
        help="print progress every N records; use 0 to disable (default: 1000000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "record-conversion worker processes; 1 keeps sequential "
            "conversion (default: 1)"
        ),
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        help=(
            "maximum records submitted to workers before ordered writing "
            "(default: two worker tasks per worker)"
        ),
    )
    parser.add_argument(
        "--records-per-task",
        type=int,
        default=256,
        help=(
            "records converted by one process-pool task (default: 256); "
            "larger values reduce multiprocessing overhead"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacement of an existing output file",
    )
    parser.add_argument(
        "--rejection-report",
        type=Path,
        help=(
            "JSONL file for skipped RDKit chemical-metadata failures "
            "(default: beside the output)"
        ),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        help="processing JSON (default: beside the canonical Parquet)",
    )
    return parser.parse_args()


def default_rejection_report_path(output_path: Path) -> Path:
    """Place the chemical-metadata audit beside the canonical Parquet."""

    return output_path.with_name(
        f"{output_path.stem}_chemical_metadata_rejections.jsonl"
    )


def main() -> None:
    args = parse_args()
    lmdb_path = resolve_input(args.input)
    if lmdb_path.resolve() == args.output.resolve():
        raise ValueError("output must not overwrite the source dataset")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be zero or greater")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.records_per_task < 1:
        raise ValueError("--records-per-task must be at least 1")
    if args.max_in_flight is not None and args.max_in_flight < args.workers:
        raise ValueError("--max-in-flight must be at least --workers")

    rejection_report_path = (
        args.rejection_report or default_rejection_report_path(args.output)
    )
    if rejection_report_path.resolve() == args.output.resolve():
        raise ValueError("rejection report must not replace the Parquet output")
    if rejection_report_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"refusing to overwrite {rejection_report_path}; pass --overwrite "
            "to replace it"
        )
    report_path, report_temporary = prepare_report_output(
        args.report_output or default_report_path(args.output),
        [args.input, args.output, rejection_report_path],
        overwrite=args.overwrite,
    )
    rejection_report_path.parent.mkdir(parents=True, exist_ok=True)
    partial_report_path = rejection_report_path.with_name(
        f".{rejection_report_path.name}.partial"
    )
    if partial_report_path.exists():
        raise FileExistsError(
            f"temporary rejection report already exists: {partial_report_path}; "
            "remove it after checking the interrupted conversion"
        )

    with partial_report_path.open("x", encoding="utf-8") as report_file:
        rejection_report = ChemicalMetadataRejectionReport(report_file)
        records = iter_converted_records(
            lmdb_path,
            rejection_report,
            workers=args.workers,
            max_in_flight=args.max_in_flight,
            records_per_task=args.records_per_task,
        )
        if args.progress_every:
            records = _records_with_progress(records, args.progress_every)

        record_count = write_canonical_parquet(
            records,
            args.output,
            source_name=SOURCE_NAME,
            converter_name="convert_nmrsolver.py",
            row_group_size=args.row_group_size,
            overwrite=args.overwrite,
        )

    partial_report_path.replace(rejection_report_path)
    report = {
        "stage": "canonicalize_simnmr_pubchem",
        "inputs": {"source": str(lmdb_path)},
        "outputs": {
            "dataset": str(args.output),
            "rejected_records": str(rejection_report_path),
        },
        "counts": {
            "output_records": record_count,
            "rejected_records": rejection_report.count,
        },
    }
    write_processing_report(report, report_path, report_temporary)
    print(f"Wrote processing report to {report_path}")


if __name__ == "__main__":
    main()

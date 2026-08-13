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
import pickle
from collections.abc import Iterator, Mapping
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any, TextIO

from data.canonicalize.common import (
    ConversionError,
    chemical_metadata_from_smiles,
    canonical_record_to_row,
    checked_canonical_record,
    finite_float,
    iter_lmdb_records,
    lmdb_key_text,
    require_mapping,
    write_canonical_rows,
)
from data.console import (
    add_console_arguments,
    configure_console,
    print_stage_complete,
    print_stage_start,
    progress_bar,
)
from data.reporting import (
    default_report_path,
    prepare_report_output,
    write_processing_report,
)
from data.schema import CanonicalRecord, ensure_record


SOURCE_NAME = "SimNMR-PubChem"
HYDROGEN_ATOMIC_NUMBER = 1
CARBON_ATOMIC_NUMBER = 6


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


def _record_for_rejection_report(
    raw_record: Mapping[str, Any],
) -> dict[str, object]:
    """Keep only source SMILES fields needed by the rejection audit."""

    return {
        field_name: raw_record[field_name]
        for field_name in ("smiles", "canonical_smiles")
        if field_name in raw_record
    }


def iter_converted_records(
    input_path: str | Path,
    rejection_report: ChemicalMetadataRejectionReport | None = None,
    *,
    workers: int = 1,
    max_in_flight: int | None = None,
    records_per_task: int = 1,
) -> Iterator[CanonicalRecord]:
    """Yield canonical records through the stable programmatic API."""

    if workers < 1:
        raise ValueError("workers must be at least 1")
    if records_per_task < 1:
        raise ValueError("records_per_task must be at least 1")
    if max_in_flight is not None and max_in_flight < workers:
        raise ValueError("max_in_flight must be at least workers")

    lmdb_path = resolve_input(input_path)
    source_records = iter(iter_lmdb_records(lmdb_path, readahead=True))
    if workers > 1:
        serialized_records = (
            (key, pickle.dumps(raw_record, protocol=pickle.HIGHEST_PROTOCOL))
            for key, raw_record in source_records
        )
        rows = _iter_parallel_converted_rows(
            serialized_records,
            lmdb_path,
            rejection_report,
            workers,
            max_in_flight or workers * records_per_task * 2,
            records_per_task,
            False,
            None,
        )
        for row in rows:
            yield ensure_record(row)
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


def _iter_serialized_lmdb_records(path: Path) -> Iterator[tuple[bytes, bytes]]:
    """Yield LMDB keys and still-pickled values in source order."""

    try:
        import lmdb
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "reading SimNMR requires lmdb; install it in the conversion environment"
        ) from error

    environment = lmdb.open(
        str(path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=True,
        meminit=False,
        max_readers=256,
    )
    try:
        with environment.begin() as transaction:
            for key, value in transaction.cursor():
                yield bytes(key), bytes(value)
    finally:
        environment.close()


def _lmdb_entry_count(path: Path) -> int:
    """Return the number of source records for an exact progress total."""

    try:
        import lmdb
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "reading SimNMR requires lmdb; install it in the conversion environment"
        ) from error

    environment = lmdb.open(
        str(path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    try:
        with environment.begin() as transaction:
            return int(transaction.stat()["entries"])
    finally:
        environment.close()


def _convert_serialized_batch_in_worker(
    records: list[tuple[bytes, bytes]],
    lmdb_path_text: str,
) -> list[
    tuple[dict[str, object] | None, str | None, dict[str, object] | None]
]:
    """Unpickle, convert, and prepare Arrow rows inside one worker."""

    results = []
    for key, serialized_record in records:
        key_name = lmdb_key_text(key)
        record_id = f"nmrsolver-simnmr-pubchem:{key_name}"
        location = f"{lmdb_path_text} key {key_name}"
        try:
            raw_record = pickle.loads(serialized_record)
        except Exception as error:
            raise ConversionError(f"could not unpickle {location}") from error
        raw_record = require_mapping(raw_record, location)

        try:
            canonical_record = convert_record(raw_record, record_id, location)
        except ChemicalMetadataConversionError as error:
            results.append((
                None,
                str(error),
                _record_for_rejection_report(raw_record),
            ))
        else:
            results.append((canonical_record_to_row(canonical_record), None, None))
    return results


def _iter_parallel_converted_rows(
    source_records: Iterator[tuple[bytes, bytes]],
    lmdb_path: Path,
    rejection_report: ChemicalMetadataRejectionReport | None,
    workers: int,
    max_in_flight: int,
    records_per_task: int,
    quiet: bool,
    progress,
) -> Iterator[dict[str, object]]:
    """Convert serialized LMDB values concurrently in deterministic order."""

    pending: list[tuple[Future[list], list[bytes]]] = []
    pending_record_count = 0
    source_exhausted = False
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("spawn"),
        initializer=configure_console,
        initargs=(quiet,),
    ) as executor:
        while pending or not source_exhausted:
            while not source_exhausted and pending_record_count < max_in_flight:
                task_size = min(
                    records_per_task,
                    max_in_flight - pending_record_count,
                )
                task_records = []
                task_keys = []
                while len(task_records) < task_size:
                    try:
                        key, serialized_record = next(source_records)
                    except StopIteration:
                        source_exhausted = True
                        break
                    task_records.append((key, serialized_record))
                    task_keys.append(key)

                if task_records:
                    future = executor.submit(
                        _convert_serialized_batch_in_worker,
                        task_records,
                        str(lmdb_path),
                    )
                    pending.append((future, task_keys))
                    pending_record_count += len(task_records)

            if not pending:
                continue

            future, task_keys = pending.pop(0)
            results = future.result()
            pending_record_count -= len(task_keys)
            if len(results) != len(task_keys):
                raise RuntimeError(
                    "SimNMR worker returned a different number of results than submitted"
                )

            for key, (row, reason, report_record) in zip(task_keys, results):
                if progress is not None:
                    progress.update(1)
                if reason is not None:
                    if rejection_report is None or report_record is None:
                        raise ChemicalMetadataConversionError(reason)
                    key_name = lmdb_key_text(key)
                    rejection_report.write(
                        f"nmrsolver-simnmr-pubchem:{key_name}",
                        report_record,
                        reason,
                    )
                elif row is None:
                    raise RuntimeError(
                        "SimNMR worker returned neither a row nor a rejection reason"
                    )
                else:
                    yield row


def iter_converted_rows(
    input_path: str | Path,
    rejection_report: ChemicalMetadataRejectionReport | None = None,
    *,
    workers: int = 1,
    max_in_flight: int | None = None,
    records_per_task: int = 256,
    quiet: bool = False,
    progress=None,
) -> Iterator[dict[str, object]]:
    """Yield Arrow-ready rows, using serialized worker input when parallel."""

    if workers < 1:
        raise ValueError("workers must be at least 1")
    if records_per_task < 1:
        raise ValueError("records_per_task must be at least 1")
    if max_in_flight is not None and max_in_flight < workers:
        raise ValueError("max_in_flight must be at least workers")

    lmdb_path = resolve_input(input_path)
    if workers > 1:
        yield from _iter_parallel_converted_rows(
            iter(_iter_serialized_lmdb_records(lmdb_path)),
            lmdb_path,
            rejection_report,
            workers,
            max_in_flight or workers * records_per_task * 2,
            records_per_task,
            quiet,
            progress,
        )
        return

    for key, raw_record in iter_lmdb_records(lmdb_path, readahead=True):
        key_name = lmdb_key_text(key)
        record_id = f"nmrsolver-simnmr-pubchem:{key_name}"
        location = f"{lmdb_path} key {key_name}"
        try:
            yield canonical_record_to_row(
                convert_record(raw_record, record_id, location)
            )
        except ChemicalMetadataConversionError as error:
            if rejection_report is None:
                raise
            rejection_report.write(record_id, raw_record, str(error))
        finally:
            if progress is not None:
                progress.update(1)


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
        help=(
            "minimum records between progress refreshes; 0 selects automatic "
            "refreshing (default: 1000000)"
        ),
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
    add_console_arguments(parser)
    return parser.parse_args()


def default_rejection_report_path(output_path: Path) -> Path:
    """Place the chemical-metadata audit beside the canonical Parquet."""

    return output_path.with_name(
        f"{output_path.stem}_chemical_metadata_rejections.jsonl"
    )


def main() -> None:
    args = parse_args()
    configure_console(args.quiet)
    print_stage_start("canonicalize SimNMR-PubChem")
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
        progress = progress_bar(
            _lmdb_entry_count(lmdb_path),
            "Canonicalizing SimNMR",
            not args.no_progress,
            min_iterations=args.progress_every or None,
        )
        try:
            rows = iter_converted_rows(
                lmdb_path,
                rejection_report,
                workers=args.workers,
                max_in_flight=args.max_in_flight,
                records_per_task=args.records_per_task,
                quiet=args.quiet,
                progress=progress,
            )
            record_count = write_canonical_rows(
                rows,
                args.output,
                source_name=SOURCE_NAME,
                converter_name="convert_nmrsolver.py",
                row_group_size=args.row_group_size,
                overwrite=args.overwrite,
            )
        finally:
            if progress is not None:
                progress.close()

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
    print_stage_complete("canonicalize SimNMR-PubChem", report_path)


if __name__ == "__main__":
    main()

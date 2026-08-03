"""Convert SimNMR-PubChem from NMR-Solver into canonical Parquet.

The source LMDB stores one predicted shift per atom. Hydrogen and carbon atoms
are grouped by the supplied ``equi_class`` value, never by shift proximity.
Each equivalence group becomes one canonical resonance at the mean member
shift. Proton groups also retain their original shifts and group size.

The database has no published train, validation, or test split, so one input
produces one physical Parquet file without inventing split labels.

Example:
    PYTHONPATH=scripts python scripts/canonicalize/convert_nmrsolver.py \
        models/NMR-Solver/database/metadata/PubChem_merged_id.lmdb \
        datasets/nmrsolver/all.parquet
"""

import argparse
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from canonicalize.common import (
    ConversionError,
    checked_canonical_record,
    finite_float,
    iter_lmdb_records,
    lmdb_key_text,
    optional_text,
    require_mapping,
    write_canonical_parquet,
)
from data.schema import CanonicalRecord


SOURCE_NAME = "SimNMR-PubChem"
HYDROGEN_ATOMIC_NUMBER = 1
CARBON_ATOMIC_NUMBER = 6


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

    canonical_data: dict[str, object] = {
        "record_id": record_id,
        "source": SOURCE_NAME,
        "smiles": optional_text(record.get("smiles"), f"{location}.smiles"),
        "smiles_canonical": optional_text(
            record.get("canonical_smiles"), f"{location}.canonical_smiles"
        ),
        "molecular_formula": None,
        "nmr_frequency": None,
        "nmr_solvent": None,
        "atoms": None,
        "coordinates": None,
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


def iter_converted_records(input_path: str | Path) -> Iterator[CanonicalRecord]:
    """Yield every LMDB entry exactly once in the stored key order."""

    lmdb_path = resolve_input(input_path)
    for key, raw_record in iter_lmdb_records(lmdb_path, readahead=True):
        key_name = lmdb_key_text(key)
        record_id = f"nmrsolver-simnmr-pubchem:{key_name}"
        location = f"{lmdb_path} key {key_name}"
        yield convert_record(raw_record, record_id, location)


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
        "--overwrite",
        action="store_true",
        help="allow replacement of an existing output file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lmdb_path = resolve_input(args.input)
    if lmdb_path.resolve() == args.output.resolve():
        raise ValueError("output must not overwrite the source dataset")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be zero or greater")

    records = iter_converted_records(lmdb_path)
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
    print(f"wrote {record_count} SimNMR-PubChem records to {args.output}")


if __name__ == "__main__":
    main()

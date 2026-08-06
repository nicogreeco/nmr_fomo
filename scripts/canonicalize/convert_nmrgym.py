#!/usr/bin/env python3
"""Convert NMRGym pickles into one schema-v2 canonical Parquet file.

NMRGym supplies one canonical SMILES and paired lists of 1H and 13C shifts per
molecule. It does not supply integration, multiplicity, ranges, J couplings,
solvent, or frequency for individual peak records. Those unavailable fields
are therefore written as null; this converter does not infer them.

Pass one NMRGym split pickle to make its matching Parquet, or pass the
directory containing train/val/test pickles to make their ordered union.

Examples:
    PYTHONPATH=scripts python dataset_overlap_audit/convert_nmrgym.py \
        dataset_overlap_audit/NMRGym_train_balanced_dedup.pkl \
        datasets/nmrgym/train.parquet

    PYTHONPATH=scripts python dataset_overlap_audit/convert_nmrgym.py \
        dataset_overlap_audit datasets/nmrgym/all.parquet
"""

import argparse
import pickle
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from canonicalize.common import (
    ConversionError,
    chemical_metadata_from_smiles,
    checked_canonical_record,
    finite_float,
    require_mapping,
    write_canonical_parquet,
)
from data.schema import CanonicalRecord


SOURCE_NAME = "NMRGym"
SPLIT_FILENAMES = {
    "train": "NMRGym_train_balanced_dedup.pkl",
    "val": "NMRGym_val_balanced_dedup.pkl",
    "test": "NMRGym_test_balanced_dedup.pkl",
}


def resolve_inputs(input_path: str | Path) -> list[tuple[str, Path]]:
    """Return source split labels and pickle paths in release order."""

    path = Path(input_path)
    if path.is_file():
        for split_name, filename in SPLIT_FILENAMES.items():
            if path.name == filename:
                return [(split_name, path)]
        raise ValueError(
            "an NMRGym pickle must be named one of: "
            + ", ".join(SPLIT_FILENAMES.values())
        )

    if not path.is_dir():
        raise FileNotFoundError(f"NMRGym input not found: {path}")

    inputs = []
    for split_name, filename in SPLIT_FILENAMES.items():
        split_path = path / filename
        if not split_path.is_file():
            raise FileNotFoundError(
                f"NMRGym input directory is missing {split_path}"
            )
        inputs.append((split_name, split_path))
    return inputs


def load_split_records(path: Path) -> list[object]:
    """Load one release pickle; conversion retains only this split at a time."""

    try:
        with path.open("rb") as handle:
            records = pickle.load(handle)
    except Exception as error:
        raise ConversionError(f"could not unpickle {path}") from error

    if not isinstance(records, list):
        raise ConversionError(f"{path} must contain one list of records")
    return records


def shift_list(
    raw_record: Mapping[str, Any], field_name: str, location: str
) -> list[object] | None:
    """Preserve an available shift list, an empty list, or a missing value."""

    if field_name not in raw_record:
        raise ConversionError(f"{location} is missing {field_name}")
    value = raw_record[field_name]
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ConversionError(f"{location}.{field_name} must be an array or None")
    return list(value)


def proton_peak(shift: object, location: str) -> dict[str, object]:
    """Map a shift-only NMRGym proton observation without invented features."""

    return {
        "shift": finite_float(shift, location),
        "integration": None,
        "multiplicity_raw": None,
        "multiplicity": None,
        "j_values": None,
        "range_min": None,
        "range_max": None,
        "range_half_span": None,
        "equivalence_class": None,
        "member_shifts": None,
    }


def carbon_peak(shift: object, location: str) -> dict[str, object]:
    return {
        "shift": finite_float(shift, location),
        "integral": None,
        "intensity": None,
        "width": None,
    }


def convert_record(
    raw_record: object,
    record_id: str,
    location: str,
) -> CanonicalRecord:
    """Map one NMRGym molecule to the shared canonical schema."""

    record = require_mapping(raw_record, location)
    h_shifts = shift_list(record, "h_shift", location)
    c_shifts = shift_list(record, "c_shift", location)
    canonical_data: dict[str, object] = {
        "record_id": record_id,
        "source": SOURCE_NAME,
        **chemical_metadata_from_smiles(record.get("smiles"), f"{location}.smiles"),
        "nmr_frequency": None,
        "nmr_solvent": None,
        "h_nmr_peaks": (
            [
                proton_peak(shift, f"{location}.h_shift[{index}]")
                for index, shift in enumerate(h_shifts)
            ]
            if h_shifts is not None
            else None
        ),
        "c_nmr_peaks": (
            [
                carbon_peak(shift, f"{location}.c_shift[{index}]")
                for index, shift in enumerate(c_shifts)
            ]
            if c_shifts is not None
            else None
        ),
    }
    return checked_canonical_record(canonical_data, location)


def iter_converted_records(input_path: str | Path) -> Iterator[CanonicalRecord]:
    """Yield source rows in train, validation, test order when given a directory."""

    for split_name, split_path in resolve_inputs(input_path):
        raw_records = load_split_records(split_path)
        try:
            for row_index, raw_record in enumerate(raw_records):
                record_id = f"nmrgym:{split_name}:{row_index}"
                location = f"{split_path} row {row_index}"
                yield convert_record(raw_record, record_id, location)
        finally:
            del raw_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="NMRGym split pickle or directory containing all three split pickles",
    )
    parser.add_argument("output", type=Path, help="single canonical .parquet output")
    parser.add_argument(
        "--row-group-size",
        type=int,
        default=50_000,
        help="rows per internal Parquet row group (default: 50000)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacement of an existing output file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input.is_file() and args.input.resolve() == args.output.resolve():
        raise ValueError("output must not overwrite the source dataset")

    record_count = write_canonical_parquet(
        iter_converted_records(args.input),
        args.output,
        source_name=SOURCE_NAME,
        converter_name="dataset_overlap_audit/convert_nmrgym.py",
        row_group_size=args.row_group_size,
        overwrite=args.overwrite,
    )
    print(f"wrote {record_count} NMRGym records to {args.output}")


if __name__ == "__main__":
    main()

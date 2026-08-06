"""Convert NMRPeak's processed NMRexp LMDB into one canonical Parquet file.

The input may be ``all.lmdb``, its ``lmdb_dataset`` directory, or the NMRexp
parent directory. A directory uses train, valid, and test in that order so the
combined output has the same ``record_id`` values as separately converted
splits. Pass ``all.lmdb`` directly only when that source file is specifically
needed. No records are filtered, deduplicated, or split by this script.

Example:
    PYTHONPATH=scripts python scripts/data/canonicalize/convert_nmrexp.py \
        models/NMRPeak/data/NMRexp/lmdb_dataset \
        datasets/canonical/nmrpeak_nmrexp.parquet
"""

import argparse
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from data.canonicalize.common import (
    checked_canonical_record,
    iter_lmdb_records,
    lmdb_key_text,
    nmrpeak_carbon_peak,
    nmrpeak_metadata,
    nmrpeak_proton_peak,
    require_mapping,
    resolve_lmdb_inputs,
    write_canonical_parquet,
)
from data.schema import CanonicalRecord


SOURCE_NAME = "NMRPeak-NMRexp"


def _peak_list(
    raw_record: Mapping[str, Any], field_name: str, location: str
) -> list[object]:
    value = raw_record.get(field_name)
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{location}.{field_name} must be an array or None")
    return list(value)


def convert_record(
    raw_record: object,
    record_id: str,
    location: str,
) -> CanonicalRecord:
    """Map one NMRPeak NMRexp record into the shared canonical schema."""

    record = require_mapping(raw_record, location)
    h_peaks = _peak_list(record, "h_nmr_peaks", location)
    c_peaks = _peak_list(record, "c_nmr_peaks", location)

    canonical_data: dict[str, object] = {
        "record_id": record_id,
        "source": SOURCE_NAME,
        **nmrpeak_metadata(record, location),
        "h_nmr_peaks": [
            nmrpeak_proton_peak(peak, f"{location}.h_nmr_peaks[{index}]")
            for index, peak in enumerate(h_peaks)
        ],
        "c_nmr_peaks": [
            nmrpeak_carbon_peak(
                peak,
                f"{location}.c_nmr_peaks[{index}]",
                keep_properties=False,
            )
            for index, peak in enumerate(c_peaks)
        ],
    }
    return checked_canonical_record(canonical_data, location)


def iter_converted_records(input_path: str | Path) -> Iterator[CanonicalRecord]:
    """Yield every source LMDB record exactly once as a canonical record."""

    for lmdb_path in resolve_lmdb_inputs(input_path, prefer_all_file=False):
        database_name = lmdb_path.stem
        for key, raw_record in iter_lmdb_records(lmdb_path):
            key_name = lmdb_key_text(key)
            record_id = f"nmrpeak-nmrexp:{database_name}:{key_name}"
            location = f"{lmdb_path} key {key_name}"
            yield convert_record(raw_record, record_id, location)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="NMRexp LMDB file or directory")
    parser.add_argument("output", type=Path, help="single canonical .parquet output")
    parser.add_argument(
        "--row-group-size",
        type=int,
        default=50000,
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
        converter_name="convert_nmrexp.py",
        row_group_size=args.row_group_size,
        overwrite=args.overwrite,
    )
    print(f"wrote {record_count} NMRexp records to {args.output}")


if __name__ == "__main__":
    main()

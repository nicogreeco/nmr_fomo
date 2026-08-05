"""Convert NMRTrans/NMRSpec's compressed files into one canonical Parquet file.

The input may be one ``.pkl.lz4`` split or the NMRTrans data directory. A
source directory is read in train, val, test order and written into one
physical Parquet file. No source records are filtered, deduplicated, or kept
as train/validation/test labels.

Example:
    PYTHONPATH=scripts python scripts/canonicalize/convert_nmrtrans.py \
        models/NMRTrans/data \
        datasets/canonical/nmrtrans_nmrspec.parquet
"""

import argparse
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from canonicalize.common import (
    ConversionError,
    chemical_metadata_from_smiles,
    checked_canonical_record,
    finite_float,
    iter_lz4_pickle_records,
    nmrtrans_file_label,
    optional_float,
    optional_text,
    parse_nmrtrans_j_values,
    non_negative_integer,
    require_mapping,
    resolve_nmrtrans_inputs,
    write_canonical_parquet,
)
from data.schema import CanonicalRecord, normalize_multiplicity


SOURCE_NAME = "NMRTrans-NMRSpec"


def _tokenized_spectra(raw_record: Mapping[str, Any], location: str) -> Mapping[str, Any]:
    value = raw_record.get("tokenized_input")
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as error:
            raise ConversionError(f"{location}.tokenized_input is not valid JSON") from error
    elif isinstance(value, Mapping):
        parsed = value
    else:
        raise ConversionError(f"{location}.tokenized_input must be a JSON object")
    return require_mapping(parsed, f"{location}.tokenized_input")


def _peak_list(
    spectra: Mapping[str, Any], field_name: str, location: str
) -> list[object] | None:
    if field_name not in spectra:
        return None
    value = spectra[field_name]
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ConversionError(f"{location}.{field_name} must be an array or None")
    return list(value)


def nmrtrans_proton_peak(raw_peak: object, location: str) -> dict[str, object]:
    """Map NMRTrans's [shift, half-span, multiplicity, integral, J] peak."""

    if not isinstance(raw_peak, (list, tuple)) or len(raw_peak) < 5:
        raise ConversionError(
            f"{location} must contain [shift, half_span, multiplicity, integration, J_values]"
        )

    shift = finite_float(raw_peak[0], f"{location}[0]")
    range_half_span = optional_float(raw_peak[1], f"{location}[1]")
    if range_half_span is not None and range_half_span < 0:
        raise ConversionError(f"{location}[1] must be non-negative")

    if range_half_span is None:
        range_min = None
        range_max = None
    else:
        range_min = shift - range_half_span
        range_max = shift + range_half_span

    multiplicity_raw = optional_text(raw_peak[2], f"{location}[2]")
    return {
        "shift": shift,
        "integration": non_negative_integer(raw_peak[3], f"{location}[3]"),
        "multiplicity_raw": multiplicity_raw,
        "multiplicity": normalize_multiplicity(multiplicity_raw),
        "j_values": parse_nmrtrans_j_values(raw_peak[4], f"{location}[4]"),
        "range_min": range_min,
        "range_max": range_max,
        "range_half_span": range_half_span,
        "equivalence_class": None,
        "member_shifts": None,
    }


def convert_record(
    raw_record: object,
    record_id: str,
    location: str,
) -> CanonicalRecord:
    """Map one NMRTrans/NMRSpec record into the shared canonical schema."""

    record = require_mapping(raw_record, location)
    spectra = _tokenized_spectra(record, location)
    h_peaks = _peak_list(spectra, "1HNMR", f"{location}.tokenized_input")
    c_peaks = _peak_list(spectra, "13CNMR", f"{location}.tokenized_input")

    original_smiles = optional_text(
        record.get("original_smiles"), f"{location}.original_smiles"
    )
    if original_smiles is not None:
        source_smiles = original_smiles
        smiles_location = f"{location}.original_smiles"
    else:
        source_smiles = optional_text(record.get("smiles"), f"{location}.smiles")
        smiles_location = f"{location}.smiles"

    canonical_data: dict[str, object] = {
        "record_id": record_id,
        "source": SOURCE_NAME,
        **chemical_metadata_from_smiles(source_smiles, smiles_location),
        "nmr_frequency": None,
        "nmr_solvent": None,
        "h_nmr_peaks": (
            [
                nmrtrans_proton_peak(
                    peak, f"{location}.tokenized_input.1HNMR[{index}]"
                )
                for index, peak in enumerate(h_peaks)
            ]
            if h_peaks is not None
            else None
        ),
        "c_nmr_peaks": (
            [
                {
                    "shift": finite_float(
                        peak, f"{location}.tokenized_input.13CNMR[{index}]"
                    ),
                    "integral": None,
                    "intensity": None,
                    "width": None,
                }
                for index, peak in enumerate(c_peaks)
            ]
            if c_peaks is not None
            else None
        ),
    }
    return checked_canonical_record(canonical_data, location)


def _source_identifier(raw_record: Mapping[str, Any], row_index: int) -> str:
    raw_identifier = raw_record.get("id")
    if raw_identifier is None:
        return f"row-{row_index}"
    return str(raw_identifier)


def iter_converted_records(input_path: str | Path) -> Iterator[CanonicalRecord]:
    """Yield every NMRTrans release row exactly once as a canonical record."""

    for input_file in resolve_nmrtrans_inputs(input_path):
        file_label = nmrtrans_file_label(input_file)
        for row_index, raw_record in enumerate(iter_lz4_pickle_records(input_file)):
            source_identifier = _source_identifier(raw_record, row_index)
            record_id = (
                f"nmrtrans-nmrspec:{file_label}:{source_identifier}:row-{row_index}"
            )
            location = f"{input_file} row {row_index} (id={source_identifier})"
            yield convert_record(raw_record, record_id, location)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="NMRTrans .pkl.lz4 file or directory")
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
        converter_name="convert_nmrtrans.py",
        row_group_size=args.row_group_size,
        overwrite=args.overwrite,
    )
    print(f"wrote {record_count} NMRTrans/NMRSpec records to {args.output}")


if __name__ == "__main__":
    main()

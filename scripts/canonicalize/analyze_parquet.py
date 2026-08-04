#!/usr/bin/env python3
"""Inspect canonical NMR Parquet files without loading them fully into memory."""

import argparse
from collections import Counter
import json
from pathlib import Path

from canonicalize.common import canonical_parquet_schema
from data.schema import ensure_record
from data.validation import (
    IncompatibleRecordError,
    validate_canonical_record,
)


def update_minimum(current_value: float | None, new_value: float) -> float:
    if current_value is None:
        return new_value
    return min(current_value, new_value)


def update_maximum(current_value: float | None, new_value: float) -> float:
    if current_value is None:
        return new_value
    return max(current_value, new_value)


def decode_schema_metadata(metadata: dict | None) -> dict[str, str]:
    """Convert Arrow's byte-string metadata to printable text."""

    if not metadata:
        return {}

    decoded = {}
    for key, value in metadata.items():
        if isinstance(key, bytes):
            key = key.decode("utf-8", errors="replace")
        else:
            key = str(key)

        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        else:
            value = str(value)
        decoded[key] = value
    return dict(sorted(decoded.items()))


def checked_record(value: object, path: Path, row_number: int):
    """Decode and validate one canonical row, adding file/row context on error."""

    try:
        record = ensure_record(value)
        validation = validate_canonical_record(record)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"could not decode row {row_number} of {path}: {error}"
        ) from error

    if not validation.is_valid:
        error = IncompatibleRecordError(validation.record_id, validation.issues)
        raise ValueError(
            f"invalid canonical row {row_number} of {path}: {error}"
        ) from error
    return record


def analyze_parquet_file(
    path: str | Path,
    arrow_batch_size: int = 2048,
    show_progress: bool = False,
) -> dict[str, object]:
    """Stream one canonical Parquet file and return aggregate statistics."""

    if arrow_batch_size < 1:
        raise ValueError("arrow_batch_size must be at least 1")

    try:
        import pyarrow.parquet as parquet
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "analyzing canonical Parquet requires pyarrow; install it in the "
            "environment used for analysis"
        ) from error

    parquet_path = Path(path)
    if not parquet_path.is_file():
        raise FileNotFoundError(f"canonical Parquet input not found: {parquet_path}")

    parquet_file = parquet.ParquetFile(parquet_path)
    file_metadata = parquet_file.metadata

    progress_bar = None
    if show_progress:
        try:
            from tqdm import tqdm
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "showing analysis progress requires tqdm; install it in the "
                "environment used for analysis"
            ) from error

        progress_bar = tqdm(
            total=file_metadata.num_rows,
            desc=f"Analyzing {parquet_path.name}",
            unit="records",
        )

    rows = 0
    seen_record_ids: set[str] = set()
    duplicate_record_id_count = 0
    source_counts: Counter[str] = Counter()
    multiplicity_counts: Counter[str] = Counter()

    h_null_records = 0
    h_empty_records = 0
    c_null_records = 0
    c_empty_records = 0
    both_modality_usable_records = 0

    total_h_peaks = 0
    maximum_h_peaks = 0
    total_c_peaks = 0
    maximum_c_peaks = 0

    proton_integration_null_count = 0
    proton_integration_zero_count = 0
    j_values_null_count = 0
    j_values_empty_count = 0
    j_values_nonempty_count = 0

    h_shift_min = None
    h_shift_max = None
    c_shift_min = None
    c_shift_max = None

    for arrow_batch in parquet_file.iter_batches(batch_size=arrow_batch_size):
        for value in arrow_batch.to_pylist():
            rows += 1
            record = checked_record(value, parquet_path, rows)

            if record.record_id in seen_record_ids:
                duplicate_record_id_count += 1
            else:
                seen_record_ids.add(record.record_id)

            source_name = record.source if record.source is not None else "<null>"
            source_counts[source_name] += 1

            h_peaks = record.h_nmr_peaks
            c_peaks = record.c_nmr_peaks

            if h_peaks is None:
                h_null_records += 1
            elif len(h_peaks) == 0:
                h_empty_records += 1

            if c_peaks is None:
                c_null_records += 1
            elif len(c_peaks) == 0:
                c_empty_records += 1

            if h_peaks and c_peaks:
                both_modality_usable_records += 1

            h_peak_count = len(h_peaks) if h_peaks is not None else 0
            c_peak_count = len(c_peaks) if c_peaks is not None else 0
            total_h_peaks += h_peak_count
            total_c_peaks += c_peak_count
            maximum_h_peaks = max(maximum_h_peaks, h_peak_count)
            maximum_c_peaks = max(maximum_c_peaks, c_peak_count)

            if h_peaks is not None:
                for peak in h_peaks:
                    h_shift_min = update_minimum(h_shift_min, peak.shift)
                    h_shift_max = update_maximum(h_shift_max, peak.shift)

                    if peak.integration is None:
                        proton_integration_null_count += 1
                    elif peak.integration == 0:
                        proton_integration_zero_count += 1

                    if peak.j_values is None:
                        j_values_null_count += 1
                    elif len(peak.j_values) == 0:
                        j_values_empty_count += 1
                    else:
                        j_values_nonempty_count += 1

                    multiplicity = (
                        peak.multiplicity
                        if peak.multiplicity is not None
                        else "<null>"
                    )
                    multiplicity_counts[multiplicity] += 1

            if c_peaks is not None:
                for peak in c_peaks:
                    c_shift_min = update_minimum(c_shift_min, peak.shift)
                    c_shift_max = update_maximum(c_shift_max, peak.shift)

        if progress_bar is not None:
            progress_bar.update(arrow_batch.num_rows)

    if progress_bar is not None:
        progress_bar.close()

    expected_rows = file_metadata.num_rows
    if rows != expected_rows:
        raise RuntimeError(
            f"read {rows} rows from {parquet_path}, but Parquet metadata reports "
            f"{expected_rows}"
        )

    return {
        "path": str(parquet_path),
        "bytes": parquet_path.stat().st_size,
        "rows": rows,
        "row_groups": file_metadata.num_row_groups,
        "schema_matches_canonical": parquet_file.schema_arrow.remove_metadata().equals(
            canonical_parquet_schema()
        ),
        "schema_metadata": decode_schema_metadata(
            parquet_file.schema_arrow.metadata
        ),
        "duplicate_record_id_count": duplicate_record_id_count,
        "source_values": dict(sorted(source_counts.items())),
        "h_nmr_peaks": {
            "null_record_count": h_null_records,
            "empty_record_count": h_empty_records,
            "total_peak_count": total_h_peaks,
            "maximum_peaks_per_record": maximum_h_peaks,
            "shift_min": h_shift_min,
            "shift_max": h_shift_max,
        },
        "c_nmr_peaks": {
            "null_record_count": c_null_records,
            "empty_record_count": c_empty_records,
            "total_peak_count": total_c_peaks,
            "maximum_peaks_per_record": maximum_c_peaks,
            "shift_min": c_shift_min,
            "shift_max": c_shift_max,
        },
        "both_modality_usable_count": both_modality_usable_records,
        "proton_integration": {
            "null_count": proton_integration_null_count,
            "zero_count": proton_integration_zero_count,
        },
        "proton_j_values": {
            "null_count": j_values_null_count,
            "empty_count": j_values_empty_count,
            "nonempty_count": j_values_nonempty_count,
        },
        "canonical_multiplicity_counts": dict(
            sorted(multiplicity_counts.items())
        ),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream and summarize canonical NMR Parquet files."
    )
    parser.add_argument(
        "parquet_files",
        nargs="+",
        help="one or more canonical Parquet files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional path for the JSON report",
    )
    parser.add_argument(
        "--arrow-batch-size",
        type=int,
        default=2048,
        help="number of rows decoded at once (default: 2048)",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()

    input_paths = [Path(path) for path in args.parquet_files]
    if args.output is not None:
        output_path = args.output.resolve()
        input_paths_resolved = [path.resolve() for path in input_paths]
        if output_path in input_paths_resolved:
            raise ValueError("the JSON output path cannot replace an input file")

    report = {
        "files": [
            analyze_parquet_file(
                path,
                args.arrow_batch_size,
                show_progress=True,
            )
            for path in input_paths
        ]
    }
    report_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(report_text, end="")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_text, encoding="utf-8")


if __name__ == "__main__":
    main()

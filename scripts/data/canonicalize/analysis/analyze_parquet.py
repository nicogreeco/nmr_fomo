#!/usr/bin/env python3
"""Inspect canonical NMR Parquet files without loading them fully into memory."""

import argparse
from collections import Counter
import json
from pathlib import Path

from data.canonicalize.common import canonical_parquet_schema


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


def count_true(values) -> int:
    """Return the number of true values in an Arrow boolean array."""

    import pyarrow.compute as compute

    result = compute.sum(values)
    return 0 if not result.is_valid else int(result.as_py())


def update_counts(counter: Counter[str], values, null_name: str = "<null>") -> None:
    """Add Arrow value counts to a small Python counter.

    This only converts distinct values (for example, source names), not every
    record in the batch.
    """

    import pyarrow.compute as compute

    for item in compute.value_counts(values).to_pylist():
        name = item["values"] if item["values"] is not None else null_name
        counter[str(name)] += int(item["counts"])


def update_minimum(current_value: float | None, values) -> float | None:
    import pyarrow.compute as compute

    result = compute.min(values)
    if not result.is_valid:
        return current_value
    value = float(result.as_py())
    return value if current_value is None else min(current_value, value)


def update_maximum(current_value: float | None, values) -> float | None:
    import pyarrow.compute as compute

    result = compute.max(values)
    if not result.is_valid:
        return current_value
    value = float(result.as_py())
    return value if current_value is None else max(current_value, value)


def analyze_parquet_file(
    path: str | Path,
    arrow_batch_size: int = 50_000,
    show_progress: bool = False,
    check_duplicate_record_ids: bool = False,
    validate_records: bool = False,
) -> dict[str, object]:
    """Stream one canonical Parquet file and return aggregate statistics.

    The default uses Arrow column operations, so memory stays bounded even for
    very large files.  Exact duplicate-ID checking and full Python validation
    are optional because they are intentionally much slower.
    """

    if arrow_batch_size < 1:
        raise ValueError("arrow_batch_size must be at least 1")

    try:
        import pyarrow.parquet as parquet
        import pyarrow.compute as compute
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
    seen_record_ids: set[str] | None = set() if check_duplicate_record_ids else None
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

    columns = ["source", "h_nmr_peaks", "c_nmr_peaks"]
    if check_duplicate_record_ids:
        columns.append("record_id")
    if validate_records:
        columns = None

    for arrow_batch in parquet_file.iter_batches(
        batch_size=arrow_batch_size,
        columns=columns,
        use_threads=True,
    ):
        rows += arrow_batch.num_rows

        if validate_records:
            from data.schema import ensure_record
            from data.validation import IncompatibleRecordError, validate_canonical_record

            first_row = rows - arrow_batch.num_rows + 1
            for row_offset, value in enumerate(arrow_batch.to_pylist()):
                row_number = first_row + row_offset
                try:
                    record = ensure_record(value)
                    validation = validate_canonical_record(record)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"could not decode row {row_number} of {parquet_path}: {error}"
                    ) from error
                if not validation.is_valid:
                    raise ValueError(
                        f"invalid canonical row {row_number} of {parquet_path}: "
                        f"{IncompatibleRecordError(validation.record_id, validation.issues)}"
                    )

        source_values = arrow_batch.column("source")
        h_peaks = arrow_batch.column("h_nmr_peaks")
        c_peaks = arrow_batch.column("c_nmr_peaks")
        h_lengths = compute.list_value_length(h_peaks)
        c_lengths = compute.list_value_length(c_peaks)

        update_counts(source_counts, source_values)
        h_null_records += count_true(compute.is_null(h_peaks))
        h_empty_records += count_true(compute.equal(h_lengths, 0))
        c_null_records += count_true(compute.is_null(c_peaks))
        c_empty_records += count_true(compute.equal(c_lengths, 0))
        both_modality_usable_records += count_true(
            compute.and_(compute.greater(h_lengths, 0), compute.greater(c_lengths, 0))
        )
        total_h_peaks += int(compute.sum(h_lengths).as_py() or 0)
        total_c_peaks += int(compute.sum(c_lengths).as_py() or 0)
        h_length_max = compute.max(h_lengths)
        c_length_max = compute.max(c_lengths)
        if h_length_max.is_valid:
            maximum_h_peaks = max(maximum_h_peaks, int(h_length_max.as_py()))
        if c_length_max.is_valid:
            maximum_c_peaks = max(maximum_c_peaks, int(c_length_max.as_py()))

        flat_h_peaks = compute.list_flatten(h_peaks)
        if len(flat_h_peaks):
            h_shift = compute.struct_field(flat_h_peaks, "shift")
            integration = compute.struct_field(flat_h_peaks, "integration")
            multiplicity = compute.struct_field(flat_h_peaks, "multiplicity")
            j_values = compute.struct_field(flat_h_peaks, "j_values")
            j_lengths = compute.list_value_length(j_values)

            h_shift_min = update_minimum(h_shift_min, h_shift)
            h_shift_max = update_maximum(h_shift_max, h_shift)
            proton_integration_null_count += count_true(compute.is_null(integration))
            proton_integration_zero_count += count_true(compute.equal(integration, 0))
            j_values_null_count += count_true(compute.is_null(j_values))
            j_values_empty_count += count_true(compute.equal(j_lengths, 0))
            j_values_nonempty_count += count_true(compute.greater(j_lengths, 0))
            update_counts(multiplicity_counts, multiplicity)

        flat_c_peaks = compute.list_flatten(c_peaks)
        if len(flat_c_peaks):
            c_shift = compute.struct_field(flat_c_peaks, "shift")
            c_shift_min = update_minimum(c_shift_min, c_shift)
            c_shift_max = update_maximum(c_shift_max, c_shift)

        if seen_record_ids is not None:
            for record_id in arrow_batch.column("record_id").to_pylist():
                if record_id in seen_record_ids:
                    duplicate_record_id_count += 1
                else:
                    seen_record_ids.add(record_id)

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
        "duplicate_record_id_count": (
            duplicate_record_id_count if check_duplicate_record_ids else None
        ),
        "duplicate_record_ids_checked": check_duplicate_record_ids,
        "records_fully_validated": validate_records,
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
        default=50_000,
        help="number of rows read at once (default: 50000)",
    )
    parser.add_argument(
        "--check-duplicate-record-ids",
        action="store_true",
        help="use large memory to count duplicate IDs exactly",
    )
    parser.add_argument(
        "--validate-records",
        action="store_true",
        help="decode every record and run the slow full Python validation",
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
                check_duplicate_record_ids=args.check_duplicate_record_ids,
                validate_records=args.validate_records,
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

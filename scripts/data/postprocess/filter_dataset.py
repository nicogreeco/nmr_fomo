#!/usr/bin/env python3
"""Filter and deduplicate one canonical NMR Parquet dataset.

The input is never modified. By default, ``records.parquet`` produces
``records_cleaned.parquet`` and a small ``records_removed.parquet`` audit.
Duplicate candidates have the same canonical SMILES and the same sorted exact
proton and carbon shift lists. A hash narrows the grouping, but the real keys
are also compared before any row is removed.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import polars as pl
import pyarrow as pa
from pyarrow import compute, parquet

from data.canonicalize.common import (
    CANONICAL_PARQUET_SCHEMA_VERSION,
    canonical_parquet_schema,
    normalize_modality_lists,
)


H_SHIFT_MIN = -5.0
H_SHIFT_MAX = 20.0
C_SHIFT_MIN = -50.0
C_SHIFT_MAX = 300.0
MAX_PEAKS_PER_MODALITY = 60
MAX_J_VALUES_PER_PEAK = 6

FILTER_REASONS = (
    "no_nmr_peaks",
    "non_finite_h_shift",
    "h_shift_out_of_range",
    "non_finite_c_shift",
    "c_shift_out_of_range",
    "too_many_h_peaks",
    "too_many_c_peaks",
    "too_many_j_values",
    "invalid_j_value",
    "non_positive_h_integration",
    "multifragment_smiles",
)

REMOVAL_REPORT_SCHEMA = pa.schema(
    [
        pa.field("record_id", pa.string(), nullable=False),
        pa.field("removal_reason", pa.string(), nullable=False),
        pa.field("duplicate_of", pa.string()),
    ]
)


def default_output_paths(input_path: str | Path) -> tuple[Path, Path]:
    """Return the default cleaned dataset and removal-report paths."""

    path = Path(input_path)
    if path.suffix.lower() != ".parquet":
        raise ValueError(f"input must have a .parquet suffix: {path}")
    cleaned = path.with_name(f"{path.stem}_cleaned.parquet")
    removed = path.with_name(f"{path.stem}_removed.parquet")
    return cleaned, removed


def validate_canonical_input(path: Path) -> parquet.ParquetFile:
    """Open a schema-v2 canonical Parquet file."""

    if not path.is_file():
        raise FileNotFoundError(f"canonical Parquet input not found: {path}")

    parquet_file = parquet.ParquetFile(path)
    expected_schema = canonical_parquet_schema()
    input_schema = parquet_file.schema_arrow
    if not input_schema.remove_metadata().equals(expected_schema):
        raise ValueError(f"{path} does not use the current canonical schema")

    metadata = input_schema.metadata or {}
    schema_version = metadata.get(b"canonical_schema_version", b"").decode(
        "utf-8"
    )
    if schema_version != CANONICAL_PARQUET_SCHEMA_VERSION:
        raise ValueError(
            f"{path} uses canonical schema version {schema_version!r}; "
            f"expected {CANONICAL_PARQUET_SCHEMA_VERSION!r}"
        )
    return parquet_file


def _audit_frame(input_path: Path) -> pl.LazyFrame:
    """Build the compact lazy audit used for filtering and deduplication."""

    h_peaks = pl.col("h_nmr_peaks")
    c_peaks = pl.col("c_nmr_peaks")
    h_peak = pl.element()
    c_peak = pl.element()

    h_shift = h_peak.struct.field("shift")
    c_shift = c_peak.struct.field("shift")
    integration = h_peak.struct.field("integration")
    multiplicity = h_peak.struct.field("multiplicity")
    j_values = h_peak.struct.field("j_values")
    range_known = h_peak.struct.field("range_half_span").is_not_null() | (
        h_peak.struct.field("range_min").is_not_null()
        & h_peak.struct.field("range_max").is_not_null()
    )

    annotation_count = h_peaks.list.eval(
        integration.is_not_null().cast(pl.Int64)
        + multiplicity.is_not_null().cast(pl.Int64)
        + j_values.is_not_null().cast(pl.Int64)
        + range_known.cast(pl.Int64)
    ).list.sum().fill_null(0)

    missing_or_unknown = h_peaks.list.eval(
        integration.is_null().cast(pl.Int64)
        + (multiplicity.is_null() | (multiplicity == "<unk>")).cast(pl.Int64)
        + j_values.is_null().cast(pl.Int64)
        + (~range_known).cast(pl.Int64)
    ).list.sum().fill_null(0)

    audit = (
        pl.scan_parquet(input_path)
        .select(
            "record_id",
            "smiles_canonical",
            "h_nmr_peaks",
            "c_nmr_peaks",
        )
        .with_columns(
            h_peaks.list.len().fill_null(0).alias("_h_peak_count"),
            c_peaks.list.len().fill_null(0).alias("_c_peak_count"),
            h_peaks.list.eval(h_shift).list.sort().fill_null([]).alias("_h_shifts"),
            c_peaks.list.eval(c_shift).list.sort().fill_null([]).alias("_c_shifts"),
            annotation_count.alias("_annotation_count"),
            missing_or_unknown.alias("_missing_or_unknown"),
        )
        .with_columns(
            (
                (pl.col("_h_peak_count") == 0)
                & (pl.col("_c_peak_count") == 0)
            ).alias("no_nmr_peaks"),
            h_peaks.list.eval(~h_shift.is_finite())
            .list.any()
            .fill_null(False)
            .alias("non_finite_h_shift"),
            h_peaks.list.eval(
                h_shift.is_finite()
                & ((h_shift < H_SHIFT_MIN) | (h_shift > H_SHIFT_MAX))
            )
            .list.any()
            .fill_null(False)
            .alias("h_shift_out_of_range"),
            c_peaks.list.eval(~c_shift.is_finite())
            .list.any()
            .fill_null(False)
            .alias("non_finite_c_shift"),
            c_peaks.list.eval(
                c_shift.is_finite()
                & ((c_shift < C_SHIFT_MIN) | (c_shift > C_SHIFT_MAX))
            )
            .list.any()
            .fill_null(False)
            .alias("c_shift_out_of_range"),
            (pl.col("_h_peak_count") > MAX_PEAKS_PER_MODALITY).alias(
                "too_many_h_peaks"
            ),
            (pl.col("_c_peak_count") > MAX_PEAKS_PER_MODALITY).alias(
                "too_many_c_peaks"
            ),
            h_peaks.list.eval(
                j_values.list.len().fill_null(0) > MAX_J_VALUES_PER_PEAK
            )
            .list.any()
            .fill_null(False)
            .alias("too_many_j_values"),
            h_peaks.list.eval(
                j_values.list.eval(
                    (~pl.element().is_finite()) | (pl.element() < 0)
                )
                .list.any()
                .fill_null(False)
            )
            .list.any()
            .fill_null(False)
            .alias("invalid_j_value"),
            h_peaks.list.eval(
                integration.is_not_null() & (integration <= 0)
            )
            .list.any()
            .fill_null(False)
            .alias("non_positive_h_integration"),
            pl.col("smiles_canonical")
            .str.contains(".", literal=True)
            .fill_null(False)
            .alias("multifragment_smiles"),
        )
        .with_columns(
            pl.any_horizontal(FILTER_REASONS).alias("_fails_filter"),
            pl.when(
                pl.col("smiles_canonical").is_not_null()
                & (pl.col("smiles_canonical").str.len_chars() > 0)
            )
            .then(pl.col("smiles_canonical"))
            .otherwise(pl.concat_str(pl.lit("__missing_smiles__:"), "record_id"))
            .alias("_molecule_key"),
        )
        .with_columns(
            pl.struct("_molecule_key", "_h_shifts", "_c_shifts")
            .hash(seed=0)
            .alias("_signature_hash")
        )
    )
    return audit


def find_removed_records(input_path: str | Path) -> list[dict[str, str | None]]:
    """Return record IDs, reasons, and duplicate winners without writing files."""

    path = Path(input_path)
    validate_canonical_input(path)
    audit = _audit_frame(path)

    failed = (
        audit.filter("_fails_filter")
        .select("record_id", *FILTER_REASONS)
        .collect(engine="streaming")
    )

    removed: list[dict[str, str | None]] = []
    for row in failed.iter_rows(named=True):
        reasons = [reason for reason in FILTER_REASONS if row[reason]]
        removed.append(
            {
                "record_id": row["record_id"],
                "removal_reason": ";".join(reasons),
                "duplicate_of": None,
            }
        )

    duplicate_hashes = (
        # This cheap first pass makes exact duplicate verification feasible for
        # very large inputs: only repeated hashes reach the full-key grouping.
        audit.filter(~pl.col("_fails_filter"))
        .group_by("_signature_hash")
        .len()
        .filter(pl.col("len") > 1)
        .select("_signature_hash")
        .collect(engine="streaming")
    )

    if duplicate_hashes.height > 0:
        candidate_hashes = duplicate_hashes.get_column("_signature_hash")
        signature_columns = [
            "_signature_hash",
            "_molecule_key",
            "_h_shifts",
            "_c_shifts",
        ]
        duplicate_rows = (
            audit.filter(
                ~pl.col("_fails_filter")
                & pl.col("_signature_hash").is_in(candidate_hashes.implode())
            )
            .select(
                "record_id",
                *signature_columns,
                "_annotation_count",
                "_missing_or_unknown",
            )
            .group_by(signature_columns)
            .agg(
                pl.col("record_id")
                .sort_by(
                    "_annotation_count",
                    "_missing_or_unknown",
                    "record_id",
                    descending=[True, False, False],
                )
                .alias("_ranked_record_ids"),
                pl.len().alias("_group_size"),
            )
            .filter(pl.col("_group_size") > 1)
            .select(
                pl.col("_ranked_record_ids").list.first().alias("duplicate_of"),
                pl.col("_ranked_record_ids").list.slice(1).alias("record_id"),
            )
            .explode("record_id", empty_as_null=False)
            .collect(engine="streaming")
        )

        for row in duplicate_rows.iter_rows(named=True):
            removed.append(
                {
                    "record_id": row["record_id"],
                    "removal_reason": "duplicate_shift_signature",
                    "duplicate_of": row["duplicate_of"],
                }
            )

    removed.sort(key=lambda row: str(row["record_id"]))
    return removed


def _write_outputs(
    input_file: parquet.ParquetFile,
    input_path: Path,
    cleaned_path: Path,
    removed_path: Path,
    removed_rows: list[dict[str, str | None]],
    batch_size: int,
) -> int:
    """Write both outputs to temporary files and promote them on success."""

    cleaned_temporary = cleaned_path.with_name(f".{cleaned_path.name}.partial")
    removed_temporary = removed_path.with_name(f".{removed_path.name}.partial")
    for temporary in (cleaned_temporary, removed_temporary):
        if temporary.exists():
            raise FileExistsError(
                f"temporary output already exists: {temporary}; inspect or remove it "
                "before starting another run"
            )

    removed_ids = {str(row["record_id"]) for row in removed_rows}
    if len(removed_ids) != len(removed_rows):
        raise RuntimeError("the removal decisions contain duplicate record IDs")
    reason_counts = Counter()
    for row in removed_rows:
        reason_counts.update(str(row["removal_reason"]).split(";"))

    metadata = dict(input_file.schema_arrow.metadata or {})
    metadata.update(
        {
            b"cleaning_policy_version": b"1",
            b"cleaned_from": input_path.name.encode("utf-8"),
            b"cleaning_script": str(Path(__file__).resolve()).encode("utf-8"),
            b"cleaning_removed_record_count": str(len(removed_ids)).encode("utf-8"),
            b"cleaning_reason_counts": json.dumps(
                dict(sorted(reason_counts.items())), sort_keys=True
            ).encode("utf-8"),
            b"duplicate_identity": (
                b"exact canonical SMILES + sorted exact 1H shifts + sorted exact "
                b"13C shifts"
            ),
        }
    )
    output_schema = input_file.schema_arrow.with_metadata(metadata)

    report_table = pa.Table.from_pylist(removed_rows, schema=REMOVAL_REPORT_SCHEMA)
    value_set = pa.array(sorted(removed_ids), type=pa.string())
    rows_written = 0
    writer = None

    try:
        parquet.write_table(report_table, removed_temporary, compression="zstd")
        writer = parquet.ParquetWriter(
            cleaned_temporary,
            output_schema,
            compression="zstd",
        )
        for batch in input_file.iter_batches(batch_size=batch_size, use_threads=True):
            table = pa.Table.from_batches([batch], schema=output_schema)
            remove_mask = compute.is_in(table["record_id"], value_set=value_set)
            filtered = table.filter(compute.invert(remove_mask))
            if filtered.num_rows == 0:
                continue
            filtered = normalize_modality_lists(filtered)
            writer.write_table(filtered, row_group_size=filtered.num_rows)
            rows_written += filtered.num_rows
        writer.close()
        writer = None

        expected_rows = input_file.metadata.num_rows - len(removed_ids)
        if rows_written != expected_rows:
            raise RuntimeError(
                "cleaned row count does not match input rows minus removed records"
            )

        removed_temporary.replace(removed_path)
        cleaned_temporary.replace(cleaned_path)
    except Exception:
        if writer is not None:
            writer.close()
        cleaned_temporary.unlink(missing_ok=True)
        removed_temporary.unlink(missing_ok=True)
        raise

    return rows_written


def clean_parquet(
    input_path: str | Path,
    *,
    batch_size: int = 50_000,
    overwrite: bool = False,
) -> dict[str, object]:
    """Create a cleaned canonical Parquet and its minimal removal report."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    input_file_path = Path(input_path)
    input_file = validate_canonical_input(input_file_path)
    cleaned_path, removed_path = default_output_paths(input_file_path)

    for output_path in (cleaned_path, removed_path):
        if output_path.resolve() == input_file_path.resolve():
            raise ValueError("output must not replace the input Parquet file")
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"refusing to overwrite {output_path}; pass --overwrite to replace it"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)

    removed_rows = find_removed_records(input_file_path)
    rows_written = _write_outputs(
        input_file,
        input_file_path,
        cleaned_path,
        removed_path,
        removed_rows,
        batch_size,
    )

    input_rows = input_file.metadata.num_rows
    if rows_written != input_rows - len(removed_rows):
        raise RuntimeError(
            "cleaned row count does not match input rows minus removed records"
        )

    reason_counts = Counter()
    for row in removed_rows:
        reason_counts.update(str(row["removal_reason"]).split(";"))

    return {
        "input": str(input_file_path),
        "output": str(cleaned_path),
        "removed_output": str(removed_path),
        "input_rows": input_rows,
        "output_rows": rows_written,
        "removed_rows": len(removed_rows),
        "reason_counts": dict(sorted(reason_counts.items())),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="input canonical Parquet file")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
        help="Parquet rows copied at once (default: 50000)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing cleaned and removal-report files",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    result = clean_parquet(
        args.input,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    reasons = ", ".join(
        f"{reason}={count:,}"
        for reason, count in result["reason_counts"].items()
    )
    print(
        f"Wrote {result['output_rows']:,} of {result['input_rows']:,} records "
        f"to {result['output']}"
    )
    print(
        f"Wrote {result['removed_rows']:,} removed record IDs to "
        f"{result['removed_output']} ({reasons})"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Merge compatible canonical Parquet files in the requested order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
from pyarrow import parquet

from data.canonicalize.common import normalize_modality_lists
from data.postprocess.common import (
    open_compatible_parquets,
    prepare_parquet_output,
    progress_bar,
)


SCRIPT_PATH = "scripts/data/postprocess/merge_datasets.py"


def merge_datasets(
    input_paths: list[str | Path],
    output_path: str | Path,
    *,
    batch_size: int = 50_000,
    overwrite: bool = False,
    show_progress: bool = True,
) -> dict[str, object]:
    """Copy canonical input files into one output using bounded batches."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    paths = [Path(path) for path in input_paths]
    if not paths:
        raise ValueError("at least one input Parquet file is required")

    output, temporary = prepare_parquet_output(
        output_path,
        paths,
        overwrite=overwrite,
    )
    parquet_files = open_compatible_parquets(paths)
    reference_schema = parquet_files[0].schema_arrow
    metadata = dict(reference_schema.metadata or {})
    metadata.update(
        {
            b"postprocess_step": b"merge_datasets",
            b"postprocess_script": SCRIPT_PATH.encode("utf-8"),
            b"postprocess_inputs": json.dumps(
                [str(path) for path in paths]
            ).encode("utf-8"),
        }
    )
    output_schema = reference_schema.with_metadata(metadata)

    total_input_rows = sum(file.metadata.num_rows for file in parquet_files)
    progress = progress_bar(total_input_rows, f"Writing {output.name}", show_progress)
    writer = parquet.ParquetWriter(temporary, output_schema, compression="zstd")
    rows_written = 0
    row_groups_written = 0
    try:
        for parquet_file in parquet_files:
            for batch in parquet_file.iter_batches(
                batch_size=batch_size,
                use_threads=True,
            ):
                table = pa.Table.from_batches([batch], schema=output_schema)
                table = normalize_modality_lists(table)
                writer.write_table(table, row_group_size=table.num_rows)
                rows_written += table.num_rows
                row_groups_written += 1
                if progress is not None:
                    progress.update(batch.num_rows)
    except Exception:
        writer.close()
        temporary.unlink(missing_ok=True)
        raise
    else:
        writer.close()
    finally:
        if progress is not None:
            progress.close()

    if rows_written != total_input_rows:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("merged row count does not equal the sum of input rows")
    temporary.replace(output)

    return {
        "inputs": [str(path) for path in paths],
        "output": str(output),
        "input_files": len(paths),
        "rows": rows_written,
        "row_groups": row_groups_written,
        "bytes": output.stat().st_size,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="canonical Parquet files in merge order",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    result = merge_datasets(
        args.inputs,
        args.output,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
        show_progress=not args.no_progress,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

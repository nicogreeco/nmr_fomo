#!/usr/bin/env python3
"""Merge canonical Parquet files into one canonical Parquet file."""

import argparse
import json
from pathlib import Path

from canonicalize.common import (
    CANONICAL_PARQUET_SCHEMA_VERSION,
    canonical_parquet_schema,
)


def merge_datasets(
    parquet_file_paths: list[Path],
    output_path: str | Path,
    arrow_batch_size: int = 50_000,
    overwrite: bool = False,
) -> dict[str, object]:
    """Copy compatible Parquet files into one output file in bounded batches."""

    if not parquet_file_paths:
        raise ValueError("at least one input Parquet file is required")
    if arrow_batch_size < 1:
        raise ValueError("arrow_batch_size must be at least 1")

    input_paths = [Path(path) for path in parquet_file_paths]
    missing_paths = [path for path in input_paths if not path.is_file()]
    if missing_paths:
        missing_text = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"input Parquet file not found: {missing_text}")

    output = Path(output_path)
    output_resolved = output.resolve()
    if any(path.resolve() == output_resolved for path in input_paths):
        raise ValueError("output must not replace one of the input files")
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite {output}; pass --overwrite to replace it"
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        import pyarrow as pa
        import pyarrow.parquet as parquet
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "merging canonical Parquet files requires pyarrow; install it in "
            "the environment used for postprocessing"
        ) from error

    parquet_files = [parquet.ParquetFile(path) for path in input_paths]
    reference_schema = parquet_files[0].schema_arrow
    expected_schema = canonical_parquet_schema()
    rdkit_versions: set[str] = set()

    for path, parquet_file in zip(input_paths, parquet_files):
        input_schema = parquet_file.schema_arrow
        if not input_schema.remove_metadata().equals(expected_schema):
            raise ValueError(
                f"input does not use the current canonical schema: {path}"
            )

        input_metadata = input_schema.metadata or {}
        schema_version = input_metadata.get(b"canonical_schema_version", b"").decode(
            "utf-8"
        )
        if schema_version != CANONICAL_PARQUET_SCHEMA_VERSION:
            raise ValueError(
                f"{path} uses canonical schema version {schema_version!r}; "
                f"expected {CANONICAL_PARQUET_SCHEMA_VERSION!r}"
            )

        rdkit_version = input_metadata.get(b"rdkit_version", b"").decode("utf-8")
        if not rdkit_version:
            raise ValueError(f"{path} has no rdkit_version metadata")
        rdkit_versions.add(rdkit_version)

    if len(rdkit_versions) != 1:
        versions_text = ", ".join(sorted(rdkit_versions))
        raise ValueError(f"input RDKit versions differ: {versions_text}")
    rdkit_version = next(iter(rdkit_versions))

    source_datasets = [
        f"{path.parent.name}/{path.name}" for path in input_paths
    ]
    metadata = {
        b"canonical_schema_version": CANONICAL_PARQUET_SCHEMA_VERSION.encode(
            "utf-8"
        ),
        b"source_datasets": json.dumps(source_datasets).encode("utf-8"),
        b"merger_script": str(Path(__file__).resolve()).encode("utf-8"),
        b"rdkit_version": rdkit_version.encode("utf-8"),
    }
    output_schema = reference_schema.with_metadata(metadata)

    temporary_path = output.with_name(f".{output.name}.partial")
    if temporary_path.exists():
        raise FileExistsError(
            f"temporary output already exists: {temporary_path}; remove it "
            "after checking the interrupted merge"
        )

    total_rows = 0
    total_row_groups = 0
    writer = parquet.ParquetWriter(
        temporary_path,
        output_schema,
        compression="zstd",
    )

    try:
        for parquet_file in parquet_files:
            for batch in parquet_file.iter_batches(
                batch_size=arrow_batch_size,
                use_threads=True,
            ):
                table = pa.Table.from_batches([batch], schema=output_schema)
                writer.write_table(table, row_group_size=batch.num_rows)
                total_rows += batch.num_rows
                total_row_groups += 1
    except Exception:
        writer.close()
        raise

    writer.close()
    temporary_path.replace(output)

    return {
        "output": str(output),
        "input_files": source_datasets,
        "rows": total_rows,
        "row_groups": total_row_groups,
        "bytes": output.stat().st_size,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge compatible canonical Parquet files."
    )
    parser.add_argument(
        "parquet_files",
        nargs="+",
        type=Path,
        help="input canonical Parquet files, in the order to merge them",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="output canonical Parquet file",
    )
    parser.add_argument(
        "--arrow-batch-size",
        type=int,
        default=50_000,
        help="rows copied at once (default: 50000)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output file",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    result = merge_datasets(
        args.parquet_files,
        args.output,
        arrow_batch_size=args.arrow_batch_size,
        overwrite=args.overwrite,
    )
    print(
        f"merged {result['rows']:,} rows from "
        f"{len(result['input_files'])} files into {result['output']}"
    )


if __name__ == "__main__":
    main()

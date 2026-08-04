#!/usr/bin/env python3
"""..."""

import argparse
import json
from pathlib import Path

from canonicalize.common import CANONICAL_PARQUET_SCHEMA_VERSION


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

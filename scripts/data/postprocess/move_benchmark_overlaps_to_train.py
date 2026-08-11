#!/usr/bin/env python3
"""Move benchmark records matching a reference dataset into train/validation.

The reference dataset is streamed using exact ``smiles_canonical`` equality.
Matching benchmark records are removed from the test output and appended,
unchanged, to the train/validation output. Only the benchmark molecular index
is retained in memory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
from pyarrow import compute, parquet

from data.canonicalize.common import normalize_modality_lists
from data.postprocess.common import (
    build_exact_smiles_index,
    find_exact_smiles_matches,
    open_compatible_parquets,
    prepare_parquet_output,
    progress_bar,
    record_ids_for_smiles,
    write_filtered_parquet,
)
from data.reporting import (
    default_report_path,
    prepare_report_output,
    write_processing_report,
)


SCRIPT_PATH = "scripts/data/postprocess/move_benchmark_overlaps_to_train.py"
IDENTITY_DESCRIPTION = "exact smiles_canonical equality"


def write_extended_train(
    train_file: parquet.ParquetFile,
    benchmark_file: parquet.ParquetFile,
    output_schema: pa.Schema,
    temporary_path: Path,
    record_ids_to_append: set[str],
    *,
    batch_size: int,
    show_progress: bool,
) -> tuple[int, int]:
    """Copy train rows, then append the selected benchmark rows."""

    value_set = pa.array(
        sorted(record_ids_to_append),
        type=output_schema.field("record_id").type,
    )
    writer = parquet.ParquetWriter(
        temporary_path,
        output_schema,
        compression="zstd",
    )
    progress = progress_bar(
        train_file.metadata.num_rows + benchmark_file.metadata.num_rows,
        "Writing extended train/validation",
        show_progress,
    )
    base_rows = 0
    appended_rows = 0

    try:
        for batch in train_file.iter_batches(batch_size=batch_size, use_threads=True):
            table = pa.Table.from_batches([batch], schema=output_schema)
            duplicate_ids = compute.is_in(table["record_id"], value_set=value_set)
            if compute.any(duplicate_ids).as_py():
                raise ValueError(
                    "a benchmark record_id selected for transfer already exists "
                    "in train/validation"
                )
            table = normalize_modality_lists(table)
            writer.write_table(table, row_group_size=table.num_rows)
            base_rows += table.num_rows
            if progress is not None:
                progress.update(batch.num_rows)

        for batch in benchmark_file.iter_batches(
            batch_size=batch_size,
            use_threads=True,
        ):
            table = pa.Table.from_batches([batch], schema=output_schema)
            selected = table.filter(
                compute.is_in(table["record_id"], value_set=value_set)
            )
            selected = normalize_modality_lists(selected)
            if selected.num_rows:
                writer.write_table(selected, row_group_size=selected.num_rows)
                appended_rows += selected.num_rows
            if progress is not None:
                progress.update(batch.num_rows)
    except Exception:
        writer.close()
        temporary_path.unlink(missing_ok=True)
        raise
    else:
        writer.close()
    finally:
        if progress is not None:
            progress.close()
    return base_rows, appended_rows


def move_benchmark_overlaps_to_train(
    benchmark_path: str | Path,
    reference_path: str | Path,
    train_path: str | Path,
    test_output_path: str | Path,
    train_output_path: str | Path,
    *,
    batch_size: int = 50_000,
    overwrite: bool = False,
    show_progress: bool = True,
) -> dict[str, object]:
    """Move benchmark rows whose canonical SMILES occurs in the reference."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    benchmark = Path(benchmark_path)
    reference = Path(reference_path)
    train = Path(train_path)
    input_paths = [benchmark, reference, train]
    test_output, test_temporary = prepare_parquet_output(
        test_output_path,
        input_paths,
        overwrite=overwrite,
    )
    train_output, train_temporary = prepare_parquet_output(
        train_output_path,
        input_paths,
        overwrite=overwrite,
    )
    if test_output.resolve() == train_output.resolve():
        raise ValueError("test and train outputs must be different files")

    benchmark_file, reference_file, train_file = open_compatible_parquets(
        input_paths
    )
    benchmark_index = build_exact_smiles_index(
        benchmark_file,
        batch_size=batch_size,
        source_name=benchmark.name,
        show_progress=show_progress,
    )
    matching_smiles = find_exact_smiles_matches(
        reference_file,
        set(benchmark_index),
        batch_size=batch_size,
        source_name=reference.name,
        show_progress=show_progress,
    )
    record_ids_to_move = record_ids_for_smiles(
        benchmark_index,
        matching_smiles,
    )

    common_metadata = {
        b"postprocess_step": b"move_benchmark_overlaps_to_train",
        b"postprocess_script": SCRIPT_PATH.encode("utf-8"),
        b"postprocess_inputs": json.dumps(
            [str(path) for path in input_paths]
        ).encode("utf-8"),
        b"molecule_identity": IDENTITY_DESCRIPTION.encode("utf-8"),
        b"moved_molecule_count": str(len(matching_smiles)).encode("utf-8"),
        b"moved_record_count": str(len(record_ids_to_move)).encode("utf-8"),
    }
    test_metadata = dict(benchmark_file.schema_arrow.metadata or {})
    test_metadata.update(common_metadata)
    test_metadata[b"postprocess_output_role"] = b"benchmark_without_matches"
    test_schema = benchmark_file.schema_arrow.with_metadata(test_metadata)

    train_metadata = dict(train_file.schema_arrow.metadata or {})
    train_metadata.update(common_metadata)
    train_metadata[b"postprocess_output_role"] = b"extended_train_validation"
    train_schema = train_file.schema_arrow.with_metadata(train_metadata)

    try:
        test_rows = write_filtered_parquet(
            benchmark_file,
            test_schema,
            test_temporary,
            record_ids_to_move,
            keep=False,
            batch_size=batch_size,
            description="Writing benchmark without transferred records",
            show_progress=show_progress,
        )
        base_train_rows, appended_rows = write_extended_train(
            train_file,
            benchmark_file,
            train_schema,
            train_temporary,
            record_ids_to_move,
            batch_size=batch_size,
            show_progress=show_progress,
        )
        if appended_rows != len(record_ids_to_move):
            raise RuntimeError(
                f"selected {len(record_ids_to_move):,} benchmark records but "
                f"appended {appended_rows:,}"
            )
        if test_rows != benchmark_file.metadata.num_rows - appended_rows:
            raise RuntimeError("benchmark output row count is inconsistent")

        train_temporary.replace(train_output)
        test_temporary.replace(test_output)
    except Exception:
        train_temporary.unlink(missing_ok=True)
        test_temporary.unlink(missing_ok=True)
        raise

    return {
        "stage": "move_benchmark_overlaps_to_train",
        "inputs": {
            "benchmark": str(benchmark),
            "reference": str(reference),
            "train_val": str(train),
        },
        "outputs": {
            "benchmark": str(test_output),
            "train_val": str(train_output),
        },
        "counts": {
            "benchmark_input_records": benchmark_file.metadata.num_rows,
            "benchmark_output_records": test_rows,
            "train_input_records": base_train_rows,
            "moved_records": appended_rows,
            "moved_molecules": len(matching_smiles),
            "train_output_records": base_train_rows + appended_rows,
        },
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("train", type=Path)
    parser.add_argument("--test-output", required=True, type=Path)
    parser.add_argument("--train-output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument(
        "--report-output",
        type=Path,
        help="processing JSON (default: beside --test-output)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    report_path, report_temporary = prepare_report_output(
        args.report_output or default_report_path(args.test_output),
        [
            args.benchmark,
            args.reference,
            args.train,
            args.test_output,
            args.train_output,
        ],
        overwrite=args.overwrite,
    )
    result = move_benchmark_overlaps_to_train(
        args.benchmark,
        args.reference,
        args.train,
        args.test_output,
        args.train_output,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
        show_progress=not args.no_progress,
    )
    write_processing_report(result, report_path, report_temporary)
    print(f"Wrote processing report to {report_path}")


if __name__ == "__main__":
    main()

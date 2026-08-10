#!/usr/bin/env python3
"""Create an exact-SMILES double-disjoint test and an extended train pool.

The first comparison removes benchmark molecules already present in the rich
train/validation reference. The second comparison scans only the remaining
benchmark SMILES against NMR-Solver. Those NMR-Solver-only benchmark records
are removed from the test and appended to an already ADMET-disjoint rich
train/validation file.

All comparisons use exact equality of the existing ``smiles_canonical``
strings. The comparison files are streamed one column at a time; only the
benchmark index is retained in memory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
from pyarrow import compute, parquet

from data.canonicalize.common import (
    CANONICAL_PARQUET_SCHEMA_VERSION,
    canonical_parquet_schema,
    normalize_modality_lists,
)


def make_progress_bar(
    total: int,
    description: str,
    show_progress: bool,
):
    if not show_progress:
        return None

    try:
        from tqdm import tqdm
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "showing preparation progress requires tqdm; install it in the "
            "environment used for post-processing"
        ) from error

    return tqdm(total=total, desc=description, unit="records")


def validate_inputs(paths: list[Path]) -> list[parquet.ParquetFile]:
    expected_schema = canonical_parquet_schema()
    rdkit_versions: set[str] = set()
    parquet_files = []

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"canonical Parquet input not found: {path}")

        parquet_file = parquet.ParquetFile(path)
        input_schema = parquet_file.schema_arrow
        if not input_schema.remove_metadata().equals(expected_schema):
            raise ValueError(f"input does not use the current canonical schema: {path}")

        metadata = input_schema.metadata or {}
        schema_version = metadata.get(b"canonical_schema_version", b"").decode(
            "utf-8"
        )
        if schema_version != CANONICAL_PARQUET_SCHEMA_VERSION:
            raise ValueError(
                f"{path} uses canonical schema version {schema_version!r}; "
                f"expected {CANONICAL_PARQUET_SCHEMA_VERSION!r}"
            )

        rdkit_version = metadata.get(b"rdkit_version", b"").decode("utf-8")
        if not rdkit_version:
            raise ValueError(f"{path} has no rdkit_version metadata")
        rdkit_versions.add(rdkit_version)
        parquet_files.append(parquet_file)

    if len(rdkit_versions) != 1:
        versions_text = ", ".join(sorted(rdkit_versions))
        raise ValueError(f"input RDKit versions differ: {versions_text}")

    return parquet_files


def build_benchmark_index(
    parquet_file: parquet.ParquetFile,
    batch_size: int,
    source_name: str,
    show_progress: bool,
) -> dict[str, list[str]]:
    smiles_index: dict[str, list[str]] = {}
    missing_smiles = 0
    progress_bar = make_progress_bar(
        parquet_file.metadata.num_rows,
        f"Indexing {source_name}",
        show_progress,
    )

    try:
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=["smiles_canonical", "record_id"],
        ):
            smiles_values = batch.column("smiles_canonical").to_pylist()
            record_ids = batch.column("record_id").to_pylist()
            for smiles, record_id in zip(smiles_values, record_ids):
                if not isinstance(smiles, str) or not smiles:
                    missing_smiles += 1
                    continue
                smiles_index.setdefault(smiles, []).append(record_id)

            if progress_bar is not None:
                progress_bar.update(batch.num_rows)
    finally:
        if progress_bar is not None:
            progress_bar.close()

    print(
        f"Indexed {len(smiles_index):,} canonical SMILES from "
        f"{parquet_file.metadata.num_rows:,} records in {source_name}; "
        f"{missing_smiles:,} rows had no canonical SMILES",
        flush=True,
    )
    return smiles_index


def find_exact_smiles_matches(
    parquet_file: parquet.ParquetFile,
    candidate_smiles: set[str],
    batch_size: int,
    source_name: str,
    show_progress: bool,
) -> set[str]:
    unmatched_smiles = set(candidate_smiles)
    value_set = pa.array(
        sorted(unmatched_smiles),
        type=parquet_file.schema_arrow.field("smiles_canonical").type,
    )
    processed_rows = 0
    progress_bar = make_progress_bar(
        parquet_file.metadata.num_rows,
        f"Scanning {source_name}",
        show_progress,
    )

    try:
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=["smiles_canonical"],
        ):
            smiles_column = batch.column("smiles_canonical")
            match_mask = compute.is_in(smiles_column, value_set=value_set)
            matched_values = compute.filter(smiles_column, match_mask).to_pylist()
            unmatched_smiles.difference_update(matched_values)
            processed_rows += batch.num_rows

            if progress_bar is not None:
                progress_bar.update(batch.num_rows)
            if not unmatched_smiles:
                break
    finally:
        if progress_bar is not None:
            progress_bar.close()

    matched_smiles = candidate_smiles - unmatched_smiles
    print(
        f"Scanned {processed_rows:,} records in {source_name}; matched "
        f"{len(matched_smiles):,} canonical SMILES",
        flush=True,
    )
    return matched_smiles


def prepare_output_path(path: Path, inputs: list[Path], overwrite: bool) -> Path:
    if path.suffix.lower() != ".parquet":
        raise ValueError(f"output must have a .parquet suffix: {path}")
    if any(path.resolve() == input_path.resolve() for input_path in inputs):
        raise ValueError(f"output must not replace an input file: {path}")
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.partial")
    if temporary_path.exists():
        raise FileExistsError(
            f"temporary output already exists: {temporary_path}; inspect or "
            "remove it before starting another run"
        )
    return temporary_path


def record_ids_for_smiles(
    smiles_index: dict[str, list[str]],
    selected_smiles: set[str],
) -> set[str]:
    return {
        record_id
        for smiles in selected_smiles
        for record_id in smiles_index[smiles]
    }


def write_filtered_benchmark(
    benchmark_file: parquet.ParquetFile,
    output_schema: pa.Schema,
    temporary_path: Path,
    record_ids_to_remove: set[str],
    batch_size: int,
    show_progress: bool,
) -> int:
    value_set = pa.array(
        sorted(record_ids_to_remove),
        type=output_schema.field("record_id").type,
    )
    writer = parquet.ParquetWriter(temporary_path, output_schema, compression="zstd")
    rows_written = 0
    progress_bar = make_progress_bar(
        benchmark_file.metadata.num_rows,
        "Writing double-disjoint test",
        show_progress,
    )

    try:
        for batch in benchmark_file.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([batch], schema=output_schema)
            remove_mask = compute.is_in(table["record_id"], value_set=value_set)
            filtered = normalize_modality_lists(
                table.filter(compute.invert(remove_mask))
            )
            if filtered.num_rows:
                writer.write_table(filtered, row_group_size=filtered.num_rows)
                rows_written += filtered.num_rows
            if progress_bar is not None:
                progress_bar.update(batch.num_rows)
    finally:
        writer.close()
        if progress_bar is not None:
            progress_bar.close()

    return rows_written


def write_extended_train(
    train_file: parquet.ParquetFile,
    benchmark_file: parquet.ParquetFile,
    output_schema: pa.Schema,
    temporary_path: Path,
    record_ids_to_append: set[str],
    batch_size: int,
    show_progress: bool,
) -> tuple[int, int]:
    append_value_set = pa.array(
        sorted(record_ids_to_append),
        type=output_schema.field("record_id").type,
    )
    total_progress = train_file.metadata.num_rows + benchmark_file.metadata.num_rows
    progress_bar = make_progress_bar(
        total_progress,
        "Writing extended train/validation",
        show_progress,
    )
    writer = parquet.ParquetWriter(temporary_path, output_schema, compression="zstd")
    base_rows = 0
    appended_rows = 0

    try:
        for batch in train_file.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([batch], schema=output_schema)
            duplicate_mask = compute.is_in(
                table["record_id"],
                value_set=append_value_set,
            )
            if compute.any(duplicate_mask).as_py():
                raise ValueError(
                    "a benchmark record_id selected for extension already exists "
                    "in the base train/validation file"
                )
            table = normalize_modality_lists(table)
            writer.write_table(table, row_group_size=table.num_rows)
            base_rows += table.num_rows
            if progress_bar is not None:
                progress_bar.update(batch.num_rows)

        for batch in benchmark_file.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([batch], schema=output_schema)
            append_mask = compute.is_in(
                table["record_id"],
                value_set=append_value_set,
            )
            selected = normalize_modality_lists(table.filter(append_mask))
            if selected.num_rows:
                writer.write_table(selected, row_group_size=selected.num_rows)
                appended_rows += selected.num_rows
            if progress_bar is not None:
                progress_bar.update(batch.num_rows)
    finally:
        writer.close()
        if progress_bar is not None:
            progress_bar.close()

    return base_rows, appended_rows


def create_double_disjoint_extended(
    benchmark_test: str | Path,
    rich_train_reference: str | Path,
    nmrsolver_reference: str | Path,
    train_val_to_extend: str | Path,
    test_output: str | Path,
    train_output: str | Path,
    *,
    scan_batch_size: int = 1_000_000,
    write_batch_size: int = 50_000,
    overwrite: bool = False,
    show_progress: bool = True,
) -> dict[str, object]:
    if scan_batch_size < 1 or write_batch_size < 1:
        raise ValueError("batch sizes must be at least 1")

    input_paths = [
        Path(benchmark_test),
        Path(rich_train_reference),
        Path(nmrsolver_reference),
        Path(train_val_to_extend),
    ]
    test_output_path = Path(test_output)
    train_output_path = Path(train_output)
    if test_output_path.resolve() == train_output_path.resolve():
        raise ValueError("test and train outputs must be different files")

    test_temporary = prepare_output_path(test_output_path, input_paths, overwrite)
    train_temporary = prepare_output_path(train_output_path, input_paths, overwrite)
    benchmark_file, rich_file, nmrsolver_file, train_file = validate_inputs(
        input_paths
    )

    benchmark_index = build_benchmark_index(
        benchmark_file,
        scan_batch_size,
        input_paths[0].name,
        show_progress,
    )
    benchmark_smiles = set(benchmark_index)
    rich_overlap_smiles = find_exact_smiles_matches(
        rich_file,
        benchmark_smiles,
        scan_batch_size,
        input_paths[1].name,
        show_progress,
    )
    nmrsolver_candidates = benchmark_smiles - rich_overlap_smiles
    nmrsolver_only_smiles = find_exact_smiles_matches(
        nmrsolver_file,
        nmrsolver_candidates,
        scan_batch_size,
        input_paths[2].name,
        show_progress,
    )

    rich_overlap_ids = record_ids_for_smiles(benchmark_index, rich_overlap_smiles)
    nmrsolver_only_ids = record_ids_for_smiles(
        benchmark_index,
        nmrsolver_only_smiles,
    )
    removed_ids = rich_overlap_ids | nmrsolver_only_ids

    test_metadata = dict(benchmark_file.schema_arrow.metadata or {})
    test_metadata.update(
        {
            b"disjointed_from": str(input_paths[0]).encode("utf-8"),
            b"disjointed_against": json.dumps(
                [str(input_paths[1]), str(input_paths[2])]
            ).encode("utf-8"),
            b"disjoin_identity": b"exact canonical SMILES",
            b"disjoin_removed_record_count": str(len(removed_ids)).encode("utf-8"),
            b"rich_overlap_record_count": str(len(rich_overlap_ids)).encode("utf-8"),
            b"nmrsolver_only_overlap_record_count": str(
                len(nmrsolver_only_ids)
            ).encode("utf-8"),
            b"preparation_script": str(Path(__file__).resolve()).encode("utf-8"),
        }
    )
    test_schema = benchmark_file.schema_arrow.with_metadata(test_metadata)

    train_metadata = dict(train_file.schema_arrow.metadata or {})
    train_metadata.update(
        {
            b"extended_from": str(input_paths[3]).encode("utf-8"),
            b"extended_with": str(input_paths[0]).encode("utf-8"),
            b"extension_reference": str(input_paths[2]).encode("utf-8"),
            b"extension_identity": b"exact canonical SMILES",
            b"extension_selection": (
                b"benchmark absent from rich reference and present in NMR-Solver"
            ),
            b"extension_record_count": str(len(nmrsolver_only_ids)).encode("utf-8"),
            b"preparation_script": str(Path(__file__).resolve()).encode("utf-8"),
        }
    )
    train_schema = train_file.schema_arrow.with_metadata(train_metadata)

    try:
        test_rows = write_filtered_benchmark(
            benchmark_file,
            test_schema,
            test_temporary,
            removed_ids,
            write_batch_size,
            show_progress,
        )
        base_train_rows, appended_rows = write_extended_train(
            train_file,
            benchmark_file,
            train_schema,
            train_temporary,
            nmrsolver_only_ids,
            write_batch_size,
            show_progress,
        )
        if appended_rows != len(nmrsolver_only_ids):
            raise RuntimeError(
                f"selected {len(nmrsolver_only_ids):,} benchmark records but "
                f"appended {appended_rows:,}"
            )

        test_temporary.replace(test_output_path)
        train_temporary.replace(train_output_path)
    except Exception:
        test_temporary.unlink(missing_ok=True)
        train_temporary.unlink(missing_ok=True)
        raise

    result = {
        "benchmark_rows": benchmark_file.metadata.num_rows,
        "rich_overlap_smiles": len(rich_overlap_smiles),
        "rich_overlap_records": len(rich_overlap_ids),
        "nmrsolver_only_overlap_smiles": len(nmrsolver_only_smiles),
        "nmrsolver_only_overlap_records": len(nmrsolver_only_ids),
        "test_rows": test_rows,
        "base_train_rows": base_train_rows,
        "appended_train_rows": appended_rows,
        "extended_train_rows": base_train_rows + appended_rows,
        "test_output": str(test_output_path),
        "train_output": str(train_output_path),
    }
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_test", type=Path)
    parser.add_argument("rich_train_reference", type=Path)
    parser.add_argument("nmrsolver_reference", type=Path)
    parser.add_argument("train_val_to_extend", type=Path)
    parser.add_argument("--test-output", required=True, type=Path)
    parser.add_argument("--train-output", required=True, type=Path)
    parser.add_argument("--scan-batch-size", type=int, default=1_000_000)
    parser.add_argument("--write-batch-size", type=int, default=50_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    result = create_double_disjoint_extended(
        args.benchmark_test,
        args.rich_train_reference,
        args.nmrsolver_reference,
        args.train_val_to_extend,
        args.test_output,
        args.train_output,
        scan_batch_size=args.scan_batch_size,
        write_batch_size=args.write_batch_size,
        overwrite=args.overwrite,
        show_progress=not args.no_progress,
    )
    print(
        f"Wrote {result['test_rows']:,} double-disjoint test records to "
        f"{result['test_output']}",
        flush=True,
    )
    print(
        f"Wrote {result['extended_train_rows']:,} extended train/validation "
        f"records to {result['train_output']} "
        f"({result['appended_train_rows']:,} appended)",
        flush=True,
    )


if __name__ == "__main__":
    main()

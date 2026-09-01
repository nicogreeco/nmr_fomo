#!/usr/bin/env python3
"""Create fixed molecule-safe MACCS probe splits from rich validation data."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
from pyarrow import compute, parquet

from data.console import (
    add_console_arguments,
    configure_console,
    print_stage_complete,
    print_stage_start,
    progress_bar,
)
from data.postprocess.common import open_canonical_parquet, prepare_parquet_output
from data.postprocess.split_foundation_datasets import smiles_hash
from data.reporting import prepare_report_output, write_processing_report


def select_probe_molecules(
    nmr_file: parquet.ParquetFile,
    train_records: int,
    eval_records: int,
    seed: int,
    batch_size: int,
) -> tuple[dict[str, set[str]], dict[str, Counter]]:
    """Select deterministic molecule splits stratified by rich source."""

    total_records = nmr_file.metadata.num_rows
    if train_records < 1 or eval_records < 1:
        raise ValueError("probe train and eval targets must be positive")
    if train_records + eval_records > total_records:
        raise ValueError("probe targets exceed the available rich validation records")

    molecule_sources: dict[str, Counter] = defaultdict(Counter)
    for batch in nmr_file.iter_batches(
        batch_size=batch_size,
        columns=["source", "smiles_canonical"],
        use_threads=True,
    ):
        for source, smiles in zip(
            batch.column("source").to_pylist(),
            batch.column("smiles_canonical").to_pylist(),
        ):
            if not source or not smiles:
                raise ValueError("rich validation contains an empty source or molecule")
            molecule_sources[smiles][source] += 1

    strata = defaultdict(list)
    for smiles, source_counts in molecule_sources.items():
        strata[tuple(sorted(source_counts.items()))].append(smiles)

    selected = {"train": set(), "eval": set()}
    for molecules in strata.values():
        molecules.sort(key=lambda value: (smiles_hash(value, seed), value))
        train_count = round(len(molecules) * train_records / total_records)
        eval_count = round(len(molecules) * eval_records / total_records)
        selected["train"].update(molecules[:train_count])
        selected["eval"].update(
            molecules[train_count : train_count + eval_count]
        )

    return selected, molecule_sources


def write_probe_splits(
    nmr_path: Path,
    molecular_path: Path,
    temporary_paths: dict[str, Path],
    selected: dict[str, set[str]],
    show_progress: bool,
) -> dict[str, int]:
    """Write aligned NMR and molecular-property files for both splits."""

    nmr_file = open_canonical_parquet(nmr_path)
    molecular_file = parquet.ParquetFile(molecular_path)
    if nmr_file.num_row_groups != molecular_file.num_row_groups:
        raise ValueError("rich validation and its sidecar have different row groups")

    writers = {
        "train_nmr": parquet.ParquetWriter(
            temporary_paths["train_nmr"], nmr_file.schema_arrow, compression="zstd"
        ),
        "train_molecular": parquet.ParquetWriter(
            temporary_paths["train_molecular"],
            molecular_file.schema_arrow,
            compression="zstd",
        ),
        "eval_nmr": parquet.ParquetWriter(
            temporary_paths["eval_nmr"], nmr_file.schema_arrow, compression="zstd"
        ),
        "eval_molecular": parquet.ParquetWriter(
            temporary_paths["eval_molecular"],
            molecular_file.schema_arrow,
            compression="zstd",
        ),
    }
    value_sets = {
        split: pa.array(
            sorted(molecules),
            type=nmr_file.schema_arrow.field("smiles_canonical").type,
        )
        for split, molecules in selected.items()
    }
    counts = {"train": 0, "eval": 0}
    progress = progress_bar(
        nmr_file.metadata.num_rows,
        "Writing MACCS probe splits",
        show_progress,
    )

    try:
        for row_group in range(nmr_file.num_row_groups):
            nmr_table = nmr_file.read_row_group(row_group, use_threads=True)
            molecular_table = molecular_file.read_row_group(row_group, use_threads=True)
            if nmr_table["record_id"].to_pylist() != molecular_table[
                "record_id"
            ].to_pylist():
                raise ValueError(f"record_id mismatch in row group {row_group}")
            if nmr_table["smiles_canonical"].to_pylist() != molecular_table[
                "smiles_canonical"
            ].to_pylist():
                raise ValueError(f"smiles mismatch in row group {row_group}")

            smiles_column = nmr_table["smiles_canonical"]
            for split in ("train", "eval"):
                mask = compute.is_in(smiles_column, value_set=value_sets[split])
                selected_nmr = nmr_table.filter(mask)
                if not selected_nmr.num_rows:
                    continue
                selected_molecular = molecular_table.filter(mask)
                writers[f"{split}_nmr"].write_table(
                    selected_nmr, row_group_size=selected_nmr.num_rows
                )
                writers[f"{split}_molecular"].write_table(
                    selected_molecular, row_group_size=selected_molecular.num_rows
                )
                counts[split] += selected_nmr.num_rows

            if progress is not None:
                progress.update(nmr_table.num_rows)
    except Exception:
        for writer in writers.values():
            writer.close()
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)
        raise
    else:
        for writer in writers.values():
            writer.close()
    finally:
        if progress is not None:
            progress.close()

    return counts


def run(args) -> None:
    print_stage_start("MACCS probe split")
    nmr_path = Path(args.nmr_input)
    molecular_path = Path(args.molecular_input)
    output_root = Path(args.output_root)
    input_paths = [nmr_path, molecular_path]
    outputs = {
        "train_nmr": output_root / "train.parquet",
        "train_molecular": output_root / "train_mol_properties.parquet",
        "eval_nmr": output_root / "eval.parquet",
        "eval_molecular": output_root / "eval_mol_properties.parquet",
    }
    temporary_paths = {}
    for name, path in outputs.items():
        outputs[name], temporary_paths[name] = prepare_parquet_output(
            path, input_paths, overwrite=args.overwrite
        )

    report_path = Path(args.report_output or output_root / "split_report.json")
    report_path, report_temporary = prepare_report_output(
        report_path,
        [*input_paths, *outputs.values()],
        overwrite=args.overwrite,
    )

    nmr_file = open_canonical_parquet(nmr_path)
    selected, molecule_sources = select_probe_molecules(
        nmr_file,
        args.train_records,
        args.eval_records,
        args.seed,
        args.batch_size,
    )
    counts = write_probe_splits(
        nmr_path,
        molecular_path,
        temporary_paths,
        selected,
        not args.no_progress,
    )
    for name, path in outputs.items():
        temporary_paths[name].replace(path)

    source_counts = {}
    for split, molecules in selected.items():
        counts_by_source = Counter()
        for molecule in molecules:
            counts_by_source.update(molecule_sources[molecule])
        source_counts[split] = dict(sorted(counts_by_source.items()))

    write_processing_report(
        {
            "stage": "split_maccs_probe",
            "inputs": {
                "nmr": str(nmr_path),
                "molecular_properties": str(molecular_path),
            },
            "outputs": {name: str(path) for name, path in outputs.items()},
            "counts": {
                split: {
                    "records": counts[split],
                    "molecules": len(selected[split]),
                    "sources": source_counts[split],
                }
                for split in ("train", "eval")
            },
            "details": {
                "seed": args.seed,
                "target_records": {
                    "train": args.train_records,
                    "eval": args.eval_records,
                },
                "molecule_identity": "exact smiles_canonical",
                "stratification": "exact per-molecule rich source record counts",
            },
        },
        report_path,
        report_temporary,
    )
    print_stage_complete("MACCS probe split", report_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nmr_input")
    parser.add_argument("molecular_input")
    parser.add_argument("--output-root", default="datasets/train_splits/maccs_probe")
    parser.add_argument("--train-records", type=int, default=20_000)
    parser.add_argument("--eval-records", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--report-output")
    parser.add_argument("--overwrite", action="store_true")
    add_console_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_console(args.quiet)
    run(args)


if __name__ == "__main__":
    main()

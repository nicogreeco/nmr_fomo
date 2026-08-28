#!/usr/bin/env python3
"""Create molecule-safe train/validation files for foundation-model training."""

from __future__ import annotations

import argparse
import hashlib
import heapq
from pathlib import Path

import pyarrow as pa
from pyarrow import compute, parquet

from data.console import add_console_arguments, configure_console, progress_bar
from data.postprocess.common import open_canonical_parquet, prepare_parquet_output
from data.reporting import prepare_report_output, write_processing_report


SOURCES = {
    "simnmr": "simnmr",
    "rich": "train_val",
    "nmrgym": "nmrgym",
}


def smiles_hash(smiles: str, seed: int) -> int:
    value = f"{seed}\0{smiles}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def select_validation_molecules(
    molecular_path: Path,
    target_records: int,
    seed: int,
    batch_size: int,
    show_progress: bool,
) -> set[str]:
    """Select the lowest-hash records while keeping only a small heap."""

    molecular_file = parquet.ParquetFile(molecular_path)
    selected = []
    progress = progress_bar(
        molecular_file.metadata.num_rows,
        f"Selecting {molecular_path.stem}",
        show_progress,
    )

    try:
        for batch in molecular_file.iter_batches(
            batch_size=batch_size,
            columns=["smiles_canonical"],
            use_threads=True,
        ):
            for smiles in batch.column("smiles_canonical").to_pylist():
                score = smiles_hash(smiles, seed)
                candidate = (-score, smiles)
                if len(selected) < target_records:
                    heapq.heappush(selected, candidate)
                elif candidate > selected[0]:
                    heapq.heapreplace(selected, candidate)

            if progress is not None:
                progress.update(batch.num_rows)
    finally:
        if progress is not None:
            progress.close()

    return {smiles for _, smiles in selected}


def split_source(
    nmr_path: Path,
    molecular_path: Path,
    temporary_paths: dict[str, Path],
    validation_molecules: set[str],
    show_progress: bool,
) -> dict[str, int]:
    """Write one aligned NMR/sidecar pair into train and validation pairs."""

    nmr_file = open_canonical_parquet(nmr_path)
    molecular_file = parquet.ParquetFile(molecular_path)
    if nmr_file.num_row_groups != molecular_file.num_row_groups:
        raise ValueError(f"row-group mismatch for {nmr_path.name}")

    writers = {
        "train_nmr": parquet.ParquetWriter(
            temporary_paths["train_nmr"],
            nmr_file.schema_arrow,
            compression="zstd",
        ),
        "train_molecular": parquet.ParquetWriter(
            temporary_paths["train_molecular"],
            molecular_file.schema_arrow,
            compression="zstd",
        ),
        "val_nmr": parquet.ParquetWriter(
            temporary_paths["val_nmr"],
            nmr_file.schema_arrow,
            compression="zstd",
        ),
        "val_molecular": parquet.ParquetWriter(
            temporary_paths["val_molecular"],
            molecular_file.schema_arrow,
            compression="zstd",
        ),
    }
    counts = {"train": 0, "validation": 0}
    progress = progress_bar(
        nmr_file.metadata.num_rows,
        f"Writing {nmr_path.stem}",
        show_progress,
    )

    try:
        for row_group in range(nmr_file.num_row_groups):
            nmr_table = nmr_file.read_row_group(row_group, use_threads=True)
            molecular_table = molecular_file.read_row_group(
                row_group,
                use_threads=True,
            )

            nmr_ids = nmr_table.column("record_id").to_pylist()
            molecular_ids = molecular_table.column("record_id").to_pylist()
            if nmr_ids != molecular_ids:
                raise ValueError(
                    f"record_id mismatch in {nmr_path.name}, row group {row_group}"
                )

            nmr_smiles = nmr_table.column("smiles_canonical").to_pylist()
            molecular_smiles = molecular_table.column(
                "smiles_canonical"
            ).to_pylist()
            if nmr_smiles != molecular_smiles:
                raise ValueError(
                    f"smiles mismatch in {nmr_path.name}, row group {row_group}"
                )

            val_mask = pa.array(
                [smiles in validation_molecules for smiles in nmr_smiles],
                type=pa.bool_(),
            )
            masks = {
                "train": compute.invert(val_mask),
                "val": val_mask,
            }

            for split, mask in masks.items():
                filtered_nmr = nmr_table.filter(mask)
                if not filtered_nmr.num_rows:
                    continue
                filtered_molecular = molecular_table.filter(mask)
                writers[f"{split}_nmr"].write_table(
                    filtered_nmr,
                    row_group_size=filtered_nmr.num_rows,
                )
                writers[f"{split}_molecular"].write_table(
                    filtered_molecular,
                    row_group_size=filtered_molecular.num_rows,
                )
                count_name = "validation" if split == "val" else "train"
                counts[count_name] += filtered_nmr.num_rows

            if progress is not None:
                progress.update(nmr_table.num_rows)
    finally:
        for writer in writers.values():
            writer.close()
        if progress is not None:
            progress.close()

    return counts


def run(args) -> None:
    cleaned_root = Path(args.cleaned_root)
    output_root = Path(args.output_root)
    targets = {
        "simnmr": args.simnmr_validation_records,
        "rich": args.rich_validation_records,
        "nmrgym": args.nmrgym_validation_records,
    }
    inputs = {
        label: {
            "nmr": cleaned_root / f"{input_name}.parquet",
            "molecular": (
                cleaned_root / f"{input_name}_mol_properties.parquet"
            ),
        }
        for label, input_name in SOURCES.items()
    }

    validation_molecules = set()
    selected_by_source = {}
    for label, paths in inputs.items():
        selected = select_validation_molecules(
            paths["molecular"],
            targets[label],
            args.seed,
            args.batch_size,
            not args.no_progress,
        )
        selected_by_source[label] = len(selected)
        validation_molecules.update(selected)

    outputs = {}
    temporary = {}
    input_paths = [path for paths in inputs.values() for path in paths.values()]
    for label in SOURCES:
        paths = {
            "train_nmr": output_root / f"{label}_train.parquet",
            "train_molecular": (
                output_root / f"{label}_train_mol_properties.parquet"
            ),
            "val_nmr": output_root / f"{label}_val.parquet",
            "val_molecular": output_root / f"{label}_val_mol_properties.parquet",
        }
        outputs[label] = paths
        temporary[label] = {}
        for name, path in paths.items():
            final_path, temporary_path = prepare_parquet_output(
                path,
                input_paths,
                overwrite=args.overwrite,
            )
            outputs[label][name] = final_path
            temporary[label][name] = temporary_path

    report_path = Path(args.report_output or output_root / "split_report.json")
    report_path, report_temporary = prepare_report_output(
        report_path,
        [*input_paths, *(path for paths in outputs.values() for path in paths.values())],
        overwrite=args.overwrite,
    )

    counts = {}
    try:
        for label, paths in inputs.items():
            counts[label] = split_source(
                paths["nmr"],
                paths["molecular"],
                temporary[label],
                validation_molecules,
                not args.no_progress,
            )
    except Exception:
        for paths in temporary.values():
            for path in paths.values():
                path.unlink(missing_ok=True)
        raise

    for label in SOURCES:
        for name, path in outputs[label].items():
            temporary[label][name].replace(path)

    write_processing_report(
        {
            "stage": "split_foundation_datasets",
            "inputs": {
                label: {name: str(path) for name, path in paths.items()}
                for label, paths in inputs.items()
            },
            "outputs": {
                label: {name: str(path) for name, path in paths.items()}
                for label, paths in outputs.items()
            },
            "counts": counts,
            "details": {
                "seed": args.seed,
                "target_validation_records": targets,
                "candidate_molecules_by_source": selected_by_source,
                "validation_molecules": len(validation_molecules),
            },
        },
        report_path,
        report_temporary,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned-root", default="datasets/cleaned")
    parser.add_argument("--output-root", default="datasets/train_splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--simnmr-validation-records", type=int, default=50_000)
    parser.add_argument("--rich-validation-records", type=int, default=50_000)
    parser.add_argument("--nmrgym-validation-records", type=int, default=10_000)
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

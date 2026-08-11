#!/usr/bin/env python3
"""Remove exact molecule overlaps from the first canonical dataset.

The first Parquet is the dataset being filtered. Every later Parquet is a
read-only comparison dataset. Molecular identity is exact equality of the
existing ``smiles_canonical`` strings, so comparison files can be streamed
without re-parsing every molecule with RDKit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.postprocess.common import (
    build_exact_smiles_index,
    find_exact_smiles_matches,
    open_compatible_parquets,
    prepare_parquet_output,
    record_ids_for_smiles,
    write_filtered_parquet,
)
from data.reporting import (
    default_report_path,
    prepare_report_output,
    write_processing_report,
)


SCRIPT_PATH = "scripts/data/postprocess/remove_benchmark_overlaps.py"
IDENTITY_DESCRIPTION = "exact smiles_canonical equality"


def remove_benchmark_overlaps(
    input_path: str | Path,
    comparison_paths: list[str | Path],
    output_path: str | Path,
    *,
    batch_size: int = 50_000,
    overwrite: bool = False,
    show_progress: bool = True,
) -> dict[str, object]:
    """Remove rows whose canonical SMILES occurs in any comparison dataset."""

    if not comparison_paths:
        raise ValueError("at least one comparison dataset is required")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    input_file_path = Path(input_path)
    references = [Path(path) for path in comparison_paths]
    all_inputs = [input_file_path, *references]
    output, temporary = prepare_parquet_output(
        output_path,
        all_inputs,
        overwrite=overwrite,
    )
    input_file, *comparison_files = open_compatible_parquets(all_inputs)

    smiles_index = build_exact_smiles_index(
        input_file,
        batch_size=batch_size,
        source_name=input_file_path.name,
        show_progress=show_progress,
    )
    remaining_smiles = set(smiles_index)
    matched_smiles: set[str] = set()
    matches_by_input = {str(path): 0 for path in references}

    for reference_path, reference_file in zip(references, comparison_files):
        new_matches = find_exact_smiles_matches(
            reference_file,
            remaining_smiles,
            batch_size=batch_size,
            source_name=reference_path.name,
            show_progress=show_progress,
        )
        matched_smiles.update(new_matches)
        remaining_smiles.difference_update(new_matches)
        matches_by_input[str(reference_path)] = len(new_matches)
        if not remaining_smiles:
            break

    record_ids_to_remove = record_ids_for_smiles(smiles_index, matched_smiles)
    metadata = dict(input_file.schema_arrow.metadata or {})
    metadata.update(
        {
            b"postprocess_step": b"remove_benchmark_overlaps",
            b"postprocess_script": SCRIPT_PATH.encode("utf-8"),
            b"postprocess_inputs": json.dumps(
                [str(path) for path in all_inputs]
            ).encode("utf-8"),
            b"molecule_identity": IDENTITY_DESCRIPTION.encode("utf-8"),
            b"removed_molecule_count": str(len(matched_smiles)).encode("utf-8"),
            b"removed_record_count": str(len(record_ids_to_remove)).encode("utf-8"),
        }
    )
    output_schema = input_file.schema_arrow.with_metadata(metadata)

    try:
        rows_written = write_filtered_parquet(
            input_file,
            output_schema,
            temporary,
            record_ids_to_remove,
            keep=False,
            batch_size=batch_size,
            description=f"Writing {output.name}",
            show_progress=show_progress,
        )
        expected_rows = input_file.metadata.num_rows - len(record_ids_to_remove)
        if rows_written != expected_rows:
            raise RuntimeError(
                "output row count does not equal input rows minus removed records"
            )
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return {
        "stage": "remove_benchmark_overlaps",
        "inputs": {
            "dataset": str(input_file_path),
            "comparisons": [str(path) for path in references],
        },
        "outputs": {"dataset": str(output)},
        "counts": {
            "input_records": input_file.metadata.num_rows,
            "output_records": rows_written,
            "removed_records": len(record_ids_to_remove),
            "removed_molecules": len(matched_smiles),
        },
        "details": {"new_molecule_matches_by_comparison": matches_by_input},
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="canonical dataset to filter")
    parser.add_argument(
        "comparisons",
        nargs="+",
        type=Path,
        help="canonical datasets whose molecules must be removed from input",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument(
        "--report-output",
        type=Path,
        help="processing JSON (default: <output>_report.json)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    report_path, report_temporary = prepare_report_output(
        args.report_output or default_report_path(args.output),
        [args.input, *args.comparisons, args.output],
        overwrite=args.overwrite,
    )
    result = remove_benchmark_overlaps(
        args.input,
        args.comparisons,
        args.output,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
        show_progress=not args.no_progress,
    )
    write_processing_report(result, report_path, report_temporary)
    print(f"Wrote processing report to {report_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Measure canonical-record compatibility with the four model processors."""

import argparse
from collections import Counter
import json
from pathlib import Path

from data import CanonicalParquetDataset, IncompatibleRecordError
from model_benchmarks import build_processor


MODEL_NAMES = ("nmrpeak", "nmrsolver", "nmrtrans", "ultranmr")


def analyze_processor_compatibility(
    parquet_path: Path,
    dataset_name: str,
    arrow_batch_size: int = 2048,
) -> dict:
    processors = {
        model_name: build_processor(model_name, mode="canonical")
        for model_name in MODEL_NAMES
    }
    counts = {
        model_name: {
            "accepted_records": 0,
            "rejected_records": 0,
            "issue_occurrence_counts": Counter(),
            "record_issue_code_counts": Counter(),
        }
        for model_name in MODEL_NAMES
    }

    row_count = 0
    dataset = CanonicalParquetDataset(
        parquet_path,
        arrow_batch_size=arrow_batch_size,
    )
    for record in dataset:
        row_count += 1
        for model_name, processor in processors.items():
            model_counts = counts[model_name]
            try:
                processor.prepare_record(record)
            except IncompatibleRecordError as error:
                model_counts["rejected_records"] += 1
                issue_codes = [issue.code for issue in error.issues]
                model_counts["issue_occurrence_counts"].update(issue_codes)
                model_counts["record_issue_code_counts"].update(set(issue_codes))
            else:
                model_counts["accepted_records"] += 1

    models = {}
    for model_name in MODEL_NAMES:
        model_counts = counts[model_name]
        models[model_name] = {
            "accepted_records": model_counts["accepted_records"],
            "rejected_records": model_counts["rejected_records"],
            "issue_occurrence_counts": dict(
                sorted(model_counts["issue_occurrence_counts"].items())
            ),
            "record_issue_code_counts": dict(
                sorted(model_counts["record_issue_code_counts"].items())
            ),
        }

    return {
        "dataset": dataset_name,
        "input": str(parquet_path),
        "mode": "canonical",
        "models": models,
        "rows": row_count,
        "scope": (
            "processor field validation; NMRPeak sequence length is checked "
            "later during collation"
        ),
    }


def write_report(report: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.partial")
    if temporary_path.exists():
        raise FileExistsError(f"temporary report already exists: {temporary_path}")
    temporary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet_file", type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arrow-batch-size", type=int, default=2048)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    report = analyze_processor_compatibility(
        args.parquet_file,
        args.dataset_name,
        args.arrow_batch_size,
    )
    write_report(report, args.output)
    print(f"Wrote processor compatibility report to {args.output}")


if __name__ == "__main__":
    main()

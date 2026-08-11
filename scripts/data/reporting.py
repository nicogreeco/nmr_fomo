"""Small helpers for deterministic processing-report JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


def default_report_path(output_path: str | Path) -> Path:
    """Place a processing report beside the stage's primary output."""

    output = Path(output_path)
    return output.with_name(f"{output.stem}_report.json")


def prepare_report_output(
    report_path: str | Path,
    protected_paths: list[str | Path],
    *,
    overwrite: bool,
) -> tuple[Path, Path]:
    """Validate a JSON report path before an expensive processing stage."""

    output = Path(report_path)
    if output.suffix.lower() != ".json":
        raise ValueError(f"processing report must have a .json suffix: {output}")
    if output.resolve() in {Path(path).resolve() for path in protected_paths}:
        raise ValueError("processing report must not replace a data input or output")
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite {output}; pass --overwrite to replace it"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.partial")
    if temporary.exists():
        raise FileExistsError(
            f"temporary processing report already exists: {temporary}"
        )
    return output, temporary


def write_processing_report(
    report: Mapping[str, object],
    output_path: Path,
    temporary_path: Path,
) -> None:
    """Write one report atomically with stable, human-readable formatting."""

    temporary_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)

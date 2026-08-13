"""Small shared helpers for readable data-pipeline command output."""

from __future__ import annotations

import argparse
import warnings
from collections.abc import Iterable, Iterator
from typing import TypeVar


T = TypeVar("T")


def add_console_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the common quiet/progress switches to one pipeline command."""

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress non-essential warnings while keeping progress and errors",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable the progress bar",
    )


def configure_console(quiet: bool) -> None:
    """Suppress noisy library warnings without hiding Python exceptions."""

    if not quiet:
        return
    warnings.filterwarnings("ignore")
    try:
        from rdkit import RDLogger
    except ModuleNotFoundError:
        return
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")


def progress_bar(
    total: int | None,
    description: str,
    enabled: bool,
    *,
    min_iterations: int | None = None,
    unit: str = "records",
):
    """Return a compact tqdm progress bar, or ``None`` when disabled."""

    if not enabled:
        return None
    try:
        from tqdm import tqdm
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "progress bars require tqdm; install it or pass --no-progress"
        ) from error
    return tqdm(
        total=total,
        desc=description,
        unit=unit,
        dynamic_ncols=True,
        mininterval=0.5,
        miniters=min_iterations,
    )


def progress_iter(
    values: Iterable[T],
    description: str,
    *,
    enabled: bool,
    total: int | None = None,
) -> Iterator[T]:
    """Yield values unchanged while updating one compact progress bar."""

    progress = progress_bar(total, description, enabled)
    try:
        for value in values:
            yield value
            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()


def print_stage_start(stage: str) -> None:
    print(f"Starting {stage}", flush=True)


def print_stage_complete(stage: str, report_path: object) -> None:
    print(f"Completed {stage}; report: {report_path}", flush=True)

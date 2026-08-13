"""Small shared helpers for canonical Parquet post-processing."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pyarrow as pa
from pyarrow import compute, parquet

from data.canonicalize.common import (
    CANONICAL_PARQUET_SCHEMA_VERSION,
    canonical_parquet_schema,
    normalize_modality_lists,
)
from data.console import progress_bar


def open_canonical_parquet(
    path: str | Path,
    *,
    require_rdkit_version: bool = False,
) -> parquet.ParquetFile:
    """Open one current schema-v2 canonical Parquet file."""

    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"canonical Parquet input not found: {input_path}")

    parquet_file = parquet.ParquetFile(input_path)
    input_schema = parquet_file.schema_arrow
    if not input_schema.remove_metadata().equals(canonical_parquet_schema()):
        raise ValueError(f"{input_path} does not use the current canonical schema")

    metadata = input_schema.metadata or {}
    schema_version = metadata.get(b"canonical_schema_version", b"").decode(
        "utf-8"
    )
    if schema_version != CANONICAL_PARQUET_SCHEMA_VERSION:
        raise ValueError(
            f"{input_path} uses canonical schema version {schema_version!r}; "
            f"expected {CANONICAL_PARQUET_SCHEMA_VERSION!r}"
        )

    if require_rdkit_version and not metadata.get(b"rdkit_version"):
        raise ValueError(f"{input_path} has no rdkit_version metadata")
    return parquet_file


def open_compatible_parquets(
    paths: Iterable[str | Path],
) -> list[parquet.ParquetFile]:
    """Open canonical files generated with one common RDKit version."""

    input_paths = [Path(path) for path in paths]
    if not input_paths:
        raise ValueError("at least one canonical Parquet input is required")

    parquet_files = [
        open_canonical_parquet(path, require_rdkit_version=True)
        for path in input_paths
    ]
    rdkit_versions = {
        parquet_file.schema_arrow.metadata[b"rdkit_version"]
        for parquet_file in parquet_files
    }
    if len(rdkit_versions) != 1:
        versions = ", ".join(
            sorted(version.decode("utf-8") for version in rdkit_versions)
        )
        raise ValueError(f"input RDKit versions differ: {versions}")
    return parquet_files


def prepare_parquet_output(
    output_path: str | Path,
    input_paths: Iterable[str | Path],
    *,
    overwrite: bool,
) -> tuple[Path, Path]:
    """Validate one output and return its final and hidden temporary paths."""

    output = Path(output_path)
    if output.suffix.lower() != ".parquet":
        raise ValueError(f"output must have a .parquet suffix: {output}")

    output_resolved = output.resolve()
    if any(Path(path).resolve() == output_resolved for path in input_paths):
        raise ValueError(f"output must not replace an input file: {output}")
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite {output}; pass --overwrite to replace it"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.partial")
    if temporary.exists():
        raise FileExistsError(
            f"temporary output already exists: {temporary}; inspect or remove it "
            "before starting another run"
        )
    return output, temporary


def build_exact_smiles_index(
    parquet_file: parquet.ParquetFile,
    *,
    batch_size: int,
    source_name: str,
    show_progress: bool,
) -> dict[str, list[str]]:
    """Index record IDs from one dataset by exact ``smiles_canonical``."""

    index: dict[str, list[str]] = {}
    seen_record_ids: set[str] = set()
    processed_rows = 0
    progress = progress_bar(
        parquet_file.metadata.num_rows,
        f"Indexing {source_name}",
        show_progress,
    )
    try:
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=["record_id", "smiles_canonical"],
            use_threads=True,
        ):
            record_ids = batch.column("record_id").to_pylist()
            smiles_values = batch.column("smiles_canonical").to_pylist()
            for offset, (record_id, smiles) in enumerate(
                zip(record_ids, smiles_values)
            ):
                row_number = processed_rows + offset + 1
                if not isinstance(record_id, str) or not record_id:
                    raise ValueError(
                        f"{source_name} has an empty or non-string record_id at "
                        f"row {row_number:,}"
                    )
                if record_id in seen_record_ids:
                    raise ValueError(
                        f"{source_name} contains duplicate record_id {record_id!r}"
                    )
                if not isinstance(smiles, str) or not smiles:
                    raise ValueError(
                        f"{source_name} has an empty smiles_canonical at row "
                        f"{row_number:,}"
                    )
                seen_record_ids.add(record_id)
                index.setdefault(smiles, []).append(record_id)

            processed_rows += batch.num_rows
            if progress is not None:
                progress.update(batch.num_rows)
    finally:
        if progress is not None:
            progress.close()
    return index


def find_exact_smiles_matches(
    parquet_file: parquet.ParquetFile,
    candidate_smiles: set[str],
    *,
    batch_size: int,
    source_name: str,
    show_progress: bool,
) -> set[str]:
    """Stream one dataset and return candidate canonical SMILES found in it."""

    if not candidate_smiles:
        return set()

    value_set = pa.array(
        sorted(candidate_smiles),
        type=parquet_file.schema_arrow.field("smiles_canonical").type,
    )
    matched_smiles: set[str] = set()
    progress = progress_bar(
        parquet_file.metadata.num_rows,
        f"Scanning {source_name}",
        show_progress,
    )
    try:
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=["smiles_canonical"],
            use_threads=True,
        ):
            smiles_column = batch.column("smiles_canonical")
            has_empty = compute.any(
                compute.equal(smiles_column, "")
            ).as_py()
            if smiles_column.null_count or has_empty:
                raise ValueError(
                    f"{source_name} contains empty smiles_canonical values"
                )
            match_mask = compute.is_in(smiles_column, value_set=value_set)
            matched_smiles.update(
                value
                for value in compute.filter(smiles_column, match_mask).to_pylist()
                if value is not None
            )
            if progress is not None:
                progress.update(batch.num_rows)
            if len(matched_smiles) == len(candidate_smiles):
                break
    finally:
        if progress is not None:
            progress.close()
    return matched_smiles


def record_ids_for_smiles(
    smiles_index: dict[str, list[str]],
    selected_smiles: set[str],
) -> set[str]:
    """Expand selected molecular keys to every corresponding record ID."""

    return {
        record_id
        for smiles in selected_smiles
        for record_id in smiles_index[smiles]
    }


def write_filtered_parquet(
    input_file: parquet.ParquetFile,
    output_schema: pa.Schema,
    temporary_path: Path,
    record_ids: set[str],
    *,
    keep: bool,
    batch_size: int,
    description: str,
    show_progress: bool,
) -> int:
    """Stream a canonical Parquet while keeping or removing record IDs."""

    value_set = pa.array(
        sorted(record_ids),
        type=output_schema.field("record_id").type,
    )
    writer = parquet.ParquetWriter(
        temporary_path,
        output_schema,
        compression="zstd",
    )
    rows_written = 0
    progress = progress_bar(
        input_file.metadata.num_rows,
        description,
        show_progress,
    )
    try:
        for batch in input_file.iter_batches(batch_size=batch_size, use_threads=True):
            table = pa.Table.from_batches([batch], schema=output_schema)
            selected = compute.is_in(table["record_id"], value_set=value_set)
            mask = selected if keep else compute.invert(selected)
            filtered = normalize_modality_lists(table.filter(mask))
            if filtered.num_rows:
                writer.write_table(filtered, row_group_size=filtered.num_rows)
                rows_written += filtered.num_rows
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
    return rows_written

import argparse
import json
from pathlib import Path
import pyarrow as pa
from pyarrow import parquet, compute
from rdkit import Chem

from data.canonicalize.common import (
    CANONICAL_PARQUET_SCHEMA_VERSION,
    canonical_parquet_schema,
    normalize_modality_lists,
)


def structure_identifiers(smiles: object) -> dict[str, str | None]:
    if not isinstance(smiles, str) or not smiles.strip():
        return {
            "canonical_smiles_rdkit": None,
            "full_inchikey": None,
            "connectivity_inchikey": None,
        }

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return {
            "canonical_smiles_rdkit": None,
            "full_inchikey": None,
            "connectivity_inchikey": None,
        }

    canonical_smiles = Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True,
    )
    try:
        full_inchikey = Chem.MolToInchiKey(molecule)
    except Exception:
        full_inchikey = None

    return {
        "canonical_smiles_rdkit": canonical_smiles,
        "full_inchikey": full_inchikey,
        "connectivity_inchikey": (
            full_inchikey.split("-")[0] if full_inchikey else None
        ),
    }


def make_progress_bar(
    parquet_file: parquet.ParquetFile,
    description: str,
    show_progress: bool,
):
    if not show_progress:
        return None

    try:
        from tqdm import tqdm
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "showing disjoin progress requires tqdm; install it in the "
            "environment used for post-processing"
        ) from error

    return tqdm(
        total=parquet_file.metadata.num_rows,
        desc=description,
        unit="records",
    )


def build_connectivity_index(
    parquet_file: parquet.ParquetFile,
    batch_size: int = 50_000,
    source_name: str = "dataset",
    show_progress: bool = True,
) -> dict[str, list[str]]:

    if "smiles_canonical" not in parquet_file.schema_arrow.names:
        raise KeyError(
            "Dataset has no 'smiles_canonical' column"
        )

    connectivity_index: dict[str, list[str]] = {}
    invalid_or_unkeyed_rows = 0
    processed_rows = 0
    progress_bar = make_progress_bar(
        parquet_file,
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
                connectivity_key = structure_identifiers(smiles)[
                    "connectivity_inchikey"
                ]
                if connectivity_key is None:
                    invalid_or_unkeyed_rows += 1
                    continue
                connectivity_index.setdefault(connectivity_key, []).append(record_id)

            processed_rows += batch.num_rows
            if progress_bar is not None:
                progress_bar.update(batch.num_rows)
    finally:
        if progress_bar is not None:
            progress_bar.close()

    print(
        f"Finished {source_name}: {processed_rows:,} records, "
        f"{len(connectivity_index):,} unique connectivity keys, "
        f"{invalid_or_unkeyed_rows:,} invalid/unkeyed rows"
    )
    return connectivity_index


def find_connectivity_intersections(
    parquet_file: parquet.ParquetFile,
    candidate_keys: set[str],
    batch_size: int = 50_000,
    source_name: str = "dataset",
    show_progress: bool = True,
) -> set[str]:
    """Stream one comparison file and return matching connectivity keys."""

    if "smiles_canonical" not in parquet_file.schema_arrow.names:
        raise KeyError("Dataset has no 'smiles_canonical' column")

    unmatched_keys = set(candidate_keys)
    invalid_or_unkeyed_rows = 0
    processed_rows = 0
    progress_bar = make_progress_bar(
        parquet_file,
        f"Scanning {source_name}",
        show_progress,
    )

    try:
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=["smiles_canonical"],
        ):
            for smiles in batch.column("smiles_canonical").to_pylist():
                connectivity_key = structure_identifiers(smiles)[
                    "connectivity_inchikey"
                ]
                if connectivity_key is None:
                    invalid_or_unkeyed_rows += 1
                else:
                    unmatched_keys.discard(connectivity_key)

            processed_rows += batch.num_rows
            if progress_bar is not None:
                progress_bar.update(batch.num_rows)

            if not unmatched_keys:
                break
    finally:
        if progress_bar is not None:
            progress_bar.close()

    matched_keys = candidate_keys - unmatched_keys
    print(
        f"Finished {source_name}: scanned {processed_rows:,} records, "
        f"matched {len(matched_keys):,} connectivity keys, "
        f"{invalid_or_unkeyed_rows:,} invalid/unkeyed rows"
    )
    return matched_keys


def disjoint_datasets(
    parquet_file_to_disjoint: list[Path],
    output_path: Path,
    arrow_batch_size: int = 50_000,
    overwrite: bool = False,
    show_progress: bool = True,
) -> dict[str, object]:
    """
    Remove connectivity intersections from the first canonical Parquet file.

    Args:
        parquet_file_to_disjoint (list[Path]): List of input canonical Parquet files.
            The first file will be used as the base dataset from which the intersection
            will be removed. The others will remain unchanged.
        output_path (Path): Output canonical Parquet file.
        arrow_batch_size (int, optional): Number of rows read at once.
            Defaults to 50,000.
        overwrite (bool, optional): If True, replace an existing output file. Defaults to False.
    """

    if not parquet_file_to_disjoint:
        raise ValueError("at least one input Parquet file is required")
    if arrow_batch_size < 1:
        raise ValueError("arrow_batch_size must be at least 1")

    parquet_file_to_disjoint = [Path(path) for path in parquet_file_to_disjoint]
    missing_paths = [path for path in parquet_file_to_disjoint if not path.is_file()]
    if missing_paths:
        missing_text = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"input Parquet file not found: {missing_text}")

    output = Path(output_path)
    output_resolved = output.resolve()
    if any(path.resolve() == output_resolved for path in parquet_file_to_disjoint):
        raise ValueError("output must not replace one of the input files")
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite {output}; pass --overwrite to replace it"
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    parquet_files = [parquet.ParquetFile(path) for path in parquet_file_to_disjoint]

    expected_schema = canonical_parquet_schema()
    rdkit_versions: set[str] = set()

    for path, parquet_file in zip(parquet_file_to_disjoint, parquet_files):
        input_schema = parquet_file.schema_arrow
        if not input_schema.remove_metadata().equals(expected_schema):
            raise ValueError(
                f"input does not use the current canonical schema: {path}"
            )

        input_metadata = input_schema.metadata or {}
        schema_version = input_metadata.get(b"canonical_schema_version", b"").decode(
            "utf-8"
        )
        if schema_version != CANONICAL_PARQUET_SCHEMA_VERSION:
            raise ValueError(
                f"{path} uses canonical schema version {schema_version!r}; "
                f"expected {CANONICAL_PARQUET_SCHEMA_VERSION!r}"
            )

        rdkit_version = input_metadata.get(b"rdkit_version", b"").decode("utf-8")
        if not rdkit_version:
            raise ValueError(f"{path} has no rdkit_version metadata")
        rdkit_versions.add(rdkit_version)

    if len(rdkit_versions) != 1:
        versions_text = ", ".join(sorted(rdkit_versions))
        raise ValueError(f"input RDKit versions differ: {versions_text}")
    to_disjoint = parquet_files.pop(0)
    to_disjoint_index = build_connectivity_index(
        to_disjoint,
        batch_size=arrow_batch_size,
        source_name=parquet_file_to_disjoint[0].name,
        show_progress=show_progress,
    )

    intersection_keys: set[str] = set()
    remaining_keys = set(to_disjoint_index)
    for path, parquet_file in zip(parquet_file_to_disjoint[1:], parquet_files):
        if not remaining_keys:
            print("All first-dataset connectivity keys matched; stopping early")
            break

        matched_keys = find_connectivity_intersections(
            parquet_file,
            remaining_keys,
            batch_size=arrow_batch_size,
            source_name=path.name,
            show_progress=show_progress,
        )
        intersection_keys.update(matched_keys)
        remaining_keys.difference_update(matched_keys)

    record_ids_to_remove = {
        record_id
        for key in intersection_keys
        for record_id in to_disjoint_index[key]
    }

    input_metadata = dict(to_disjoint.schema_arrow.metadata or {})
    input_metadata.update(
        {
            b"disjointed_from": str(parquet_file_to_disjoint[0]).encode("utf-8"),
            b"disjointed_against": json.dumps(
                [str(path) for path in parquet_file_to_disjoint[1:]]
            ).encode("utf-8"),
            b"disjoin_identity": b"RDKit connectivity InChIKey",
            b"disjoin_removed_record_count": str(len(record_ids_to_remove)).encode(
                "utf-8"
            ),
            b"disjoin_script": str(Path(__file__).resolve()).encode("utf-8"),
        }
    )
    output_schema = to_disjoint.schema_arrow.with_metadata(input_metadata)

    temporary_output = output.with_name(f".{output.name}.partial")
    if temporary_output.exists():
        raise FileExistsError(
            f"temporary output already exists: {temporary_output}; inspect or "
            "remove it before starting another run"
        )

    value_set = pa.array(
        sorted(record_ids_to_remove),
        type=output_schema.field("record_id").type,
    )
    writer = parquet.ParquetWriter(
        temporary_output,
        output_schema,
        compression="zstd",
    )

    final_rows = 0
    total_rows = 0
    try:
        for batch in to_disjoint.iter_batches(
            batch_size=arrow_batch_size,
        ):
            table = pa.Table.from_batches([batch], schema=output_schema)
            remove_mask = compute.is_in(table["record_id"], value_set=value_set)
            filtered_table = table.filter(compute.invert(remove_mask))
            filtered_table = normalize_modality_lists(filtered_table)

            final_rows += filtered_table.num_rows
            total_rows += table.num_rows

            if filtered_table.num_rows == 0:
                continue

            writer.write_table(
                filtered_table,
                row_group_size=filtered_table.num_rows,
            )
    except Exception:
        writer.close()
        temporary_output.unlink(missing_ok=True)
        raise

    writer.close()
    temporary_output.replace(output)

    return {
        "output": str(output),
        "input_files": parquet_file_to_disjoint,
        "rows": total_rows,
        "intersection_rows": total_rows - final_rows,
        "bytes": output.stat().st_size,
    }
def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Remove connectivity overlap from the first canonical Parquet input."
        )
    )
    parser.add_argument(
        "parquet_file_to_disjoint",
        nargs="+",
        type=Path,
        help=(
            "canonical Parquet inputs; rows are removed only from the first "
            "file when their connectivity occurs in a later input"
        ),
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
        help="rows read at once (default: 50000)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output file",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable progress bars",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    result = disjoint_datasets(
        args.parquet_file_to_disjoint,
        args.output,
        arrow_batch_size=args.arrow_batch_size,
        overwrite=args.overwrite,
        show_progress=not args.no_progress,
    )
    print(
        f"Disjoint dataset written to {args.output} with "
        f"{result['intersection_rows']:,} overlapping rows removed"
    )


if __name__ == "__main__":
    main()

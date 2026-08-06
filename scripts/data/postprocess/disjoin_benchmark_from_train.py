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
            'canonical_smiles_rdkit': None,
            'full_inchikey': None,
            'connectivity_inchikey': None,
        }

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return {
            'canonical_smiles_rdkit': None,
            'full_inchikey': None,
            'connectivity_inchikey': None,
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

    connectivity_inchikey = (
        full_inchikey.split('-')[0]
        if full_inchikey
        else None
    )

    return {
        'canonical_smiles_rdkit': canonical_smiles,
        'full_inchikey': full_inchikey,
        'connectivity_inchikey': connectivity_inchikey,
    }

def build_connectivity_index(
    parquet_file: parquet.ParquetFile,
    batch_size: int = 10_000,
) -> dict[str, list[dict[str, object]]]:

    if "smiles_canonical" not in parquet_file.schema_arrow.names:
        raise KeyError(
            "Dataset has no 'smiles_canonical' column"
        )

    connectivity_index: dict[str, list[dict[str, object]]] = {}
    invalid_or_unkeyed_rows = 0
    processed_rows = 0

    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=["smiles_canonical", "record_id"],
    ):
        smiles_values = batch.column("smiles_canonical").to_pylist()
        record_ids = batch.column("record_id").to_pylist()

        for row_offset, (smiles, record_id) in enumerate(
            zip(smiles_values, record_ids)
        ):
            parquet_row_index = processed_rows + row_offset

            identifiers = structure_identifiers(smiles)
            connectivity_inchikey = identifiers["connectivity_inchikey"]

            if connectivity_inchikey is None:
                invalid_or_unkeyed_rows += 1
                continue

            connectivity_index.setdefault(connectivity_inchikey, []).append({
                "index": parquet_row_index,
                "record_id": record_id,
            })

        processed_rows += batch.num_rows

        if processed_rows % 50_000 == 0:
            print(
                f"Processed {processed_rows:,} dataset records"
            )

    print(
        f"Finished: {processed_rows:,} records, "
        f"{len(connectivity_index):,} unique connectivity keys, "
        f"{invalid_or_unkeyed_rows:,} invalid/unkeyed rows"
    )

    return connectivity_index

def disjoint_datasets(
    parquet_file_to_disjoint: list[Path],
    output_path: Path,
    arrow_batch_size: int = 50_000,
    overwrite: bool = False,
) -> dict[str, object]:
    """
    Remove intersection of molecules from the first canonical Parquet file in the list
    
    
    and write the result to a new canonical Parquet file.

    Args:
        parquet_file_to_disjoint (list[Path]): List of input canonical Parquet files.
            The first file will be used as the base dataset from which the intersection
            will be removed. The others will remain unchanged.
        output_path (Path): Output canonical Parquet file.
        arrow_batch_size (int, optional): Number of rows copied at once. Defaults to 50_000.
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
    to_disjoint_index = build_connectivity_index(to_disjoint, batch_size=arrow_batch_size)
    parquet_files_indexes = [build_connectivity_index(parquet_file, batch_size=arrow_batch_size) for parquet_file in parquet_files]

    intersection_keys = set()
    for parquet_file_index in parquet_files_indexes:
        intersection_keys.update(
            to_disjoint_index.keys() & parquet_file_index.keys()
        )

    record_ids_to_remove = {
        entry["record_id"]
        for key in intersection_keys
        for entry in to_disjoint_index[key]
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
        help="rows copied at once (default: 50000)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output file",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    result = disjoint_datasets(
        args.parquet_file_to_disjoint,
        args.output,
        arrow_batch_size=args.arrow_batch_size,
        overwrite=args.overwrite,
    )
    print(
        f"Disjoint dataset written to {args.output} with "
        f"{result['intersection_rows']:,} overlapping rows removed"
    )


if __name__ == "__main__":
    main()

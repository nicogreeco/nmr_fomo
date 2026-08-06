#!/usr/bin/env python3
"""Extract exact TDC/NMR overlaps and write disjoint canonical records.

The audit notebook already establishes the exact full-InChIKey matches for
each TDC split.  This script uses that audit manifest, adds every matching
canonical ``record_id`` to the source property rows, and copies the matching
canonical records into endpoint-specific Parquet files.

Connectivity-only matches are deliberately not considered anywhere here.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pyarrow as pa
from pyarrow import compute, parquet
from rdkit import Chem, RDLogger

from data.canonicalize.common import (
    CANONICAL_PARQUET_SCHEMA_VERSION,
    canonical_parquet_schema,
)


RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")

ENDPOINTS = ("solubility_aqsoldb", "ld50_zhu", "ames")
SPLITS = ("train_val", "test")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract exact full-InChIKey TDC/NMR overlaps and create a "
            "disjoint merged canonical dataset."
        )
    )
    parser.add_argument(
        "--merged",
        type=Path,
        default=Path("datasets/merged/merged_train_val_all.parquet"),
        help="merged canonical input Parquet file",
    )
    parser.add_argument(
        "--audit-matches",
        type=Path,
        default=Path("datasets/properties/admet_overlap/exact_matches.csv"),
        help="exact-match manifest written by admet_overlap_audit.ipynb",
    )
    parser.add_argument(
        "--audit-summary",
        type=Path,
        default=Path("datasets/properties/admet_overlap/overlap_summary.csv"),
        help="overlap summary written by admet_overlap_audit.ipynb",
    )
    parser.add_argument(
        "--tdc-root",
        type=Path,
        default=Path("datasets/properties/tdc_admet/admet_group"),
        help="directory containing endpoint train_val.csv and test.csv files",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/properties/annonated_peaks"),
        help="directory for endpoint-specific annotated outputs",
    )
    parser.add_argument(
        "--disjoint-output",
        type=Path,
        default=Path("datasets/merged/merged_train_val_all_disjoint.parquet"),
        help="output merged Parquet file with all matched records removed",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
        help="Parquet batch size (default: 50000)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace outputs that already exist",
    )
    return parser


def full_inchikey(smiles: object) -> str | None:
    """Return the RDKit full InChIKey, or None for an unusable structure."""

    if not isinstance(smiles, str) or not smiles.strip():
        return None

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None

    try:
        return Chem.MolToInchiKey(molecule)
    except Exception:
        return None


def load_audit_matches(
    path: Path,
) -> dict[tuple[str, str], list[dict[str, object]]]:
    """Load the notebook's exact-match rows for the three target endpoints."""

    required_columns = {
        "source",
        "dataset",
        "split",
        "_source_row",
        "full_inchikey",
        "nmr_record_count",
    }
    matches: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing_text = ", ".join(sorted(missing_columns))
            raise ValueError(f"{path} is missing columns: {missing_text}")

        for row in reader:
            endpoint = row["dataset"]
            split = row["split"]
            if row["source"] != "TDC ADMET":
                continue
            if endpoint not in ENDPOINTS or split not in SPLITS:
                continue

            key = row["full_inchikey"].strip()
            if not key:
                raise ValueError(
                    f"{path} contains an empty full_inchikey for "
                    f"{endpoint}/{split}"
                )

            try:
                source_row = int(row["_source_row"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid _source_row in {path}: {row['_source_row']!r}"
                ) from error

            expected_count = None
            if row["nmr_record_count"].strip():
                expected_count = int(row["nmr_record_count"])

            matches[(endpoint, split)].append(
                {
                    "source_row": source_row,
                    "full_inchikey": key,
                    "expected_record_count": expected_count,
                }
            )

    for release, rows in matches.items():
        rows.sort(key=lambda row: int(row["source_row"]))

    return matches


def load_audit_summary(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    """Load summary values for provenance without recomputing property labels."""

    summary: dict[tuple[str, str], dict[str, str]] = {}
    if not path.is_file():
        return summary

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            endpoint = row.get("dataset", "")
            split = row.get("split", "")
            if row.get("source") == "TDC ADMET" and endpoint in ENDPOINTS:
                summary[(endpoint, split)] = row
    return summary


def build_record_index(
    path: Path,
    batch_size: int,
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Index merged record IDs by exact full InChIKey in streaming batches."""

    parquet_file = parquet.ParquetFile(path)
    required_columns = {"record_id", "smiles_canonical"}
    missing_columns = required_columns - set(parquet_file.schema_arrow.names)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"{path} is missing columns: {missing_text}")

    record_ids_by_key: dict[str, list[str]] = defaultdict(list)
    seen_record_ids: set[str] = set()
    invalid_or_unkeyed_rows = 0
    processed_rows = 0

    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=["record_id", "smiles_canonical"],
    ):
        record_ids = batch.column("record_id").to_pylist()
        smiles_values = batch.column("smiles_canonical").to_pylist()

        for record_id, smiles in zip(record_ids, smiles_values):
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(
                    f"{path} contains an empty or non-string record_id at row "
                    f"{processed_rows}"
                )
            if record_id in seen_record_ids:
                raise ValueError(f"duplicate record_id in {path}: {record_id}")
            seen_record_ids.add(record_id)

            key = full_inchikey(smiles)
            if key is None:
                invalid_or_unkeyed_rows += 1
                continue
            record_ids_by_key[key].append(record_id)

        processed_rows += batch.num_rows
        if processed_rows % 250_000 == 0:
            print(f"Indexed {processed_rows:,} merged records")

    return record_ids_by_key, {
        "merged_records": processed_rows,
        "merged_invalid_or_unkeyed_rows": invalid_or_unkeyed_rows,
        "merged_unique_full_inchikeys": len(record_ids_by_key),
    }


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"{path} has no CSV header")
        if "record_id" in fieldnames:
            raise ValueError(f"{path} already contains a record_id column")
        return fieldnames, list(reader)


def write_csv_atomically(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[dict[str, str]],
    overwrite: bool,
) -> int:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.partial")
    if temporary_path.exists():
        raise FileExistsError(f"temporary output already exists: {temporary_path}")

    row_count = 0
    try:
        with temporary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
                row_count += 1
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    temporary_path.replace(path)
    return row_count


def validate_canonical_input(path: Path) -> parquet.ParquetFile:
    parquet_file = parquet.ParquetFile(path)
    expected_schema = canonical_parquet_schema()
    if not parquet_file.schema_arrow.remove_metadata().equals(expected_schema):
        raise ValueError(f"{path} does not use the current canonical schema")

    metadata = parquet_file.schema_arrow.metadata or {}
    version = metadata.get(b"canonical_schema_version", b"").decode("utf-8")
    if version != CANONICAL_PARQUET_SCHEMA_VERSION:
        raise ValueError(
            f"{path} uses canonical schema version {version!r}; "
            f"expected {CANONICAL_PARQUET_SCHEMA_VERSION!r}"
        )
    return parquet_file


def write_filtered_canonical(
    input_path: Path,
    output_path: Path,
    *,
    keep_record_ids: set[str] | None = None,
    remove_record_ids: set[str] | None = None,
    metadata_updates: dict[bytes, str],
    batch_size: int,
    overwrite: bool,
) -> int:
    """Copy a canonical Parquet file while filtering by record ID."""

    if (keep_record_ids is None) == (remove_record_ids is None):
        raise ValueError("provide exactly one of keep_record_ids or remove_record_ids")
    if output_path.resolve() == input_path.resolve():
        raise ValueError("output must not replace the input Parquet file")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_file = validate_canonical_input(input_path)
    input_metadata = dict(input_file.schema_arrow.metadata or {})
    for key, value in metadata_updates.items():
        input_metadata[key] = value.encode("utf-8")
    output_schema = input_file.schema_arrow.with_metadata(input_metadata)

    temporary_path = output_path.with_name(f".{output_path.name}.partial")
    if temporary_path.exists():
        raise FileExistsError(f"temporary output already exists: {temporary_path}")

    record_count = 0
    writer = parquet.ParquetWriter(
        temporary_path,
        output_schema,
        compression="zstd",
    )
    try:
        for batch in input_file.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([batch], schema=output_schema)
            record_id_column = table["record_id"]

            if keep_record_ids is not None:
                value_set = pa.array(
                    sorted(keep_record_ids),
                    type=record_id_column.type,
                )
                mask = compute.is_in(record_id_column, value_set=value_set)
            else:
                value_set = pa.array(
                    sorted(remove_record_ids or set()),
                    type=record_id_column.type,
                )
                mask = compute.invert(
                    compute.is_in(record_id_column, value_set=value_set)
                )

            filtered_table = table.filter(mask)
            if filtered_table.num_rows == 0:
                continue

            writer.write_table(
                filtered_table,
                row_group_size=filtered_table.num_rows,
            )
            record_count += filtered_table.num_rows
    except Exception:
        writer.close()
        temporary_path.unlink(missing_ok=True)
        raise

    writer.close()
    temporary_path.replace(output_path)
    return record_count


def write_dict_csv(path: Path, rows: list[dict[str, object]], overwrite: bool) -> None:
    if not rows:
        raise ValueError(f"cannot infer columns for empty report: {path}")
    fieldnames = list(rows[0])
    serialised_rows = [
        {key: "" if value is None else str(value) for key, value in row.items()}
        for row in rows
    ]
    write_csv_atomically(path, fieldnames, serialised_rows, overwrite)


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    for path in [args.merged, args.audit_matches, args.tdc_root]:
        if not path.exists():
            raise FileNotFoundError(path)

    audit_matches = load_audit_matches(args.audit_matches)
    audit_summary = load_audit_summary(args.audit_summary)
    record_ids_by_key, merged_summary = build_record_index(
        args.merged,
        batch_size=args.batch_size,
    )

    endpoint_sets: dict[str, set[str]] = {endpoint: set() for endpoint in ENDPOINTS}
    split_record_ids: dict[tuple[str, str], set[str]] = {}
    split_molecule_keys: dict[tuple[str, str], set[str]] = {}
    endpoint_stats: list[dict[str, object]] = []

    for endpoint in ENDPOINTS:
        endpoint_output = args.output_root / endpoint
        for split in SPLITS:
            release = (endpoint, split)
            source_path = args.tdc_root / endpoint / f"{split}.csv"
            if not source_path.is_file():
                raise FileNotFoundError(source_path)

            fieldnames, property_rows = read_csv_rows(source_path)
            entries = audit_matches.get(release, [])
            rows_by_source_index = {
                index: row for index, row in enumerate(property_rows)
            }
            expanded_rows: list[dict[str, str]] = []
            matched_record_ids: set[str] = set()
            matched_molecule_keys: set[str] = set()
            seen_source_rows: set[int] = set()

            for entry in entries:
                source_row = int(entry["source_row"])
                if source_row not in rows_by_source_index:
                    raise ValueError(
                        f"audit source row {source_row} is outside {source_path}"
                    )
                if source_row in seen_source_rows:
                    raise ValueError(
                        f"audit contains duplicate source row {source_row} for "
                        f"{endpoint}/{split}"
                    )
                seen_source_rows.add(source_row)

                key = str(entry["full_inchikey"])
                record_ids = record_ids_by_key.get(key, [])
                if not record_ids:
                    raise ValueError(
                        f"audit full InChIKey {key} is not present in the current "
                        f"merged file for {endpoint}/{split}"
                    )

                expected_count = entry["expected_record_count"]
                if expected_count is not None and len(record_ids) != expected_count:
                    raise ValueError(
                        f"record count mismatch for {endpoint}/{split}, {key}: "
                        f"audit={expected_count}, current={len(record_ids)}"
                    )

                property_row = rows_by_source_index[source_row]
                for record_id in record_ids:
                    expanded_row = dict(property_row)
                    expanded_row["record_id"] = record_id
                    expanded_rows.append(expanded_row)
                    matched_record_ids.add(record_id)

                matched_molecule_keys.add(key)

            csv_path = endpoint_output / f"{split}.csv"
            expanded_csv_rows = write_csv_atomically(
                csv_path,
                fieldnames + ["record_id"],
                expanded_rows,
                args.overwrite,
            )

            split_record_ids[release] = matched_record_ids
            split_molecule_keys[release] = matched_molecule_keys
            endpoint_sets[endpoint].update(matched_molecule_keys)

            summary_row = audit_summary.get(release, {})
            endpoint_stats.append(
                {
                    "dataset": endpoint,
                    "split": split,
                    "property_rows": len(property_rows),
                    "audit_property_unique_molecules": summary_row.get(
                        "property_unique_molecules", ""
                    ),
                    "audit_exact_overlap_unique_molecules": summary_row.get(
                        "exact_overlap_unique_molecules", ""
                    ),
                    "matched_property_rows": len(entries),
                    "matched_unique_molecules": len(matched_molecule_keys),
                    "matched_merged_records": len(matched_record_ids),
                    "expanded_csv_rows": expanded_csv_rows,
                }
            )

    for endpoint in ENDPOINTS:
        for split in SPLITS:
            release = (endpoint, split)
            output_path = args.output_root / endpoint / f"{split}.parquet"
            parquet_rows = write_filtered_canonical(
                args.merged,
                output_path,
                keep_record_ids=split_record_ids[release],
                metadata_updates={
                    b"source_dataset": (
                        f"merged_train_val_all exact matches for "
                        f"TDC ADMET {endpoint}/{split}"
                    ),
                    b"property_endpoint": endpoint,
                    b"property_split": split,
                    b"match_identity": "exact RDKit full InChIKey equality",
                    b"extraction_script": str(Path(__file__).resolve()),
                },
                batch_size=args.batch_size,
                overwrite=args.overwrite,
            )

            for row in endpoint_stats:
                if row["dataset"] == endpoint and row["split"] == split:
                    row["canonical_parquet_rows"] = parquet_rows
                    break

    removed_record_ids: set[str] = set()
    for record_ids in split_record_ids.values():
        removed_record_ids.update(record_ids)

    disjoint_rows = write_filtered_canonical(
        args.merged,
        args.disjoint_output,
        remove_record_ids=removed_record_ids,
        metadata_updates={
            b"source_dataset": "merged_train_val_all minus exact TDC property matches",
            b"match_identity": "exact RDKit full InChIKey equality",
            b"removed_record_count": str(len(removed_record_ids)),
            b"removed_property_endpoints": json.dumps(list(ENDPOINTS)),
            b"extraction_script": str(Path(__file__).resolve()),
        },
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )

    intersection_rows: list[dict[str, object]] = []
    for endpoint in ENDPOINTS:
        intersection_rows.append(
            {
                "intersection_type": "individual",
                "datasets": endpoint,
                "unique_molecules": len(endpoint_sets[endpoint]),
            }
        )
    for first_index, first_endpoint in enumerate(ENDPOINTS):
        for second_endpoint in ENDPOINTS[first_index + 1 :]:
            shared = endpoint_sets[first_endpoint] & endpoint_sets[second_endpoint]
            intersection_rows.append(
                {
                    "intersection_type": "pairwise",
                    "datasets": f"{first_endpoint}|{second_endpoint}",
                    "unique_molecules": len(shared),
                }
            )

    triple_shared = set.intersection(*(endpoint_sets[endpoint] for endpoint in ENDPOINTS))
    union_keys = set.union(*(endpoint_sets[endpoint] for endpoint in ENDPOINTS))
    intersection_rows.extend(
        [
            {
                "intersection_type": "triple",
                "datasets": "|".join(ENDPOINTS),
                "unique_molecules": len(triple_shared),
            },
            {
                "intersection_type": "union",
                "datasets": "|".join(ENDPOINTS),
                "unique_molecules": len(union_keys),
            },
        ]
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_dict_csv(
        args.output_root / "intersection_summary.csv",
        intersection_rows,
        args.overwrite,
    )
    write_dict_csv(
        args.output_root / "production_summary.csv",
        endpoint_stats,
        args.overwrite,
    )

    manifest = {
        "match_policy": "exact RDKit full InChIKey equality only",
        "merged_input": str(args.merged),
        "audit_matches": str(args.audit_matches),
        "tdc_root": str(args.tdc_root),
        "output_root": str(args.output_root),
        "disjoint_output": str(args.disjoint_output),
        "merged_summary": merged_summary,
        "endpoint_stats": endpoint_stats,
        "intersections": intersection_rows,
        "removed_record_ids": len(removed_record_ids),
        "disjoint_output_rows": disjoint_rows,
        "removed_record_check": merged_summary["merged_records"]
        - disjoint_rows
        == len(removed_record_ids),
    }
    manifest_path = args.output_root / "provenance.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create cleaned ADMET cohorts and remove their records from train/validation.

The small TDC property tables are read first and converted to RDKit full
InChIKeys. The canonical NMR input is then streamed once while retaining only
records matching those requested keys. Repeated agreeing labels are collapsed;
molecules with discordant labels are excluded from the supervised cohort but
remain excluded from pretraining to avoid label leakage.

Run this step after extending and filtering the rich train/validation dataset.
All Parquet outputs therefore remain compliant with the common cleaning policy
without another filtering pass.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from rdkit import Chem, RDLogger

from data.postprocess.common import (
    open_canonical_parquet,
    prepare_parquet_output,
    progress_bar,
    write_filtered_parquet,
)


RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")

ENDPOINTS = ("solubility_aqsoldb", "ld50_zhu", "ames")
SPLITS = ("train_val", "test")
SCRIPT_PATH = "scripts/data/postprocess/prepare_admet_datasets.py"
IDENTITY_DESCRIPTION = "exact RDKit full InChIKey equality"


def full_inchikey(smiles: object) -> str | None:
    """Return the RDKit full InChIKey, or ``None`` for an unusable structure."""

    if not isinstance(smiles, str) or not smiles.strip():
        return None
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    try:
        return Chem.MolToInchiKey(molecule)
    except Exception:
        return None


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read one small property CSV while preserving its original columns."""

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"{path} has no CSV header")
        if "record_id" in fieldnames:
            raise ValueError(f"{path} already contains a record_id column")
        return fieldnames, list(reader)


def load_property_releases(
    tdc_root: Path,
    smiles_column: str,
    label_column: str,
) -> tuple[
    dict[tuple[str, str], dict[str, object]],
    set[str],
]:
    """Load the small ADMET files and collect the molecular keys to request."""

    releases: dict[tuple[str, str], dict[str, object]] = {}
    requested_keys: set[str] = set()
    endpoint_split_keys: dict[tuple[str, str], set[str]] = {}

    for endpoint in ENDPOINTS:
        for split in SPLITS:
            path = tdc_root / endpoint / f"{split}.csv"
            if not path.is_file():
                raise FileNotFoundError(path)
            fieldnames, rows = read_csv_rows(path)
            for required_column in (smiles_column, label_column):
                if required_column not in fieldnames:
                    raise ValueError(
                        f"{path} is missing required column {required_column!r}"
                    )

            keyed_rows: list[tuple[dict[str, str], str]] = []
            invalid_rows = 0
            release_keys: set[str] = set()
            for row in rows:
                key = full_inchikey(row.get(smiles_column))
                if key is None:
                    invalid_rows += 1
                    continue
                keyed_rows.append((row, key))
                release_keys.add(key)

            release = (endpoint, split)
            endpoint_split_keys[release] = release_keys
            requested_keys.update(release_keys)
            releases[release] = {
                "path": path,
                "fieldnames": fieldnames,
                "rows": rows,
                "keyed_rows": keyed_rows,
                "invalid_rows": invalid_rows,
                "unique_keys": len(release_keys),
            }

        shared_keys = (
            endpoint_split_keys[(endpoint, "train_val")]
            & endpoint_split_keys[(endpoint, "test")]
        )
        if shared_keys:
            raise ValueError(
                f"{endpoint} has {len(shared_keys):,} molecular identities in "
                "both train_val and test"
            )

    return releases, requested_keys


def index_requested_nmr_records(
    input_path: Path,
    requested_keys: set[str],
    *,
    batch_size: int,
    show_progress: bool,
):
    """Stream the NMR dataset and retain record IDs only for requested keys."""

    input_file = open_canonical_parquet(input_path)
    record_ids_by_key: dict[str, list[str]] = defaultdict(list)
    matched_record_ids: set[str] = set()
    invalid_or_unkeyed_rows = 0
    processed_rows = 0
    progress = progress_bar(
        input_file.metadata.num_rows,
        f"Matching {input_path.name} to ADMET",
        show_progress,
    )
    try:
        for batch in input_file.iter_batches(
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
                        f"{input_path} has an invalid record_id at row "
                        f"{row_number:,}"
                    )
                key = full_inchikey(smiles)
                if key is None:
                    invalid_or_unkeyed_rows += 1
                    continue
                if key not in requested_keys:
                    continue
                if record_id in matched_record_ids:
                    raise ValueError(
                        f"{input_path} contains duplicate matched record_id "
                        f"{record_id!r}"
                    )
                matched_record_ids.add(record_id)
                record_ids_by_key[key].append(record_id)

            processed_rows += batch.num_rows
            if progress is not None:
                progress.update(batch.num_rows)
    finally:
        if progress is not None:
            progress.close()

    summary = {
        "input_records": processed_rows,
        "input_invalid_or_unkeyed_records": invalid_or_unkeyed_rows,
        "requested_molecules": len(requested_keys),
        "matched_molecules": len(record_ids_by_key),
        "matched_records": len(matched_record_ids),
    }
    return input_file, record_ids_by_key, summary


def numeric_label(value: object, location: str) -> float:
    """Parse one finite endpoint label for agreement checks."""

    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid endpoint label at {location}: {value!r}") from error
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"non-finite endpoint label at {location}: {value!r}")
    return number


def prepare_property_release(
    keyed_rows: list[tuple[dict[str, str], str]],
    record_ids_by_key: dict[str, list[str]],
    *,
    label_column: str,
    location: str,
) -> tuple[list[dict[str, str]], set[str], set[str], dict[str, int]]:
    """Expand one release after collapsing or rejecting repeated labels."""

    property_rows_by_key: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row, key in keyed_rows:
        property_rows_by_key[key].append(row)

    output_rows: list[dict[str, str]] = []
    valid_record_ids: set[str] = set()
    all_matched_record_ids: set[str] = set()
    matched_property_rows = 0
    agreeing_duplicate_molecules = 0
    discordant_molecules = 0

    for key, property_rows in property_rows_by_key.items():
        record_ids = record_ids_by_key.get(key, [])
        if not record_ids:
            continue
        matched_property_rows += len(property_rows)
        all_matched_record_ids.update(record_ids)

        labels = {
            numeric_label(row.get(label_column), f"{location}/{key}")
            for row in property_rows
        }
        if len(labels) > 1:
            discordant_molecules += 1
            continue
        if len(property_rows) > 1:
            agreeing_duplicate_molecules += 1

        representative = property_rows[0]
        for record_id in record_ids:
            output_row = dict(representative)
            output_row["record_id"] = record_id
            output_rows.append(output_row)
            valid_record_ids.add(record_id)

    stats = {
        "matched_property_rows": matched_property_rows,
        "matched_records_before_label_cleanup": len(all_matched_record_ids),
        "output_records": len(valid_record_ids),
        "agreeing_duplicate_molecules": agreeing_duplicate_molecules,
        "discordant_molecules": discordant_molecules,
    }
    return output_rows, valid_record_ids, all_matched_record_ids, stats


def prepare_temporary_file(path: Path, overwrite: bool) -> Path:
    """Validate a non-Parquet output and return its temporary path."""

    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    return temporary


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[dict[str, str]],
) -> int:
    """Write rows to a prepared temporary CSV path."""

    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def prepare_admet_datasets(
    input_path: str | Path,
    tdc_root: str | Path,
    output_root: str | Path,
    train_output_path: str | Path,
    *,
    smiles_column: str = "Drug",
    label_column: str = "Y",
    batch_size: int = 50_000,
    overwrite: bool = False,
    show_progress: bool = True,
) -> dict[str, object]:
    """Create final ADMET cohorts and their ADMET-disjoint training pool."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    input_file_path = Path(input_path)
    property_root = Path(tdc_root)
    cohort_root = Path(output_root)
    train_output = Path(train_output_path)

    releases, requested_keys = load_property_releases(
        property_root,
        smiles_column,
        label_column,
    )
    input_file, record_ids_by_key, input_summary = index_requested_nmr_records(
        input_file_path,
        requested_keys,
        batch_size=batch_size,
        show_progress=show_progress,
    )

    planned_parquets: dict[tuple[str, str] | str, tuple[Path, Path]] = {}
    planned_csvs: dict[tuple[str, str], tuple[Path, Path]] = {}
    all_temporary_paths: list[Path] = []
    for endpoint in ENDPOINTS:
        for split in SPLITS:
            release = (endpoint, split)
            parquet_output, parquet_temporary = prepare_parquet_output(
                cohort_root / endpoint / f"{split}.parquet",
                [input_file_path],
                overwrite=overwrite,
            )
            csv_output = cohort_root / endpoint / f"{split}.csv"
            csv_temporary = prepare_temporary_file(csv_output, overwrite)
            planned_parquets[release] = (parquet_output, parquet_temporary)
            planned_csvs[release] = (csv_output, csv_temporary)
            all_temporary_paths.extend([parquet_temporary, csv_temporary])

    final_train, train_temporary = prepare_parquet_output(
        train_output,
        [input_file_path],
        overwrite=overwrite,
    )
    planned_parquets["train"] = (final_train, train_temporary)
    all_temporary_paths.append(train_temporary)
    report_output = cohort_root / "preparation_report.json"
    report_temporary = prepare_temporary_file(report_output, overwrite)
    all_temporary_paths.append(report_temporary)

    final_outputs = [
        *[output for output, _ in planned_parquets.values()],
        *[output for output, _ in planned_csvs.values()],
        report_output,
    ]
    resolved_outputs = [path.resolve() for path in final_outputs]
    if len(set(resolved_outputs)) != len(resolved_outputs):
        raise ValueError(
            "ADMET Parquet, CSV, train, and report outputs must all use "
            "different paths"
        )

    all_matched_record_ids: set[str] = set()
    release_results: list[dict[str, object]] = []
    try:
        for endpoint in ENDPOINTS:
            for split in SPLITS:
                release = (endpoint, split)
                release_data = releases[release]
                (
                    label_rows,
                    valid_record_ids,
                    matched_record_ids,
                    label_stats,
                ) = prepare_property_release(
                    release_data["keyed_rows"],
                    record_ids_by_key,
                    label_column=label_column,
                    location=f"{endpoint}/{split}",
                )
                all_matched_record_ids.update(matched_record_ids)

                csv_output, csv_temporary = planned_csvs[release]
                csv_rows = write_csv(
                    csv_temporary,
                    [*release_data["fieldnames"], "record_id"],
                    label_rows,
                )
                parquet_output, parquet_temporary = planned_parquets[release]
                metadata = dict(input_file.schema_arrow.metadata or {})
                metadata.update(
                    {
                        b"postprocess_step": b"prepare_admet_datasets",
                        b"postprocess_script": SCRIPT_PATH.encode("utf-8"),
                        b"postprocess_output_role": b"admet_cohort",
                        b"property_endpoint": endpoint.encode("utf-8"),
                        b"property_split": split.encode("utf-8"),
                        b"molecule_identity": IDENTITY_DESCRIPTION.encode("utf-8"),
                    }
                )
                parquet_rows = write_filtered_parquet(
                    input_file,
                    input_file.schema_arrow.with_metadata(metadata),
                    parquet_temporary,
                    valid_record_ids,
                    keep=True,
                    batch_size=batch_size,
                    description=f"Writing ADMET {endpoint}/{split}",
                    show_progress=show_progress,
                )
                if csv_rows != parquet_rows:
                    raise RuntimeError(
                        f"{endpoint}/{split} CSV and Parquet row counts differ"
                    )

                release_results.append(
                    {
                        "endpoint": endpoint,
                        "split": split,
                        "property_rows": len(release_data["rows"]),
                        "property_invalid_or_unkeyed_rows": release_data[
                            "invalid_rows"
                        ],
                        "property_unique_molecules": release_data["unique_keys"],
                        **label_stats,
                        "csv_output": str(csv_output),
                        "parquet_output": str(parquet_output),
                    }
                )

        train_metadata = dict(input_file.schema_arrow.metadata or {})
        train_metadata.update(
            {
                b"postprocess_step": b"prepare_admet_datasets",
                b"postprocess_script": SCRIPT_PATH.encode("utf-8"),
                b"postprocess_output_role": b"admet_disjoint_train_validation",
                b"molecule_identity": IDENTITY_DESCRIPTION.encode("utf-8"),
                b"removed_record_count": str(len(all_matched_record_ids)).encode(
                    "utf-8"
                ),
            }
        )
        train_rows = write_filtered_parquet(
            input_file,
            input_file.schema_arrow.with_metadata(train_metadata),
            train_temporary,
            all_matched_record_ids,
            keep=False,
            batch_size=batch_size,
            description="Writing ADMET-disjoint train/validation",
            show_progress=show_progress,
        )
        expected_train_rows = input_file.metadata.num_rows - len(
            all_matched_record_ids
        )
        if train_rows != expected_train_rows:
            raise RuntimeError("ADMET-disjoint train row count is inconsistent")

        report = {
            "input": str(input_file_path),
            "tdc_root": str(property_root),
            "output_root": str(cohort_root),
            "train_output": str(final_train),
            "match_policy": IDENTITY_DESCRIPTION,
            "input_summary": input_summary,
            "release_results": release_results,
            "removed_pretraining_records": len(all_matched_record_ids),
            "train_output_rows": train_rows,
        }
        report_temporary.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )

        for endpoint in ENDPOINTS:
            for split in SPLITS:
                release = (endpoint, split)
                csv_output, csv_temporary = planned_csvs[release]
                parquet_output, parquet_temporary = planned_parquets[release]
                csv_temporary.replace(csv_output)
                parquet_temporary.replace(parquet_output)
        train_temporary.replace(final_train)
        report_temporary.replace(report_output)
    except Exception:
        for temporary in all_temporary_paths:
            temporary.unlink(missing_ok=True)
        raise

    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="cleaned rich train/validation")
    parser.add_argument("--tdc-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--train-output", required=True, type=Path)
    parser.add_argument("--smiles-column", default="Drug")
    parser.add_argument("--label-column", default="Y")
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    report = prepare_admet_datasets(
        args.input,
        args.tdc_root,
        args.output_root,
        args.train_output,
        smiles_column=args.smiles_column,
        label_column=args.label_column,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
        show_progress=not args.no_progress,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

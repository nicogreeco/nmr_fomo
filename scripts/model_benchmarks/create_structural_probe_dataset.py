#!/usr/bin/env python3
"""Build a fixed molecule-disjoint train/validation set for structural probes.

The input is the existing Rich foundation-training split.  A molecule is the
exact ``smiles_canonical`` value used by the NMR split pipeline.  For every
eligible molecule this script keeps the lexicographically smallest record ID;
eligibility requires non-empty H and C peak lists and a complete, successful
RDKit property sidecar row.  Molecules are ordered by a seeded SHA-256 score,
then divided into 50k train and 10k validation molecules.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

import pyarrow as pa
from pyarrow import compute, parquet

from data.console import add_console_arguments, configure_console, progress_bar
from data.postprocess.common import open_canonical_parquet, prepare_parquet_output
from data.reporting import prepare_report_output, write_processing_report


FUNCTIONAL_GROUPS = (
    "has_amine", "has_amide", "has_alcohol_or_phenol", "has_ester",
    "has_carboxylic_acid", "has_aldehyde_or_ketone", "has_nitrile",
    "has_halogenated_group", "has_heteroaromatic_ring",
)
DESCRIPTORS = (
    "exact_molecular_weight", "tpsa", "hba", "hbd", "fraction_csp3",
    "aromatic_atom_fraction", "rotatable_bonds", "calculated_logp",
)


def molecule_score(smiles: str, seed: int) -> int:
    """Return a stable random ordering key without depending on row order."""

    digest = hashlib.sha256(f"{seed}\0{smiles}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def valid_property_row(row: dict[str, object]) -> tuple[bool, str | None]:
    if row["rdkit_status"] != "ok":
        return False, "rdkit_status"
    for field in FUNCTIONAL_GROUPS:
        if row[field] not in (0, 1):
            return False, field
    for field in DESCRIPTORS:
        value = row[field]
        if not isinstance(value, (float, int)) or not float("-inf") < float(value) < float("inf"):
            return False, field
    return True, None


def index_candidates(nmr_path: Path, property_path: Path, batch_size: int, show_progress: bool):
    """Choose the documented representative while checking sidecar alignment."""

    nmr_file = open_canonical_parquet(nmr_path)
    property_file = parquet.ParquetFile(property_path)
    if nmr_file.metadata.num_rows != property_file.metadata.num_rows:
        raise ValueError("NMR and property sidecar row counts differ")
    if nmr_file.num_row_groups != property_file.num_row_groups:
        raise ValueError("NMR and property sidecar row groups differ")
    required = {"record_id", "smiles_canonical", "rdkit_status", *FUNCTIONAL_GROUPS, *DESCRIPTORS}
    missing = required - set(property_file.schema_arrow.names)
    if missing:
        raise ValueError(f"property sidecar is missing fields: {sorted(missing)}")

    representatives: dict[str, dict[str, str]] = {}
    invalid = Counter()
    progress = progress_bar(nmr_file.metadata.num_rows, "Indexing Rich molecules", show_progress)
    try:
        for index in range(nmr_file.num_row_groups):
            nmr = nmr_file.read_row_group(index, columns=["record_id", "smiles_canonical", "source", "h_nmr_peaks", "c_nmr_peaks"])
            props = property_file.read_row_group(index, columns=sorted(required))
            nmr_ids = nmr["record_id"].to_pylist()
            prop_ids = props["record_id"].to_pylist()
            nmr_smiles = nmr["smiles_canonical"].to_pylist()
            prop_smiles = props["smiles_canonical"].to_pylist()
            if nmr_ids != prop_ids or nmr_smiles != prop_smiles:
                raise ValueError(f"NMR/property alignment failure in row group {index}")
            rows = props.to_pylist()
            for record_id, smiles, source, h_peaks, c_peaks, prop_row in zip(
                nmr_ids, nmr_smiles, nmr["source"].to_pylist(),
                nmr["h_nmr_peaks"].to_pylist(), nmr["c_nmr_peaks"].to_pylist(), rows,
            ):
                if not isinstance(smiles, str) or not smiles or not isinstance(record_id, str) or not record_id:
                    raise ValueError("Rich input has an empty record_id or smiles_canonical")
                if not h_peaks or not c_peaks:
                    invalid["missing_modality"] += 1
                    continue
                valid, reason = valid_property_row(prop_row)
                if not valid:
                    invalid[f"invalid_{reason}"] += 1
                    continue
                previous = representatives.get(smiles)
                if previous is None or record_id < previous["record_id"]:
                    representatives[smiles] = {"record_id": record_id, "source": source or ""}
            if progress is not None:
                progress.update(nmr.num_rows)
    finally:
        if progress is not None:
            progress.close()
    return representatives, dict(sorted(invalid.items()))


def write_selected_pairs(nmr_path: Path, property_path: Path, output_paths: dict[str, Path], selected: dict[str, dict[str, dict[str, str]]], show_progress: bool):
    nmr_file = open_canonical_parquet(nmr_path)
    property_file = parquet.ParquetFile(property_path)
    record_split = {entry["record_id"]: split for split, records in selected.items() for entry in records.values()}
    writers = {
        key: parquet.ParquetWriter(output_paths[key], schema, compression="zstd")
        for key, schema in {
            "train": nmr_file.schema_arrow, "validation": nmr_file.schema_arrow,
            "train_properties": property_file.schema_arrow, "validation_properties": property_file.schema_arrow,
        }.items()
    }
    counts = Counter()
    progress = progress_bar(nmr_file.metadata.num_rows, "Writing structural probe splits", show_progress)
    try:
        for index in range(nmr_file.num_row_groups):
            nmr = nmr_file.read_row_group(index)
            props = property_file.read_row_group(index)
            if nmr["record_id"].to_pylist() != props["record_id"].to_pylist():
                raise ValueError(f"record_id alignment failure in row group {index}")
            ids = nmr["record_id"].to_pylist()
            for split in ("train", "validation"):
                mask = pa.array([record_split.get(record_id) == split for record_id in ids])
                selected_nmr = nmr.filter(mask)
                if selected_nmr.num_rows:
                    writers[split].write_table(selected_nmr)
                    writers[f"{split}_properties"].write_table(props.filter(mask))
                    counts[split] += selected_nmr.num_rows
            if progress is not None:
                progress.update(nmr.num_rows)
    finally:
        for writer in writers.values():
            writer.close()
        if progress is not None:
            progress.close()
    return dict(counts)


def test_overlap(test_path: Path, selected_smiles: set[str], batch_size: int) -> set[str]:
    test_file = open_canonical_parquet(test_path)
    values = pa.array(sorted(selected_smiles), type=test_file.schema_arrow.field("smiles_canonical").type)
    overlap = set()
    for batch in test_file.iter_batches(batch_size=batch_size, columns=["smiles_canonical"]):
        matched = compute.filter(batch["smiles_canonical"], compute.is_in(batch["smiles_canonical"], value_set=values))
        overlap.update(value for value in matched.to_pylist() if value)
    return overlap


def run(args):
    nmr_path, property_path, test_path = Path(args.nmr_input), Path(args.property_input), Path(args.test_input)
    output_root = Path(args.output_root)
    representatives, invalid = index_candidates(nmr_path, property_path, args.batch_size, not args.no_progress)
    required = args.train_molecules + args.validation_molecules
    if len(representatives) < required:
        raise ValueError(f"only {len(representatives):,} eligible molecules; need {required:,}")
    ordered = sorted(representatives, key=lambda smiles: (molecule_score(smiles, args.seed), smiles))
    selected = {
        "train": {smiles: representatives[smiles] for smiles in ordered[:args.train_molecules]},
        "validation": {smiles: representatives[smiles] for smiles in ordered[args.train_molecules:required]},
    }
    if set(selected["train"]) & set(selected["validation"]):
        raise RuntimeError("train and validation molecule selections overlap")
    overlap = test_overlap(test_path, set(selected["train"]) | set(selected["validation"]), args.batch_size)
    if overlap:
        raise ValueError(f"selected molecules overlap held-out benchmark: {len(overlap)}")

    finals = {
        "train": output_root / "train.parquet", "validation": output_root / "validation.parquet",
        "train_properties": output_root / "train_mol_properties.parquet",
        "validation_properties": output_root / "validation_mol_properties.parquet",
    }
    temporaries = {}
    for name, output in finals.items():
        finals[name], temporaries[name] = prepare_parquet_output(output, [nmr_path, property_path, test_path], overwrite=args.overwrite)
    manifest_path, manifest_tmp = prepare_report_output(output_root / "split_manifest.json", [nmr_path, property_path, test_path, *finals.values()], overwrite=args.overwrite)
    report_path, report_tmp = prepare_report_output(Path(args.report_output or output_root / "build_report.json"), [nmr_path, property_path, test_path, *finals.values()], overwrite=args.overwrite)
    try:
        counts = write_selected_pairs(nmr_path, property_path, temporaries, selected, not args.no_progress)
        if counts != {"train": args.train_molecules, "validation": args.validation_molecules}:
            raise RuntimeError(f"written records do not match selected molecules: {counts}")
        for name in finals:
            temporaries[name].replace(finals[name])
    except Exception:
        for path in temporaries.values():
            path.unlink(missing_ok=True)
        raise
    manifest_records = [
        {"molecule_id": smiles, "split": split, "record_id": entry["record_id"], "source": entry["source"]}
        for split, entries in selected.items() for smiles, entry in sorted(entries.items())
    ]
    manifest = {"stage": "create_structural_probe_dataset", "seed": args.seed, "molecule_identity": "exact smiles_canonical", "representative_rule": "lexicographically smallest eligible record_id", "inputs": {"rich_train": str(nmr_path), "rich_properties": str(property_path), "held_out_test": str(test_path)}, "records": manifest_records}
    write_processing_report(manifest, manifest_path, manifest_tmp)
    write_processing_report({"stage": "create_structural_probe_dataset", "inputs": manifest["inputs"], "outputs": {name: str(path) for name, path in finals.items()}, "counts": {"train": {"records": counts["train"], "unique_molecules": len(selected["train"])}, "validation": {"records": counts["validation"], "unique_molecules": len(selected["validation"])}, "duplicate_molecule_ids": 0, "held_out_test_overlap": 0}, "details": {"seed": args.seed, "invalid_or_ineligible_rows": invalid, "required_functional_groups": list(FUNCTIONAL_GROUPS), "required_descriptors": list(DESCRIPTORS), "manifest": str(manifest_path)}}, report_path, report_tmp)
    print(f"saved {counts['train']:,} train and {counts['validation']:,} validation spectra under {output_root}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nmr-input", default="datasets/train_splits/rich_train.parquet")
    parser.add_argument("--property-input", default="datasets/train_splits/rich_train_mol_properties.parquet")
    parser.add_argument("--test-input", default="datasets/cleaned/test_benchmark.parquet")
    parser.add_argument("--output-root", default="datasets/downstream_structural_information")
    parser.add_argument("--train-molecules", type=int, default=50_000)
    parser.add_argument("--validation-molecules", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--report-output")
    parser.add_argument("--overwrite", action="store_true")
    add_console_arguments(parser)
    args = parser.parse_args()
    if args.train_molecules < 1 or args.validation_molecules < 1 or args.batch_size < 1:
        parser.error("molecule counts and batch size must be positive")
    configure_console(args.quiet)
    run(args)


if __name__ == "__main__":
    main()

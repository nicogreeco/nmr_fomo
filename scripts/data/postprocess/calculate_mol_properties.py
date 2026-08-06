#!/usr/bin/env python3
"""Export RDKit molecular properties from a canonical Parquet dataset.

The input is streamed in Arrow batches and only ``record_id`` and
``smiles_canonical`` are read. Each batch is divided into small tasks for a
process pool, while the resulting rows are written to CSV immediately. This
keeps memory bounded for large canonical datasets.

The CSV contains one row per input record. Invalid or missing SMILES are not
silently dropped: their descriptor fields are empty and ``rdkit_status`` plus
``rdkit_error`` explain why.

Morgan ECFP4 fingerprints use radius 2 and 2,048 bits. They are stored as
RDKit binary fingerprint bytes encoded as hexadecimal (512 characters), which
is reversible and substantially smaller than a 2,048-character bit string.
RDKit's MACCS vector has 167 positions because bit 0 is unused; the output
contains positions 1--166 as ``maccs_keys_166_bits``.

Example:
    PYTHONPATH=scripts python scripts/data/postprocess/calculate_mol_properties.py \\
        datasets/merged/merged_train_val_all.parquet --workers 8
"""

from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Iterator

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import (
    Crippen,
    Descriptors,
    MACCSkeys,
    rdFingerprintGenerator,
    rdMolDescriptors,
)


# Invalid source structures are represented explicitly in the CSV. Suppress
# RDKit's per-record stderr output so a very large run remains readable.
RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")

MORGAN_RADIUS = 2
MORGAN_FP_SIZE = 2048
MACCS_OUTPUT_BITS = 166

FUNCTIONAL_GROUP_SMARTS = {
    "has_amine": "[NX3;!$(N[C,S]=O);!$(N=*)]",
    "has_amide": "[NX3][CX3](=[OX1])",
    "has_alcohol_or_phenol": "[$([OX2H][CX4]),$([OX2H][c])]",
    "has_ester": "[CX3](=[OX1])[OX2][#6]",
    "has_carboxylic_acid": "[CX3](=[OX1])[OX2H1]",
    "has_aldehyde_or_ketone": "[$([CX3H1](=O)),$([CX3](=O)([#6])[#6])]",
    "has_nitrile": "[CX2]#[NX1]",
    "has_halogenated_group": "[#6][F,Cl,Br,I]",
    "has_heteroaromatic_ring": "[n,o,s,p]",
}

FUNCTIONAL_GROUP_PATTERNS = {
    name: Chem.MolFromSmarts(smarts)
    for name, smarts in FUNCTIONAL_GROUP_SMARTS.items()
}
for _name, _pattern in FUNCTIONAL_GROUP_PATTERNS.items():
    if _pattern is None:
        raise RuntimeError(f"could not compile SMARTS for {_name}")

MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS,
    fpSize=MORGAN_FP_SIZE,
)

PROPERTY_FIELDS = [
    "exact_molecular_weight",
    "calculated_logp",
    "tpsa",
    "hba",
    "hbd",
    "rotatable_bonds",
    "fraction_csp3",
    "aromatic_atom_fraction",
]

CSV_FIELDS = [
    "record_id",
    "smiles_canonical",
    "rdkit_status",
    "rdkit_error",
    *PROPERTY_FIELDS,
    *FUNCTIONAL_GROUP_SMARTS,
    "morgan_ecfp4_2048_hex",
    "maccs_keys_166_bits",
]


def default_output_path(input_path: Path) -> Path:
    """Return the same-directory CSV name requested for one Parquet input."""

    if input_path.suffix.lower() != ".parquet":
        raise ValueError(f"input must have a .parquet suffix: {input_path}")
    return input_path.with_name(f"{input_path.stem}_mol_properties.csv")


def empty_property_values() -> dict[str, object]:
    """Return blank descriptor and fingerprint values for an unusable SMILES."""

    return {field: None for field in CSV_FIELDS[4:]}


def unusable_row(
    record_id: str,
    smiles: object,
    status: str,
    error: str,
) -> dict[str, object]:
    """Represent a missing or invalid structure without omitting its record."""

    row = {
        "record_id": record_id,
        "smiles_canonical": smiles if isinstance(smiles, str) else None,
        "rdkit_status": status,
        "rdkit_error": error,
    }
    row.update(empty_property_values())
    return row


def calculate_row(record_id: str, smiles: object) -> dict[str, object]:
    """Calculate all requested properties for one canonical SMILES string."""

    if not isinstance(smiles, str) or not smiles.strip():
        return unusable_row(
            record_id,
            smiles,
            "missing_smiles",
            "smiles_canonical is missing or empty",
        )

    smiles = smiles.strip()
    try:
        molecule = Chem.MolFromSmiles(smiles)
    except Exception as error:  # RDKit normally returns None, but retain context.
        return unusable_row(record_id, smiles, "invalid_smiles", str(error))

    if molecule is None:
        return unusable_row(
            record_id,
            smiles,
            "invalid_smiles",
            "RDKit could not parse smiles_canonical",
        )

    try:
        heavy_atoms = [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1]
        aromatic_heavy_atoms = sum(atom.GetIsAromatic() for atom in heavy_atoms)
        aromatic_atom_fraction = (
            aromatic_heavy_atoms / len(heavy_atoms) if heavy_atoms else None
        )

        maccs_bits = DataStructs.BitVectToText(MACCSkeys.GenMACCSKeys(molecule))
        if len(maccs_bits) != MACCS_OUTPUT_BITS + 1:
            raise RuntimeError(
                "unexpected RDKit MACCS vector length "
                f"{len(maccs_bits)}; expected {MACCS_OUTPUT_BITS + 1}"
            )

        row: dict[str, object] = {
            "record_id": record_id,
            "smiles_canonical": smiles,
            "rdkit_status": "ok",
            "rdkit_error": None,
            "exact_molecular_weight": Descriptors.ExactMolWt(molecule),
            "calculated_logp": Crippen.MolLogP(molecule),
            "tpsa": rdMolDescriptors.CalcTPSA(molecule),
            "hba": rdMolDescriptors.CalcNumHBA(molecule),
            "hbd": rdMolDescriptors.CalcNumHBD(molecule),
            "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(molecule),
            "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(molecule),
            "aromatic_atom_fraction": aromatic_atom_fraction,
            # Recover the original 2,048-bit RDKit vector with:
            # DataStructs.CreateFromBinaryText(bytes.fromhex(morgan_hex)).
            "morgan_ecfp4_2048_hex": DataStructs.BitVectToBinaryText(
                MORGAN_GENERATOR.GetFingerprint(molecule)
            ).hex(),
            "maccs_keys_166_bits": maccs_bits[1:],
        }
        for name, pattern in FUNCTIONAL_GROUP_PATTERNS.items():
            row[name] = int(molecule.HasSubstructMatch(pattern))
        return row
    except Exception as error:
        return unusable_row(record_id, smiles, "calculation_error", str(error))


def calculate_chunk(records: list[tuple[str, object]]) -> list[dict[str, object]]:
    """Worker entry point that calculates a small ordered group of records."""

    return [calculate_row(record_id, smiles) for record_id, smiles in records]


def iter_chunks(
    record_ids: list[object],
    smiles_values: list[object],
    records_per_task: int,
    first_row_index: int,
) -> Iterator[list[tuple[str, object]]]:
    """Split one Arrow batch into bounded process-pool tasks in input order."""

    if len(record_ids) != len(smiles_values):
        raise RuntimeError("record_id and smiles_canonical columns have different lengths")

    current_chunk: list[tuple[str, object]] = []
    for offset, (record_id, smiles) in enumerate(zip(record_ids, smiles_values)):
        if not isinstance(record_id, str) or not record_id:
            row_number = first_row_index + offset + 1
            raise ValueError(
                f"input has an empty or non-string record_id at row {row_number}"
            )
        current_chunk.append((record_id, smiles))
        if len(current_chunk) == records_per_task:
            yield current_chunk
            current_chunk = []

    if current_chunk:
        yield current_chunk


def _write_rows(
    writer: csv.DictWriter,
    row_groups: Iterable[list[dict[str, object]]],
    status_counts: dict[str, int],
) -> int:
    """Write result groups immediately and count their explicit RDKit statuses."""

    row_count = 0
    for rows in row_groups:
        writer.writerows(rows)
        row_count += len(rows)
        for row in rows:
            status = str(row["rdkit_status"])
            status_counts[status] = status_counts.get(status, 0) + 1
    return row_count


def export_molecular_properties(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    batch_size: int = 4096,
    records_per_task: int = 512,
    workers: int | None = None,
    progress_every: int = 250_000,
    overwrite: bool = False,
) -> dict[str, object]:
    """Stream a canonical Parquet file into a CSV of RDKit properties.

    ``workers=1`` processes batches in the parent process. Higher values use
    a process pool; at most one Arrow batch and its submitted task results are
    retained at a time.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if records_per_task < 1:
        raise ValueError("records_per_task must be at least 1")
    if progress_every < 1:
        raise ValueError("progress_every must be at least 1")

    input_file_path = Path(input_path)
    if not input_file_path.is_file():
        raise FileNotFoundError(f"canonical Parquet input not found: {input_file_path}")
    output_file_path = (
        default_output_path(input_file_path)
        if output_path is None
        else Path(output_path)
    )
    if output_file_path.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite {output_file_path}; pass --overwrite to replace it"
        )
    if output_file_path.resolve() == input_file_path.resolve():
        raise ValueError("CSV output must not replace the input Parquet file")

    if workers is None:
        workers = min(4, os.cpu_count() or 1)
    if workers < 1:
        raise ValueError("workers must be at least 1")

    try:
        import pyarrow.parquet as parquet
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "molecular-property export requires pyarrow; install it in the "
            "environment used for postprocessing"
        ) from error

    parquet_file = parquet.ParquetFile(input_file_path)
    required_columns = {"record_id", "smiles_canonical"}
    missing_columns = required_columns - set(parquet_file.schema_arrow.names)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"{input_file_path} is missing required columns: {missing_text}")

    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_file_path.with_name(f".{output_file_path.name}.partial")
    if temporary_path.exists():
        raise FileExistsError(
            f"temporary output already exists: {temporary_path}; inspect or remove it "
            "before starting another run"
        )

    rows_written = 0
    status_counts: dict[str, int] = {}
    next_progress = progress_every
    try:
        with temporary_path.open("w", newline="", encoding="utf-8") as output_handle:
            writer = csv.DictWriter(output_handle, fieldnames=CSV_FIELDS)
            writer.writeheader()

            executor = (
                ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
            )
            try:
                for arrow_batch in parquet_file.iter_batches(
                    batch_size=batch_size,
                    columns=["record_id", "smiles_canonical"],
                    use_threads=True,
                ):
                    chunks = list(
                        iter_chunks(
                            arrow_batch.column("record_id").to_pylist(),
                            arrow_batch.column("smiles_canonical").to_pylist(),
                            records_per_task,
                            rows_written,
                        )
                    )
                    if executor is None:
                        row_groups = (calculate_chunk(chunk) for chunk in chunks)
                    else:
                        row_groups = executor.map(calculate_chunk, chunks)
                    rows_written += _write_rows(writer, row_groups, status_counts)

                    while rows_written >= next_progress:
                        print(f"Processed {rows_written:,} records")
                        next_progress += progress_every
            finally:
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=True)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    temporary_path.replace(output_file_path)
    return {
        "input": str(input_file_path),
        "output": str(output_file_path),
        "rows": rows_written,
        "status_counts": dict(sorted(status_counts.items())),
        "workers": workers,
        "batch_size": batch_size,
        "records_per_task": records_per_task,
        "morgan_radius": MORGAN_RADIUS,
        "morgan_bits": MORGAN_FP_SIZE,
        "maccs_output_bits": MACCS_OUTPUT_BITS,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="input canonical Parquet file")
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "output CSV path (default: input name with _mol_properties.csv in "
            "the same directory)"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50000,
        help="Parquet rows read per bounded batch (default: 50000)",
    )
    parser.add_argument(
        "--records-per-task",
        type=int,
        default=5000,
        help="records sent to one worker task at once (default: 5000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="RDKit worker processes; use 1 to disable multiprocessing (default: up to 4)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250_000,
        help="print progress after this many records (default: 250000)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output CSV after a successful run",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    result = export_molecular_properties(
        args.input,
        args.output,
        batch_size=args.batch_size,
        records_per_task=args.records_per_task,
        workers=args.workers,
        progress_every=args.progress_every,
        overwrite=args.overwrite,
    )
    statuses = ", ".join(
        f"{name}={count}" for name, count in result["status_counts"].items()
    )
    print(
        f"Wrote {result['rows']:,} molecular-property rows to {result['output']} "
        f"({statuses})"
    )


if __name__ == "__main__":
    main()

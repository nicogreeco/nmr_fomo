#!/usr/bin/env python3
"""Export RDKit molecular properties from a canonical Parquet dataset.

The output is a typed Parquet sidecar with exactly one row for every input
record. Input order and physical row-group boundaries are preserved so the NMR
file and its molecular sidecar can be streamed together during training.

Invalid or missing SMILES are never silently dropped. Their descriptor and
fingerprint fields are null, while rdkit_status and rdkit_error retain the
reason the structure could not be processed.

Morgan ECFP4 fingerprints use radius 2 and 2,048 bits. They are stored as 256
raw bytes in a fixed-size binary column. Expanding the bits is intentionally
left to the model collator, avoiding a 2,048-value representation on disk.

Example:
    PYTHONPATH=scripts python scripts/data/postprocess/calculate_mol_properties.py \\
        datasets/cleaned/rich.parquet --workers 8
"""

from __future__ import annotations

import argparse
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as parquet
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import (
    Crippen,
    Descriptors,
    MACCSkeys,
    rdFingerprintGenerator,
    rdMolDescriptors,
)

from data.console import (
    add_console_arguments,
    configure_console,
    print_stage_complete,
    print_stage_start,
    progress_bar,
)
from data.postprocess.common import open_canonical_parquet
from data.reporting import (
    default_report_path,
    prepare_report_output,
    write_processing_report,
)


# Invalid source structures are represented explicitly in the output. Suppress
# RDKit's per-record stderr output so a very large run remains readable.
RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")

MOLECULAR_PROPERTIES_SCHEMA_VERSION = "1"
MORGAN_RADIUS = 2
MORGAN_FP_SIZE = 2048
MORGAN_FP_BYTES = MORGAN_FP_SIZE // 8
MORGAN_FIELD = "morgan_ecfp4_2048"
MACCS_OUTPUT_BITS = 166
MACCS_FIELD = "maccs_keys_166_bits"

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

INTEGER_PROPERTY_FIELDS = ["hba", "hbd", "rotatable_bonds"]
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


def molecular_properties_schema() -> pa.Schema:
    """Return the stable Arrow schema shared by calculation and migration."""

    fields = [
        pa.field("record_id", pa.string(), nullable=False),
        pa.field("smiles_canonical", pa.string()),
        pa.field("rdkit_status", pa.string(), nullable=False),
        pa.field("rdkit_error", pa.string()),
    ]
    for name in PROPERTY_FIELDS:
        field_type = pa.int32() if name in INTEGER_PROPERTY_FIELDS else pa.float64()
        fields.append(pa.field(name, field_type))
    fields.extend(pa.field(name, pa.int8()) for name in FUNCTIONAL_GROUP_SMARTS)
    fields.extend(
        [
            pa.field(MORGAN_FIELD, pa.binary(MORGAN_FP_BYTES)),
            pa.field(MACCS_FIELD, pa.string()),
        ]
    )
    return pa.schema(
        fields,
        metadata={
            b"molecular_properties_schema_version": (
                MOLECULAR_PROPERTIES_SCHEMA_VERSION.encode("utf-8")
            ),
            b"morgan_radius": str(MORGAN_RADIUS).encode("ascii"),
            b"morgan_num_bits": str(MORGAN_FP_SIZE).encode("ascii"),
            b"morgan_encoding": b"rdkit_binary_text",
        },
    )


MOLECULAR_PROPERTY_FIELDS = molecular_properties_schema().names


def default_output_path(input_path: Path) -> Path:
    """Return the same-directory molecular-property Parquet path."""

    if input_path.suffix.lower() != ".parquet":
        raise ValueError(f"input must have a .parquet suffix: {input_path}")
    return input_path.with_name(f"{input_path.stem}_mol_properties.parquet")


def empty_property_values() -> dict[str, object]:
    """Return null descriptor and fingerprint values for an unusable SMILES."""

    return {field: None for field in MOLECULAR_PROPERTY_FIELDS[4:]}


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

        morgan_bytes = DataStructs.BitVectToBinaryText(
            MORGAN_GENERATOR.GetFingerprint(molecule)
        )
        if len(morgan_bytes) != MORGAN_FP_BYTES:
            raise RuntimeError(
                f"unexpected Morgan byte length {len(morgan_bytes)}; "
                f"expected {MORGAN_FP_BYTES}"
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
            MORGAN_FIELD: morgan_bytes,
            MACCS_FIELD: maccs_bits[1:],
        }
        for name, pattern in FUNCTIONAL_GROUP_PATTERNS.items():
            row[name] = int(molecule.HasSubstructMatch(pattern))
        return row
    except Exception as error:
        return unusable_row(record_id, smiles, "calculation_error", str(error))


def calculate_chunk(records: list[tuple[str, object]]) -> list[dict[str, object]]:
    """Calculate a small ordered group of records."""

    return [calculate_row(record_id, smiles) for record_id, smiles in records]


def iter_chunks(
    record_ids: list[object],
    smiles_values: list[object],
    records_per_task: int,
    first_row_index: int,
) -> Iterator[list[tuple[str, object]]]:
    """Split one row group into bounded process-pool tasks in input order."""

    if len(record_ids) != len(smiles_values):
        raise RuntimeError(
            "record_id and smiles_canonical columns have different lengths"
        )

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


def _merge_status_counts(
    target: dict[str, int], source: dict[str, int]
) -> None:
    for status, count in source.items():
        target[status] = target.get(status, 0) + count


def _status_counts(rows: Iterable[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row["rdkit_status"])
        counts[status] = counts.get(status, 0) + 1
    return counts


def _calculate_row_group(
    chunks: Iterable[list[tuple[str, object]]],
    executor: ProcessPoolExecutor | None,
    max_pending_tasks: int,
) -> list[dict[str, object]]:
    """Calculate one row group while preserving task submission order."""

    rows: list[dict[str, object]] = []
    if executor is None:
        for chunk in chunks:
            rows.extend(calculate_chunk(chunk))
        return rows

    pending = deque()
    for chunk in chunks:
        pending.append(executor.submit(calculate_chunk, chunk))
        if len(pending) >= max_pending_tasks:
            rows.extend(pending.popleft().result())
    while pending:
        rows.extend(pending.popleft().result())
    return rows


def validate_aligned_row_groups(
    nmr_file,
    molecular_path: str | Path,
) -> None:
    """Check that a molecular sidecar retained the source row-group layout."""

    molecular_file = parquet.ParquetFile(molecular_path)
    if molecular_file.metadata.num_rows != nmr_file.metadata.num_rows:
        raise RuntimeError("molecular-property row count does not match the input")
    if molecular_file.num_row_groups != nmr_file.num_row_groups:
        raise RuntimeError(
            "molecular-property row-group count does not match the input"
        )
    for index in range(nmr_file.num_row_groups):
        input_rows = nmr_file.metadata.row_group(index).num_rows
        output_rows = molecular_file.metadata.row_group(index).num_rows
        if input_rows != output_rows:
            raise RuntimeError(
                f"molecular-property row group {index} has {output_rows} rows; "
                f"expected {input_rows}"
            )


def export_molecular_properties(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    records_per_task: int = 1_000,
    workers: int | None = None,
    progress_every: int = 250_000,
    show_progress: bool = True,
    quiet: bool = False,
    overwrite: bool = False,
) -> dict[str, object]:
    """Calculate a row-group-aligned molecular-property Parquet sidecar."""

    if records_per_task < 1:
        raise ValueError("records_per_task must be at least 1")
    if progress_every < 1:
        raise ValueError("progress_every must be at least 1")

    input_file_path = Path(input_path)
    parquet_file = open_canonical_parquet(input_file_path)
    output_file_path = (
        default_output_path(input_file_path)
        if output_path is None
        else Path(output_path)
    )
    if output_file_path.suffix.lower() != ".parquet":
        raise ValueError(
            f"molecular-property output must be a .parquet file: {output_file_path}"
        )
    if output_file_path.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite {output_file_path}; pass --overwrite to replace it"
        )
    if output_file_path.resolve() == input_file_path.resolve():
        raise ValueError("molecular-property output must not replace the NMR input")

    if workers is None:
        workers = min(8, os.cpu_count() or 1)
    if workers < 1:
        raise ValueError("workers must be at least 1")

    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_file_path.with_name(f".{output_file_path.name}.partial")
    if temporary_path.exists():
        raise FileExistsError(
            f"temporary output already exists: {temporary_path}; inspect or remove it "
            "before starting another run"
        )

    input_rows = parquet_file.metadata.num_rows
    rows_written = 0
    status_counts: dict[str, int] = {}
    schema = molecular_properties_schema()
    progress = progress_bar(
        input_rows,
        f"Calculating {input_file_path.name}",
        show_progress,
        min_iterations=progress_every,
    )
    executor = None
    try:
        if workers > 1:
            executor = ProcessPoolExecutor(
                max_workers=workers,
                initializer=configure_console,
                initargs=(quiet,),
            )

        with parquet.ParquetWriter(
            temporary_path,
            schema,
            compression="zstd",
            use_dictionary=["rdkit_status"],
            write_statistics=True,
        ) as writer:
            first_row_index = 0
            for row_group_index in range(parquet_file.num_row_groups):
                input_table = parquet_file.read_row_group(
                    row_group_index,
                    columns=["record_id", "smiles_canonical"],
                    use_threads=True,
                )
                chunks = iter_chunks(
                    input_table.column("record_id").to_pylist(),
                    input_table.column("smiles_canonical").to_pylist(),
                    records_per_task,
                    first_row_index,
                )
                rows = _calculate_row_group(
                    chunks,
                    executor,
                    max_pending_tasks=max(1, workers * 2),
                )
                if len(rows) != input_table.num_rows:
                    raise RuntimeError(
                        f"row group {row_group_index} produced {len(rows)} rows; "
                        f"expected {input_table.num_rows}"
                    )

                output_table = pa.Table.from_pylist(rows, schema=schema)
                writer.write_table(
                    output_table,
                    row_group_size=input_table.num_rows,
                )
                group_counts = _status_counts(rows)
                _merge_status_counts(status_counts, group_counts)
                rows_written += len(rows)
                first_row_index += input_table.num_rows
                if progress is not None:
                    progress.update(len(rows))
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        if progress is not None:
            progress.close()

    if rows_written != input_rows:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            "molecular-property row count does not match the input dataset"
        )
    try:
        validate_aligned_row_groups(parquet_file, temporary_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    temporary_path.replace(output_file_path)
    return {
        "stage": "calculate_mol_properties",
        "inputs": {"dataset": str(input_file_path)},
        "outputs": {"molecular_properties": str(output_file_path)},
        "counts": {
            "input_records": input_rows,
            "output_records": rows_written,
            "row_groups": parquet_file.num_row_groups,
        },
        "details": {
            "rdkit_status": dict(sorted(status_counts.items())),
            "morgan": {
                "radius": MORGAN_RADIUS,
                "num_bits": MORGAN_FP_SIZE,
                "storage_bytes": MORGAN_FP_BYTES,
            },
        },
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="input canonical Parquet file")
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "output Parquet path (default: input name with "
            "_mol_properties.parquet in the same directory)"
        ),
    )
    parser.add_argument(
        "--records-per-task",
        type=int,
        default=1000,
        help="records sent to one RDKit worker task at once (default: 1000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help=(
            "RDKit worker processes; use 1 to disable multiprocessing "
            "(default: up to 8)"
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250_000,
        help="minimum records between progress refreshes (default: 250000)",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        help="processing JSON (default: beside the output Parquet)",
    )
    add_console_arguments(parser)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output Parquet after a successful run",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    configure_console(args.quiet)
    print_stage_start("calculate molecular properties")
    output = args.output or default_output_path(args.input)
    report_path, report_temporary = prepare_report_output(
        args.report_output or default_report_path(output),
        [args.input, output],
        overwrite=args.overwrite,
    )
    result = export_molecular_properties(
        args.input,
        args.output,
        records_per_task=args.records_per_task,
        workers=args.workers,
        progress_every=args.progress_every,
        show_progress=not args.no_progress,
        quiet=args.quiet,
        overwrite=args.overwrite,
    )
    write_processing_report(result, report_path, report_temporary)
    print_stage_complete("calculate molecular properties", report_path)


if __name__ == "__main__":
    main()

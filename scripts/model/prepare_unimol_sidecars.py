"""Experimental Rich teacher sidecars; run in the existing UniMol2 environment.

Each inference batch contains ONE molecule. Completed raw row groups are cached
for restart. Final sidecars retain the original properties and row-group layout,
adding embeddings centered with the full Rich TRAIN mean and L2-normalized.
Neither the canonical Parquets nor their original sidecars are modified.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

from model_benchmarks.embedders.unimol2 import UniMol2Embedder
from model_benchmarks.processors.unimol2 import UniMol2Processor


_processor = None


def singleton_batch(rows):
    # The existing processor stores modules and cannot be pickled with spawn.
    global _processor
    if _processor is None:
        _processor = UniMol2Processor(strict=False)
    from data.validation import IncompatibleRecordError
    from rdkit.Chem.rdchem import AtomValenceException

    try:
        return _processor(rows)
    except IncompatibleRecordError as error:
        if not isinstance(error.__cause__, AtomValenceException):
            raise
        return _processor.collate([restore_canonical_graph(row) for row in rows])


def restore_canonical_graph(row):
    """Keep generated coordinates, restoring bonds damaged by conformer setup."""
    from rdkit import Chem
    from unimol_tools.data.conformer import inner_smi2coords, mol2unimolv2

    smiles = row["smiles_canonical"]
    original = Chem.AddHs(Chem.MolFromSmiles(smiles))
    generated = inner_smi2coords(
        smiles, seed=42, mode="fast", remove_hs=True, return_mol=True
    )
    original_atoms = [(a.GetAtomicNum(), a.GetIsotope()) for a in original.GetAtoms()]
    generated_atoms = [(a.GetAtomicNum(), a.GetIsotope()) for a in generated.GetAtoms()]
    if original_atoms != generated_atoms or generated.GetNumConformers() != 1:
        raise ValueError(f"Cannot safely restore atom mapping for {row['record_id']}")
    original.AddConformer(Chem.Conformer(generated.GetConformer()), assignId=True)
    features = mol2unimolv2(original, max_atoms=128, remove_hs=True)
    if not np.isfinite(features["src_coord"]).all():
        raise ValueError(f"Non-finite conformer coordinates for {row['record_id']}")
    print(f"Restored canonical graph after conformer valence failure: {row['record_id']}", flush=True)
    return {"record_id": row["record_id"], "features": features}


def file_signature(path):
    path = Path(path).resolve()
    stat = path.stat()
    return [str(path), stat.st_size, stat.st_mtime_ns]


def embedding_array(table):
    return table.column("unimol_embedding").combine_chunks().values.to_numpy().reshape(-1, 768)


def extract_parts(root, output, split, embedder, workers, checkpoint_hash):
    nmr_path = root / f"rich_{split}.parquet"
    properties_path = root / f"rich_{split}_mol_properties.parquet"
    nmr = pq.ParquetFile(nmr_path)
    properties = pq.ParquetFile(properties_path)
    if nmr.num_row_groups != properties.num_row_groups:
        raise ValueError("NMR and property row-group counts differ")
    signature = json.dumps({
        "nmr": file_signature(nmr_path),
        "properties": file_signature(properties_path),
        "checkpoint_sha256": checkpoint_hash,
        "batch_size": 1, "seed": 42,
    }, sort_keys=True).encode()
    schema = pa.schema([
        ("record_id", pa.string()),
        ("unimol_embedding", pa.list_(pa.float32(), 768)),
    ], metadata={b"extraction_signature": signature})
    part_dir = output / "raw_parts" / split
    part_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    start = time.monotonic()
    for index in range(nmr.num_row_groups):
        path = part_dir / f"row_group_{index:04d}.parquet"
        rows = nmr.read_row_group(index, columns=["record_id", "smiles_canonical"])
        ids = rows.column("record_id").to_pylist()
        property_ids = properties.read_row_group(index, columns=["record_id"])
        if ids != property_ids.column("record_id").to_pylist():
            raise ValueError(f"Unaligned input row group {split}/{index}")
        if path.exists():
            cached = pq.ParquetFile(path)
            if cached.schema_arrow.metadata != schema.metadata:
                raise ValueError(f"Stale extraction cache: {path}; use a new output directory")
            if cached.read(columns=["record_id"]).column("record_id").to_pylist() != ids:
                raise ValueError(f"Invalid cached record IDs: {path}")
            print(f"Reuse {split} row group {index + 1}/{nmr.num_row_groups}", flush=True)
        else:
            loader = DataLoader(
                rows.to_pylist(), batch_size=1, num_workers=workers,
                collate_fn=singleton_batch,
                multiprocessing_context="spawn" if workers else None,
            )
            vectors = np.empty((len(rows), 768), dtype=np.float32)
            for row_index, batch in enumerate(loader):
                if batch.record_ids != [ids[row_index]]:
                    raise ValueError("UniMol loader changed record ordering")
                vectors[row_index] = embedder.encode(batch).embeddings[0].numpy()
                if (row_index + 1) % 5000 == 0:
                    print(f"{split} RG {index + 1}: {row_index + 1}/{len(rows)} records", flush=True)
            table = pa.Table.from_arrays([
                pa.array(ids), pa.FixedSizeListArray.from_arrays(pa.array(vectors.ravel()), 768)
            ], schema=schema)
            partial = path.with_suffix(".partial")
            pq.write_table(table, partial, row_group_size=len(table), compression="zstd")
            partial.replace(path)
            print(f"Saved {split} row group {index + 1}/{nmr.num_row_groups}; elapsed {time.monotonic()-start:.0f}s", flush=True)
        paths.append(path)
    return paths, signature


def training_center(paths):
    total = np.zeros(768, dtype=np.float64)
    count = 0
    for path in paths:
        values = embedding_array(pq.read_table(path))
        total += values.sum(axis=0, dtype=np.float64)
        count += len(values)
    if count == 0:
        raise ValueError("Cannot fit a teacher center on an empty training set")
    return (total / count).astype(np.float32)


def write_sidecar(root, output, split, paths, center, signature):
    properties = pq.ParquetFile(root / f"rich_{split}_mol_properties.parquet")
    metadata = dict(properties.schema_arrow.metadata or {})
    metadata.update({
        b"unimol_batch_size": b"1",
        b"unimol_transform": b"rich_train_center_l2",
        b"unimol_center_sha256": hashlib.sha256(center.tobytes()).hexdigest().encode(),
        b"extraction_signature": signature,
    })
    schema = properties.schema_arrow.append(
        pa.field("unimol_embedding", pa.list_(pa.float32(), 768))
    ).with_metadata(metadata)
    path = output / f"rich_{split}_mol_properties.parquet"
    if path.exists():
        existing = pq.ParquetFile(path)
        if (existing.schema_arrow == schema and existing.schema_arrow.metadata == metadata
                and existing.metadata.num_rows == properties.metadata.num_rows
                and existing.num_row_groups == properties.num_row_groups):
            print(f"Already complete: {path}", flush=True)
            return
        raise ValueError(f"Refusing to overwrite incompatible output: {path}")
    partial = path.with_suffix(".partial")
    with pq.ParquetWriter(partial, schema, compression="zstd") as writer:
        for index, raw_path in enumerate(paths):
            raw = pq.read_table(raw_path)
            original = properties.read_row_group(index)
            if raw.column("record_id").to_pylist() != original.column("record_id").to_pylist():
                raise ValueError("Raw teacher IDs differ from property IDs")
            values = embedding_array(raw) - center
            norms = np.linalg.norm(values, axis=1, keepdims=True)
            if not np.isfinite(values).all() or (norms == 0).any():
                raise ValueError("Non-finite or zero centered teacher embedding")
            values /= norms
            column = pa.FixedSizeListArray.from_arrays(pa.array(values.ravel()), 768)
            table = original.append_column("unimol_embedding", column).replace_schema_metadata(metadata)
            writer.write_table(table, row_group_size=len(table))
    partial.replace(path)
    print(f"Complete: {path} ({properties.metadata.num_rows} records)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("datasets/train_splits"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/experimental/unimol2_rich"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if args.output_dir.resolve() == args.root.resolve():
        raise ValueError("Use a separate experimental output directory")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.checkpoint.open("rb") as file:
        checkpoint_hash = hashlib.file_digest(file, "sha256").hexdigest()
    torch.set_num_threads(4)
    embedder = UniMol2Embedder(checkpoint_path=args.checkpoint, device=args.device)
    train_paths, train_signature = extract_parts(
        args.root, args.output_dir, "train", embedder, args.num_workers, checkpoint_hash
    )
    center = training_center(train_paths)
    np.save(args.output_dir / "rich_train_center.npy", center)
    write_sidecar(args.root, args.output_dir, "train", train_paths, center, train_signature)
    val_paths, val_signature = extract_parts(
        args.root, args.output_dir, "val", embedder, args.num_workers, checkpoint_hash
    )
    write_sidecar(args.root, args.output_dir, "val", val_paths, center, val_signature)


if __name__ == "__main__":
    main()

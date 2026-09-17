#!/usr/bin/env python3
"""Train frozen NMR structural-information probes from cached embeddings only.

This command never loads an encoder or extracts embeddings.  It expects the
standard extractor Parquets (``record_id``, fixed-size ``embedding``, and
``nmr_embedding_metadata`` footer) for the fixed downstream train/validation
splits and for an explicitly supplied held-out benchmark cache.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as parquet
import torch
from dotenv import load_dotenv
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score, roc_auc_score


FUNCTIONAL_GROUPS = (
    "has_amine", "has_amide", "has_alcohol_or_phenol", "has_ester",
    "has_carboxylic_acid", "has_aldehyde_or_ketone", "has_nitrile",
    "has_halogenated_group", "has_heteroaromatic_ring",
)
DESCRIPTORS = (
    "exact_molecular_weight", "tpsa", "hba", "hbd", "fraction_csp3",
    "aromatic_atom_fraction", "rotatable_bonds", "calculated_logp",
)
MODELS = ("fomonmr", "nmrpeak", "nmrtrans", "ultranmr", "nmrsolver")
DEFAULT_COMPARISON_REPRESENTATIONS = (
    "nmrpeak",
    "nmrtrans",
    "ultranmr",
    "nmrsolver",
    "fomonmr-pre-shifts",
    "fomonmr-post-shifts",
    "fomonmr-post-rich",
    "fomonmr-post-unimol-shifts",
    "fomonmr-post-unimol-rich",
)


def embedding_metadata(path: Path) -> dict[str, object]:
    # The streaming extractor adds metadata to the footer after writing batches.
    # The serialized Arrow schema does not include those late additions.
    metadata = parquet.ParquetFile(path).metadata.metadata or {}
    raw = metadata.get(b"nmr_embedding_metadata")
    if raw is None:
        raise ValueError(f"{path} is missing nmr_embedding_metadata")
    return json.loads(raw.decode("utf-8"))


def read_embedding(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    table = parquet.read_table(path, columns=["record_id", "embedding"])
    ids = table["record_id"].to_pylist()
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} contains duplicate embedding record IDs")
    return {record_id: np.asarray(value, dtype=np.float32) for record_id, value in zip(ids, table["embedding"].to_pylist())}, embedding_metadata(path)


def embedding_ids(path: Path) -> tuple[set[str], dict[str, object]]:
    """Read only IDs while constructing the shared comparison cohort."""

    if not path.is_file():
        raise FileNotFoundError(path)
    ids = parquet.read_table(path, columns=["record_id"])["record_id"].to_pylist()
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} contains duplicate embedding record IDs")
    return set(ids), embedding_metadata(path)


def common_cohort(args, representation_id: str) -> tuple[dict[str, set[str]], dict[str, object]]:
    """Load or create a frozen ID intersection across configured caches."""

    path = args.common_cohort_manifest
    if path.exists() and not args.rebuild_common_cohort:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if tuple(manifest.get("comparison_representations", ())) != tuple(args.comparison_representations):
            raise ValueError("common cohort was created for different representations")
        return {split: set(ids) for split, ids in manifest["record_ids"].items()}, manifest

    manifest = {
        "comparison_representations": args.comparison_representations,
        "record_ids": {},
        "availability": {},
    }
    for split in ("train", "validation", "test"):
        id_sets = []
        manifest["availability"][split] = {}
        for representation in args.comparison_representations:
            cache_path = args.embeddings_root / representation / f"{split}.parquet"
            if representation == representation_id:
                cache_path = getattr(args, f"{split}_embeddings")
            record_ids, metadata = embedding_ids(cache_path)
            id_sets.append(record_ids)
            manifest["availability"][split][representation] = {
                "path": str(cache_path),
                "records": len(record_ids),
                "metadata": metadata,
            }
        common_ids = set.intersection(*id_sets)
        if not common_ids:
            raise ValueError(f"no common {split} records across comparison representations")
        manifest["record_ids"][split] = sorted(common_ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {split: set(ids) for split, ids in manifest["record_ids"].items()}, manifest


def read_properties(path: Path, fields: tuple[str, ...]) -> dict[str, dict[str, object]]:
    table = parquet.read_table(path, columns=["record_id", "smiles_canonical", "rdkit_status", *fields])
    rows = {}
    for row in table.to_pylist():
        record_id = row["record_id"]
        if record_id in rows:
            raise ValueError(f"{path} contains duplicate property record_id {record_id!r}")
        if row["rdkit_status"] != "ok":
            raise ValueError(f"{path} contains non-ok RDKit status for {record_id!r}")
        rows[record_id] = row
    return rows


def split_ids_from_manifest(path: Path) -> dict[str, set[str]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("split manifest has no records list")
    result = {"train": set(), "validation": set()}
    molecule_splits = {}
    for row in records:
        split, record_id, molecule = row.get("split"), row.get("record_id"), row.get("molecule_id")
        if split not in result or not isinstance(record_id, str) or not isinstance(molecule, str):
            raise ValueError("invalid structural-probe split manifest row")
        if record_id in result[split] or molecule in molecule_splits:
            raise ValueError("manifest repeats a record or molecule")
        result[split].add(record_id)
        molecule_splits[molecule] = split
    if result["train"] & result["validation"]:
        raise ValueError("manifest train/validation record overlap")
    return result


def validate_cache(model: str, metadata: dict[str, object], args) -> None:
    if metadata.get("model_name") != model:
        raise ValueError(f"embedding metadata model is {metadata.get('model_name')!r}, expected {model!r}")
    if model != "fomonmr":
        return
    if metadata.get("input_mode") != args.input_mode:
        raise ValueError(f"FoMoNMR cache input_mode is {metadata.get('input_mode')!r}, expected {args.input_mode!r}")
    expected = None
    if args.run_id:
        expected = f"runs:/{args.run_id}/model"
    elif args.checkpoint_path:
        expected = str(Path(args.checkpoint_path).expanduser().resolve())
    if expected and str(metadata.get("checkpoint")) != expected:
        raise ValueError("FoMoNMR cache checkpoint does not match --run-id/--checkpoint-path")


def load_split(name: str, embedding_path: Path, property_path: Path, expected_ids: set[str], model: str, args, fields):
    embeddings, metadata = read_embedding(embedding_path)
    validate_cache(model, metadata, args)
    properties = read_properties(property_path, fields)
    if expected_ids - set(embeddings) or expected_ids - set(properties):
        raise ValueError(f"{name} does not cover the common comparison cohort")
    ids = sorted(expected_ids)
    dimensions = {vector.shape for vector in embeddings.values()}
    if len(dimensions) != 1:
        raise ValueError(f"{name} embeddings have inconsistent dimensions")
    return ids, np.stack([embeddings[record_id] for record_id in ids]), properties, metadata


class Probe(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, kind: str):
        super().__init__()
        if kind == "linear":
            self.layers = torch.nn.Linear(input_dim, output_dim)
        else:
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 256), torch.nn.GELU(), torch.nn.Dropout(0.1),
                torch.nn.Linear(256, output_dim),
            )

    def forward(self, values):
        return self.layers(values)


def standardize(train, validation, test):
    mean, std = train.mean(axis=0), train.std(axis=0)
    if np.any(std == 0):
        raise ValueError("an embedding coordinate has zero training variance")
    return (train - mean) / std, (validation - mean) / std, (test - mean) / std


def score_functional(target, logits):
    probabilities = 1 / (1 + np.exp(-logits))
    rows = []
    for index, name in enumerate(FUNCTIONAL_GROUPS):
        truth, score = target[:, index], probabilities[:, index]
        positives = int(truth.sum())
        rows.append({"target": name, "support": len(truth), "positives": positives, "prevalence": positives / len(truth), "roc_auc": roc_auc_score(truth, score) if 0 < positives < len(truth) else np.nan, "average_precision": average_precision_score(truth, score) if positives else np.nan})
    return rows


def score_descriptors(target, prediction):
    return [{"target": name, "support": len(target), "mae": mean_absolute_error(target[:, index], prediction[:, index]), "r2": r2_score(target[:, index], prediction[:, index])} for index, name in enumerate(DESCRIPTORS)]


def train_one(kind, task, train_x, train_y, validation_x, validation_y, test_x, target_mean, target_std, seed, weight_decays, epochs, device):
    torch.manual_seed(seed)
    device = torch.device(device)
    tensors = [torch.as_tensor(value, dtype=torch.float32, device=device) for value in (train_x, train_y, validation_x, validation_y, test_x)]
    train_x_t, train_y_t, val_x_t, val_y_t, test_x_t = tensors
    best = None
    for weight_decay in weight_decays:
        torch.manual_seed(seed)
        model = Probe(train_x.shape[1], train_y.shape[1], kind).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
        loss_function = torch.nn.BCEWithLogitsLoss() if task == "functional" else torch.nn.MSELoss()
        best_state, best_value, stale = None, -np.inf, 0
        for epoch in range(epochs):
            model.train(); optimizer.zero_grad(); loss = loss_function(model(train_x_t), train_y_t); loss.backward(); optimizer.step()
            model.eval()
            with torch.no_grad():
                prediction = model(val_x_t).cpu().numpy()
            if task == "functional":
                values = score_functional(validation_y, prediction)
                value = float(np.nanmean([row["roc_auc"] for row in values]))
            else:
                value = float(np.mean([r2_score(validation_y[:, i], prediction[:, i]) for i in range(validation_y.shape[1])]))
            if value > best_value + 1e-5:
                best_value, best_state, stale = value, {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, 0
            else:
                stale += 1
                if stale >= 25:
                    break
        if best is None or best_value > best[0]:
            best = (best_value, weight_decay, best_state)
    value, weight_decay, state = best
    model = Probe(train_x.shape[1], train_y.shape[1], kind).to(device); model.load_state_dict(state); model.eval()
    with torch.no_grad():
        prediction = model(test_x_t).cpu().numpy()
    return model, prediction, {"validation_score": value, "weight_decay": weight_decay}


def write_csv(path: Path, rows):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def maybe_mlflow(args, parameters, metrics, output_dir):
    if args.no_mlflow:
        return
    try:
        import mlflow
    except ModuleNotFoundError as error:
        raise RuntimeError("MLflow is unavailable; pass --no-mlflow to disable logging") from error
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params(parameters); mlflow.log_metrics(metrics); mlflow.log_artifacts(str(output_dir), artifact_path="structural_information")


def parse_arguments():
    """Parse CLI options and resolve representation-specific cache paths."""

    load_dotenv("mlflow.env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--representation-id")
    parser.add_argument("--input-mode", choices=("shifts", "rich"))
    parser.add_argument("--run-id")
    parser.add_argument("--checkpoint-path", type=Path)
    for split in ("train", "validation", "test"):
        parser.add_argument(f"--{split}-embeddings", type=Path)
    parser.add_argument("--train-properties", type=Path, default=Path("datasets/downstream_structural_information/train_mol_properties.parquet"))
    parser.add_argument("--validation-properties", type=Path, default=Path("datasets/downstream_structural_information/validation_mol_properties.parquet"))
    parser.add_argument("--test-properties", type=Path, default=Path("datasets/cleaned/test_benchmark_mol_properties.parquet"))
    parser.add_argument("--split-manifest", type=Path, default=Path("datasets/downstream_structural_information/split_manifest.json"))
    parser.add_argument("--embeddings-root", type=Path, default=Path("embeddings/structural_information"))
    parser.add_argument("--comparison-representations", nargs="+", default=list(DEFAULT_COMPARISON_REPRESENTATIONS))
    parser.add_argument("--common-cohort-manifest", type=Path)
    parser.add_argument("--rebuild-common-cohort", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/structural_information"))
    parser.add_argument("--mlp-seeds", type=int, nargs="+", default=[13, 42, 73])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-mlp", action="store_true")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    parser.add_argument("--mlflow-experiment", default="structural-information-probes")
    parser.add_argument("--run-name")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    if args.model == "fomonmr":
        if args.input_mode is None:
            parser.error("FoMoNMR requires --input-mode shifts or rich")
        if bool(args.run_id) == bool(args.checkpoint_path):
            parser.error("FoMoNMR requires exactly one of --run-id or --checkpoint-path")
    elif args.input_mode or args.run_id or args.checkpoint_path:
        parser.error("FoMoNMR-specific options require --model fomonmr")

    representation_id = args.representation_id or args.model
    if args.model == "fomonmr" and args.representation_id is None:
        parser.error("FoMoNMR requires --representation-id")
    if representation_id not in args.comparison_representations:
        parser.error("--representation-id must be in --comparison-representations")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    for split in ("train", "validation", "test"):
        if getattr(args, f"{split}_embeddings") is None:
            setattr(args, f"{split}_embeddings", args.embeddings_root / representation_id / f"{split}.parquet")
    args.common_cohort_manifest = args.common_cohort_manifest or args.embeddings_root / "structural_common_cohort.json"
    return args, representation_id


def load_benchmark_splits(args, representation_id):
    """Load the common cohort and its embeddings and labels for all splits."""

    manifest_ids = split_ids_from_manifest(args.split_manifest)
    common_ids, _ = common_cohort(args, representation_id)
    if not common_ids["train"] <= manifest_ids["train"] or not common_ids["validation"] <= manifest_ids["validation"]:
        raise ValueError("common cohort contains train/validation records outside the split manifest")

    fields = FUNCTIONAL_GROUPS + DESCRIPTORS
    splits = {}
    for name in ("train", "validation", "test"):
        splits[name] = load_split(
            name, getattr(args, f"{name}_embeddings"), getattr(args, f"{name}_properties"),
            common_ids[name], args.model, args, fields,
        )
    train_ids, train_x, train_props, metadata = splits["train"]
    val_ids, val_x, val_props, _ = splits["validation"]
    test_ids, test_x, test_props, _ = splits["test"]
    if set(row["smiles_canonical"] for row in train_props.values()) & set(row["smiles_canonical"] for row in val_props.values()):
        raise ValueError("train and validation molecular identities overlap")
    if train_x.shape[1] != val_x.shape[1] or train_x.shape[1] != test_x.shape[1]:
        raise ValueError("embedding dimensions differ across splits")
    train_x, val_x, test_x = standardize(train_x, val_x, test_x)
    return (train_ids, train_x, train_props, metadata), (val_ids, val_x, val_props), (test_ids, test_x, test_props)


def train_all_probes(args, splits, output_dir):
    """Train fresh linear and optional MLP heads and return test metric rows."""

    (train_ids, train_x, train_props, _), (val_ids, val_x, val_props), (test_ids, test_x, test_props) = splits
    all_rows = []
    device = "cuda" if args.device == "cuda" else "cpu"
    for task, targets, scorer in (("functional", FUNCTIONAL_GROUPS, score_functional), ("descriptors", DESCRIPTORS, score_descriptors)):
        train_y = np.asarray([[train_props[key][field] for field in targets] for key in train_ids], dtype=np.float32)
        val_y = np.asarray([[val_props[key][field] for field in targets] for key in val_ids], dtype=np.float32)
        test_y = np.asarray([[test_props[key][field] for field in targets] for key in test_ids], dtype=np.float32)
        mean, std = (train_y.mean(0), train_y.std(0)) if task == "descriptors" else (np.zeros(len(targets)), np.ones(len(targets)))
        if task == "descriptors" and np.any(std == 0):
            raise ValueError("a descriptor has zero train variance")
        for kind, seeds in (("linear", [42]), ("mlp", [] if args.no_mlp else args.mlp_seeds)):
            for seed in seeds:
                model, prediction, details = train_one(kind, task, train_x, (train_y - mean) / std, val_x, (val_y - mean) / std, test_x, mean, std, seed, [0.0, 1e-5, 1e-4, 1e-3], args.epochs, device)
                if task == "descriptors":
                    prediction = prediction * std + mean
                for row in scorer(test_y, prediction):
                    row.update({"model": args.model, "input_mode": args.input_mode or "native", "probe": kind, "seed": seed, **details})
                    all_rows.append(row)
                torch.save({"state_dict": model.state_dict(), "metadata": {"task": task, "probe": kind, "seed": seed, **details}}, output_dir / f"{task}_{kind}_seed{seed}.pt")
    return all_rows


def run(args, representation_id):
    splits = load_benchmark_splits(args, representation_id)
    output_dir = args.output_dir / representation_id
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)

    all_rows = train_all_probes(args, splits, output_dir)
    (train_ids, _, _, metadata), (val_ids, _, _), (test_ids, _, _) = splits
    write_csv(output_dir / "test_metrics.csv", all_rows)
    summary = {"model": args.model, "representation_id": representation_id, "input_mode": args.input_mode or "native", "embedding_metadata": metadata, "common_cohort_manifest": str(args.common_cohort_manifest), "n_train": len(train_ids), "n_validation": len(val_ids), "n_test": len(test_ids), "mlp_seeds": args.mlp_seeds, "metrics_file": "test_metrics.csv"}
    (output_dir / "run_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    numeric = {f"{row['probe']}/{row['target']}/{key}": float(value) for row in all_rows for key, value in row.items() if key in {"roc_auc", "average_precision", "mae", "r2"} and np.isfinite(value)}
    maybe_mlflow(args, {"model": args.model, "input_mode": args.input_mode or "native", "n_train": len(train_ids), "n_validation": len(val_ids), "n_test": len(test_ids)}, numeric, output_dir)
    print(f"saved structural-probe artifacts to {output_dir}")


def main():
    args, representation_id = parse_arguments()
    run(args, representation_id)


if __name__ == "__main__":
    main()

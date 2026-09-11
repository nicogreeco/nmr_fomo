#!/usr/bin/env python3
"""Fine-tune a FoMoNMR MLflow model on one or more molecular properties."""

import argparse
import copy
import csv
import os
from pathlib import Path

import mlflow
import numpy as np
import torch
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader

from data import CanonicalParquetDataset
from model.FoMoNMR import FoMoNMR
from model.processor import FoundationNMRProcessor
from model_benchmarks.run_property_prediction import (
    CANONICAL_FILES,
    CLASSIFICATION_DATASETS,
    TARGET_FILES,
    classification_metrics,
    molecule_group,
    parse_names,
    regression_metrics,
)


def parse_arguments(argv=None):
    load_dotenv("mlflow.env")

    parser = argparse.ArgumentParser(description=__doc__)
    model_source = parser.add_mutually_exclusive_group(required=True)
    model_source.add_argument("--run-id")
    model_source.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI"),
    )
    parser.add_argument(
        "--datasets-dir", type=Path, default=Path("datasets/cleaned/admet")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/fomonmr_finetune")
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but it is unavailable")
    return torch.device(name)


def load_model(run_id, checkpoint_path, tracking_uri):
    if checkpoint_path:
        checkpoint_path = checkpoint_path.resolve()
        run_name = checkpoint_path.stem
        if (checkpoint_path.parent.name in ("best", "latest")
                and checkpoint_path.parent.parent.name == "checkpoints"):
            run_name = checkpoint_path.parent.parent.parent.name
        print(f"Loading {run_name} from {checkpoint_path}", flush=True)
        return FoMoNMR.load_from_checkpoint(checkpoint_path, map_location="cpu"), run_name

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    run = MlflowClient().get_run(run_id)
    run_name = run.data.tags.get("mlflow.runName", run_id)
    model_uri = f"runs:/{run_id}/model"
    print(f"Loading {run_name} from {model_uri}", flush=True)
    return mlflow.pytorch.load_model(model_uri, map_location="cpu"), run_name


def load_records(dataset, split, datasets_dir):
    datasets_dir = Path(datasets_dir)
    target_path = datasets_dir / dataset / TARGET_FILES[split]
    nmr_path = datasets_dir / dataset / CANONICAL_FILES[split]

    with target_path.open(newline="") as file:
        targets = {
            row["record_id"]: float(row["Y"])
            for row in csv.DictReader(file)
        }

    records = [
        (record, targets[record.record_id])
        for record in CanonicalParquetDataset(nmr_path)
        if record.record_id in targets
    ]
    if not records:
        raise ValueError(f"No labelled NMR records found for {dataset}/{split}")
    return records


def split_train_validation(records, classification, fraction, seed):
    groups = np.asarray([molecule_group(record.smiles_canonical) for record, _ in records])
    targets = np.asarray([target for _, target in records])

    if classification:
        splitter = StratifiedGroupKFold(
            n_splits=round(1 / fraction),
            shuffle=True,
            random_state=seed,
        )
        train_indices, validation_indices = next(
            splitter.split(np.zeros(len(records)), targets, groups)
        )
    else:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=fraction,
            random_state=seed,
        )
        train_indices, validation_indices = next(
            splitter.split(np.zeros(len(records)), targets, groups)
        )

    train_records = [records[index] for index in train_indices]
    validation_records = [records[index] for index in validation_indices]
    return train_records, validation_records


def collate_records(records, processor):
    nmr_records, targets = zip(*records)
    batch = processor(nmr_records)
    batch["targets"] = torch.tensor(targets, dtype=torch.float32)
    return batch


def make_loader(records, processor, batch_size, num_workers, shuffle):
    return DataLoader(
        records,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_records(batch, processor),
        pin_memory=True,
    )


def apply_input_policy(batch, model):
    if model.config.stage == "pretrain" or not model.config.use_rich_input:
        batch["h"]["availability"].zero_()
        batch["h"]["j_mask"].zero_()


def make_head(d_model, hidden_size):
    return nn.Sequential(
        nn.Linear(d_model, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 1),
    )


def evaluate(model, head, loader, device, classification, target_mean, target_std):
    model.eval()
    head.eval()

    total_loss = 0.0
    total_count = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            batch = model.transfer_batch_to_device(batch, device, 0)
            target = batch.pop("targets")
            apply_input_policy(batch, model)

            pooled, _, _ = model(batch)
            output = head(pooled).squeeze(1)

            if classification:
                loss = nn.functional.binary_cross_entropy_with_logits(output, target)
                prediction = output.sigmoid()
            else:
                normalized_target = (target - target_mean) / target_std
                loss = nn.functional.mse_loss(output, normalized_target)
                prediction = output * target_std + target_mean

            total_loss += loss.item() * len(target)
            total_count += len(target)
            predictions.append(prediction.cpu())
            targets.append(target.cpu())

    predictions = torch.cat(predictions).numpy()
    targets = torch.cat(targets).numpy()
    if classification:
        metrics = classification_metrics(targets, predictions >= 0.5, predictions)
    else:
        metrics = regression_metrics(targets, predictions)
    return total_loss / total_count, metrics


def fine_tune_property(
    model,
    initial_state,
    train_records,
    validation_records,
    test_records,
    classification,
    arguments,
    device,
):
    torch.manual_seed(arguments.seed)
    model.load_state_dict(initial_state)
    model.to(device)
    head = make_head(model.config.d_model, arguments.hidden_size).to(device)

    train_targets = torch.tensor([target for _, target in train_records])
    target_mean = train_targets.mean().to(device)
    target_std = train_targets.std().clamp_min(1e-6).to(device)

    backbone_parameters = [
        parameter
        for module in (model.embedding, model.transformer_encoder)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": arguments.backbone_lr},
            {"params": head.parameters(), "lr": arguments.head_lr},
        ],
        weight_decay=arguments.weight_decay,
    )

    processor = FoundationNMRProcessor()
    train_loader = make_loader(
        train_records,
        processor,
        arguments.batch_size,
        arguments.num_workers,
        shuffle=True,
    )
    validation_loader = make_loader(
        validation_records,
        processor,
        arguments.batch_size,
        arguments.num_workers,
        shuffle=False,
    )
    test_loader = make_loader(
        test_records,
        processor,
        arguments.batch_size,
        arguments.num_workers,
        shuffle=False,
    )

    best_loss = float("inf")
    best_model_state = None
    best_head_state = None
    for epoch in range(1, arguments.epochs + 1):
        model.train()
        head.train()

        for batch in train_loader:
            batch = model.transfer_batch_to_device(batch, device, 0)
            target = batch.pop("targets")
            apply_input_policy(batch, model)

            optimizer.zero_grad()
            pooled, _, _ = model(batch)
            output = head(pooled).squeeze(1)
            if classification:
                loss = nn.functional.binary_cross_entropy_with_logits(output, target)
            else:
                normalized_target = (target - target_mean) / target_std
                loss = nn.functional.mse_loss(output, normalized_target)
            loss.backward()
            optimizer.step()

        validation_loss, _ = evaluate(
            model,
            head,
            validation_loader,
            device,
            classification,
            target_mean,
            target_std,
        )
        print(f"epoch {epoch}: validation loss {validation_loss:.4f}", flush=True)

        if validation_loss < best_loss - 1e-4:
            best_loss = validation_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_head_state = copy.deepcopy(head.state_dict())

    model.load_state_dict(best_model_state)
    head.load_state_dict(best_head_state)
    test_loss, test_metrics = evaluate(
        model,
        head,
        test_loader,
        device,
        classification,
        target_mean,
        target_std,
    )
    return best_loss, test_loss, test_metrics, epoch


def main(argv=None):
    arguments = parse_arguments(argv)
    torch.manual_seed(arguments.seed)
    device = resolve_device(arguments.device)
    datasets = parse_names(arguments.datasets)

    model, run_name = load_model(
        arguments.run_id, arguments.checkpoint_path, arguments.tracking_uri
    )
    initial_state = copy.deepcopy(model.state_dict())
    output_dir = arguments.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        output_path = output_dir / f"{dataset}.csv"
        if output_path.exists() and not arguments.overwrite:
            raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")

        classification = dataset in CLASSIFICATION_DATASETS
        records = load_records(dataset, "train", arguments.datasets_dir)
        train_records, validation_records = split_train_validation(
            records,
            classification,
            arguments.validation_fraction,
            arguments.seed,
        )
        test_records = load_records(dataset, "test", arguments.datasets_dir)

        print(
            f"{dataset}: {len(train_records)} train, "
            f"{len(validation_records)} validation, {len(test_records)} test",
            flush=True,
        )
        best_loss, test_loss, metrics, epochs = fine_tune_property(
            model,
            initial_state,
            train_records,
            validation_records,
            test_records,
            classification,
            arguments,
            device,
        )

        row = {
            "dataset": dataset,
            "task": "classification" if classification else "regression",
            "run_id": arguments.run_id or str(arguments.checkpoint_path),
            "run_name": run_name,
            "n_train": len(train_records),
            "n_validation": len(validation_records),
            "n_test": len(test_records),
            "best_validation_loss": best_loss,
            "test_loss": test_loss,
            "epochs": epochs,
            "backbone_lr": arguments.backbone_lr,
            "head_lr": arguments.head_lr,
            **metrics,
        }
        with output_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=row.keys())
            writer.writeheader()
            writer.writerow(row)
        print(f"Saved test metrics to {output_path}", flush=True)


if __name__ == "__main__":
    main()

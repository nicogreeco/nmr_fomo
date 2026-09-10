#!/usr/bin/env python3
"""Extract FoMoNMR embeddings and run comparable property probes."""

import argparse
import json
import os
from pathlib import Path

import torch
from dotenv import load_dotenv

from data import CanonicalParquetDataset
from model.FoMoNMR import FoMoNMR
from model.processor import FoundationNMRProcessor
from model_benchmarks.extract_embeddings import _write_embedding_batch
from model_benchmarks.run_property_prediction import (
    CANONICAL_FILES,
    TARGET_FILES,
    common_record_ids,
    main as run_property_prediction,
    parse_names,
)


BASELINE_MODELS = (
    "morgan",
    "nmrpeak",
    "nmrsolver",
    "nmrtrans",
    "ultranmr",
    "unimol2",
)


def parse_arguments(argv=None):
    load_dotenv("mlflow.env")
    parser = argparse.ArgumentParser(description=__doc__)
    model_source = parser.add_mutually_exclusive_group(required=True)
    model_source.add_argument(
        "--checkpoint-path",
        "--checkpoints",
        dest="checkpoints",
        nargs="+",
        type=Path,
        help="local .ckpt file(s) under runs/fomonmr/<run-name>/",
    )
    model_source.add_argument(
        "--run-id",
        help="MLflow run ID containing the logged FoMoNMR model",
    )
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument(
        "--shift-only",
        action="store_true",
        help="hide rich 1H annotations, even for a posttrain checkpoint",
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/fomonmr"))
    parser.add_argument(
        "--datasets-dir", type=Path, default=Path("datasets/cleaned/admet")
    )
    parser.add_argument(
        "--embeddings-dir", type=Path, default=Path("embeddings/admet")
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI"),
        help="MLflow tracking URI (defaults to MLFLOW_TRACKING_URI)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def run_name_from_checkpoint(checkpoint, runs_dir):
    """Get the run folder name from an explicit checkpoint path."""

    checkpoint = checkpoint.resolve()
    runs_dir = runs_dir.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if checkpoint.suffix != ".ckpt":
        raise ValueError(f"checkpoint must be a .ckpt file: {checkpoint}")

    try:
        relative = checkpoint.relative_to(runs_dir)
    except ValueError as error:
        raise ValueError(
            f"checkpoint must be inside {runs_dir}: {checkpoint}"
        ) from error
    if len(relative.parts) < 4 or relative.parts[1] != "checkpoints":
        raise ValueError(
            "expected checkpoint path runs/fomonmr/<run-name>/checkpoints/..."
        )

    run_name = relative.parts[0]
    if "+" in run_name:
        raise ValueError("FoMoNMR run names cannot contain '+'")
    if run_name in BASELINE_MODELS:
        raise ValueError(f"FoMoNMR run name conflicts with a baseline: {run_name}")
    return run_name


def load_mlflow_model(run_id, tracking_uri):
    """Load the model artifact and run name for one MLflow run."""

    import mlflow
    from mlflow.tracking import MlflowClient

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    run = MlflowClient().get_run(run_id)
    run_name = run.data.tags.get("mlflow.runName", run_id)
    if Path(run_name).name != run_name or "+" in run_name:
        raise ValueError(f"MLflow run name cannot be used as a model name: {run_name}")
    if run_name in BASELINE_MODELS:
        raise ValueError(f"FoMoNMR run name conflicts with a baseline: {run_name}")

    model_uri = f"runs:/{run_id}/model"
    print(f"loading {run_name} from MLflow: {model_uri}", flush=True)
    model = mlflow.pytorch.load_model(model_uri, map_location="cpu")
    return model, run_name, model_uri


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but it is unavailable")
    return torch.device(name)


def apply_checkpoint_input_policy(batch, model, shift_only=False):
    """Match the rich-input policy used while training this checkpoint."""

    if shift_only or model.config.stage == "pretrain" or not model.config.use_rich_input:
        batch["h"]["availability"].zero_()
        batch["h"]["j_mask"].zero_()


def batched_selected_records(input_path, selected_ids, batch_size):
    """Stream only the baseline-common records in small Python lists."""

    batch = []
    for record in CanonicalParquetDataset(input_path):
        if record.record_id not in selected_ids:
            continue
        batch.append(record)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def extract_embeddings(
    model,
    checkpoint,
    run_name,
    input_path,
    output_path,
    selected_ids,
    batch_size,
    device,
    shift_only=False,
    overwrite=False,
):
    """Extract one FoMoNMR split into the standard embedding Parquet schema."""

    import pyarrow as pa
    import pyarrow.parquet as parquet

    partial_output = Path(str(output_path) + ".partial")
    existing = [path for path in (output_path, partial_output) if path.exists()]
    if existing and not overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"refusing to overwrite {paths}; pass --overwrite to replace it"
        )

    dimension = int(model.config.d_model)
    schema = pa.schema(
        [
            pa.field("record_id", pa.string(), nullable=False),
            pa.field("embedding", pa.list_(pa.float32(), dimension), nullable=False),
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = parquet.ParquetWriter(partial_output, schema, compression="zstd")
    processor = FoundationNMRProcessor()
    written_ids = set()
    written_count = 0

    try:
        for records in batched_selected_records(input_path, selected_ids, batch_size):
            batch = processor(records)
            batch = model.transfer_batch_to_device(batch, device, 0)
            apply_checkpoint_input_policy(batch, model, shift_only=shift_only)
            with torch.inference_mode():
                pooled, _, _ = model(batch)
            embeddings = pooled.detach().to(device="cpu", dtype=torch.float32)
            record_ids = batch["record_ids"]
            _write_embedding_batch(writer, schema, pa, embeddings, record_ids)
            written_ids.update(record_ids)
            written_count += len(record_ids)

        missing = selected_ids - written_ids
        unexpected = written_ids - selected_ids
        if missing or unexpected or written_count != len(selected_ids):
            raise RuntimeError(
                "FoMoNMR extraction did not preserve the baseline-common cohort: "
                f"expected={len(selected_ids)}, written={written_count}, "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )

        uses_rich_input = (
            not shift_only
            and model.config.stage != "pretrain"
            and model.config.use_rich_input
        )
        metadata = {
            "model_name": run_name,
            "checkpoint": str(
                checkpoint.resolve() if isinstance(checkpoint, Path) else checkpoint
            ),
            "dimension": dimension,
            "pooling": "mean of valid 1H+13C transformer peak states",
            "modality": "1H+13C",
            "stage": model.config.stage,
            "use_rich_input": bool(uses_rich_input),
            "cohort_models": list(BASELINE_MODELS),
            "accepted_count": written_count,
        }
        writer.add_key_value_metadata(
            {"nmr_embedding_metadata": json.dumps(metadata, sort_keys=True)}
        )
    finally:
        writer.close()

    partial_output.replace(output_path)
    print(f"saved {written_count} embeddings to {output_path}", flush=True)


def check_arguments(arguments):
    if arguments.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if Path(arguments.experiment_name).name != arguments.experiment_name:
        raise ValueError("--experiment-name must be a directory name, not a path")

    datasets = parse_names(arguments.datasets)
    if not datasets:
        raise ValueError("at least one property dataset is required")

    checkpoints = []
    run_names = []
    if arguments.checkpoints:
        for checkpoint in arguments.checkpoints:
            checkpoint = checkpoint.resolve()
            checkpoints.append(checkpoint)
            run_names.append(run_name_from_checkpoint(checkpoint, arguments.runs_dir))
        if len(run_names) != len(set(run_names)):
            raise ValueError("give at most one checkpoint for each FoMoNMR run")

    for dataset in datasets:
        dataset_dir = arguments.datasets_dir / dataset
        for filename in (*CANONICAL_FILES.values(), *TARGET_FILES.values()):
            path = dataset_dir / filename
            if not path.is_file():
                raise FileNotFoundError(f"property dataset split not found: {path}")
    return checkpoints, run_names, datasets


def main(argv=None):
    arguments = parse_arguments(argv)
    checkpoints, run_names, datasets = check_arguments(arguments)
    device = resolve_device(arguments.device)
    print(f"FoMoNMR extraction device: {device}", flush=True)

    common = {
        dataset: {
            split: common_record_ids(
                dataset, BASELINE_MODELS, split, arguments.embeddings_dir
            )
            for split in ("train", "test")
        }
        for dataset in datasets
    }
    for dataset, splits in common.items():
        for split, record_ids in splits.items():
            if not record_ids:
                raise ValueError(
                    f"no records common to the baseline models for {dataset}/{split}"
                )

    if arguments.run_id:
        model, run_name, model_source = load_mlflow_model(
            arguments.run_id, arguments.tracking_uri
        )
        if arguments.shift_only:
            run_name = f"{run_name}-shift-only"
        models = [(model, run_name, model_source)]
        run_names = [run_name]
    else:
        models = []
        model_names = []
        for checkpoint, run_name in zip(checkpoints, run_names):
            if arguments.shift_only:
                run_name = f"{run_name}-shift-only"
            print(f"loading {run_name}: {checkpoint}", flush=True)
            model = FoMoNMR.load_from_checkpoint(checkpoint, map_location="cpu")
            models.append((model, run_name, checkpoint))
            model_names.append(run_name)
        run_names = model_names

    for model, run_name, model_source in models:
        model.eval().to(device)

        for dataset in datasets:
            for split in ("train", "test"):
                input_path = (
                    arguments.datasets_dir / dataset / CANONICAL_FILES[split]
                )
                output_path = (
                    arguments.embeddings_dir
                    / dataset
                    / run_name
                    / f"{split}.parquet"
                )
                extract_embeddings(
                    model=model,
                    checkpoint=model_source,
                    run_name=run_name,
                    input_path=input_path,
                    output_path=output_path,
                    selected_ids=common[dataset][split],
                    batch_size=arguments.batch_size,
                    device=device,
                    shift_only=arguments.shift_only,
                    overwrite=arguments.overwrite,
                )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    experiment_name = arguments.experiment_name
    if arguments.shift_only and not experiment_name.endswith("-shift-only"):
        experiment_name = f"{experiment_name}-shift-only"
    output_dir = Path(f"results/property_prediction_fomonmr_{experiment_name}")
    run_property_prediction(
        [
            "--models",
            *run_names,
            "--datasets",
            *datasets,
            "--embeddings-dir",
            str(arguments.embeddings_dir),
            "--datasets-dir",
            str(arguments.datasets_dir),
            "--output-dir",
            str(output_dir),
            "--cv-folds",
            str(arguments.cv_folds),
            "--epochs",
            str(arguments.epochs),
            "--device",
            arguments.device,
        ],
        cohort_representations=BASELINE_MODELS,
    )


if __name__ == "__main__":
    main()

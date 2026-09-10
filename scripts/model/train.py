"""Simple FoMoNMR pretraining and continued-pretraining entry point."""

import argparse
import csv
import os
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import lightning as L
import mlflow
import torch
import yaml
from dotenv import load_dotenv
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader

from model import (
    FoundationNMRProcessor,
    MixedFoundationDataset,
    PairedFoundationDataset,
)
from model.FoMoNMR import FoMoNMR, ModelConfig
from model.maccs_probe import MaccsLinearProbe


class ResumedRunLogger(L.Callback):
    def on_train_start(self, trainer, model):
        if not trainer.is_global_zero:
            return
        logger = trainer.logger
        logger.experiment.update_run(logger.run_id, status="RUNNING")
        logger.experiment.set_tag(logger.run_id, "resume_step", trainer.global_step)
        logger.experiment.set_tag(logger.run_id, "slurm_job_id", os.getenv("SLURM_JOB_ID", ""))
        print(f"[FoMoNMR] Resumed MLflow run {logger.run_id} at step {trainer.global_step}", flush=True)


class MemoryLogger(L.Callback):
    def on_train_epoch_start(self, trainer, model):
        if model.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(model.device)

    def on_train_epoch_end(self, trainer, model):
        if model.device.type != "cuda":
            return

        megabyte = 1024 ** 2
        model.log_dict(
            {
                "memory/gpu_peak_allocated_mb": (
                    torch.cuda.max_memory_allocated(model.device) / megabyte
                ),
                "memory/gpu_peak_reserved_mb": (
                    torch.cuda.max_memory_reserved(model.device) / megabyte
                ),
            },
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )


class FingerprintValidationLogger(L.Callback):
    def __init__(self, run_dir):
        self.output_dir = run_dir / "artifacts/fingerprint_validation"

    def on_validation_end(self, trainer, model):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"epoch_{trainer.current_epoch}_step_{trainer.global_step}.csv"
        )
        output_path = self.output_dir / filename
        with output_path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(("range", "mae", "count", "mean_predicted_similarity"))
            writer.writerows(model.fingerprint_validation_rows())

        if isinstance(trainer.logger, MLFlowLogger):
            trainer.logger.experiment.log_artifact(
                trainer.logger.run_id,
                str(output_path),
                artifact_path="fingerprint_validation",
            )


class ShiftValidationLogger(L.Callback):
    def __init__(self, run_dir):
        self.output_dir = run_dir / "artifacts/shift_validation"

    def on_validation_end(self, trainer, model):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"epoch_{trainer.current_epoch}_step_{trainer.global_step}.csv"
        for nucleus in ("h", "c"):
            output_path = self.output_dir / f"{nucleus}_{suffix}"
            with output_path.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(("range", "mae", "count", "mean_predicted_shift"))
                writer.writerows(model.shift_validation_rows(nucleus))

            if isinstance(trainer.logger, MLFlowLogger):
                trainer.logger.experiment.log_artifact(
                    trainer.logger.run_id,
                    str(output_path),
                    artifact_path="shift_validation",
                )


class FinalModelLogger(L.Callback):
    def __init__(self, latest_checkpoints, best_checkpoints, tracking_uri):
        self.latest_checkpoints = latest_checkpoints
        self.best_checkpoints = best_checkpoints
        self.tracking_uri = tracking_uri

    def _log_checkpoints(self, trainer, checkpoint_callback, artifact_path):
        for checkpoint_path in checkpoint_callback.best_k_models:
            path = Path(checkpoint_path)
            if path.is_file():
                trainer.logger.experiment.log_artifact(
                    trainer.logger.run_id,
                    str(path),
                    artifact_path=artifact_path,
                )

    def on_fit_end(self, trainer, model):
        if not trainer.is_global_zero or not isinstance(trainer.logger, MLFlowLogger):
            return

        self._log_checkpoints(
            trainer,
            self.latest_checkpoints,
            artifact_path="checkpoints/latest",
        )
        self._log_checkpoints(
            trainer,
            self.best_checkpoints,
            artifact_path="checkpoints/best",
        )

        best_path = Path(self.best_checkpoints.best_model_path)
        if not best_path.is_file():
            print("No best checkpoint is available to log as an MLflow model.")
            return

        best_model = FoMoNMR.load_from_checkpoint(
            best_path,
            config=model.config,
            map_location="cpu",
        )
        best_model.eval()

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model"
            mlflow.pytorch.save_model(
                best_model,
                model_path,
                serialization_format="pickle",
            )
            trainer.logger.experiment.log_artifacts(
                trainer.logger.run_id,
                str(model_path),
                artifact_path="model",
            )


def parse_args():
    load_dotenv('mlflow.env')
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pretrain", "posttrain"), default="pretrain")
    parser.add_argument(
        "--logging",
        dest="logging_mode",
        choices=("minimal", "complete"),
        default="minimal",
    )
    parser.add_argument("--config")
    parser.add_argument("--pretrained-checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--validation-interval", type=int, default=40_000)
    parser.add_argument("--checkpoint-interval", type=int, default=40_000)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv('MLFLOW_TRACKING_URI'),
    )
    parser.add_argument("--experiment-name")
    parser.add_argument("--run-name")
    parser.add_argument("--mlflow-run-id", help="Existing MLflow run ID for checkpoint resume")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--no-maccs-probe", action="store_true")
    parser.add_argument("--maccs-probe-every-n-validations", type=int, default=1)
    return parser.parse_args()


def paired_dataset(
    source,
    split,
    shuffle,
    shift_only,
    seed,
    *,
    shard_across_ranks=True,
    replicate_when_too_small=False,
):
    root = Path("datasets/train_splits")
    return PairedFoundationDataset(
        root / f"{source}_{split}.parquet",
        root / f"{source}_{split}_mol_properties.parquet",
        shuffle=shuffle,
        shuffle_buffer_size=65536,
        seed=seed,
        shift_only=shift_only,
        source_name=source,
        shard_across_ranks=shard_across_ranks,
        replicate_when_too_small=replicate_when_too_small,
    )


def make_dataloaders(stage, batch_size, num_workers, seed):
    if stage == "pretrain":
        train_data = MixedFoundationDataset(
            datasets=[
                paired_dataset(
                    "simnmr", 
                    "train", 
                    True, 
                    True, 
                    seed
                ),
                paired_dataset(
                    "rich_shuffle",
                    "train",
                    True,
                    True,
                    seed,
                    replicate_when_too_small=True,
                ),
                paired_dataset(
                    "nmrgym",
                    "train",
                    True,
                    True,
                    seed,
                    replicate_when_too_small=True,
                ),
            ],
            proportions=[0.90, 0.09, 0.01],
            shift_only=[True, True, True],
            seed=seed,
        )
    else:
        train_data = MixedFoundationDataset(
            datasets=[
                paired_dataset(
                    "rich_shuffle", 
                    "train", 
                    True, 
                    False, 
                    seed
                ),
                paired_dataset(
                    "simnmr",
                    "train",
                    True,
                    True,
                    seed,
                    replicate_when_too_small=True,
                ),
                paired_dataset(
                    "nmrgym",
                    "train",
                    True,
                    True,
                    seed,
                    replicate_when_too_small=True,
                ),
            ],
            proportions=[0.90, 0.05, 0.05],
            shift_only=[False, True, True],
            seed=seed,
        )

    validation_data = [
        paired_dataset(
            "rich",
            "val",
            False,
            False,
            seed,
            shard_across_ranks=True,
        ),
    ]
    if stage == "pretrain":
        validation_data.append(
            paired_dataset(
                "simnmr",
                "val",
                False,
                True,
                seed,
                shard_across_ranks=True,
            )
        )
    validation_data.append(
        paired_dataset(
            "nmrgym",
            "val",
            False,
            True,
            seed,
            shard_across_ranks=True,
        )
    )

    processor = FoundationNMRProcessor()
    train_loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": processor,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_data, drop_last=True, **train_loader_options)

    val_loader_options = {
        "batch_size": 2048,
        "num_workers": 0,
        "collate_fn": processor,
        "pin_memory": torch.cuda.is_available(),
    }

    val_loaders = [
        DataLoader(dataset, drop_last=False, **val_loader_options)
        for dataset in validation_data
    ]
    return train_loader, val_loaders


def make_probe_dataloaders(batch_size, num_workers, seed):
    root = Path("datasets/train_splits/maccs_probe")
    processor = FoundationNMRProcessor()
    loaders = []
    for split in ("train", "eval"):
        dataset = PairedFoundationDataset(
            root / f"{split}.parquet",
            root / f"{split}_mol_properties.parquet",
            shuffle=False,
            seed=seed,
            shift_only=True,
            source_name=f"maccs_probe_{split}",
            shard_across_ranks=True,
        )
        probe_workers = min(num_workers, dataset.num_row_groups)
        loaders.append(
            DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=probe_workers,
                collate_fn=processor,
                pin_memory=torch.cuda.is_available(),
                drop_last=False,
            )
        )
    return loaders

def main():
    args = parse_args()
    if args.mlflow_run_id and (not args.resume or not args.run_name or args.no_mlflow):
        raise ValueError("--mlflow-run-id requires --resume, --run-name and MLflow logging")
    if args.pretrained_checkpoint and args.resume:
        raise ValueError("Use --pretrained-checkpoint or --resume, not both")
    if args.pretrained_checkpoint and args.stage != "posttrain":
        raise ValueError("--pretrained-checkpoint is only used for posttrain")

    L.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    config_path = args.config or f"scripts/model/configs/{args.stage}.yaml"
    with open(config_path) as file:
        config_values = yaml.safe_load(file)
    config_values["stage"] = args.stage
    config_values["logging_mode"] = args.logging_mode
    if args.max_steps is not None:
        config_values["max_steps"] = args.max_steps
    config = ModelConfig(**config_values)

    train_loader, val_loaders = make_dataloaders(
        args.stage,
        args.batch_size,
        args.num_workers,
        args.seed,
    )

    if args.pretrained_checkpoint:
        model = FoMoNMR.load_from_checkpoint(
            args.pretrained_checkpoint,
            config=config,
        )
    else:
        model = FoMoNMR(config)

    run_name = args.run_name or (
        f"{args.stage}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    run_dir = Path("runs/fomonmr") / run_name

    latest_checkpoints = ModelCheckpoint(
        dirpath=run_dir / "checkpoints/latest",
        filename="step={step}",
        monitor="step",
        mode="max",
        save_top_k=3,
        save_last="link",
        save_on_exception=True,
        every_n_train_steps=args.checkpoint_interval,
        auto_insert_metric_name=False,
    )
    best_checkpoints = ModelCheckpoint(
        dirpath=run_dir / "checkpoints/best",
        filename="best-step={step}",
        monitor="val/loss",
        mode="min",
        save_top_k=2,
        save_on_train_epoch_end=False,
        auto_insert_metric_name=False,
    )
    early_stopping = EarlyStopping(
        monitor="val/loss",
        mode="min",
        patience=config.early_stopping_patience,
        check_on_train_epoch_end=False,
    )
    callbacks = [
        latest_checkpoints,
        best_checkpoints,
        early_stopping,
        FingerprintValidationLogger(run_dir),
        ShiftValidationLogger(run_dir),
    ]
    if not args.no_maccs_probe:
        probe_train_loader, probe_eval_loader = make_probe_dataloaders(
            args.batch_size,
            args.num_workers,
            args.seed,
        )
        callbacks.append(
            MaccsLinearProbe(
                probe_train_loader,
                probe_eval_loader,
                seed=args.seed,
                every_n_validations=args.maccs_probe_every_n_validations,
            )
        )

    logger = False
    if not args.no_mlflow:
        logger = MLFlowLogger(
            experiment_name=args.experiment_name or f"fomonmr-{args.stage}",
            run_name=run_name,
            run_id=args.mlflow_run_id,
            tracking_uri=args.tracking_uri,
            log_model=False,
            tags={"stage": args.stage},
        )
        if args.mlflow_run_id:
            callbacks.append(ResumedRunLogger())
        logger.log_hyperparams(
            {
                **asdict(config),
                "accumulate_grad_batches": args.accumulate_grad_batches,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "seed": args.seed,
                "precision": args.precision,
                "validation_interval": args.validation_interval,
                "checkpoint_interval": args.checkpoint_interval,
                "maccs_probe": not args.no_maccs_probe,
                "maccs_probe_every_n_validations": (
                    args.maccs_probe_every_n_validations
                ),
            }
        )
        callbacks.extend(
            [
                LearningRateMonitor(logging_interval="step"),
                MemoryLogger(),
                FinalModelLogger(
                    latest_checkpoints,
                    best_checkpoints,
                    args.tracking_uri,
                ),
            ]
        )

    validation_batches = args.validation_interval * args.accumulate_grad_batches
    strategy = "auto"
    if int(os.getenv("SLURM_NTASKS", "1")) > 1:
        # Sharded IterableDatasets can yield a different final batch count on
        # each rank, so validation forwards must not broadcast DDP buffers.
        strategy = DDPStrategy(
            broadcast_buffers=False,
            # Posttraining losses are conditional on the annotations/tasks
            # present in each batch, so some heads can be unused on a rank.
            find_unused_parameters=args.stage == "posttrain",
        )

    # The training mixture is infinite; max_steps is the authoritative limit.
    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=strategy,
        precision=args.precision,
        max_steps=config.max_steps,
        max_epochs=-1,
        val_check_interval=validation_batches,
        check_val_every_n_epoch=None,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        log_every_n_steps=16,
        logger=logger,
        callbacks=callbacks,
        default_root_dir=run_dir,
    )
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loaders,
        ckpt_path=args.resume,
    )

if __name__ == "__main__":
    main()

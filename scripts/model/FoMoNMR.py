from dataclasses import asdict, dataclass
import yaml

import lightning as L
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .embedding import NmrEmbedder
from .losses import (
    focal_cross_entropy,
    gaussian_soft_cross_entropy,
    symmetric_fingerprint_pairs,
    tanimoto_to_bins,
)
from .processor import MULTIPLICITY_TO_ID
from .utils.corruption import (
    apply_global_shift,
    apply_local_jitter,
    mask_annotations,
    select_masked_peak,
)


@dataclass
class ModelConfig:
    stage: str = "pretrain"
    logging_mode: str = "minimal"

    d_model: int = 512
    fourier_strategy: str = "log_spaced"
    num_fourier_freqs: int = 256
    num_multiplicities: int | None = None
    j_min: float = 0.0
    j_max: float = 20.0
    num_rbf_centers: int = 32
    rbf_sigma: float = 2.0

    nhead: int = 8
    num_encoder_layers: int = 12
    dim_feedforward: int = 2048
    dropout: float = 0.1

    h_min: float = -5.5
    h_max: float = 20.0
    h_bin_size: float = 0.02
    c_min: float = -40.0
    c_max: float = 300.0
    c_bin_size: float = 0.2

    fp_sim_bin_size: float = 0.05
    fp_focal_gamma: float = 5.0
    balanced_fp_pairs: bool = False

    global_shift_probability: float = 0.2
    h_global_shift_max: float = 0.01
    c_global_shift_max: float = 0.1
    local_jitter_probability: float = 0.5
    h_jitter_sigma: float = 0.05
    c_jitter_sigma: float = 0.5
    modality_dropout_probability: float = 0.1
    annotation_mask_probability: float = 0.15
    use_rich_input: bool = True
    use_annotation_loss: bool = True
    h_soft_label_sigma: float = 0.05
    c_soft_label_sigma: float = 0.5

    lambda_fp: float = 0.1
    lambda_annotation: float = 0.2

    lr: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    warmup_steps: int = 5_000
    plateau_factor: float = 0.5
    plateau_patience: int = 2
    early_stopping_patience: int = 8
    max_steps: int = 1_000_000
    validation_seed: int = 42


class FoMoNMR(L.LightningModule):
    validation_metric_names = {
        "loss": "loss",
        "shift": "shift",
        "shift_h": "shift_h",
        "shift_c": "shift_c",
        "fingerprint": "fp_loss",
        "fingerprint_mae": "fp_mae",
        "annotation": "annotation",
        "integration": "annotation/integration",
        "multiplicity": "annotation/multiplicity",
        "width": "annotation/width",
        "j": "annotation/j",
        "j_count": "annotation/j_count",
        "j_value": "annotation/j_value",
        "shift_h_mae": "shift/h_mae_ppm",
        "shift_c_mae": "shift/c_mae_ppm",
    }
    training_metric_names = {
        "loss": "train/loss",
        "shift": "train/shift",
        "shift_h": "train/shift/h",
        "shift_c": "train/shift/c",
        "fingerprint": "train/fp_loss",
        "annotation": "train/rich",
        "integration": "train/rich/integration",
        "multiplicity": "train/rich/multiplicity",
        "width": "train/rich/width",
        "j": "train/rich/j",
        "j_count": "train/rich/j_count",
        "j_value": "train/rich/j_value",
    }
    base_loss_names = ("loss", "shift", "fingerprint")
    shift_component_names = ("shift_h", "shift_c")
    rich_loss_names = (
        "annotation",
        "integration",
        "multiplicity",
        "width",
        "j",
        "j_count",
        "j_value",
    )
    fingerprint_range_labels = (
        "0.0-0.2",
        "0.2-0.4",
        "0.4-0.6",
        "0.6-0.8",
        "0.8-1.0",
    )

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = ModelConfig()
        elif isinstance(config, dict):
            config = ModelConfig(**config)
        self.config = config
        if config.logging_mode not in ("minimal", "complete"):
            raise ValueError("logging_mode must be minimal or complete")
        self.save_hyperparameters({"config": asdict(config)}, logger=False)
        self.validation_metric_sums = {}
        self.validation_metric_counts = {}
        self.validation_source_metric_sums = {}
        self.validation_source_metric_counts = {}

        self.validation_source_names = []
        self.fp_source_range_error_sums = {}
        self.fp_source_range_counts = {}
        self.register_buffer(
            "fp_range_error_sums", torch.zeros(5), persistent=False
        )
        self.register_buffer(
            "fp_range_prediction_sums", torch.zeros(5), persistent=False
        )
        self.register_buffer(
            "fp_range_counts", torch.zeros(5, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "shift_range_error_sums", torch.zeros(2, 20), persistent=False
        )
        self.register_buffer(
            "shift_range_prediction_sums", torch.zeros(2, 20), persistent=False
        )
        self.register_buffer(
            "shift_range_counts",
            torch.zeros(2, 20, dtype=torch.long),
            persistent=False,
        )

        self.embedding = NmrEmbedder(
            d_model=self.config.d_model,
            fourier_strategy=self.config.fourier_strategy,
            h_x_min=self.config.h_min,
            h_x_max=self.config.h_max,
            h_resolution=self.config.h_bin_size,
            c_x_min=self.config.c_min,
            c_x_max=self.config.c_max,
            c_resolution=self.config.c_bin_size,
            num_fourier_freqs=self.config.num_fourier_freqs,
            num_multiplicities=self.config.num_multiplicities,
            j_min=self.config.j_min,
            j_max=self.config.j_max,
            num_rbf_centers=self.config.num_rbf_centers,
            rbf_sigma=self.config.rbf_sigma,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.num_encoder_layers,
        )

        h_num_bins = round(
            (self.config.h_max - self.config.h_min) / self.config.h_bin_size
        ) + 1
        c_num_bins = round(
            (self.config.c_max - self.config.c_min) / self.config.c_bin_size
        ) + 1
        self.h_classification_head = nn.Linear(self.config.d_model, h_num_bins)
        self.c_classification_head = nn.Linear(self.config.d_model, c_num_bins)

        self.fp_sim_num_bins = int(1.0 / self.config.fp_sim_bin_size) + 1
        self.fp_sim_classifier = nn.Sequential(
            nn.Linear(2 * self.config.d_model, self.config.d_model),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.d_model, self.fp_sim_num_bins),
        )

        self.i_regression_head = nn.Linear(self.config.d_model, 1)
        self.m_classification_head = nn.Linear(
            self.config.d_model,
            len(MULTIPLICITY_TO_ID),
        )
        self.w_regression_head = nn.Linear(self.config.d_model, 1)
        self.j_count_head = nn.Linear(self.config.d_model, 7)
        self.j_value_head = nn.Linear(self.config.d_model, 6)

        if self.config.stage == "pretrain":
            h_embedder = self.embedding.h_embedder
            rich_modules = (
                h_embedder.integration_embedder,
                h_embedder.multiplicity_embedder,
                h_embedder.j_embedder,
                h_embedder.width_embedder,
                h_embedder.rich_mlp,
                self.i_regression_head,
                self.m_classification_head,
                self.w_regression_head,
                self.j_count_head,
                self.j_value_head,
            )
            for module in rich_modules:
                module.requires_grad_(False)

    @classmethod
    def from_config(cls, path):
        with open(path) as f:
            config = yaml.safe_load(f)

        return cls(ModelConfig(**config))

    def forward(
        self,
        batch,
        h_shift_prediction_mask=None,
        c_shift_prediction_mask=None,
    ):
        peak_embeddings, valid_peak_mask = self.embedding(
            batch,
            h_shift_prediction_mask=h_shift_prediction_mask,
            c_shift_prediction_mask=c_shift_prediction_mask,
        )
        peak_states = self.transformer_encoder(
            peak_embeddings,
            src_key_padding_mask=~valid_peak_mask,
        )

        mask = valid_peak_mask.unsqueeze(-1)
        pooled = (peak_states * mask).sum(dim=1)
        pooled = pooled / mask.sum(dim=1).clamp_min(1)
        return pooled, peak_states, valid_peak_mask

    def corrupt_batch(self, batch):
        corrupted = {
            "h": {name: value.clone() for name, value in batch["h"].items()},
            "c": {name: value.clone() for name, value in batch["c"].items()},
        }
        h = corrupted["h"]
        c = corrupted["c"]

        h["shift"] = apply_global_shift(
            h["shift"],
            h["peak_mask"],
            self.config.h_min,
            self.config.h_max,
            self.config.h_global_shift_max,
            self.config.global_shift_probability,
        )
        h_shift_targets = h["shift"].clone()

        c["shift"] = apply_global_shift(
            c["shift"],
            c["peak_mask"],
            self.config.c_min,
            self.config.c_max,
            self.config.c_global_shift_max,
            self.config.global_shift_probability,
        )
        c_shift_targets = c["shift"].clone()

        has_h = h["peak_mask"].any(dim=1)
        has_c = c["peak_mask"].any(dim=1)
        drop_modality = (
            torch.rand_like(has_h, dtype=torch.float32)
            < self.config.modality_dropout_probability
        ) & has_h & has_c
        drop_h = drop_modality & (
            torch.rand_like(has_h, dtype=torch.float32) < 0.5
        )
        drop_c = drop_modality & ~drop_h

        h["peak_mask"][drop_h] = False
        h["availability"][drop_h] = False
        h["j_mask"][drop_h] = False
        c["peak_mask"][drop_c] = False

        h_shift_mask = select_masked_peak(h["peak_mask"])
        c_shift_mask = select_masked_peak(c["peak_mask"])

        h["shift"] = apply_local_jitter(
            h["shift"],
            h["peak_mask"],
            h_shift_mask,
            self.config.local_jitter_probability,
            self.config.h_jitter_sigma,
            self.config.h_min,
            self.config.h_max,
        )
        c["shift"] = apply_local_jitter(
            c["shift"],
            c["peak_mask"],
            c_shift_mask,
            self.config.local_jitter_probability,
            self.config.c_jitter_sigma,
            self.config.c_min,
            self.config.c_max,
        )

        if self.config.stage == "pretrain":
            h["availability"].fill_(False)
            h["j_mask"].fill_(False)
            annotation_mask = torch.zeros_like(h["availability"])
        else:
            shift_only = batch["shift_only"]
            h["availability"][shift_only] = False
            h["j_mask"][shift_only] = False
            annotation_mask = mask_annotations(
                h,
                self.config.annotation_mask_probability,
            )
            if not self.config.use_rich_input:
                h["availability"].fill_(False)
                h["j_mask"].fill_(False)

        corruption = {
            "h_shift_mask": h_shift_mask,
            "c_shift_mask": c_shift_mask,
            "h_shift_targets": h_shift_targets,
            "c_shift_targets": c_shift_targets,
            "annotation_mask": annotation_mask,
        }
        return corrupted, corruption

    def compute_losses(
        self,
        outputs,
        clean_batch,
        corruption,
        compute_mae=False,
        fp_source=None,
    ):
        pooled, peak_states, _ = outputs
        zero = pooled.sum() * 0.0

        h_peak_count = clean_batch["h"]["shift"].shape[1]
        h_states = peak_states[:, :h_peak_count]
        c_states = peak_states[:, h_peak_count:]

        h_shift_mask = corruption["h_shift_mask"]
        h_shift_loss = zero
        h_shift_mae = zero
        if h_shift_mask.any():
            h_logits = self.h_classification_head(h_states[h_shift_mask])
            h_targets = corruption["h_shift_targets"][h_shift_mask]
            h_bin_centers = self.config.h_min + torch.arange(
                self.h_classification_head.out_features,
                device=h_logits.device,
            ) * self.config.h_bin_size
            h_shift_loss = gaussian_soft_cross_entropy(
                h_logits,
                h_targets,
                h_bin_centers,
                self.config.h_soft_label_sigma,
            )
            if compute_mae:
                h_prediction = (
                    F.softmax(h_logits.float(), dim=1) * h_bin_centers
                ).sum(dim=1)
                h_shift_mae = F.l1_loss(h_prediction, h_targets)
                self.update_shift_range_statistics(
                    0, h_targets, h_prediction, self.config.h_min, self.config.h_max
                )

        c_shift_mask = corruption["c_shift_mask"]
        c_shift_loss = zero
        c_shift_mae = zero
        if c_shift_mask.any():
            c_logits = self.c_classification_head(c_states[c_shift_mask])
            c_targets = corruption["c_shift_targets"][c_shift_mask]
            c_bin_centers = self.config.c_min + torch.arange(
                self.c_classification_head.out_features,
                device=c_logits.device,
            ) * self.config.c_bin_size
            c_shift_loss = gaussian_soft_cross_entropy(
                c_logits,
                c_targets,
                c_bin_centers,
                self.config.c_soft_label_sigma,
            )

            if compute_mae:
                c_prediction = (
                    F.softmax(c_logits.float(), dim=1) * c_bin_centers
                ).sum(dim=1)
                c_shift_mae = F.l1_loss(c_prediction, c_targets)
                self.update_shift_range_statistics(
                    1, c_targets, c_prediction, self.config.c_min, self.config.c_max
                )

        shift_losses = []
        if h_shift_mask.any():
            shift_losses.append(h_shift_loss)
        if c_shift_mask.any():
            shift_losses.append(c_shift_loss)
        shift_loss = torch.stack(shift_losses).mean() if shift_losses else zero

        # Validation keeps every natural pair and the fixed weights for comparison.
        balance_fp_pairs = self.training and self.config.balanced_fp_pairs
        pair_features, similarities = symmetric_fingerprint_pairs(
            pooled,
            clean_batch["fingerprints"],
            bin_size=self.config.fp_sim_bin_size,
            max_pairs_per_bin=1024 if balance_fp_pairs else None,
        )
        fingerprint_loss = zero
        fingerprint_mae = zero
        if similarities.numel() > 0:
            fp_logits = self.fp_sim_classifier(pair_features)
            fp_targets = tanimoto_to_bins(
                similarities,
                self.config.fp_sim_bin_size,
                self.fp_sim_num_bins,
            )
            pair_weights = None
            if not balance_fp_pairs:
                range_idx = (similarities / 0.2).long().clamp(max=4)
                range_weights = similarities.new_tensor([
                    1.0,   # 0.0 - 0.2
                    2.0,   # 0.2 - 0.4
                    5.0,   # 0.4 - 0.6
                    10.0,  # 0.6 - 0.8
                    20.0,  # 0.8 - 1.0
                ])
                pair_weights = range_weights[range_idx]
            fingerprint_loss = focal_cross_entropy(
                fp_logits,
                fp_targets,
                self.config.fp_focal_gamma,
                pair_weights
            )
            if compute_mae:
                fp_values = (
                    torch.arange(self.fp_sim_num_bins, device=fp_logits.device) + 0.5
                ) * self.config.fp_sim_bin_size
                fp_values = fp_values.clamp_max(1.0)
                fp_prediction = (
                    F.softmax(fp_logits.float(), dim=1) * fp_values
                ).sum(dim=1)
                fp_errors = (fp_prediction - similarities).abs()
                fingerprint_mae = fp_errors.mean()

                range_indices = (similarities / 0.2).long().clamp(max=4)
                self.fp_range_error_sums.scatter_add_(
                    0, range_indices, fp_errors.detach()
                )
                self.fp_range_prediction_sums.scatter_add_(
                    0, range_indices, fp_prediction.detach()
                )
                self.fp_range_counts.scatter_add_(
                    0, range_indices, torch.ones_like(range_indices)
                )
                if fp_source is not None:
                    self.fp_source_range_error_sums[fp_source].scatter_add_(
                        0, range_indices, fp_errors.detach()
                    )
                    self.fp_source_range_counts[fp_source].scatter_add_(
                        0, range_indices, torch.ones_like(range_indices)
                    )

        integration_loss = zero
        multiplicity_loss = zero
        width_loss = zero
        j_loss = zero
        j_count_loss = zero
        j_value_loss = zero
        annotation_loss = zero

        if self.config.stage == "posttrain" and self.config.use_annotation_loss:
            h = clean_batch["h"]
            annotation_mask = corruption["annotation_mask"]
            annotation_losses = []

            integration_mask = annotation_mask[:, :, 0]
            if integration_mask.any():
                prediction = self.i_regression_head(
                    h_states[integration_mask]
                ).squeeze(-1)
                target = torch.log1p(h["integration"][integration_mask])
                integration_loss = F.smooth_l1_loss(prediction, target)
                annotation_losses.append(integration_loss)

            multiplicity_mask = annotation_mask[:, :, 1]
            if multiplicity_mask.any():
                logits = self.m_classification_head(h_states[multiplicity_mask])
                target = h["multiplicity"][multiplicity_mask]
                multiplicity_loss = F.cross_entropy(logits, target)
                annotation_losses.append(multiplicity_loss)

            width_mask = annotation_mask[:, :, 3]
            if width_mask.any():
                prediction = self.w_regression_head(
                    h_states[width_mask]
                ).squeeze(-1)
                target = h["range_half_span"][width_mask]
                width_loss = F.smooth_l1_loss(prediction, target)
                annotation_losses.append(width_loss)

            masked_j = annotation_mask[:, :, 2]
            if masked_j.any():
                selected_states = h_states[masked_j]
                valid_j = h["j_mask"][masked_j]
                target_counts = valid_j.sum(dim=1)

                count_logits = self.j_count_head(selected_states)
                j_count_loss = F.cross_entropy(count_logits, target_counts)

                predicted_values = self.j_value_head(selected_states)
                target_values = h["j_values"][masked_j].masked_fill(
                    ~valid_j,
                    float("inf"),
                )
                target_values = target_values.sort(dim=1).values
                value_slots = (
                    torch.arange(6, device=predicted_values.device).unsqueeze(0)
                    < target_counts.unsqueeze(1)
                )

                if value_slots.any():
                    j_value_loss = F.smooth_l1_loss(
                        predicted_values[value_slots],
                        target_values[value_slots],
                    )
                    j_loss = (j_count_loss + j_value_loss) / 2
                else:
                    j_loss = j_count_loss
                annotation_losses.append(j_loss)

            if annotation_losses:
                annotation_loss = torch.stack(annotation_losses).mean()

        total_loss = shift_loss + self.config.lambda_fp * fingerprint_loss
        if self.config.stage == "posttrain" and self.config.use_annotation_loss:
            total_loss += self.config.lambda_annotation * annotation_loss

        return {
            "loss": total_loss,
            "shift": shift_loss,
            "shift_h": h_shift_loss,
            "shift_c": c_shift_loss,
            "fingerprint": fingerprint_loss,
            "fingerprint_mae": fingerprint_mae,
            "annotation": annotation_loss,
            "integration": integration_loss,
            "multiplicity": multiplicity_loss,
            "width": width_loss,
            "j": j_loss,
            "j_count": j_count_loss,
            "j_value": j_value_loss,
            "shift_h_mae": h_shift_mae,
            "shift_c_mae": c_shift_mae,
        }

    def losses_to_log(self, losses, include_annotations=True):
        names = list(self.base_loss_names)
        if self.config.logging_mode == "complete":
            names.extend(self.shift_component_names)
        if (
            self.config.stage == "posttrain"
            and self.config.use_annotation_loss
            and include_annotations
        ):
            names.append("annotation")
            if self.config.logging_mode == "complete":
                names.extend(self.rich_loss_names[1:])
        return {name: losses[name] for name in names}

    def training_step(self, batch, batch_idx):
        corrupted, corruption = self.corrupt_batch(batch)
        outputs = self(
            corrupted,
            h_shift_prediction_mask=corruption["h_shift_mask"],
            c_shift_prediction_mask=corruption["c_shift_mask"],
        )
        losses = self.compute_losses(
            outputs,
            batch,
            corruption,
        )
        batch_size = batch["h"]["shift"].shape[0]
        for name, value in self.losses_to_log(losses).items():
            self.log(
                self.training_metric_names[name],
                value,
                on_step=True,
                on_epoch=False,
                prog_bar=name == "loss",
                batch_size=batch_size,
                sync_dist=True,
            )
        return losses["loss"]

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx < len(self.validation_source_names):
            source = self.validation_source_names[dataloader_idx]
        else:
            source = f"dataloader_{dataloader_idx}"

        if source not in self.validation_source_metric_sums:
            metric_names = tuple(self.validation_metric_names)
            self.validation_source_metric_sums[source] = {
                name: torch.zeros((), device=self.device)
                for name in metric_names
            }
            self.validation_source_metric_counts[source] = {
                name: torch.zeros((), device=self.device, dtype=torch.long)
                for name in metric_names
            }

        # The same validation record receives the same corruption every run.
        with torch.random.fork_rng():
            seed = self.config.validation_seed + dataloader_idx * 100_000 + batch_idx
            torch.manual_seed(seed)
            corrupted, corruption = self.corrupt_batch(batch)
            outputs = self(
                corrupted,
                h_shift_prediction_mask=corruption["h_shift_mask"],
                c_shift_prediction_mask=corruption["c_shift_mask"],
            )
            losses = self.compute_losses(
                outputs,
                batch,
                corruption,
                compute_mae=True,
                fp_source=(
                    source if source in self.fp_source_range_counts else None
                ),
            )

        batch_size = batch["h"]["shift"].shape[0]
        annotations_active = (
            self.config.stage == "posttrain"
            and self.config.use_annotation_loss
            and not batch["shift_only"].all().item()
        )
        global_names = [*self.base_loss_names, *self.shift_component_names]
        if annotations_active:
            global_names.append("annotation")
        global_metrics = {name: losses[name] for name in global_names}
        metric_weights = {name: batch_size for name in global_metrics}
        metric_counts = {
            "shift_h_mae": int(corruption["h_shift_mask"].sum()),
            "shift_c_mae": int(corruption["c_shift_mask"].sum()),
            "fingerprint_mae": batch_size * (batch_size - 1) // 2,
        }
        for name, count in metric_counts.items():
            if count > 0:
                global_metrics[name] = losses[name]
                metric_weights[name] = count

        for name, value in global_metrics.items():
            weight = metric_weights[name]
            if name == "fingerprint_mae":
                continue
            self.validation_metric_sums[name] = (
                self.validation_metric_sums.get(name, 0.0)
                + value.detach() * weight
            )
            self.validation_metric_counts[name] = (
                self.validation_metric_counts.get(name, 0) + weight
            )

        source_metrics = {"loss": losses["loss"]}
        if self.config.logging_mode == "complete":
            source_metrics = {name: losses[name] for name in global_names}
            if annotations_active:
                source_metrics.update(
                    {name: losses[name] for name in self.rich_loss_names[1:]}
                )
            for name, count in metric_counts.items():
                if count > 0:
                    source_metrics[name] = losses[name]

        for name, value in source_metrics.items():
            weight = metric_weights.get(name, batch_size)
            self.validation_source_metric_sums[source][name] += (
                value.detach() * weight
            )
            self.validation_source_metric_counts[source][name] += weight
        return losses

    def on_validation_epoch_start(self):
        if self._trainer is not None:
            loaders = self.trainer.val_dataloaders
            if not isinstance(loaders, (list, tuple)):
                loaders = [loaders]
            self.validation_source_names = [
                loader.dataset.source_name
                for loader in loaders
            ]
        metric_names = tuple(self.validation_metric_names)
        self.validation_metric_sums = {
            name: torch.zeros((), device=self.device)
            for name in metric_names
        }
        self.validation_metric_counts = {
            name: torch.zeros((), device=self.device, dtype=torch.long)
            for name in metric_names
        }
        self.validation_source_metric_sums = {
            source: {
                name: torch.zeros((), device=self.device)
                for name in metric_names
            }
            for source in self.validation_source_names
        }
        self.validation_source_metric_counts = {
            source: {
                name: torch.zeros((), device=self.device, dtype=torch.long)
                for name in metric_names
            }
            for source in self.validation_source_names
        }
        self.fp_source_range_error_sums = {
            source: torch.zeros_like(self.fp_range_error_sums)
            for source in self.validation_source_names
        }
        self.fp_source_range_counts = {
            source: torch.zeros_like(self.fp_range_counts)
            for source in self.validation_source_names
        }
        self.fp_range_error_sums.zero_()
        self.fp_range_prediction_sums.zero_()
        self.fp_range_counts.zero_()
        self.shift_range_error_sums.zero_()
        self.shift_range_prediction_sums.zero_()
        self.shift_range_counts.zero_()

    def update_shift_range_statistics(
        self, nucleus_index, targets, predictions, minimum, maximum
    ):
        range_width = (maximum - minimum) / 20
        range_indices = ((targets - minimum) / range_width).long().clamp(0, 19)
        self.shift_range_error_sums[nucleus_index].scatter_add_(
            0, range_indices, (predictions - targets).abs().detach()
        )
        self.shift_range_prediction_sums[nucleus_index].scatter_add_(
            0, range_indices, predictions.detach()
        )
        self.shift_range_counts[nucleus_index].scatter_add_(
            0, range_indices, torch.ones_like(range_indices)
        )

    def fingerprint_validation_rows(self):
        error_sums = self.fp_range_error_sums.detach().cpu().tolist()
        prediction_sums = self.fp_range_prediction_sums.detach().cpu().tolist()
        counts = self.fp_range_counts.detach().cpu().tolist()
        rows = []
        for label, error_sum, prediction_sum, count in zip(
            self.fingerprint_range_labels,
            error_sums,
            prediction_sums,
            counts,
        ):
            mae = error_sum / count if count else float("nan")
            mean_prediction = prediction_sum / count if count else float("nan")
            rows.append((label, mae, int(count), mean_prediction))
        return rows

    def shift_validation_rows(self, nucleus):
        nucleus_index = 0 if nucleus == "h" else 1
        minimum = self.config.h_min if nucleus == "h" else self.config.c_min
        maximum = self.config.h_max if nucleus == "h" else self.config.c_max
        range_width = (maximum - minimum) / 20
        error_sums = self.shift_range_error_sums[nucleus_index].cpu().tolist()
        prediction_sums = self.shift_range_prediction_sums[nucleus_index].cpu().tolist()
        counts = self.shift_range_counts[nucleus_index].cpu().tolist()
        rows = []
        for index, (error_sum, prediction_sum, count) in enumerate(
            zip(error_sums, prediction_sums, counts)
        ):
            lower = minimum + index * range_width
            upper = minimum + (index + 1) * range_width
            mae = error_sum / count if count else float("nan")
            mean_prediction = prediction_sum / count if count else float("nan")
            rows.append(
                (f"{lower:.3f}-{upper:.3f}", mae, int(count), mean_prediction)
            )
        return rows

    def on_validation_epoch_end(self):
        if dist.is_available() and dist.is_initialized():
            for name in self.validation_metric_sums:
                dist.all_reduce(
                    self.validation_metric_sums[name],
                    op=dist.ReduceOp.SUM,
                )
                dist.all_reduce(
                    self.validation_metric_counts[name],
                    op=dist.ReduceOp.SUM,
                )

            for source in self.validation_source_names:
                for name in self.validation_source_metric_sums[source]:
                    dist.all_reduce(
                        self.validation_source_metric_sums[source][name],
                        op=dist.ReduceOp.SUM,
                    )
                    dist.all_reduce(
                        self.validation_source_metric_counts[source][name],
                        op=dist.ReduceOp.SUM,
                    )
                dist.all_reduce(
                    self.fp_source_range_error_sums[source],
                    op=dist.ReduceOp.SUM,
                )
                dist.all_reduce(
                    self.fp_source_range_counts[source],
                    op=dist.ReduceOp.SUM,
                )

            for tensor in (
                self.fp_range_error_sums,
                self.fp_range_prediction_sums,
                self.fp_range_counts,
                self.shift_range_error_sums,
                self.shift_range_prediction_sums,
                self.shift_range_counts,
            ):
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        mean_metrics = {}
        for name, total in self.validation_metric_sums.items():
            count = self.validation_metric_counts[name]
            if count == 0:
                continue
            mean_metrics[name] = total / count
            self.log(
                f"val/{self.validation_metric_names[name]}",
                mean_metrics[name],
                prog_bar=name == "loss",
                sync_dist=True,
            )

        for source in self.validation_source_names:
            names = ["loss"]
            if self.config.logging_mode == "complete":
                names = list(self.validation_metric_names)
            for name in names:
                count = self.validation_source_metric_counts[source][name]
                if count == 0:
                    continue
                value = self.validation_source_metric_sums[source][name] / count
                self.log(
                    f"val/datasets/{source}/{self.validation_metric_names[name]}",
                    value,
                    prog_bar=name == "loss",
                    sync_dist=True,
                )

        valid_ranges = self.fp_range_counts > 0
        if valid_ranges.any():
            range_maes = (
                self.fp_range_error_sums[valid_ranges]
                / self.fp_range_counts[valid_ranges]
            )
            fingerprint_mae = (
                self.fp_range_error_sums.sum()
                / self.fp_range_counts.sum()
            )
            self.log("val/fp_mae", fingerprint_mae, sync_dist=True)
            self.log("val/fp_macro_mae", range_maes.mean(), sync_dist=True)

        mean_loss = mean_metrics.get("loss")
        if self.config.logging_mode == "complete":
            for source, counts in self.fp_source_range_counts.items():
                valid_ranges = counts > 0
                if valid_ranges.any():
                    range_maes = (
                        self.fp_source_range_error_sums[source][valid_ranges]
                        / counts[valid_ranges]
                    )
                    self.log(
                        f"val/datasets/{source}/fp_macro_mae",
                        range_maes.mean(),
                        sync_dist=True,
                    )
        if mean_loss is not None:

            if (
                not self.trainer.sanity_checking
                and self.global_step >= self.config.warmup_steps
            ):
                self.lr_schedulers().step(mean_loss)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        if self.global_step < self.config.warmup_steps:
            warmup_lr = self.config.lr * (
                (self.global_step + 1) / self.config.warmup_steps
            )
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = warmup_lr

        super().optimizer_step(
            epoch, batch_idx, optimizer, optimizer_closure
        )

    def configure_optimizers(self):
        trainable_parameters = [
            parameter for parameter in self.parameters() if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=self.config.lr,
            betas=(0.9, 0.99),
            weight_decay=self.config.weight_decay,
            eps=1e-6,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.config.plateau_factor,
            patience=self.config.plateau_patience,
            min_lr=self.config.min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }

    def lr_scheduler_step(self, scheduler, metric):
        # ReduceLROnPlateau is stepped after each validation above.
        pass

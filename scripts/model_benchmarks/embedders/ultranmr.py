"""Extract embeddings with the released UltraNMR implementation."""

import importlib
import sys
from collections.abc import Mapping
from pathlib import Path

from model_benchmarks.processors import ModelBatch

from . import EmbeddingResult


def _default_repo_path() -> Path:
    return Path(__file__).resolve().parents[3] / "models" / "UltraNMR"


def _load_official_model_class(repo_path: Path):
    """Import the cloned repository only when UltraNMR is requested."""

    repo_path = repo_path.resolve()
    if not repo_path.is_dir():
        raise FileNotFoundError(f"UltraNMR repository not found: {repo_path}")
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    module = importlib.import_module("models.UltraNMR")
    module_file = Path(getattr(module, "__file__", "")).resolve()
    try:
        module_file.relative_to(repo_path)
    except ValueError as error:
        raise ImportError(
            "models.UltraNMR resolved outside the cloned repository; "
            "run one model family per process"
        ) from error
    return module.UltraNMR


class UltraNMREmbedder:
    """Load UltraNMR and mean-pool its final valid token states."""

    model_name = "ultranmr"
    dimension = 768
    pooling = "validity-masked mean of final Transformer token states"

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str = "cpu",
        repo_path: str | Path | None = None,
    ):
        import torch

        self._torch = torch
        self.repo_path = (
            Path(repo_path).expanduser() if repo_path else _default_repo_path()
        )
        path = checkpoint_path or (
            self.repo_path
            / "model_checkpoint"
            / "checkpoints_nce"
            / "model_epoch_1.pth"
        )
        self.checkpoint_path = Path(path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"UltraNMR checkpoint not found: {self.checkpoint_path}"
            )

        self.device = torch.device(device)
        self.model, self.loaded_parameter_count = self._load_model()

    def _load_model(self):
        torch = self._torch
        model_class = _load_official_model_class(self.repo_path)
        model = model_class(
            d_model=768,
            nhead=12,
            num_encoder_layers=16,
            dim_feedforward=3072,
            dropout=0.1,
            h_bin_size=0.01,
            h_max=16.0,
            c_bin_size=0.1,
            c_max=230.0,
            fp_sim_bin_size=0.005,
            fp_sim_mlp_hidden=512,
            use_identity_embedding=True,
            use_count_embedding=False,
        )

        checkpoint = torch.load(
            str(self.checkpoint_path), map_location="cpu", weights_only=True
        )
        if isinstance(checkpoint, Mapping):
            state_dict = checkpoint.get("model_state_dict", checkpoint)
        else:
            state_dict = checkpoint
        if not isinstance(state_dict, Mapping):
            raise ValueError("UltraNMR checkpoint does not contain a state dictionary")

        required_keys = {
            "embedding.h_shift_ffn.ff.0.weight",
            "embedding.c_shift_ffn.ff.0.weight",
            "transformer_encoder.layers.0.self_attn.in_proj_weight",
            "transformer_encoder.layers.15.norm2.weight",
        }
        missing = sorted(required_keys.difference(state_dict))
        if missing:
            raise ValueError(
                "UltraNMR checkpoint lacks required encoder weights: "
                + ", ".join(missing)
            )

        # strict=True prevents missing weights and random encoder layers.
        model.load_state_dict(state_dict, strict=True)
        probe = dict(model.named_parameters()).get(
            "transformer_encoder.layers.0.self_attn.in_proj_weight"
        )
        if probe is None or probe.shape != (2304, 768):
            raise RuntimeError("UltraNMR encoder weights were not loaded")
        if not torch.isfinite(probe.reshape(-1)[:1024]).all():
            raise RuntimeError("UltraNMR checkpoint contains non-finite encoder weights")

        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        model.to(self.device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model, parameter_count

    def encode(self, batch: ModelBatch) -> EmbeddingResult:
        torch = self._torch
        names = ("shifts", "counts", "types", "padding_mask")
        missing = [name for name in names if name not in batch.inputs]
        if missing:
            raise ValueError("UltraNMR batch is missing: " + ", ".join(missing))

        shifts, counts, types, padding_mask = (
            batch.inputs[name] for name in names
        )
        tensors = (shifts, counts, types, padding_mask)
        if not all(isinstance(value, torch.Tensor) for value in tensors):
            raise TypeError("UltraNMR batch inputs must be tensors")
        if shifts.ndim != 2 or any(value.shape != shifts.shape for value in tensors[1:]):
            raise ValueError("UltraNMR tensors must share shape [batch, tokens]")
        if shifts.shape[0] != len(batch.record_ids) or shifts.shape[1] == 0:
            raise ValueError("UltraNMR batch shape does not match record_ids")

        shifts = shifts.to(self.device, dtype=torch.float32)
        counts = counts.to(self.device, dtype=torch.float32)
        types = types.to(self.device, dtype=torch.long)
        padding_mask = padding_mask.to(self.device, dtype=torch.bool)
        if padding_mask.all(dim=1).any():
            raise ValueError("every UltraNMR record needs at least one valid token")

        with torch.inference_mode():
            # These are the spectrum-encoder operations in UltraNMR.forward.
            hidden = self.model.embedding(shifts, counts, types)
            hidden = self.model.transformer_encoder(
                hidden, src_key_padding_mask=padding_mask
            )
            valid = (~padding_mask).unsqueeze(-1).to(hidden.dtype)
            embeddings = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)

        if embeddings.shape != (len(batch.record_ids), self.dimension):
            raise RuntimeError(f"unexpected UltraNMR shape: {tuple(embeddings.shape)}")
        if not torch.isfinite(embeddings).all():
            raise RuntimeError("UltraNMR produced non-finite embeddings")

        return EmbeddingResult(
            embeddings=embeddings.detach().to(device="cpu", dtype=torch.float32),
            record_ids=list(batch.record_ids),
            model_name=self.model_name,
            checkpoint=str(self.checkpoint_path),
            dimension=self.dimension,
            pooling=self.pooling,
            metadata={
                "modality": "1H+13C",
                "processor_mode": batch.metadata.get("processor_mode"),
                "checkpoint_loading": "strict",
                "loaded_parameters": self.loaded_parameter_count,
            },
        )


__all__ = ["UltraNMREmbedder"]


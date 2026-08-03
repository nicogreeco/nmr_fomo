"""Extract NMRTrans embeddings from its released H and C encoders."""

import importlib.util
import sys
from collections.abc import Mapping
from pathlib import Path

from model_benchmarks.processors import ModelBatch

from . import EmbeddingResult


_MODEL_ROOT = Path(__file__).resolve().parents[3] / "models" / "NMRTrans"
_DEFAULT_CHECKPOINT = _MODEL_ROOT / "model" / "nmrtrans-c-h-nmr.ckpt"


def load_official_model(model_root: str | Path):
    """Load the released NMRTrans model module only when requested."""

    path = Path(model_root) / "src" / "model.py"
    if not path.is_file():
        raise FileNotFoundError(f"NMRTrans model module not found: {path}")
    specification = importlib.util.spec_from_file_location(
        "_model_benchmarks_nmrtrans_model", path
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"could not load NMRTrans model module from {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _load_checkpoint(path: Path, model_root: Path, torch):
    """Load the Lightning checkpoint with its config class available."""

    config_path = model_root / "src" / "config.py"
    specification = importlib.util.spec_from_file_location("config", config_path)
    if specification is None or specification.loader is None:
        raise ImportError(f"could not load NMRTrans config from {config_path}")

    config_module = importlib.util.module_from_spec(specification)
    previous_config = sys.modules.get("config")
    sys.modules["config"] = config_module
    try:
        specification.loader.exec_module(config_module)
        try:
            checkpoint = torch.load(
                str(path), map_location="cpu", weights_only=False, mmap=True
            )
        except TypeError:  # Older torch versions lack weights_only or mmap.
            checkpoint = torch.load(str(path), map_location="cpu")
    finally:
        if previous_config is None:
            sys.modules.pop("config", None)
        else:
            sys.modules["config"] = previous_config

    if not isinstance(checkpoint, Mapping):
        raise RuntimeError(f"NMRTrans checkpoint must contain a mapping: {path}")
    return checkpoint


def _config_value(config, name: str):
    if isinstance(config, Mapping) and name in config:
        return config[name]
    if hasattr(config, name):
        return getattr(config, name)
    raise RuntimeError(f"NMRTrans checkpoint config is missing {name}")


def _encoder_weights(state_dict: Mapping, prefix: str) -> dict:
    weights = {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    if not weights:
        raise RuntimeError(f"NMRTrans checkpoint contains no {prefix[:-1]} weights")
    return weights


class NMRTransEmbedder:
    """Return masked means of final H and C local states (512 + 512)."""

    model_name = "nmrtrans"
    pooling = "concat(masked_mean(final H local), masked_mean(final C local))"

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str = "cpu",
        model_root: str | Path | None = None,
    ):
        import torch

        self._torch = torch
        self.device = torch.device(device)
        self.model_root = Path(model_root or _MODEL_ROOT).expanduser().resolve()
        path = Path(checkpoint_path) if checkpoint_path else _DEFAULT_CHECKPOINT
        self.checkpoint_path = path.expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"NMRTrans checkpoint not found: {self.checkpoint_path}"
            )

        checkpoint = _load_checkpoint(self.checkpoint_path, self.model_root, torch)
        state_dict = checkpoint.get("state_dict", checkpoint)
        if not isinstance(state_dict, Mapping):
            raise RuntimeError("NMRTrans state_dict is not a mapping")

        hyper_parameters = checkpoint.get("hyper_parameters")
        if not isinstance(hyper_parameters, Mapping) or "config" not in hyper_parameters:
            raise RuntimeError("NMRTrans checkpoint is missing hyper_parameters.config")
        config = hyper_parameters["config"]

        d_model = int(_config_value(config, "PEAK_ENCODER_D_MODEL"))
        n_layers = int(_config_value(config, "PEAK_ENCODER_N_LAYERS"))
        n_heads = int(_config_value(config, "PEAK_ENCODER_N_HEADS"))
        feedforward_dimension = int(_config_value(config, "PEAK_ENCODER_FF_DIM"))
        dropout = float(_config_value(config, "PEAK_ENCODER_DROPOUT"))
        max_peaks = int(_config_value(config, "MAX_PEAKS"))
        if max_peaks != 60:
            raise RuntimeError(f"expected released MAX_PEAKS=60, got {max_peaks}")

        c_weights = _encoder_weights(state_dict, "c_encoder.")
        h_weights = _encoder_weights(state_dict, "h_encoder.")
        try:
            number_of_inducing_points = int(c_weights["isab_layers.0.I"].shape[1])
            number_of_seeds = int(c_weights["pma.S"].shape[1])
            split_vocabulary_size = int(h_weights["split_embedding.weight"].shape[0])
        except KeyError as error:
            raise RuntimeError(
                f"NMRTrans checkpoint has an unexpected encoder layout: {error}"
            ) from error

        official = load_official_model(self.model_root)
        self.c_encoder = official.SetTransformerPeakEncoder(
            d_model=d_model,
            n_heads=n_heads,
            num_inds=number_of_inducing_points,
            num_seeds=number_of_seeds,
            n_layers=n_layers,
            ff_dim=feedforward_dimension,
            dropout=dropout,
            max_peaks=max_peaks,
        )
        self.h_encoder = official.SetTransformer1HNMRPeakEncoder(
            d_model=d_model,
            n_heads=n_heads,
            num_inds=number_of_inducing_points,
            num_seeds=number_of_seeds,
            n_layers=n_layers,
            ff_dim=feedforward_dimension,
            dropout=dropout,
            max_peaks=max_peaks,
            split_vocab_size=split_vocabulary_size,
        )

        # Strict loading ensures no encoder layer remains randomly initialized.
        self.c_encoder.load_state_dict(c_weights, strict=True)
        self.h_encoder.load_state_dict(h_weights, strict=True)
        self.c_encoder.to(self.device).eval()
        self.h_encoder.to(self.device).eval()
        for encoder in (self.c_encoder, self.h_encoder):
            for parameter in encoder.parameters():
                parameter.requires_grad_(False)

        self.dimension = d_model * 2
        self._loaded_state_keys = {
            "h_encoder": len(h_weights),
            "c_encoder": len(c_weights),
        }

    def _batch_inputs(self, batch: ModelBatch):
        names = ("h_nmr_features", "h_nmr_mask", "c_nmr_peaks", "c_nmr_mask")
        missing = [name for name in names if name not in batch.inputs]
        if missing:
            raise KeyError("NMRTrans batch is missing: " + ", ".join(missing))

        tensors = tuple(batch.inputs[name].to(self.device) for name in names)
        if any(tensor.shape[0] != len(batch.record_ids) for tensor in tensors):
            raise ValueError("NMRTrans batch rows do not match record_ids")
        if tensors[0].shape[1:] != (60, 10) or tensors[1].shape[1:] != (60,):
            raise ValueError(
                "NMRTrans H tensors must have shapes [B,60,10] and [B,60]"
            )
        if tensors[2].shape[1:] != (60, 1) or tensors[3].shape[1:] != (60,):
            raise ValueError(
                "NMRTrans C tensors must have shapes [B,60,1] and [B,60]"
            )
        return tensors

    def _forward(self, batch: ModelBatch):
        h_features, h_mask, c_peaks, c_mask = self._batch_inputs(batch)
        h_local, h_pma, h_valid = self.h_encoder(h_features, h_mask)
        c_local, c_pma, c_valid = self.c_encoder(c_peaks, c_mask)
        return h_local, h_pma, h_valid, c_local, c_pma, c_valid

    @staticmethod
    def _masked_mean(values, valid_mask):
        weights = valid_mask.to(dtype=values.dtype).unsqueeze(-1)
        return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def encode(self, batch: ModelBatch) -> EmbeddingResult:
        torch = self._torch
        with torch.inference_mode():
            h_local, _, h_valid, c_local, _, c_valid = self._forward(batch)
            h_embedding = self._masked_mean(h_local, h_valid)
            c_embedding = self._masked_mean(c_local, c_valid)
            embeddings = torch.cat((h_embedding, c_embedding), dim=-1)

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
                "attention_mask_behavior": "official checkpoint",
                "loaded_state_keys": self._loaded_state_keys,
            },
        )

    def pma_diagnostics(self, batch: ModelBatch) -> dict:
        """Return official PMA tensors for diagnostics, not embeddings."""

        torch = self._torch
        with torch.inference_mode():
            _, h_pma, _, _, c_pma, _ = self._forward(batch)
        return {
            "h_pma": h_pma.detach().to(device="cpu", dtype=torch.float32),
            "c_pma": c_pma.detach().to(device="cpu", dtype=torch.float32),
            "record_ids": list(batch.record_ids),
        }


__all__ = ["NMRTransEmbedder", "load_official_model"]


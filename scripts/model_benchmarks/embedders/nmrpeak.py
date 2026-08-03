"""Extract NMRPeak-R embeddings with the released BART encoder."""

from pathlib import Path

from model_benchmarks.processors import ModelBatch

from . import EmbeddingResult


_CHECKPOINT_DIRECTORY = (
    "NMRexp_lr3e-4_bs512_gpu1_t0.1_unimol_pretrain1_frozen0_"
    "spec_mol_cl_bart_base_spec_mol_cl_15000_250000"
)


def default_checkpoint_path() -> Path:
    root = Path(__file__).resolve().parents[3] / "models" / "NMRPeak" / "weights"
    candidates = (
        root
        / "retrieval"
        / "all_weights"
        / _CHECKPOINT_DIRECTORY
        / "CH"
        / "checkpoint_best.pt",
        root
        / "weights"
        / "retrieval"
        / "all_weights"
        / _CHECKPOINT_DIRECTORY
        / "CH"
        / "checkpoint_best.pt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "NMRPeak CH checkpoint not found; checked: "
        + ", ".join(str(path) for path in candidates)
    )


class NMRPeakEmbedder:
    """Load the BART spectrum encoder and return normalized BOS states."""

    model_name = "nmrpeak"
    dimension = 768
    pooling = "final BART encoder BOS, L2-normalized"

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str = "cpu",
    ):
        # Imports stay lazy because NMRPeak has its own environment.
        try:
            import torch
            from transformers.models.bart.modeling_bart import BartEncoder
        except ImportError as error:
            raise ImportError(
                "NMRPeak requires torch and transformers from nmrpeak_venv"
            ) from error

        path = Path(checkpoint_path) if checkpoint_path else default_checkpoint_path()
        self.checkpoint_path = path.expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"NMRPeak checkpoint not found: {self.checkpoint_path}"
            )

        self._torch = torch
        self.device = torch.device(device)
        self.encoder, self.pad_idx, self.max_sequence_length = self._load_encoder(
            BartEncoder
        )
        self.encoder.to(self.device).eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    def _load_encoder(self, bart_encoder_class):
        torch = self._torch
        try:
            checkpoint = torch.load(
                self.checkpoint_path, map_location="cpu", weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")

        config = getattr(checkpoint.get("args"), "configuration", None)
        if config is None:
            raise ValueError("NMRPeak checkpoint has no BART configuration")
        if getattr(config, "d_model", None) != self.dimension:
            raise ValueError(
                f"unexpected NMRPeak dimension {getattr(config, 'd_model', None)!r}"
            )

        model_state = checkpoint.get("model")
        if not isinstance(model_state, dict):
            raise ValueError("NMRPeak checkpoint has no model state dictionary")
        prefix = "spec_encoder."
        encoder_state = {
            key[len(prefix) :]: value
            for key, value in model_state.items()
            if key.startswith(prefix)
        }
        if not encoder_state:
            raise ValueError("NMRPeak checkpoint contains no spectrum encoder weights")

        # Loading only this branch avoids the unrelated large molecule encoder.
        encoder = bart_encoder_class(config)
        encoder.load_state_dict(encoder_state, strict=True)
        self._loaded_state_keys = len(encoder_state)

        pad_idx = getattr(config, "pad_token_id", None)
        max_positions = getattr(config, "max_position_embeddings", None)
        if not isinstance(pad_idx, int):
            raise ValueError("NMRPeak checkpoint has no integer pad token")
        if not isinstance(max_positions, int) or max_positions <= 0:
            raise ValueError("NMRPeak checkpoint has no valid position limit")
        return encoder, pad_idx, max_positions

    def encode(self, batch: ModelBatch) -> EmbeddingResult:
        torch = self._torch
        src_tokens = batch.inputs.get("src_tokens")
        if not isinstance(src_tokens, torch.Tensor) or src_tokens.ndim != 2:
            raise TypeError("NMRPeak src_tokens must be a rank-2 tensor")
        if src_tokens.shape[0] != len(batch.record_ids):
            raise ValueError("NMRPeak batch rows do not match record_ids")
        if src_tokens.shape[1] > self.max_sequence_length:
            raise ValueError(
                f"input length {src_tokens.shape[1]} exceeds NMRPeak limit "
                f"{self.max_sequence_length}"
            )

        batch_pad_idx = batch.metadata.get("pad_idx")
        if batch_pad_idx is not None and int(batch_pad_idx) != self.pad_idx:
            raise ValueError("processor and checkpoint use different pad indices")

        src_tokens = src_tokens.to(self.device)
        attention_mask = batch.inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = src_tokens.ne(self.pad_idx)
        elif not isinstance(attention_mask, torch.Tensor):
            raise TypeError("NMRPeak attention_mask must be a tensor")
        else:
            attention_mask = attention_mask.to(self.device)

        with torch.inference_mode():
            hidden = self.encoder(
                input_ids=src_tokens,
                attention_mask=attention_mask,
                return_dict=True,
            ).last_hidden_state
            embeddings = torch.nn.functional.normalize(hidden[:, 0, :], dim=-1)

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
                "normalized": True,
                "loaded_state_keys": self._loaded_state_keys,
            },
        )


__all__ = ["NMRPeakEmbedder", "default_checkpoint_path"]


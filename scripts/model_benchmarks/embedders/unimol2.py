"""Extract molecular embeddings with the official UniMol2 implementation."""

from pathlib import Path

from model_benchmarks.processors import ModelBatch
from model_benchmarks.processors.unimol2 import FEATURE_NAMES

from . import EmbeddingResult


def normalize_model_size(value: str) -> str:
    """Return the two supported UniMol2 size names in their public form."""

    normalized = str(value).strip().lower()
    if normalized == "84m":
        return "84M"
    if normalized == "164m":
        return "164M"
    raise ValueError("UniMol2 model_size must be '84M' or '164M'")


class UniMol2Embedder:
    """Load UniMol2 and return its final molecular CLS representation."""

    model_name = "unimol2"
    modality = "molecule"
    dimension = 768
    pooling = "final UniMol2 CLS token"

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str = "cpu",
        model_size: str = "84M",
    ):
        try:
            import torch
            from unimol_tools.models import UniMolV2Model
        except ImportError as error:
            raise ImportError(
                "UniMol2 embedding requires torch and unimol_tools from "
                "unimol2_venv"
            ) from error

        self._torch = torch
        self.model_size = normalize_model_size(model_size)
        self.device = torch.device(device)

        pretrained_path = None
        if checkpoint_path is not None:
            pretrained_path = Path(checkpoint_path).expanduser().resolve()
            if not pretrained_path.is_file():
                raise FileNotFoundError(
                    f"UniMol2 checkpoint not found: {pretrained_path}"
                )

        self.model = UniMolV2Model(
            model_size=self.model_size.lower(),
            pretrained_model_path=(
                str(pretrained_path) if pretrained_path is not None else None
            ),
        )
        self.checkpoint_path = Path(self.model.pretrain_path).expanduser().resolve()
        self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def encode(self, batch: ModelBatch) -> EmbeddingResult:
        torch = self._torch
        missing = [name for name in FEATURE_NAMES if name not in batch.inputs]
        if missing:
            raise ValueError("UniMol2 batch is missing: " + ", ".join(missing))
        if not batch.record_ids:
            raise ValueError("UniMol2 cannot encode an empty batch")

        inputs = {}
        for name in FEATURE_NAMES:
            value = batch.inputs[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"UniMol2 input {name!r} must be a tensor")
            if value.shape[0] != len(batch.record_ids):
                raise ValueError(
                    f"UniMol2 input {name!r} does not match record_ids"
                )
            inputs[name] = value.to(self.device)

        with torch.inference_mode():
            embeddings = self.model(**inputs, return_repr=True)

        expected_shape = (len(batch.record_ids), self.dimension)
        if not isinstance(embeddings, torch.Tensor):
            raise RuntimeError("UniMol2 did not return a CLS tensor")
        if tuple(embeddings.shape) != expected_shape:
            raise RuntimeError(
                f"unexpected UniMol2 shape: {tuple(embeddings.shape)}"
            )
        if not torch.isfinite(embeddings).all():
            raise RuntimeError("UniMol2 produced non-finite embeddings")

        return EmbeddingResult(
            embeddings=embeddings.detach().to(device="cpu", dtype=torch.float32),
            record_ids=list(batch.record_ids),
            model_name=self.model_name,
            checkpoint=str(self.checkpoint_path),
            dimension=self.dimension,
            pooling=self.pooling,
            metadata={
                "modality": self.modality,
                "model_size": self.model_size,
                "processor_mode": batch.metadata.get("processor_mode"),
                "conformer_seed": batch.metadata.get("conformer_seed"),
                "max_atoms": batch.metadata.get("max_atoms"),
            },
        )


__all__ = ["UniMol2Embedder", "normalize_model_size"]

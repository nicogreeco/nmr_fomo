"""Shared result container for the embedding wrappers."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EmbeddingResult:
    """Embeddings and the minimal information needed to interpret them."""

    embeddings: Any
    record_ids: list[str]
    model_name: str
    checkpoint: str | None
    dimension: int
    pooling: str
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = ["EmbeddingResult"]


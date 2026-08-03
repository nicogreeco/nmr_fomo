"""Extract the official fixed NMR-Solver Gaussian representation."""

import importlib.util
from pathlib import Path

from model_benchmarks.processors import ModelBatch

from . import EmbeddingResult


def _default_repo_path() -> Path:
    return Path(__file__).resolve().parents[3] / "models" / "NMR-Solver"


def _load_official_match_module(repo_path: Path):
    module_path = repo_path.resolve() / "src" / "utils" / "nmr_match.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"NMR-Solver nmr_match.py not found: {module_path}")
    specification = importlib.util.spec_from_file_location(
        "_model_benchmarks_nmrsolver_match", module_path
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot import NMR-Solver module: {module_path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class NMRSolverEmbedder:
    """Return the fixed, non-learned 256-D H+C set2vec representation."""

    model_name = "nmrsolver"
    dimension = 256
    pooling = "concatenated normalized H/C Gaussian set2vec (128+128)"

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str = "cpu",
        repo_path: str | Path | None = None,
    ):
        if checkpoint_path is not None:
            raise ValueError("NMR-Solver is fixed and has no spectrum checkpoint")

        import torch

        if torch.device(device).type != "cpu":
            raise ValueError("the NMR-Solver NumPy/SciPy featurizer runs on CPU")
        self._torch = torch
        self.repo_path = (
            Path(repo_path).expanduser() if repo_path else _default_repo_path()
        )
        self._set2vec = _load_official_match_module(self.repo_path).set2vec

    def encode(self, batch: ModelBatch) -> EmbeddingResult:
        import numpy as np

        torch = self._torch
        try:
            h_rows = batch.inputs["h_shifts"]
            c_rows = batch.inputs["c_shifts"]
        except KeyError as error:
            raise ValueError(f"NMR-Solver batch is missing {error.args[0]}") from error
        if len(h_rows) != len(c_rows) or len(h_rows) != len(batch.record_ids):
            raise ValueError("NMR-Solver rows do not match record_ids")
        if not h_rows:
            raise ValueError("NMR-Solver cannot encode an empty batch")

        h_arrays = [np.asarray(row, dtype=np.float32) for row in h_rows]
        c_arrays = [np.asarray(row, dtype=np.float32) for row in c_rows]
        if any(row.ndim != 1 or row.size == 0 for row in h_arrays + c_arrays):
            raise ValueError(
                "every NMR-Solver record needs non-empty 1D H and C shifts"
            )
        if any(not np.isfinite(row).all() for row in h_arrays + c_arrays):
            raise ValueError("NMR-Solver shifts must be finite")

        # This is the exact fixed representation released with NMR-Solver.
        h_vectors = self._set2vec(h_arrays, "H", dim=128, normalize=True)
        c_vectors = self._set2vec(c_arrays, "C", dim=128, normalize=True)
        combined = np.concatenate((h_vectors, c_vectors), axis=1)
        embeddings = torch.from_numpy(combined).to(dtype=torch.float32).contiguous()
        if embeddings.shape != (len(batch.record_ids), self.dimension):
            raise RuntimeError(f"unexpected NMR-Solver shape: {tuple(embeddings.shape)}")

        return EmbeddingResult(
            embeddings=embeddings.cpu(),
            record_ids=list(batch.record_ids),
            model_name=self.model_name,
            checkpoint=None,
            dimension=self.dimension,
            pooling=self.pooling,
            metadata={
                "modality": "1H+13C",
                "processor_mode": batch.metadata.get("processor_mode"),
                "representation": "fixed_featurizer",
                "learned": False,
            },
        )


__all__ = ["NMRSolverEmbedder"]


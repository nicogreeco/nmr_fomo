"""Real-checkpoint smoke tests for the lightweight embedding wrappers."""

import importlib.util
import os
import unittest
from pathlib import Path

from data import CanonicalRecord, CarbonPeak, ProtonPeak
from model_benchmarks import build_embedder, build_processor


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def example_record():
    return CanonicalRecord(
        record_id="example-1",
        h_nmr_peaks=(
            ProtonPeak(
                shift=1.20,
                integration=2,
                multiplicity_raw="p",
                multiplicity="quint",
                j_values=(7.2,),
                range_min=1.19,
                range_max=1.21,
                range_half_span=0.01,
            ),
        ),
        c_nmr_peaks=(CarbonPeak(shift=42.0),),
    )


def require_modules(test_case, *module_names):
    missing = [
        name for name in module_names if importlib.util.find_spec(name) is None
    ]
    if missing:
        test_case.skipTest("missing model dependencies: " + ", ".join(missing))


class EmbedderSmokeTests(unittest.TestCase):
    def test_nmrpeak_checkpoint_returns_normalized_bos(self):
        require_modules(self, "torch", "transformers", "nmrpeak", "unicore")
        import torch

        from model_benchmarks.embedders.nmrpeak import default_checkpoint_path

        try:
            checkpoint_path = default_checkpoint_path()
        except FileNotFoundError as error:
            self.skipTest(str(error))

        batch = build_processor("nmrpeak")([example_record()])
        embedder = build_embedder(
            "nmrpeak", checkpoint_path=checkpoint_path, device="cpu"
        )
        result = embedder.encode(batch)

        self.assertEqual(result.record_ids, ["example-1"])
        self.assertEqual(tuple(result.embeddings.shape), (1, 768))
        self.assertIn("BOS", result.pooling)
        self.assertTrue(result.metadata["normalized"])
        norms = torch.linalg.vector_norm(result.embeddings, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones(1), atol=1e-5))

    def test_ultranmr_checkpoint(self):
        if os.environ.get("NMR_BENCHMARK_RUN_ULTRANMR_SMOKE") != "1":
            self.skipTest(
                "set NMR_BENCHMARK_RUN_ULTRANMR_SMOKE=1 for the large checkpoint"
            )
        require_modules(self, "torch")
        import torch

        checkpoint_path = (
            PROJECT_ROOT
            / "models"
            / "UltraNMR"
            / "model_checkpoint"
            / "checkpoints_nce"
            / "model_epoch_1.pth"
        )
        if not checkpoint_path.is_file():
            self.skipTest(f"UltraNMR checkpoint not found: {checkpoint_path}")

        batch = build_processor("ultranmr")([example_record()])
        embedder = build_embedder(
            "ultranmr", checkpoint_path=checkpoint_path, device="cpu"
        )
        result = embedder.encode(batch)

        self.assertEqual(result.record_ids, ["example-1"])
        self.assertEqual(tuple(result.embeddings.shape), (1, 768))
        self.assertTrue(torch.isfinite(result.embeddings).all().item())
        self.assertGreater(result.metadata["loaded_parameters"], 0)

    def test_nmrsolver_fixed_256_features(self):
        require_modules(self, "torch", "numpy", "scipy")
        import torch

        official_file = (
            PROJECT_ROOT / "models" / "NMR-Solver" / "src" / "utils" / "nmr_match.py"
        )
        if not official_file.is_file():
            self.skipTest(f"NMR-Solver utility not found: {official_file}")

        batch = build_processor("nmrsolver")([example_record()])
        result = build_embedder("nmrsolver", device="cpu").encode(batch)

        self.assertEqual(result.record_ids, ["example-1"])
        self.assertEqual(tuple(result.embeddings.shape), (1, 256))
        h_norm = torch.linalg.vector_norm(result.embeddings[:, :128], dim=1)
        c_norm = torch.linalg.vector_norm(result.embeddings[:, 128:], dim=1)
        self.assertTrue(torch.allclose(h_norm, torch.ones(1), atol=1e-6))
        self.assertTrue(torch.allclose(c_norm, torch.ones(1), atol=1e-6))
        self.assertIs(result.metadata["learned"], False)
        self.assertEqual(result.metadata["representation"], "fixed_featurizer")
        self.assertIsNone(result.checkpoint)


if __name__ == "__main__":
    unittest.main()

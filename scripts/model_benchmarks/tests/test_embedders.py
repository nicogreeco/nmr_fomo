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
        smiles="CCO",
        smiles_canonical="CCO",
        atoms=("C", "C", "O"),
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

    def test_morgan_fixed_2048_bit_fingerprint(self):
        require_modules(self, "torch", "rdkit")
        import torch

        batch = build_processor("morgan")([example_record()])
        result = build_embedder("morgan", device="cpu").encode(batch)

        self.assertEqual(result.record_ids, ["example-1"])
        self.assertEqual(tuple(result.embeddings.shape), (1, 2048))
        self.assertEqual(set(result.embeddings.unique().tolist()), {0.0, 1.0})
        self.assertEqual(result.metadata["radius"], 2)
        self.assertEqual(result.metadata["n_bits"], 2048)
        self.assertEqual(result.metadata["modality"], "molecule")
        self.assertIs(result.metadata["learned"], False)
        self.assertIsNone(result.checkpoint)
        self.assertEqual(result.embeddings.dtype, torch.float32)

    def test_unimol2_model_size_names(self):
        from model_benchmarks.embedders.unimol2 import normalize_model_size

        self.assertEqual(normalize_model_size("84M"), "84M")
        self.assertEqual(normalize_model_size("84m"), "84M")
        self.assertEqual(normalize_model_size("164M"), "164M")
        with self.assertRaisesRegex(ValueError, "84M.*164M"):
            normalize_model_size("310M")

    def test_unimol2_checkpoint(self):
        if os.environ.get("NMR_BENCHMARK_RUN_UNIMOL2_SMOKE") != "1":
            self.skipTest(
                "set NMR_BENCHMARK_RUN_UNIMOL2_SMOKE=1 for a local checkpoint"
            )
        require_modules(self, "torch", "rdkit", "unimol_tools")
        import torch

        checkpoint_value = os.environ.get("NMR_BENCHMARK_UNIMOL2_CHECKPOINT")
        if not checkpoint_value:
            self.skipTest("set NMR_BENCHMARK_UNIMOL2_CHECKPOINT")
        checkpoint_path = Path(checkpoint_value).expanduser()
        if not checkpoint_path.is_file():
            self.skipTest(f"UniMol2 checkpoint not found: {checkpoint_path}")

        model_size = os.environ.get("NMR_BENCHMARK_UNIMOL2_SIZE", "84M")
        batch = build_processor("unimol2")([example_record()])
        result = build_embedder(
            "unimol2",
            checkpoint_path=checkpoint_path,
            device="cpu",
            model_size=model_size,
        ).encode(batch)

        self.assertEqual(result.record_ids, ["example-1"])
        self.assertEqual(tuple(result.embeddings.shape), (1, 768))
        self.assertTrue(torch.isfinite(result.embeddings).all().item())
        self.assertEqual(result.metadata["model_size"], model_size.upper())
        self.assertEqual(result.metadata["modality"], "molecule")


if __name__ == "__main__":
    unittest.main()

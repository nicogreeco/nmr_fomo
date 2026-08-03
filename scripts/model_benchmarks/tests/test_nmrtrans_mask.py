"""Reproduce NMRTrans's padding behavior and verify the chosen representation."""

import importlib.util
import unittest
from pathlib import Path

from data import CanonicalRecord, CarbonPeak, ProtonPeak
from model_benchmarks import build_embedder, build_processor


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def example_record(record_id="example-1", h_shift=1.2, c_shift=42.0):
    return CanonicalRecord(
        record_id=record_id,
        h_nmr_peaks=(
            ProtonPeak(
                shift=h_shift,
                integration=2,
                multiplicity_raw="p",
                multiplicity="quint",
                j_values=(7.2,),
                range_min=h_shift - 0.01,
                range_max=h_shift + 0.01,
                range_half_span=0.01,
            ),
        ),
        c_nmr_peaks=(CarbonPeak(shift=c_shift),),
    )


def require_nmrtrans(test_case):
    required_modules = ("torch", "pytorch_lightning", "transformers", "rdkit")
    missing = [
        name for name in required_modules if importlib.util.find_spec(name) is None
    ]
    if missing:
        test_case.skipTest("missing model dependencies: " + ", ".join(missing))


class NMRTransMaskTests(unittest.TestCase):
    def test_masked_local_means_are_stable_across_batch_sizes(self):
        require_nmrtrans(self)
        import torch

        checkpoint_path = (
            PROJECT_ROOT / "models" / "NMRTrans" / "model" / "nmrtrans-c-h-nmr.ckpt"
        )
        if not checkpoint_path.is_file():
            self.skipTest(f"NMRTrans checkpoint not found: {checkpoint_path}")

        processor = build_processor("nmrtrans")
        first_record = example_record()
        second_record = example_record("example-2", h_shift=3.4, c_shift=128.0)
        single_batch = processor([first_record])
        paired_batch = processor([first_record, second_record])
        embedder = build_embedder(
            "nmrtrans", checkpoint_path=checkpoint_path, device="cpu"
        )

        single_result = embedder.encode(single_batch)
        paired_result = embedder.encode(paired_batch)
        diagnostics = embedder.pma_diagnostics(paired_batch)

        self.assertEqual(tuple(single_result.embeddings.shape), (1, 1024))
        self.assertEqual(tuple(paired_result.embeddings.shape), (2, 1024))
        self.assertIn("masked_mean", single_result.pooling)
        self.assertTrue(torch.isfinite(paired_result.embeddings).all().item())
        self.assertTrue(
            torch.allclose(
                single_result.embeddings[0],
                paired_result.embeddings[0],
                rtol=1e-5,
                atol=1e-5,
            )
        )
        self.assertEqual(tuple(diagnostics["h_pma"].shape), (2, 4, 512))
        self.assertEqual(tuple(diagnostics["c_pma"].shape), (2, 4, 512))

    def test_released_pma_changes_when_padding_is_added(self):
        require_nmrtrans(self)
        import torch

        from model_benchmarks.embedders.nmrtrans import load_official_model

        official = load_official_model(PROJECT_ROOT / "models" / "NMRTrans")
        torch.manual_seed(7)
        pma = official.PMA(dim=8, num_heads=2, num_seeds=2, dropout=0.0).eval()
        compact_states = torch.randn(1, 2, 8)
        padded_states = torch.cat((compact_states, torch.zeros(1, 3, 8)), dim=1)

        with torch.inference_mode():
            compact_output = pma(compact_states)
            # The released MAB interprets 1 as valid, but receives this pad mask.
            released_padding_mask = torch.tensor([[False, False, True, True, True]])
            padded_output = pma(padded_states, mask=released_padding_mask)

        self.assertEqual(tuple(compact_output.shape), (1, 2, 8))
        self.assertEqual(tuple(padded_output.shape), (1, 2, 8))
        self.assertFalse(torch.allclose(compact_output, padded_output, atol=1e-5))


if __name__ == "__main__":
    unittest.main()

"""Focused tests for the frozen MACCS linear probe."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from data import CanonicalRecord, CarbonPeak, ProtonPeak
from model import FoundationNMRProcessor, PairedFoundationRecord
from model.FoMoNMR import FoMoNMR, ModelConfig
from model.maccs_probe import MaccsLinearProbe


class MaccsLinearProbeTests(unittest.TestCase):
    def make_batch(self):
        records = []
        for index in range(8):
            fingerprint = "01" * 83 if index % 2 else "10" * 83
            records.append(
                PairedFoundationRecord(
                    record=CanonicalRecord(
                        record_id=str(index),
                        h_nmr_peaks=(
                            ProtonPeak(1.0 + index, integration=index + 1),
                        ),
                        c_nmr_peaks=(CarbonPeak(20.0 + index),),
                    ),
                    morgan_fingerprint=bytes(256),
                    maccs_fingerprint=fingerprint,
                    shift_only=True,
                )
            )
        return FoundationNMRProcessor()(records)

    def test_probe_is_shift_only_and_does_not_update_fomonmr(self):
        model = FoMoNMR(
            ModelConfig(
                stage="posttrain",
                d_model=16,
                num_fourier_freqs=8,
                num_rbf_centers=4,
                nhead=4,
                num_encoder_layers=1,
                dim_feedforward=32,
                dropout=0.0,
            )
        ).eval()
        callback = MaccsLinearProbe([], [], epochs=1, batch_size=4, seed=7)
        train_batch = self.make_batch()
        eval_batch = self.make_batch()
        callback.train_loader = [train_batch]
        callback.eval_loader = [eval_batch]
        score = callback.run(model)

        self.assertFalse(train_batch["h"]["availability"].any())
        self.assertTrue(torch.isfinite(score))
        self.assertTrue(
            all(parameter.grad is None for parameter in model.parameters())
        )

    def test_probe_can_run_every_four_validations(self):
        callback = MaccsLinearProbe([], [], every_n_validations=4)
        callback.run = Mock(return_value=torch.tensor(0.5))
        trainer = SimpleNamespace(sanity_checking=False, is_global_zero=True)
        model = Mock()

        for _ in range(4):
            callback.on_validation_epoch_end(trainer, model)

        callback.run.assert_called_once_with(model)
        model.log.assert_called_once()


if __name__ == "__main__":
    unittest.main()

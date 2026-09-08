"""Focused tests for the FoMoNMR training entry point."""

import unittest
from pathlib import Path

from model.FoMoNMR import ModelConfig
from model.train import (
    FingerprintValidationLogger,
    ShiftValidationLogger,
    validation_artifact_callbacks,
)


class TrainTests(unittest.TestCase):
    def test_similarity_csv_callback_is_only_used_for_similarity_objective(self):
        run_dir = Path("unused")
        similarity = validation_artifact_callbacks(ModelConfig(), run_dir)

        self.assertEqual(
            [type(callback) for callback in similarity],
            [FingerprintValidationLogger, ShiftValidationLogger],
        )

        bits = validation_artifact_callbacks(
            ModelConfig(fingerprint_objective="bits"), run_dir
        )
        self.assertEqual(
            [type(callback) for callback in bits],
            [ShiftValidationLogger],
        )

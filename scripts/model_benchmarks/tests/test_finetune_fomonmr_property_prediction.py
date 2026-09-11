import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from model_benchmarks.finetune_fomonmr_property_prediction import (
    load_model,
    parse_arguments,
    split_train_validation,
)


class FoMoNMRFineTuningTests(unittest.TestCase):
    def test_cli_accepts_run_id_and_dataset_list(self):
        arguments = parse_arguments(
            [
                "--run-id",
                "abc123",
                "--datasets",
                "ames,ld50_zhu",
            ]
        )

        self.assertEqual(arguments.run_id, "abc123")
        self.assertEqual(arguments.datasets, ["ames,ld50_zhu"])
        self.assertIsNone(arguments.checkpoint_path)

    def test_cli_accepts_checkpoint_path(self):
        arguments = parse_arguments(
            [
                "--checkpoint-path",
                "runs/fomonmr/example/checkpoints/best/model.ckpt",
                "--datasets",
                "ames",
            ]
        )

        self.assertEqual(arguments.checkpoint_path.suffix, ".ckpt")
        self.assertIsNone(arguments.run_id)

    def test_local_checkpoint_loading_and_output_names(self):
        for path, expected_name in (
            ("runs/fomonmr/example/checkpoints/best/model.ckpt", "example"),
            ("runs/fomonmr/example/checkpoints/latest/model.ckpt", "example"),
            ("downloaded.ckpt", "downloaded"),
            ("/model.ckpt", "model"),
        ):
            with self.subTest(path=path), patch(
                "model_benchmarks.finetune_fomonmr_property_prediction."
                "FoMoNMR.load_from_checkpoint"
            ) as load:
                model, run_name = load_model(None, Path(path), None)
                load.assert_called_once_with(Path(path).resolve(), map_location="cpu")
                self.assertIs(model, load.return_value)
                self.assertEqual(run_name, expected_name)

    def test_regression_split_keeps_molecules_together(self):
        records = [
            (SimpleNamespace(smiles_canonical="CCO"), 1.0),
            (SimpleNamespace(smiles_canonical="CCO"), 1.1),
            (SimpleNamespace(smiles_canonical="CCC"), 2.0),
            (SimpleNamespace(smiles_canonical="CCC"), 2.1),
            (SimpleNamespace(smiles_canonical="CCN"), 3.0),
            (SimpleNamespace(smiles_canonical="CCN"), 3.1),
            (SimpleNamespace(smiles_canonical="CCCl"), 4.0),
            (SimpleNamespace(smiles_canonical="CCCl"), 4.1),
            (SimpleNamespace(smiles_canonical="CCBr"), 5.0),
            (SimpleNamespace(smiles_canonical="CCBr"), 5.1),
        ]

        train, validation = split_train_validation(
            records,
            classification=False,
            fraction=0.2,
            seed=42,
        )

        train_smiles = {record.smiles_canonical for record, _ in train}
        validation_smiles = {record.smiles_canonical for record, _ in validation}
        self.assertFalse(train_smiles & validation_smiles)


if __name__ == "__main__":
    unittest.main()

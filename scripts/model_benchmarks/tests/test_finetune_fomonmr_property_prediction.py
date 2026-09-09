import unittest
from types import SimpleNamespace

from model_benchmarks.finetune_fomonmr_property_prediction import (
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

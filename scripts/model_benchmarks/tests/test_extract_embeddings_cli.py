"""Command-line defaults for model-specific embedding extraction."""

import contextlib
import io
import unittest
from unittest.mock import patch

from model_benchmarks.extract_embeddings import parse_arguments


def parse_for_model(model_name, *extra_arguments):
    command = [
        "extract_embeddings",
        "--input",
        "input.parquet",
        "--model",
        model_name,
        "--output",
        "output.parquet",
        *extra_arguments,
    ]
    with patch("sys.argv", command):
        return parse_arguments()


class ExtractEmbeddingArgumentTests(unittest.TestCase):
    def test_unimol2_defaults_to_one_record_and_84m(self):
        arguments = parse_for_model("unimol2")

        self.assertEqual(arguments.batch_size, 1)
        self.assertEqual(arguments.model_size, "84M")

    def test_unimol2_allows_an_explicit_larger_batch(self):
        arguments = parse_for_model(
            "uni-mol2", "--batch-size", "4", "--model-size", "164M"
        )

        self.assertEqual(arguments.batch_size, 4)
        self.assertEqual(arguments.model_size, "164M")

    def test_nmr_models_keep_the_existing_batch_default(self):
        arguments = parse_for_model("nmrtrans")

        self.assertEqual(arguments.batch_size, 32)
        self.assertIsNone(arguments.model_size)

    def test_morgan_uses_the_regular_batch_default(self):
        arguments = parse_for_model("morgan")

        self.assertEqual(arguments.batch_size, 32)
        self.assertIsNone(arguments.model_size)

    def test_model_size_is_only_available_for_unimol2(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_for_model("nmrtrans", "--model-size", "164M")

    def test_fomonmr_requires_one_model_source(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_for_model("fomonmr")

    def test_fomonmr_accepts_a_checkpoint_or_run_id(self):
        checkpoint = parse_for_model("fomonmr", "--checkpoint", "model.ckpt", "--input-mode", "shifts")
        run = parse_for_model("fomonmr", "--run-id", "abc123", "--input-mode", "rich")

        self.assertEqual(str(checkpoint.checkpoint), "model.ckpt")
        self.assertEqual(run.run_id, "abc123")

    def test_run_id_is_only_available_for_fomonmr(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_for_model("nmrtrans", "--run-id", "abc123")


if __name__ == "__main__":
    unittest.main()

"""Small tests for FoMoNMR property embedding extraction."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as parquet
import torch

from model_benchmarks.run_fomonmr_property_prediction import (
    apply_checkpoint_input_policy,
    extract_embeddings,
    parse_arguments,
    run_name_from_checkpoint,
)


class FakeFoMoNMR:
    def __init__(self, stage="posttrain", use_rich_input=True):
        self.config = SimpleNamespace(
            d_model=2,
            stage=stage,
            use_rich_input=use_rich_input,
        )

    def transfer_batch_to_device(self, batch, device, dataloader_index):
        return batch

    def __call__(self, batch):
        h_shift = batch["h"]["shift"].sum(dim=1)
        c_shift = batch["c"]["shift"].sum(dim=1)
        pooled = torch.stack((h_shift, c_shift), dim=1)
        return pooled, None, None


def canonical_record(record_id, h_shift, c_shift):
    return {
        "record_id": record_id,
        "source": "fixture",
        "h_nmr_peaks": [{"shift": h_shift}],
        "c_nmr_peaks": [{"shift": c_shift}],
    }


class FoMoNMRPropertyPredictionTests(unittest.TestCase):
    def test_cli_accepts_specific_checkpoint_paths(self):
        arguments = parse_arguments(
            [
                "--checkpoint-path",
                "runs/fomonmr/example/checkpoints/best/model.ckpt",
                "--datasets",
                "ames,ld50_zhu",
                "--experiment-name",
                "comparison",
            ]
        )

        self.assertEqual(len(arguments.checkpoints), 1)
        self.assertEqual(arguments.checkpoints[0].suffix, ".ckpt")
        self.assertEqual(arguments.experiment_name, "comparison")

    def test_cli_accepts_mlflow_run_id_without_checkpoint(self):
        arguments = parse_arguments(
            [
                "--run-id",
                "abc123",
                "--datasets",
                "ames",
                "--experiment-name",
                "comparison",
            ]
        )

        self.assertEqual(arguments.run_id, "abc123")
        self.assertIsNone(arguments.checkpoints)

    def test_cli_requires_one_model_source(self):
        with self.assertRaises(SystemExit):
            parse_arguments(
                [
                    "--datasets",
                    "ames",
                    "--experiment-name",
                    "comparison",
                ]
            )

    def test_run_name_comes_from_checkpoint_parent_run(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            runs_dir = Path(temporary_directory) / "runs/fomonmr"
            checkpoint = runs_dir / "my-run/checkpoints/best/model.ckpt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()

            run_name = run_name_from_checkpoint(checkpoint, runs_dir)

        self.assertEqual(run_name, "my-run")

    def test_shift_only_policy_hides_rich_annotations(self):
        batch = {
            "h": {
                "availability": torch.ones(1, 2, 4, dtype=torch.bool),
                "j_mask": torch.ones(1, 2, 6, dtype=torch.bool),
            }
        }

        apply_checkpoint_input_policy(batch, FakeFoMoNMR(stage="pretrain"))

        self.assertFalse(batch["h"]["availability"].any())
        self.assertFalse(batch["h"]["j_mask"].any())

    def test_posttrain_rich_policy_preserves_annotations(self):
        batch = {
            "h": {
                "availability": torch.ones(1, 2, 4, dtype=torch.bool),
                "j_mask": torch.ones(1, 2, 6, dtype=torch.bool),
            }
        }

        apply_checkpoint_input_policy(batch, FakeFoMoNMR())

        self.assertTrue(batch["h"]["availability"].all())
        self.assertTrue(batch["h"]["j_mask"].all())

    def test_extraction_writes_only_selected_baseline_common_records(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "records.parquet"
            output_path = root / "embeddings/run-a/train.parquet"
            checkpoint = root / "runs/fomonmr/run-a/checkpoints/best/model.ckpt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            parquet.write_table(
                pa.Table.from_pylist(
                    [
                        canonical_record("keep-a", 1.0, 10.0),
                        canonical_record("skip", 2.0, 20.0),
                        canonical_record("keep-b", 3.0, 30.0),
                    ]
                ),
                input_path,
            )

            extract_embeddings(
                model=FakeFoMoNMR(),
                checkpoint=checkpoint,
                run_name="run-a",
                input_path=input_path,
                output_path=output_path,
                selected_ids={"keep-a", "keep-b"},
                batch_size=1,
                device=torch.device("cpu"),
            )

            parquet_file = parquet.ParquetFile(output_path)
            table = parquet_file.read()
            metadata = json.loads(
                parquet_file.metadata.metadata[b"nmr_embedding_metadata"]
            )

        self.assertEqual(table["record_id"].to_pylist(), ["keep-a", "keep-b"])
        self.assertEqual(
            table["embedding"].to_pylist(), [[1.0, 10.0], [3.0, 30.0]]
        )
        self.assertEqual(metadata["accepted_count"], 2)
        self.assertEqual(metadata["model_name"], "run-a")


if __name__ == "__main__":
    unittest.main()

"""Small fixtures for the Rich-only UniMol objective."""

from dataclasses import asdict, replace
from itertools import islice
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import lightning as L
from torch.utils.data import DataLoader

from model import FoundationNMRProcessor, MixedFoundationDataset, PairedFoundationDataset
from model.FoMoNMR import FoMoNMR
from model.losses import capped_similarity_indices, relational_cosine_loss
from model.train import (
    load_pretrained_model,
    make_dataloaders,
    FingerprintValidationLogger,
    validation_artifact_callbacks,
    ShiftValidationLogger,
    UniMolValidationLogger,
)
from model.tests.test_fomonmr import example_batch, small_config
from model.tests import test_dataset


def write_fixture(root):
    nmr, properties, _ = test_dataset.PairedFoundationDatasetTest().write_pair(root)
    source = pq.ParquetFile(properties)
    schema = source.schema_arrow.append(pa.field("unimol_embedding", pa.list_(pa.float32(), 768)))
    metadata = dict(schema.metadata)
    metadata.update({b"unimol_batch_size": b"1", b"unimol_transform": b"rich_train_center_l2"})
    schema = schema.with_metadata(metadata)
    teacher_path = root / "teacher.parquet"
    with pq.ParquetWriter(teacher_path, schema) as writer:
        for index in range(source.num_row_groups):
            table = source.read_row_group(index)
            vectors = np.zeros((len(table), 768), dtype=np.float32)
            vectors[:, index] = 1
            column = pa.FixedSizeListArray.from_arrays(pa.array(vectors.ravel()), 768)
            writer.write_table(table.append_column("unimol_embedding", column).replace_schema_metadata(metadata))
    return nmr, properties, teacher_path


class UniMolDistillationTests(unittest.TestCase):
    def test_checkpoint_loading_without_teachers_and_legacy_morgan_config(self):
        batch = example_batch()
        del batch["fingerprints"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.ckpt"
            for target in ("morgan", "unimol"):
                config = small_config(stage="posttrain", molecular_target=target)
                model = FoMoNMR(config).eval()
                saved_config = asdict(config)
                if target == "morgan":
                    for name in ("molecular_target", "fingerprint_objective", "unimol_sidecar_dir"):
                        del saved_config[name]
                torch.save({
                    "state_dict": model.state_dict(),
                    "hyper_parameters": {"config": saved_config},
                    "pytorch-lightning_version": L.__version__,
                }, path)
                loaded = FoMoNMR.load_from_checkpoint(path, map_location="cpu").eval()
                with torch.inference_mode():
                    torch.testing.assert_close(loaded(batch)[0], model(batch)[0], rtol=0, atol=0)
                checkpoint = torch.load(path, weights_only=False)
                checkpoint["state_dict"]["unexpected.weight"] = torch.zeros(1)
                torch.save(checkpoint, path)
                with self.assertRaises(RuntimeError):
                    FoMoNMR.load_from_checkpoint(path, map_location="cpu")
        with self.assertRaisesRegex(ValueError, "similarity"):
            FoMoNMR(small_config(fingerprint_objective="bits"))

    def test_training_paths_and_teacher_center_validation(self):
        # Inspect loader wiring without reading or iterating any real dataset.
        for target, rich_name in (("morgan", "rich_shuffle"), ("unimol", "rich")):
            config = small_config(stage="posttrain", molecular_target=target,
                                  unimol_sidecar_dir="teachers")
            metadata = SimpleNamespace(schema_arrow=SimpleNamespace(
                metadata={b"unimol_center_sha256": b"same-center"}))
            with patch("model.train.pq.ParquetFile", return_value=metadata), patch(
                "model.train.PairedFoundationDataset"
            ) as dataset:
                make_dataloaders("posttrain", 4, 0, 42, config)
            args, kwargs = dataset.call_args_list[0]
            self.assertEqual(args[0], Path(f"datasets/train_splits/{rich_name}_train.parquet"))
            properties_root = "teachers" if target == "unimol" else "datasets/train_splits"
            self.assertEqual(args[1], Path(f"{properties_root}/{rich_name}_train_mol_properties.parquet"))
            self.assertEqual(kwargs["include_unimol"], target == "unimol")
        with patch("model.train.pq.ParquetFile") as parquet:
            parquet.return_value.schema_arrow.metadata = {}
            with self.assertRaisesRegex(ValueError, "training center"):
                make_dataloaders("posttrain", 4, 0, 42, config)
        self.assertEqual(
            [type(c) for c in validation_artifact_callbacks(small_config(), Path("unused"))],
            [FingerprintValidationLogger, ShiftValidationLogger],
        )

    def test_geometry_gradients_and_small_batches(self):
        torch.manual_seed(4)
        teacher = torch.randn(12, 8, requires_grad=True)
        identical, _ = relational_cosine_loss(teacher, teacher)
        self.assertLess(identical.item(), 1e-12)
        student = torch.randn(12, 5, requires_grad=True)
        loss, _ = relational_cosine_loss(student, teacher, max_pairs_per_bin=2)
        loss.backward()
        self.assertGreater(student.grad.abs().sum().item(), 0)
        self.assertIsNone(teacher.grad)
        collapsed, _ = relational_cosine_loss(torch.ones(12, 5), teacher)
        self.assertGreater(collapsed.item(), 0.1)
        for count in (0, 1):
            single = torch.randn(count, 5, requires_grad=True)
            zero, _ = relational_cosine_loss(single, teacher[:count])
            self.assertEqual(zero.item(), 0)
            zero.backward()

    def test_sampler_caps_bins_including_negative_cosines(self):
        cosine = torch.linspace(-1, 1, 1000)
        scaled = (cosine + 1) / 2
        selected = capped_similarity_indices(scaled, 0.05, 4)
        counts = torch.bincount((scaled[selected] / 0.05).long().clamp(max=20))
        self.assertLessEqual(counts.max().item(), 4)
        self.assertEqual(len(selected), len(selected.unique()))
        self.assertTrue((cosine[selected] < 0).any())

    def test_sidecar_mixer_and_processor_preserve_teacher_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            nmr, properties, teacher = write_fixture(Path(directory))
            rich = PairedFoundationDataset(nmr, teacher, include_unimol=True, shuffle=False, source_name="rich")
            auxiliary = PairedFoundationDataset(nmr, properties, shuffle=False, source_name="auxiliary")
            mixture = MixedFoundationDataset([rich, auxiliary], [0.9, 0.1], [False, True])
            records = list(islice(mixture, 100))
            batch = FoundationNMRProcessor(molecular_target="unimol")(records)
            self.assertTrue(batch["unimol_mask"].any())
            self.assertTrue((~batch["unimol_mask"]).any())
            self.assertTrue(torch.equal(batch["unimol_mask"], ~batch["shift_only"]))
            self.assertNotIn("fingerprints", batch)
            self.assertEqual(batch["unimol_embeddings"].shape, (100, 768))
            bad = replace(records[0], unimol_embedding=np.full(768, np.nan))
            with self.assertRaisesRegex(ValueError, "finite"):
                FoundationNMRProcessor(molecular_target="unimol")([bad])

    def test_only_rich_records_contribute_and_no_morgan_sampler(self):
        batch = example_batch()
        del batch["fingerprints"]
        batch["unimol_mask"] = torch.tensor([True, True, False])
        batch["unimol_embeddings"] = torch.randn(3, 768)
        model = FoMoNMR(small_config(stage="posttrain", molecular_target="unimol"))
        corrupted, corruption = model.corrupt_batch(batch)
        pooled, peaks, mask = model(corrupted)
        pooled.retain_grad()
        with patch("model.FoMoNMR.symmetric_fingerprint_pairs", side_effect=AssertionError("Morgan called")):
            losses = model.compute_losses((pooled, peaks, mask), batch, corruption)
            losses["fingerprint"].backward()
        self.assertGreater(pooled.grad[:2].abs().sum().item(), 0)
        self.assertEqual(pooled.grad[2].abs().sum().item(), 0)
        batch["unimol_mask"].zero_()
        loss = model.compute_losses((pooled, peaks, mask), batch, corruption)["fingerprint"]
        self.assertEqual(loss.item(), 0)

    def test_pretrain_rejected_callbacks_and_checkpoint_transfer(self):
        with self.assertRaisesRegex(ValueError, "posttrain"):
            FoMoNMR(small_config(molecular_target="unimol"))
        config = small_config(stage="posttrain", molecular_target="unimol")
        self.assertEqual(
            [type(c) for c in validation_artifact_callbacks(config, Path("unused"))],
            [UniMolValidationLogger, ShiftValidationLogger],
        )
        with tempfile.TemporaryDirectory() as directory:
            original = FoMoNMR(small_config())
            path = Path(directory) / "pretrained.ckpt"
            torch.save({"state_dict": original.state_dict()}, path)
            transferred = load_pretrained_model(path, config)
            self.assertFalse(hasattr(transferred, "fp_sim_classifier"))
            for name, value in transferred.state_dict().items():
                self.assertTrue(torch.equal(value, original.state_dict()[name]), name)
            broken = dict(original.state_dict())
            del broken["h_classification_head.weight"]
            torch.save({"state_dict": broken}, path)
            with self.assertRaises(RuntimeError):
                load_pretrained_model(path, config)

    def test_lightning_infinite_loader_with_mixed_teachers(self):
        with tempfile.TemporaryDirectory() as directory:
            nmr, properties, teacher = write_fixture(Path(directory))
            rich = PairedFoundationDataset(nmr, teacher, include_unimol=True, shuffle=False, source_name="rich")
            auxiliary = PairedFoundationDataset(nmr, properties, shuffle=False, source_name="auxiliary")
            mixed = MixedFoundationDataset([rich, auxiliary], [0.9, 0.1], [False, True])
            processor = FoundationNMRProcessor(molecular_target="unimol")
            loader = DataLoader(mixed, batch_size=4, collate_fn=processor)
            validation = [DataLoader(ds, batch_size=4, collate_fn=processor) for ds in (rich, auxiliary)]
            model = FoMoNMR(small_config(stage="posttrain", molecular_target="unimol", balanced_fp_pairs=True, warmup_steps=0))
            trainer = L.Trainer(accelerator="cpu", devices=1, max_steps=3, max_epochs=-1,
                                val_check_interval=2, check_val_every_n_epoch=None,
                                logger=False, enable_checkpointing=False, enable_progress_bar=False,
                                enable_model_summary=False, num_sanity_val_steps=0)
            trainer.fit(model, loader, validation)
            self.assertEqual(trainer.global_step, 3)
            self.assertIn("val/unimol_loss", trainer.callback_metrics)
            self.assertIn("val/unimol_cosine_mae", trainer.callback_metrics)
            self.assertIn("val/unimol_macro_cosine_mae", trainer.callback_metrics)
            # Auxiliary validation batches must not dilute teacher metrics.
            self.assertEqual(model.validation_metric_counts["fingerprint"], 6)
            rows = model.unimol_validation_rows()
            self.assertEqual(len(rows), 10)
            self.assertEqual(sum(row[1] for row in rows), 6)


if __name__ == "__main__":
    unittest.main()

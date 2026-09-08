"""Small fixtures for the experimental Rich-only UniMol objective."""

from dataclasses import replace
from itertools import islice
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import lightning as L
from torch.utils.data import DataLoader

from model import FoundationNMRProcessor, MixedFoundationDataset, PairedFoundationDataset
from model.FoMoNMR import FoMoNMR
from model.losses import capped_similarity_indices, relational_cosine_loss
from model.prepare_unimol_sidecars import training_center, write_sidecar
from model.train import load_pretrained_model, validation_artifact_callbacks, ShiftValidationLogger
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
        self.assertEqual([type(c) for c in validation_artifact_callbacks(config, Path("unused"))], [ShiftValidationLogger])
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

    def test_center_is_train_only_and_layout_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, properties, _ = write_fixture(root)
            source = pq.ParquetFile(properties)
            # Names expected by the experimental sidecar merger.
            import shutil
            for split in ("train", "val"):
                shutil.copyfile(properties, root / f"rich_{split}_mol_properties.parquet")
            paths = []
            for index in range(2):
                vectors = np.zeros((2, 768), np.float32)
                vectors[:, 0] = index * 2 + np.array([1, 2])
                table = pa.table({
                    "record_id": source.read_row_group(index).column("record_id"),
                    "unimol_embedding": pa.FixedSizeListArray.from_arrays(pa.array(vectors.ravel()), 768),
                })
                path = root / f"raw_{index}.parquet"
                pq.write_table(table, path)
                paths.append(path)
            center = training_center(paths)
            self.assertEqual(center[0], 2.5)
            output = root / "experimental"
            output.mkdir()
            for split in ("train", "val"):
                write_sidecar(root, output, split, paths, center, b"test")
                result = pq.ParquetFile(output / f"rich_{split}_mol_properties.parquet")
                self.assertEqual(result.num_row_groups, 2)
                self.assertEqual(result.metadata.num_rows, 4)
                values = result.read().column("unimol_embedding").to_pylist()
                self.assertEqual([row[0] for row in values], [-1, -1, 1, 1])

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
            # Auxiliary validation batches must not dilute teacher metrics.
            self.assertEqual(model.validation_metric_counts["fingerprint"], 6)


if __name__ == "__main__":
    unittest.main()

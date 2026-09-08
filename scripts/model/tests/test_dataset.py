"""Tests for aligned NMR and molecular-property training input."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as parquet
import torch
from torch.utils.data import DataLoader

from data import CanonicalRecord, CarbonPeak, ProtonPeak
from data.canonicalize.common import write_canonical_parquet
from data.postprocess.calculate_mol_properties import (
    MACCS_FIELD,
    MACCS_OUTPUT_BITS,
    MORGAN_FIELD,
    MORGAN_FP_BYTES,
    molecular_properties_schema,
)
from model import (
    FoundationNMRProcessor,
    MixedFoundationDataset,
    PairedFoundationDataset,
    PairedFoundationRecord,
)


class PairedFoundationDatasetTest(unittest.TestCase):
    def write_pair(
        self,
        root: Path,
        *,
        molecular_ids: list[str] | None = None,
        statuses: list[str] | None = None,
        num_records: int = 4,
        nmr_row_group_size: int = 2,
        molecular_row_group_size: int | None = None,
    ) -> tuple[Path, Path, list[bytes]]:
        records = [
            CanonicalRecord(
                record_id=f"record-{index}",
                source="fixture",
                h_nmr_peaks=(ProtonPeak(shift=1.0 + index),),
                c_nmr_peaks=(CarbonPeak(shift=10.0 + index),),
            )
            for index in range(num_records)
        ]
        nmr_path = root / "nmr.parquet"
        write_canonical_parquet(
            records,
            nmr_path,
            source_name="fixture",
            converter_name="test",
            row_group_size=nmr_row_group_size,
        )

        fingerprints = [
            bytes([1 << (index % 8)]) + bytes(MORGAN_FP_BYTES - 1)
            for index in range(num_records)
        ]
        maccs_fingerprints = [
            "0" * index + "1" + "0" * (MACCS_OUTPUT_BITS - index - 1)
            for index in range(num_records)
        ]
        ids = molecular_ids or [record.record_id for record in records]
        row_statuses = statuses or ["ok"] * len(records)
        rows = [
            {
                "record_id": record_id,
                "smiles_canonical": "C",
                "rdkit_status": status,
                MORGAN_FIELD: fingerprint if status == "ok" else None,
                MACCS_FIELD: maccs if status == "ok" else None,
            }
            for record_id, status, fingerprint, maccs in zip(
                ids,
                row_statuses,
                fingerprints,
                maccs_fingerprints,
            )
        ]
        molecular_path = root / "molecular.parquet"
        parquet.write_table(
            pa.Table.from_pylist(rows, schema=molecular_properties_schema()),
            molecular_path,
            row_group_size=molecular_row_group_size or nmr_row_group_size,
        )
        return nmr_path, molecular_path, fingerprints

    def test_dataset_and_processor_produce_training_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            nmr_path, molecular_path, fingerprints = self.write_pair(Path(directory))
            dataset = PairedFoundationDataset(
                nmr_path,
                molecular_path,
                arrow_batch_size=1,
                shuffle=False,
                source_name="custom_source",
            )
            loader = DataLoader(
                dataset,
                batch_size=4,
                collate_fn=FoundationNMRProcessor(),
            )
            batch = next(iter(loader))
            loaded_fingerprints = [
                sample.morgan_fingerprint for sample in dataset
            ]
        self.assertEqual(dataset.source_name, "custom_source")

        self.assertEqual(len(dataset), 4)
        self.assertEqual(
            batch["record_ids"],
            ["record-0", "record-1", "record-2", "record-3"],
        )
        self.assertEqual(batch["fingerprints"].shape, (4, 2048))
        self.assertEqual(batch["fingerprints"].dtype, torch.float32)
        self.assertEqual(batch["maccs_fingerprints"].shape, (4, 166))
        self.assertEqual(batch["maccs_fingerprints"].dtype, torch.float32)
        self.assertEqual(
            batch["fingerprints"][:, :8].tolist(),
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ],
        )
        self.assertEqual(loaded_fingerprints, fingerprints)
        self.assertEqual(
            batch["maccs_fingerprints"][:, :4].tolist(),
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        )

    def test_workers_receive_disjoint_row_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            nmr_path, molecular_path, _ = self.write_pair(Path(directory))
            dataset = PairedFoundationDataset(
                nmr_path,
                molecular_path,
                shuffle=False,
            )

            with patch(
                "model.dataset.get_worker_info",
                return_value=SimpleNamespace(id=0, num_workers=2),
            ):
                worker_zero = [sample.record.record_id for sample in dataset]
            with patch(
                "model.dataset.get_worker_info",
                return_value=SimpleNamespace(id=1, num_workers=2),
            ):
                worker_one = [sample.record.record_id for sample in dataset]

        self.assertEqual(worker_zero, ["record-0", "record-1"])
        self.assertEqual(worker_one, ["record-2", "record-3"])
        self.assertEqual(
            sorted(worker_zero + worker_one),
            [f"record-{index}" for index in range(4)],
        )

    def test_ranks_and_workers_receive_disjoint_interleaved_row_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            nmr_path, molecular_path, _ = self.write_pair(
                Path(directory),
                num_records=8,
                nmr_row_group_size=1,
            )
            dataset = PairedFoundationDataset(
                nmr_path,
                molecular_path,
                shuffle=False,
            )

            rank_records = []
            for rank in range(2):
                records = []
                for worker_id in range(2):
                    with patch(
                        "model.dataset._distributed_info",
                        return_value=(rank, 2),
                    ), patch(
                        "model.dataset.get_worker_info",
                        return_value=SimpleNamespace(id=worker_id, num_workers=2),
                    ):
                        records.extend(
                            sample.record.record_id for sample in dataset
                        )
                rank_records.append(records)

        self.assertEqual(sorted(rank_records[0]), [
            "record-0", "record-2", "record-4", "record-6"
        ])
        self.assertEqual(sorted(rank_records[1]), [
            "record-1", "record-3", "record-5", "record-7"
        ])
        self.assertEqual(
            sorted(rank_records[0] + rank_records[1]),
            [f"record-{index}" for index in range(8)],
        )

    def test_dataset_attaches_shift_only_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            nmr_path, molecular_path, _ = self.write_pair(Path(directory))
            dataset = PairedFoundationDataset(
                nmr_path,
                molecular_path,
                shuffle=False,
                shift_only=True,
            )
            batch = FoundationNMRProcessor()(list(dataset))

        self.assertTrue(batch["shift_only"].all().item())

    def test_shuffle_changes_order_between_iterations(self):
        with tempfile.TemporaryDirectory() as directory:
            nmr_path, molecular_path, _ = self.write_pair(Path(directory))
            first_dataset = PairedFoundationDataset(
                nmr_path,
                molecular_path,
                shuffle=True,
                shuffle_buffer_size=4,
                seed=7,
            )
            second_dataset = PairedFoundationDataset(
                nmr_path,
                molecular_path,
                shuffle=True,
                shuffle_buffer_size=4,
                seed=7,
            )

            first_order = [
                sample.record.record_id for sample in first_dataset
            ]
            repeated_order = [
                sample.record.record_id for sample in first_dataset
            ]
            reproduced_order = [
                sample.record.record_id for sample in second_dataset
            ]

        expected_ids = [f"record-{index}" for index in range(4)]
        self.assertEqual(sorted(first_order), expected_ids)
        self.assertEqual(sorted(repeated_order), expected_ids)
        self.assertEqual(first_order, reproduced_order)
        self.assertNotEqual(first_order, repeated_order)

    def test_rejects_invalid_shuffle_buffer_size(self):
        with tempfile.TemporaryDirectory() as directory:
            nmr_path, molecular_path, _ = self.write_pair(Path(directory))
            with self.assertRaisesRegex(ValueError, "shuffle_buffer_size"):
                PairedFoundationDataset(
                    nmr_path,
                    molecular_path,
                    shuffle_buffer_size=0,
                )

    def test_persistent_workers_shuffle_successive_epochs_without_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            nmr_path, molecular_path, _ = self.write_pair(
                Path(directory),
                num_records=16,
            )
            dataset = PairedFoundationDataset(
                nmr_path,
                molecular_path,
                shuffle=True,
                shuffle_buffer_size=8,
                seed=7,
            )
            loader = DataLoader(
                dataset,
                batch_size=None,
                num_workers=2,
                persistent_workers=True,
                generator=torch.Generator().manual_seed(123),
            )

            first_epoch = [
                sample.record.record_id for sample in loader
            ]
            second_epoch = [
                sample.record.record_id for sample in loader
            ]

        expected_ids = [f"record-{index}" for index in range(16)]
        self.assertEqual(sorted(first_epoch), sorted(expected_ids))
        self.assertEqual(sorted(second_epoch), sorted(expected_ids))
        self.assertNotEqual(first_epoch, second_epoch)

    def test_rejects_misaligned_ids_and_failed_fingerprints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nmr_path, molecular_path, _ = self.write_pair(
                root,
                molecular_ids=["record-0", "wrong", "record-2", "record-3"],
            )
            with self.assertRaisesRegex(ValueError, "record_id mismatch"):
                list(PairedFoundationDataset(nmr_path, molecular_path))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nmr_path, molecular_path, _ = self.write_pair(
                root,
                statuses=["ok", "invalid_smiles", "ok", "ok"],
            )
            with self.assertRaisesRegex(ValueError, "rdkit_status"):
                list(PairedFoundationDataset(nmr_path, molecular_path))

    def test_rejects_different_row_group_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            nmr_path, molecular_path, _ = self.write_pair(
                Path(directory),
                molecular_row_group_size=4,
            )
            with self.assertRaisesRegex(ValueError, "row-group counts"):
                PairedFoundationDataset(nmr_path, molecular_path)

    def test_unimol_is_explicitly_not_implemented(self):
        with tempfile.TemporaryDirectory() as directory:
            nmr_path, molecular_path, _ = self.write_pair(Path(directory))
            with self.assertRaises(NotImplementedError):
                PairedFoundationDataset(
                    nmr_path,
                    molecular_path,
                    include_unimol=True,
                )

    def test_processor_rejects_mixed_paired_and_unpaired_records(self):
        record = CanonicalRecord(
            record_id="record",
            h_nmr_peaks=(ProtonPeak(shift=1.0),),
        )
        paired = PairedFoundationRecord(
            record=record,
            morgan_fingerprint=bytes(MORGAN_FP_BYTES),
            maccs_fingerprint="0" * MACCS_OUTPUT_BITS,
        )
        with self.assertRaisesRegex(TypeError, "mix paired and unpaired"):
            FoundationNMRProcessor()([record, paired])

    def test_mixed_dataset_uses_primary_epoch_and_shift_only_flags(self):
        fingerprint = bytes(MORGAN_FP_BYTES)
        maccs_fingerprint = "0" * MACCS_OUTPUT_BITS
        primary = [
            PairedFoundationRecord(
                CanonicalRecord(
                    record_id=f"primary-{index}",
                    h_nmr_peaks=(ProtonPeak(1.0),),
                ),
                fingerprint,
                maccs_fingerprint,
            )
            for index in range(20)
        ]
        auxiliary = [
            PairedFoundationRecord(
                CanonicalRecord(
                    record_id=f"auxiliary-{index}",
                    h_nmr_peaks=(ProtonPeak(2.0),),
                ),
                fingerprint,
                maccs_fingerprint,
            )
            for index in range(2)
        ]
        dataset = MixedFoundationDataset(
            [primary, auxiliary],
            proportions=[0.5, 0.5],
            shift_only=[False, True],
            seed=5,
        )

        samples = list(dataset)
        primary_samples = [
            sample for sample in samples if sample.record.record_id.startswith("primary")
        ]
        auxiliary_samples = [
            sample for sample in samples if sample.record.record_id.startswith("auxiliary")
        ]

        self.assertEqual(len(primary_samples), 20)
        self.assertGreater(len(auxiliary_samples), 2)
        self.assertAlmostEqual(
            len(primary_samples) / len(samples),
            0.5,
            delta=0.15,
        )
        self.assertTrue(all(not sample.shift_only for sample in primary_samples))
        self.assertTrue(all(sample.shift_only for sample in auxiliary_samples))
        self.assertEqual(len(dataset), 40)

        batch = FoundationNMRProcessor()(samples[:8])
        expected_flags = [sample.shift_only for sample in samples[:8]]
        self.assertEqual(batch["shift_only"].dtype, torch.bool)
        self.assertEqual(batch["shift_only"].tolist(), expected_flags)


if __name__ == "__main__":
    unittest.main()

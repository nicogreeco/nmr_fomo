"""Tests for molecule-safe foundation-model train/validation splits."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
from pyarrow import parquet

from data import CanonicalRecord, ProtonPeak
from data.canonicalize.common import write_canonical_parquet
from data.postprocess.calculate_mol_properties import (
    MORGAN_FIELD,
    MORGAN_FP_BYTES,
    molecular_properties_schema,
)
from data.postprocess.split_foundation_datasets import run, smiles_hash


class SplitFoundationDatasetsTests(unittest.TestCase):
    def write_source(self, root, name, smiles_values):
        records = [
            CanonicalRecord(
                record_id=f"{name}-{index}",
                source=name,
                smiles=smiles,
                smiles_canonical=smiles,
                h_nmr_peaks=(ProtonPeak(1.0 + index),),
            )
            for index, smiles in enumerate(smiles_values)
        ]
        nmr_path = root / f"{name}.parquet"
        write_canonical_parquet(
            records,
            nmr_path,
            source_name=name,
            converter_name="test",
            row_group_size=2,
        )

        rows = [
            {
                "record_id": record.record_id,
                "smiles_canonical": record.smiles_canonical,
                "rdkit_status": "ok",
                MORGAN_FIELD: bytes(MORGAN_FP_BYTES),
            }
            for record in records
        ]
        molecular_path = root / f"{name}_mol_properties.parquet"
        parquet.write_table(
            pa.Table.from_pylist(rows, schema=molecular_properties_schema()),
            molecular_path,
            row_group_size=2,
        )

    def run_split(self, cleaned_root, output_root):
        run(
            SimpleNamespace(
                cleaned_root=str(cleaned_root),
                output_root=str(output_root),
                seed=7,
                simnmr_validation_records=1,
                rich_validation_records=1,
                nmrgym_validation_records=1,
                batch_size=2,
                report_output=None,
                overwrite=False,
                no_progress=True,
            )
        )

    def read_smiles(self, path):
        values = parquet.read_table(path, columns=["smiles_canonical"])[0]
        return set(values.to_pylist())

    def test_split_is_deterministic_aligned_and_cross_source_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cleaned = root / "cleaned"
            cleaned.mkdir()

            candidates = ["C", "CC", "CCC", "CCCC", "N", "O", "CO", "CN"]
            ordered = sorted(candidates, key=lambda value: smiles_hash(value, 7))
            shared = ordered[0]
            high = ordered[-6:]

            self.write_source(cleaned, "simnmr", [shared, high[0], high[1]])
            self.write_source(cleaned, "train_val", [shared, high[2], high[3]])
            self.write_source(cleaned, "nmrgym", [high[4], high[5], high[3]])

            first_output = root / "first"
            second_output = root / "second"
            self.run_split(cleaned, first_output)
            self.run_split(cleaned, second_output)

            all_train_smiles = set()
            all_val_smiles = set()
            for label in ("simnmr", "rich", "nmrgym"):
                train_nmr = first_output / f"{label}_train.parquet"
                train_molecular = (
                    first_output / f"{label}_train_mol_properties.parquet"
                )
                val_nmr = first_output / f"{label}_val.parquet"
                val_molecular = first_output / f"{label}_val_mol_properties.parquet"

                train_ids = parquet.read_table(train_nmr, columns=["record_id"])[0]
                train_molecular_ids = parquet.read_table(
                    train_molecular,
                    columns=["record_id"],
                )[0]
                val_ids = parquet.read_table(val_nmr, columns=["record_id"])[0]
                val_molecular_ids = parquet.read_table(
                    val_molecular,
                    columns=["record_id"],
                )[0]

                self.assertEqual(
                    train_ids.to_pylist(),
                    train_molecular_ids.to_pylist(),
                )
                self.assertEqual(val_ids.to_pylist(), val_molecular_ids.to_pylist())
                self.assertEqual(
                    parquet.ParquetFile(train_nmr).num_row_groups,
                    parquet.ParquetFile(train_molecular).num_row_groups,
                )
                self.assertEqual(
                    parquet.ParquetFile(val_nmr).num_row_groups,
                    parquet.ParquetFile(val_molecular).num_row_groups,
                )

                train_smiles = self.read_smiles(train_nmr)
                val_smiles = self.read_smiles(val_nmr)
                self.assertFalse(train_smiles & val_smiles)
                self.assertTrue(val_smiles)
                all_train_smiles.update(train_smiles)
                all_val_smiles.update(val_smiles)
                self.assertEqual(
                    train_smiles,
                    self.read_smiles(second_output / f"{label}_train.parquet"),
                )
                self.assertEqual(
                    val_smiles,
                    self.read_smiles(second_output / f"{label}_val.parquet"),
                )

            self.assertFalse(all_train_smiles & all_val_smiles)
            self.assertIn(shared, self.read_smiles(first_output / "simnmr_val.parquet"))
            self.assertIn(shared, self.read_smiles(first_output / "rich_val.parquet"))


if __name__ == "__main__":
    unittest.main()

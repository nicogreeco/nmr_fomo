"""Tests for the fixed molecule-safe MACCS probe split."""

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
    MACCS_FIELD,
    MACCS_OUTPUT_BITS,
    MORGAN_FIELD,
    MORGAN_FP_BYTES,
    molecular_properties_schema,
)
from data.postprocess.split_maccs_probe import run


class SplitMaccsProbeTests(unittest.TestCase):
    def write_inputs(self, root):
        records = []
        for source, prefix in (("mst", "C"), ("exp", "N"), ("trans", "O")):
            for index in range(10):
                smiles = prefix + "C" * index
                records.append(
                    CanonicalRecord(
                        record_id=f"{source}-{index}",
                        source=source,
                        smiles=smiles,
                        smiles_canonical=smiles,
                        h_nmr_peaks=(ProtonPeak(1.0),),
                    )
                )

        nmr_path = root / "rich_val.parquet"
        write_canonical_parquet(
            records,
            nmr_path,
            source_name="rich",
            converter_name="test",
            row_group_size=7,
        )
        molecular_rows = [
            {
                "record_id": record.record_id,
                "smiles_canonical": record.smiles_canonical,
                "rdkit_status": "ok",
                MORGAN_FIELD: bytes(MORGAN_FP_BYTES),
                MACCS_FIELD: "0" * MACCS_OUTPUT_BITS,
            }
            for record in records
        ]
        molecular_path = root / "rich_val_mol_properties.parquet"
        parquet.write_table(
            pa.Table.from_pylist(
                molecular_rows,
                schema=molecular_properties_schema(),
            ),
            molecular_path,
            row_group_size=7,
        )
        return nmr_path, molecular_path

    def run_split(self, nmr_path, molecular_path, output_root):
        run(
            SimpleNamespace(
                nmr_input=str(nmr_path),
                molecular_input=str(molecular_path),
                output_root=str(output_root),
                train_records=12,
                eval_records=6,
                seed=7,
                batch_size=7,
                report_output=None,
                overwrite=False,
                no_progress=True,
            )
        )

    def test_split_is_deterministic_aligned_and_stratified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nmr_path, molecular_path = self.write_inputs(root)
            first = root / "first"
            second = root / "second"
            self.run_split(nmr_path, molecular_path, first)
            self.run_split(nmr_path, molecular_path, second)

            split_smiles = {}
            for split, expected_records in (("train", 12), ("eval", 6)):
                nmr = parquet.read_table(first / f"{split}.parquet")
                molecular = parquet.read_table(
                    first / f"{split}_mol_properties.parquet"
                )
                repeated = parquet.read_table(second / f"{split}.parquet")

                self.assertEqual(nmr.num_rows, expected_records)
                self.assertEqual(
                    nmr["record_id"].to_pylist(),
                    molecular["record_id"].to_pylist(),
                )
                self.assertEqual(
                    nmr["record_id"].to_pylist(),
                    repeated["record_id"].to_pylist(),
                )
                self.assertEqual(
                    {source: nmr["source"].to_pylist().count(source)
                     for source in ("mst", "exp", "trans")},
                    {source: expected_records // 3
                     for source in ("mst", "exp", "trans")},
                )
                split_smiles[split] = set(nmr["smiles_canonical"].to_pylist())

            self.assertFalse(split_smiles["train"] & split_smiles["eval"])


if __name__ == "__main__":
    unittest.main()

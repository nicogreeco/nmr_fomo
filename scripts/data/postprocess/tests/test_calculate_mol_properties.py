from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.parquet as parquet
    from rdkit import Chem
    from data.canonicalize.common import (
        CANONICAL_PARQUET_SCHEMA_VERSION,
        canonical_parquet_schema,
    )
    from data.postprocess.calculate_mol_properties import (
        CSV_FIELDS,
        MACCS_OUTPUT_BITS,
        MORGAN_FP_SIZE,
        default_output_path,
        export_molecular_properties,
    )
except ModuleNotFoundError:
    pa = None
    parquet = None
    Chem = None


@unittest.skipUnless(pa is not None and Chem is not None, "requires pyarrow and RDKit")
class CalculateMolecularPropertiesTest(unittest.TestCase):
    def write_fixture(self, path: Path) -> None:
        record_ids = [
            "ethanol",
            "heteroaromatic",
            "amide",
            "invalid",
            "missing",
        ]
        smiles_values = [
            "CCO",
            "c1ccncc1",
            "CC(=O)N",
            "not-a-smiles",
            None,
        ]
        table = pa.Table.from_pydict(
            {
                "record_id": record_ids,
                "source": ["fixture"] * len(record_ids),
                "smiles": smiles_values,
                "smiles_canonical": smiles_values,
                "molecular_formula": [None] * len(record_ids),
                "nmr_frequency": [None] * len(record_ids),
                "nmr_solvent": [None] * len(record_ids),
                "atoms": [None] * len(record_ids),
                "h_nmr_peaks": [[] for _ in record_ids],
                "c_nmr_peaks": [[] for _ in record_ids],
            },
            schema=canonical_parquet_schema().with_metadata(
                {
                    b"canonical_schema_version": (
                        CANONICAL_PARQUET_SCHEMA_VERSION.encode("utf-8")
                    )
                }
            ),
        )
        parquet.write_table(table, path)

    def read_rows(self, path: Path) -> list[dict[str, str]]:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def test_streamed_export_has_expected_columns_and_values(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "fixture.parquet"
            self.write_fixture(input_path)

            result = export_molecular_properties(
                input_path,
                batch_size=2,
                records_per_task=1,
                workers=2,
                progress_every=100,
                show_progress=False,
            )
            output_path = default_output_path(input_path)
            rows = self.read_rows(output_path)

        self.assertEqual(result["counts"]["output_records"], 5)
        self.assertEqual(
            result["details"]["rdkit_status"],
            {"invalid_smiles": 1, "missing_smiles": 1, "ok": 3},
        )
        self.assertEqual(
            [row["record_id"] for row in rows],
            ["ethanol", "heteroaromatic", "amide", "invalid", "missing"],
        )
        self.assertEqual(set(rows[0]), set(CSV_FIELDS))

        ethanol = rows[0]
        self.assertEqual(ethanol["rdkit_status"], "ok")
        self.assertAlmostEqual(float(ethanol["exact_molecular_weight"]), 46.041864812)
        self.assertEqual(ethanol["has_alcohol_or_phenol"], "1")
        self.assertEqual(len(ethanol["morgan_ecfp4_2048_hex"]), MORGAN_FP_SIZE // 4)
        self.assertEqual(len(ethanol["maccs_keys_166_bits"]), MACCS_OUTPUT_BITS)

        heteroaromatic = rows[1]
        self.assertEqual(heteroaromatic["has_heteroaromatic_ring"], "1")
        self.assertAlmostEqual(float(heteroaromatic["aromatic_atom_fraction"]), 1.0)

        amide = rows[2]
        self.assertEqual(amide["has_amide"], "1")
        self.assertEqual(amide["has_amine"], "0")

        self.assertEqual(rows[3]["rdkit_status"], "invalid_smiles")
        self.assertEqual(rows[4]["rdkit_status"], "missing_smiles")
        self.assertEqual(rows[3]["tpsa"], "")

    def test_parallel_output_is_byte_identical_to_sequential_output(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "fixture.parquet"
            sequential_path = Path(directory) / "sequential.csv"
            parallel_path = Path(directory) / "parallel.csv"
            self.write_fixture(input_path)

            export_molecular_properties(
                input_path,
                sequential_path,
                batch_size=2,
                records_per_task=1,
                workers=1,
                show_progress=False,
            )
            export_molecular_properties(
                input_path,
                parallel_path,
                batch_size=2,
                records_per_task=1,
                workers=2,
                show_progress=False,
            )

            self.assertEqual(sequential_path.read_bytes(), parallel_path.read_bytes())

    def test_existing_output_requires_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "fixture.parquet"
            self.write_fixture(input_path)
            output_path = default_output_path(input_path)
            output_path.write_text("existing output\n", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                export_molecular_properties(input_path, workers=1)

            with self.assertRaises(ValueError):
                export_molecular_properties(
                    input_path,
                    output_path=Path(directory) / "not-a-csv.parquet",
                    workers=1,
                )


if __name__ == "__main__":
    unittest.main()

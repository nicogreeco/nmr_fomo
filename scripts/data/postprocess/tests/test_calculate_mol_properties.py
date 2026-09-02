from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.parquet as parquet
    from rdkit import Chem, DataStructs

    from data.canonicalize.common import (
        CANONICAL_PARQUET_SCHEMA_VERSION,
        canonical_parquet_schema,
    )
    from data.postprocess.calculate_mol_properties import (
        FUNCTIONAL_GROUP_SMARTS,
        MACCS_OUTPUT_BITS,
        MOLECULAR_PROPERTY_FIELDS,
        MORGAN_FIELD,
        MORGAN_FP_BYTES,
        MORGAN_FP_SIZE,
        default_output_path,
        export_molecular_properties,
        molecular_properties_schema,
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
        parquet.write_table(table, path, row_group_size=2)

    def test_export_has_expected_schema_values_and_row_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "fixture.parquet"
            self.write_fixture(input_path)

            result = export_molecular_properties(
                input_path,
                records_per_task=1,
                workers=2,
                progress_every=100,
                show_progress=False,
            )
            output_path = default_output_path(input_path)
            output_file = parquet.ParquetFile(output_path)
            table = output_file.read()
            rows = table.to_pylist()

        self.assertEqual(result["counts"]["output_records"], 5)
        self.assertEqual(result["counts"]["row_groups"], 3)
        self.assertEqual(
            result["details"]["rdkit_status"],
            {"invalid_smiles": 1, "missing_smiles": 1, "ok": 3},
        )
        self.assertTrue(
            output_file.schema_arrow.equals(
                molecular_properties_schema(),
                check_metadata=True,
            )
        )
        self.assertEqual(
            [
                output_file.metadata.row_group(index).num_rows
                for index in range(output_file.num_row_groups)
            ],
            [2, 2, 1],
        )
        self.assertEqual(
            [row["record_id"] for row in rows],
            ["ethanol", "heteroaromatic", "amide", "invalid", "missing"],
        )
        self.assertEqual(list(rows[0]), MOLECULAR_PROPERTY_FIELDS)

        ethanol = rows[0]
        self.assertEqual(ethanol["rdkit_status"], "ok")
        self.assertAlmostEqual(ethanol["exact_molecular_weight"], 46.041864812)
        self.assertEqual(ethanol["has_alcohol_or_phenol"], 1)
        self.assertEqual(len(ethanol[MORGAN_FIELD]), MORGAN_FP_BYTES)
        restored = DataStructs.CreateFromBinaryText(ethanol[MORGAN_FIELD])
        self.assertEqual(restored.GetNumBits(), MORGAN_FP_SIZE)
        self.assertEqual(
            DataStructs.BitVectToBinaryText(restored),
            ethanol[MORGAN_FIELD],
        )
        self.assertEqual(len(ethanol["maccs_keys_166_bits"]), MACCS_OUTPUT_BITS)

        self.assertEqual(rows[1]["has_heteroaromatic_ring"], 1)
        self.assertAlmostEqual(rows[1]["aromatic_atom_fraction"], 1.0)
        self.assertEqual(rows[2]["has_amide"], 1)
        self.assertEqual(rows[2]["has_amine"], 0)

        self.assertEqual(rows[3]["rdkit_status"], "invalid_smiles")
        self.assertEqual(rows[4]["rdkit_status"], "missing_smiles")
        self.assertIsNone(rows[3]["tpsa"])
        self.assertIsNone(rows[3][MORGAN_FIELD])
        self.assertTrue(all(name in rows[0] for name in FUNCTIONAL_GROUP_SMARTS))

    def test_parallel_and_sequential_outputs_are_semantically_equal(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "fixture.parquet"
            sequential_path = Path(directory) / "sequential.parquet"
            parallel_path = Path(directory) / "parallel.parquet"
            self.write_fixture(input_path)

            export_molecular_properties(
                input_path,
                sequential_path,
                records_per_task=1,
                workers=1,
                show_progress=False,
            )
            export_molecular_properties(
                input_path,
                parallel_path,
                records_per_task=1,
                workers=2,
                show_progress=False,
            )

            sequential = parquet.read_table(sequential_path)
            parallel = parquet.read_table(parallel_path)

        self.assertTrue(sequential.equals(parallel, check_metadata=True))

    def test_existing_output_requires_overwrite_and_parquet_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "fixture.parquet"
            self.write_fixture(input_path)
            output_path = default_output_path(input_path)
            output_path.write_bytes(b"existing output")

            with self.assertRaises(FileExistsError):
                export_molecular_properties(input_path, workers=1)

            with self.assertRaises(ValueError):
                export_molecular_properties(
                    input_path,
                    output_path=Path(directory) / "not-parquet.csv",
                    workers=1,
                )


if __name__ == "__main__":
    unittest.main()

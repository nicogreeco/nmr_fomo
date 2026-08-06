from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import pyarrow as pa
    import rdkit
    from pyarrow import parquet

    from data.canonicalize.common import (
        CANONICAL_PARQUET_SCHEMA_VERSION,
        canonical_parquet_schema,
    )
    from data.postprocess.disjoin_benchmark_from_train import disjoint_datasets
except ModuleNotFoundError:
    pa = None
    parquet = None
    rdkit = None


def record(record_id: str, smiles: str) -> dict[str, object]:
    return {
        "record_id": record_id,
        "source": "fixture",
        "smiles": smiles,
        "smiles_canonical": smiles,
        "molecular_formula": None,
        "nmr_frequency": None,
        "nmr_solvent": None,
        "atoms": None,
        "h_nmr_peaks": None,
        "c_nmr_peaks": None,
    }


@unittest.skipUnless(pa is not None and rdkit is not None, "requires PyArrow and RDKit")
class DisjoinBenchmarkTest(unittest.TestCase):
    def write_fixture(
        self,
        path: Path,
        records: list[dict[str, object]],
        source_name: str,
    ) -> None:
        schema = canonical_parquet_schema().with_metadata(
            {
                b"canonical_schema_version": (
                    CANONICAL_PARQUET_SCHEMA_VERSION.encode("utf-8")
                ),
                b"rdkit_version": rdkit.__version__.encode("utf-8"),
                b"source_dataset": source_name.encode("utf-8"),
                b"custom_metadata": b"preserved",
            }
        )
        parquet.write_table(pa.Table.from_pylist(records, schema=schema), path)

    def test_output_preserves_base_metadata_and_adds_disjoin_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_path = root / "base.parquet"
            comparison_path = root / "comparison.parquet"
            output_path = root / "disjoint.parquet"
            self.write_fixture(
                base_path,
                [record("remove", "CC"), record("keep", "CCO")],
                "base",
            )
            self.write_fixture(
                comparison_path,
                [record("comparison", "CC")],
                "comparison",
            )

            result = disjoint_datasets(
                [base_path, comparison_path],
                output_path,
                arrow_batch_size=1,
            )
            output = parquet.read_table(output_path)
            metadata = output.schema.metadata or {}

        self.assertEqual(result["rows"], 2)
        self.assertEqual(result["intersection_rows"], 1)
        self.assertEqual(output["record_id"].to_pylist(), ["keep"])
        self.assertTrue(
            output.schema.remove_metadata().equals(canonical_parquet_schema())
        )
        self.assertEqual(metadata[b"canonical_schema_version"], b"2")
        self.assertEqual(metadata[b"rdkit_version"], rdkit.__version__.encode())
        self.assertEqual(metadata[b"source_dataset"], b"base")
        self.assertEqual(metadata[b"custom_metadata"], b"preserved")
        self.assertEqual(metadata[b"disjointed_from"], str(base_path).encode())
        self.assertEqual(
            json.loads(metadata[b"disjointed_against"]),
            [str(comparison_path)],
        )
        self.assertEqual(metadata[b"disjoin_identity"], b"RDKit connectivity InChIKey")
        self.assertEqual(metadata[b"disjoin_removed_record_count"], b"1")
        self.assertFalse(output_path.with_name(f".{output_path.name}.partial").exists())


if __name__ == "__main__":
    unittest.main()

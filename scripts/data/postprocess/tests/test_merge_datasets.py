from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import pyarrow as pa
    from pyarrow import parquet

    from data.canonicalize.common import (
        CANONICAL_PARQUET_SCHEMA_VERSION,
        canonical_parquet_schema,
    )
    from data.postprocess.merge_datasets import merge_datasets
except ModuleNotFoundError:
    pa = None
    parquet = None


def record(record_id: str) -> dict[str, object]:
    return {
        "record_id": record_id,
        "source": "fixture",
        "smiles": "CC",
        "smiles_canonical": "CC",
        "molecular_formula": None,
        "nmr_frequency": None,
        "nmr_solvent": None,
        "atoms": None,
        "h_nmr_peaks": [],
        "c_nmr_peaks": [],
    }


@unittest.skipUnless(pa is not None, "requires PyArrow")
class MergeDatasetsTest(unittest.TestCase):
    def write_fixture(self, path: Path, record_ids: list[str]) -> None:
        schema = canonical_parquet_schema().with_metadata(
            {
                b"canonical_schema_version": (
                    CANONICAL_PARQUET_SCHEMA_VERSION.encode("utf-8")
                ),
                b"rdkit_version": b"fixture",
            }
        )
        parquet.write_table(
            pa.Table.from_pylist(
                [record(value) for value in record_ids], schema=schema
            ),
            path,
        )

    def test_preserves_input_order_and_row_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.parquet"
            second = root / "second.parquet"
            output = root / "merged.parquet"
            self.write_fixture(first, ["a", "b"])
            self.write_fixture(second, ["c"])

            result = merge_datasets(
                [first, second],
                output,
                batch_size=1,
                show_progress=False,
            )
            output_ids = parquet.read_table(output)["record_id"].to_pylist()

        self.assertEqual(output_ids, ["a", "b", "c"])
        self.assertEqual(result["stage"], "merge_datasets")
        self.assertEqual(result["counts"]["output_records"], 3)


if __name__ == "__main__":
    unittest.main()

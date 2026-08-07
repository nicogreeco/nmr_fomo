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
    from data.postprocess.disjoin_benchmark_from_train import disjoint_datasets
except ModuleNotFoundError:
    pa = None
    parquet = None


def record(record_id: str, smiles: str | None) -> dict[str, object]:
    return {
        "record_id": record_id,
        "source": "fixture",
        "smiles": smiles,
        "smiles_canonical": smiles,
        "molecular_formula": None,
        "nmr_frequency": None,
        "nmr_solvent": None,
        "atoms": None,
        "h_nmr_peaks": [],
        "c_nmr_peaks": [],
    }


@unittest.skipUnless(pa is not None, "requires PyArrow")
class DisjoinBenchmarkTest(unittest.TestCase):
    def write_fixture(
        self,
        path: Path,
        records: list[dict[str, object]],
    ) -> None:
        schema = canonical_parquet_schema().with_metadata(
            {
                b"canonical_schema_version": (
                    CANONICAL_PARQUET_SCHEMA_VERSION.encode("utf-8")
                ),
                b"rdkit_version": b"fixture",
            }
        )
        parquet.write_table(pa.Table.from_pylist(records, schema=schema), path)

    def test_streams_comparison_files_and_removes_matches_from_first(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            first = directory_path / "first.parquet"
            second = directory_path / "second.parquet"
            third = directory_path / "third.parquet"
            output = directory_path / "output.parquet"

            self.write_fixture(
                first,
                [
                    record("ethane-a", "CC"),
                    record("ethane-b", "CC"),
                    record("propane", "CCC"),
                    record("nitrogen", "N"),
                    record("missing", None),
                ],
            )
            self.write_fixture(second, [record("other-ethane", "CC")])
            self.write_fixture(third, [record("other-propane", "CCC")])

            result = disjoint_datasets(
                [first, second, third],
                output,
                arrow_batch_size=2,
                show_progress=False,
            )
            output_file = parquet.ParquetFile(output)
            output_ids = parquet.read_table(output)["record_id"].to_pylist()

        self.assertEqual(output_ids, ["nitrogen", "missing"])
        self.assertEqual(result["rows"], 5)
        self.assertEqual(result["intersection_rows"], 3)
        self.assertEqual(
            output_file.schema_arrow.metadata[b"disjoin_identity"],
            b"RDKit connectivity InChIKey",
        )


if __name__ == "__main__":
    unittest.main()

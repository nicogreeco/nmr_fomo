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
    from data.postprocess.create_double_disjoint_extended import (
        create_double_disjoint_extended,
    )
except ModuleNotFoundError:
    pa = None
    parquet = None


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
        "h_nmr_peaks": [],
        "c_nmr_peaks": [],
    }


@unittest.skipUnless(pa is not None, "requires PyArrow")
class CreateDoubleDisjointExtendedTest(unittest.TestCase):
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

    def test_moves_only_nmrsolver_only_benchmark_records_to_extended_train(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = root / "benchmark.parquet"
            rich_reference = root / "rich.parquet"
            nmrsolver = root / "nmrsolver.parquet"
            base_train = root / "base_train.parquet"
            test_output = root / "test_double.parquet"
            train_output = root / "train_extended.parquet"

            self.write_fixture(
                benchmark,
                [
                    record("rich-overlap", "CC"),
                    record("solver-a", "CCC"),
                    record("solver-b", "CCC"),
                    record("test-only", "N"),
                ],
            )
            self.write_fixture(rich_reference, [record("rich", "CC")])
            self.write_fixture(nmrsolver, [record("solver", "CCC")])
            self.write_fixture(base_train, [record("base", "O")])

            result = create_double_disjoint_extended(
                benchmark,
                rich_reference,
                nmrsolver,
                base_train,
                test_output,
                train_output,
                scan_batch_size=2,
                write_batch_size=2,
                show_progress=False,
            )
            test_ids = parquet.read_table(test_output)["record_id"].to_pylist()
            train_ids = parquet.read_table(train_output)["record_id"].to_pylist()
            test_metadata = parquet.ParquetFile(test_output).schema_arrow.metadata
            train_metadata = parquet.ParquetFile(train_output).schema_arrow.metadata

        self.assertEqual(test_ids, ["test-only"])
        self.assertEqual(train_ids, ["base", "solver-a", "solver-b"])
        self.assertEqual(result["rich_overlap_records"], 1)
        self.assertEqual(result["nmrsolver_only_overlap_records"], 2)
        self.assertEqual(result["test_rows"], 1)
        self.assertEqual(result["extended_train_rows"], 3)
        self.assertEqual(test_metadata[b"disjoin_identity"], b"exact canonical SMILES")
        self.assertEqual(train_metadata[b"extension_record_count"], b"2")


if __name__ == "__main__":
    unittest.main()

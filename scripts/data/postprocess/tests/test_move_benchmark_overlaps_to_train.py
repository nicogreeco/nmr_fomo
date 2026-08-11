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
    from data.postprocess.move_benchmark_overlaps_to_train import (
        move_benchmark_overlaps_to_train,
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
class MoveBenchmarkOverlapsToTrainTest(unittest.TestCase):
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

    def test_moves_every_matching_benchmark_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = root / "benchmark.parquet"
            reference = root / "simnmr.parquet"
            train = root / "train.parquet"
            test_output = root / "test.parquet"
            train_output = root / "train_extended.parquet"
            self.write_fixture(
                benchmark,
                [
                    record("move-a", "CCC"),
                    record("move-b", "CCC"),
                    record("keep", "N"),
                ],
            )
            self.write_fixture(reference, [record("simulated", "CCC")])
            self.write_fixture(train, [record("base", "O")])

            result = move_benchmark_overlaps_to_train(
                benchmark,
                reference,
                train,
                test_output,
                train_output,
                batch_size=2,
                show_progress=False,
            )
            test_ids = parquet.read_table(test_output)["record_id"].to_pylist()
            train_ids = parquet.read_table(train_output)["record_id"].to_pylist()
            train_metadata = parquet.ParquetFile(
                train_output
            ).schema_arrow.metadata

        self.assertEqual(test_ids, ["keep"])
        self.assertEqual(train_ids, ["base", "move-a", "move-b"])
        self.assertEqual(result["appended_records"], 2)
        self.assertEqual(result["moved_molecules"], 1)
        self.assertEqual(train_metadata[b"moved_record_count"], b"2")


if __name__ == "__main__":
    unittest.main()

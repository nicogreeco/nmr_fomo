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
    from data.postprocess.remove_benchmark_overlaps import (
        remove_benchmark_overlaps,
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
class RemoveBenchmarkOverlapsTest(unittest.TestCase):
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

    def test_removes_exact_smiles_found_in_multiple_comparisons(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = root / "benchmark.parquet"
            train = root / "train.parquet"
            nmrgym = root / "nmrgym.parquet"
            output = root / "output.parquet"
            self.write_fixture(
                benchmark,
                [
                    record("ethane-a", "CC"),
                    record("ethane-b", "CC"),
                    record("propane", "CCC"),
                    record("stereo-a", "F[C@H](Cl)Br"),
                    record("keep", "N"),
                ],
            )
            self.write_fixture(train, [record("train-ethane", "CC")])
            self.write_fixture(
                nmrgym,
                [
                    record("gym-propane", "CCC"),
                    record("stereo-b", "F[C@@H](Cl)Br"),
                ],
            )

            result = remove_benchmark_overlaps(
                benchmark,
                [train, nmrgym],
                output,
                batch_size=2,
                show_progress=False,
            )
            output_file = parquet.ParquetFile(output)
            output_ids = parquet.read_table(output)["record_id"].to_pylist()

        self.assertEqual(output_ids, ["stereo-a", "keep"])
        self.assertEqual(result["removed_records"], 3)
        self.assertEqual(result["removed_molecules"], 2)
        self.assertEqual(
            output_file.schema_arrow.metadata[b"molecule_identity"],
            b"exact smiles_canonical equality",
        )


if __name__ == "__main__":
    unittest.main()

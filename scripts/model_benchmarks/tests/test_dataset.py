"""Smoke tests for the reusable, model-independent datasets."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

from data import CanonicalNMRDataset, CanonicalParquetDataset, CanonicalRecord


def example_record():
    return {
        "record_id": "example-1",
        "source": "hand-written",
        "h_nmr_peaks": [{"shift": 1.2, "j_values": None}],
        "c_nmr_peaks": [],
    }


class DatasetSmokeTests(unittest.TestCase):
    def test_in_memory_dataset_returns_canonical_records(self):
        dataset = CanonicalNMRDataset([example_record()])

        self.assertEqual(len(dataset), 1)
        self.assertIsInstance(dataset[0], CanonicalRecord)
        self.assertEqual(dataset[0].record_id, "example-1")
        self.assertIsNone(dataset[0].h_nmr_peaks[0].j_values)
        self.assertEqual(dataset[0].c_nmr_peaks, ())

    @unittest.skipUnless(
        importlib.util.find_spec("pyarrow") is not None,
        "pyarrow is not installed in this environment",
    )
    def test_parquet_dataset_reads_a_temporary_canonical_file(self):
        import pyarrow as arrow
        import pyarrow.parquet as parquet

        with tempfile.TemporaryDirectory() as temporary_directory:
            parquet_path = Path(temporary_directory) / "records.parquet"
            table = arrow.Table.from_pylist([example_record()])
            parquet.write_table(table, parquet_path)

            records = list(CanonicalParquetDataset(parquet_path, arrow_batch_size=1))

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].record_id, "example-1")
        self.assertIsNone(records[0].h_nmr_peaks[0].j_values)

    @unittest.skipUnless(
        importlib.util.find_spec("pyarrow") is not None
        and importlib.util.find_spec("torch") is not None,
        "pyarrow or torch is not installed in this environment",
    )
    def test_embedding_batches_are_written_as_separate_row_groups(self):
        import pyarrow as arrow
        import pyarrow.parquet as parquet
        import torch

        from model_benchmarks.extract_embeddings import _write_embedding_batch

        schema = arrow.schema(
            [
                arrow.field("record_id", arrow.string(), nullable=False),
                arrow.field(
                    "embedding", arrow.list_(arrow.float32(), 2), nullable=False
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "embeddings.parquet"
            writer = parquet.ParquetWriter(output_path, schema)
            try:
                _write_embedding_batch(
                    writer, schema, arrow, torch.tensor([[1.0, 2.0]]), ["a"]
                )
                _write_embedding_batch(
                    writer, schema, arrow, torch.tensor([[3.0, 4.0]]), ["b"]
                )
                writer.add_key_value_metadata({"test_metadata": "present"})
            finally:
                writer.close()

            parquet_file = parquet.ParquetFile(output_path)
            table = parquet_file.read()

        self.assertEqual(parquet_file.metadata.num_row_groups, 2)
        self.assertEqual(table["record_id"].to_pylist(), ["a", "b"])
        self.assertEqual(table["embedding"].to_pylist(), [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(
            parquet_file.metadata.metadata[b"test_metadata"], b"present"
        )


if __name__ == "__main__":
    unittest.main()

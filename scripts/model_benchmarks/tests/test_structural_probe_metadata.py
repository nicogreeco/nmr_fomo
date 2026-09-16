"""Regression coverage for metadata in streaming embedding caches."""

import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as parquet

from model_benchmarks.run_structural_information_probes import (
    embedding_ids,
    embedding_metadata,
    read_embedding,
)


class StructuralProbeMetadataTests(unittest.TestCase):
    def test_footer_metadata_is_available_to_both_cache_readers(self):
        metadata = {"model_name": "nmrpeak", "dimension": 2, "pooling": "bos"}
        table = pa.table({
            "record_id": ["fixture"],
            "embedding": pa.array([[1.0, 2.0]], type=pa.list_(pa.float32(), 2)),
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings.parquet"
            with parquet.ParquetWriter(path, table.schema) as writer:
                writer.write_table(table)
                writer.add_key_value_metadata({
                    "nmr_embedding_metadata": json.dumps(metadata),
                })
            # Reproduce the extractor's late footer update and the old failure.
            self.assertNotIn(
                b"nmr_embedding_metadata",
                parquet.ParquetFile(path).schema_arrow.metadata or {},
            )
            self.assertEqual(embedding_ids(path), ({"fixture"}, metadata))
            embeddings, actual_metadata = read_embedding(path)
            self.assertEqual(actual_metadata, metadata)
            self.assertEqual(embeddings["fixture"].tolist(), [1.0, 2.0])

    def test_schema_metadata_remains_supported(self):
        metadata = {"model_name": "nmrtrans"}
        table = pa.table({"record_id": ["fixture"]}).replace_schema_metadata({
            b"nmr_embedding_metadata": json.dumps(metadata).encode("utf-8"),
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings.parquet"
            parquet.write_table(table, path)
            self.assertEqual(embedding_metadata(path), metadata)

    def test_missing_metadata_still_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings.parquet"
            parquet.write_table(pa.table({"record_id": ["fixture"]}), path)
            with self.assertRaisesRegex(ValueError, "missing nmr_embedding_metadata"):
                embedding_metadata(path)


if __name__ == "__main__":
    unittest.main()

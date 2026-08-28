from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import polars as pl

    from data.postprocess.analyze_cleaned_datasets import (
        FUNCTIONAL_GROUP_COLUMNS,
        PROPERTY_COLUMNS,
        molecular_frame_from_aligned_parquet,
        sample_by_group,
    )
except ModuleNotFoundError:
    pl = None


@unittest.skipUnless(pl is not None, "requires Polars and plotting dependencies")
class AnalyzeCleanedDatasetsTest(unittest.TestCase):
    def test_sampling_keeps_records_from_every_source(self):
        frame = pl.DataFrame(
            {
                "source": ["first"] * 5 + ["second"] * 5 + ["third"] * 5,
                "value": list(range(15)),
            }
        ).lazy()

        sample = sample_by_group(frame, "source", 2)
        counts = dict(
            sample.group_by("source").len().iter_rows()
        )

        self.assertEqual(counts, {"first": 2, "second": 2, "third": 2})

    def test_molecular_sidecar_must_match_record_order(self):
        records = pl.DataFrame(
            {
                "record_id": ["record-a", "record-b"],
                "source": ["source-a", "source-b"],
                "atoms": [["C"], ["C", "C"]],
            }
        ).lazy()
        property_rows = []
        for record_id in ("record-a", "wrong-record"):
            property_rows.append(
                {
                    "record_id": record_id,
                    "rdkit_status": "ok",
                    **{column: 1.0 for column in PROPERTY_COLUMNS},
                    **{column: 0 for column in FUNCTIONAL_GROUP_COLUMNS},
                }
            )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "properties.parquet"
            pl.DataFrame(property_rows).write_parquet(path)
            with self.assertRaisesRegex(ValueError, "not aligned"):
                molecular_frame_from_aligned_parquet(records, path)


if __name__ == "__main__":
    unittest.main()

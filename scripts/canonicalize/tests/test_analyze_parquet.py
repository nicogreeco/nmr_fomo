"""Small smoke test for the streaming canonical Parquet report."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

from canonicalize.analyze_parquet import analyze_parquet_file
from canonicalize.common import write_canonical_parquet
from data import CanonicalRecord, CarbonPeak, ProtonPeak


@unittest.skipIf(
    importlib.util.find_spec("pyarrow") is None,
    "pyarrow is needed for the Parquet analysis smoke test",
)
class AnalyzeParquetTest(unittest.TestCase):
    def test_streaming_summary(self):
        records = [
            CanonicalRecord(
                record_id="record-1",
                source="test-source",
                h_nmr_peaks=(
                    ProtonPeak(
                        shift=1.0,
                        integration=None,
                        multiplicity="s",
                        j_values=None,
                    ),
                ),
                c_nmr_peaks=(CarbonPeak(shift=10.0),),
            ),
            CanonicalRecord(
                record_id="record-1",
                source="test-source",
                h_nmr_peaks=(),
                c_nmr_peaks=None,
            ),
            CanonicalRecord(
                record_id="record-3",
                h_nmr_peaks=(
                    ProtonPeak(
                        shift=2.0,
                        integration=0,
                        multiplicity=None,
                        j_values=(),
                    ),
                ),
                c_nmr_peaks=(),
            ),
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            parquet_path = Path(temporary_directory) / "fixture.parquet"
            write_canonical_parquet(
                records,
                parquet_path,
                source_name="test-source",
                converter_name="test-analyzer",
                row_group_size=2,
            )
            report = analyze_parquet_file(parquet_path, arrow_batch_size=1)

        self.assertEqual(report["rows"], 3)
        self.assertEqual(report["row_groups"], 2)
        self.assertEqual(report["duplicate_record_id_count"], 1)
        self.assertEqual(
            report["source_values"], {"<null>": 1, "test-source": 2}
        )
        self.assertEqual(report["both_modality_usable_count"], 1)
        self.assertEqual(report["h_nmr_peaks"]["empty_record_count"], 1)
        self.assertEqual(report["h_nmr_peaks"]["total_peak_count"], 2)
        self.assertEqual(report["h_nmr_peaks"]["shift_min"], 1.0)
        self.assertEqual(report["h_nmr_peaks"]["shift_max"], 2.0)
        self.assertEqual(report["c_nmr_peaks"]["null_record_count"], 1)
        self.assertEqual(report["c_nmr_peaks"]["empty_record_count"], 1)
        self.assertEqual(report["c_nmr_peaks"]["total_peak_count"], 1)
        self.assertEqual(report["proton_integration"]["null_count"], 1)
        self.assertEqual(report["proton_integration"]["zero_count"], 1)
        self.assertEqual(report["proton_j_values"]["null_count"], 1)
        self.assertEqual(report["proton_j_values"]["empty_count"], 1)
        self.assertEqual(
            report["canonical_multiplicity_counts"], {"<null>": 1, "s": 1}
        )
        self.assertEqual(
            report["schema_metadata"]["canonical_schema_version"], "1"
        )
        self.assertTrue(report["schema_matches_canonical"])


if __name__ == "__main__":
    unittest.main()

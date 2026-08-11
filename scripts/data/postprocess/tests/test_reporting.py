from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from data.reporting import prepare_report_output, write_processing_report


class ProcessingReportTest(unittest.TestCase):
    def test_writes_atomic_structured_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path, temporary_path = prepare_report_output(
                root / "report.json",
                [root / "dataset.parquet"],
                overwrite=False,
            )
            report = {
                "stage": "fixture",
                "inputs": {"dataset": "input.parquet"},
                "outputs": {"dataset": "output.parquet"},
                "counts": {"output_records": 3},
            }
            write_processing_report(report, report_path, temporary_path)

            loaded = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(loaded, report)

    def test_rejects_a_data_path_as_report(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "dataset.parquet"
            with self.assertRaises(ValueError):
                prepare_report_output(
                    data_path,
                    [data_path],
                    overwrite=False,
                )


if __name__ == "__main__":
    unittest.main()

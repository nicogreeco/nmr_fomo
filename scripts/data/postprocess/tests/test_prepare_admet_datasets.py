from __future__ import annotations

import csv
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
    from data.postprocess.prepare_admet_datasets import (
        ENDPOINTS,
        full_inchikey,
        prepare_admet_datasets,
        prepare_property_release,
    )
except ModuleNotFoundError:
    pa = None
    parquet = None
    full_inchikey = None
    prepare_admet_datasets = None
    prepare_property_release = None


@unittest.skipUnless(
    full_inchikey is not None and pa is not None,
    "requires RDKit and PyArrow",
)
class PrepareAdmetDatasetsTest(unittest.TestCase):
    def test_collapses_agreeing_labels_and_rejects_discordant_labels(self):
        ethanol_key = full_inchikey("CCO")
        propane_key = full_inchikey("CCC")
        assert ethanol_key is not None
        assert propane_key is not None

        keyed_rows = [
            ({"Drug": "CCO", "Y": "1.0"}, ethanol_key),
            ({"Drug": "CCO", "Y": "1.00"}, ethanol_key),
            ({"Drug": "CCC", "Y": "2.0"}, propane_key),
            ({"Drug": "CCC", "Y": "3.0"}, propane_key),
        ]
        record_ids_by_key = {
            ethanol_key: ["ethanol-a", "ethanol-b"],
            propane_key: ["propane"],
        }

        rows, valid_ids, all_matched_ids, stats = prepare_property_release(
            keyed_rows,
            record_ids_by_key,
            label_column="Y",
            location="fixture",
        )

        self.assertEqual(
            [row["record_id"] for row in rows],
            ["ethanol-a", "ethanol-b"],
        )
        self.assertEqual(valid_ids, {"ethanol-a", "ethanol-b"})
        self.assertEqual(
            all_matched_ids,
            {"ethanol-a", "ethanol-b", "propane"},
        )
        self.assertEqual(stats["agreeing_duplicate_molecules"], 1)
        self.assertEqual(stats["discordant_molecules"], 1)

    def write_canonical_fixture(self, path: Path) -> None:
        records = []
        for record_id, smiles in (
            ("ethanol", "CCO"),
            ("propane", "CCC"),
            ("nitrogen", "N"),
        ):
            records.append(
                {
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
            )
        schema = canonical_parquet_schema().with_metadata(
            {
                b"canonical_schema_version": (
                    CANONICAL_PARQUET_SCHEMA_VERSION.encode("utf-8")
                )
            }
        )
        parquet.write_table(pa.Table.from_pylist(records, schema=schema), path)

    def write_tdc_fixture(self, root: Path) -> None:
        for endpoint in ENDPOINTS:
            endpoint_dir = root / endpoint
            endpoint_dir.mkdir(parents=True)
            (endpoint_dir / "train_val.csv").write_text(
                "Drug,Y\nCCO,1.0\n",
                encoding="utf-8",
            )
            (endpoint_dir / "test.csv").write_text(
                "Drug,Y\nCCC,2.0\n",
                encoding="utf-8",
            )

    def test_streams_input_and_writes_aligned_cohorts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "rich_train.parquet"
            tdc_root = root / "tdc"
            output_root = root / "admet"
            train_output = root / "train_val.parquet"
            self.write_canonical_fixture(input_path)
            self.write_tdc_fixture(tdc_root)

            report = prepare_admet_datasets(
                input_path,
                tdc_root,
                output_root,
                train_output,
                batch_size=1,
                show_progress=False,
            )

            self.assertEqual(report["removed_pretraining_records"], 2)
            self.assertEqual(
                parquet.read_table(train_output)["record_id"].to_pylist(),
                ["nitrogen"],
            )
            for endpoint in ENDPOINTS:
                for split, expected_id in (
                    ("train_val", "ethanol"),
                    ("test", "propane"),
                ):
                    parquet_path = output_root / endpoint / f"{split}.parquet"
                    csv_path = output_root / endpoint / f"{split}.csv"
                    parquet_ids = parquet.read_table(parquet_path)[
                        "record_id"
                    ].to_pylist()
                    with csv_path.open(newline="", encoding="utf-8") as handle:
                        csv_ids = [
                            row["record_id"]
                            for row in csv.DictReader(handle)
                        ]
                    self.assertEqual(parquet_ids, [expected_id])
                    self.assertEqual(csv_ids, parquet_ids)


if __name__ == "__main__":
    unittest.main()

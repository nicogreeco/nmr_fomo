"""Small mapping and Parquet-reader tests; no source dataset is converted."""

import importlib.util
import io
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data.canonicalize.common import (
    ConversionError,
    resolve_lmdb_inputs,
    write_canonical_parquet,
)
from data.canonicalize.convert_mst_nmr import convert_record as convert_mst_record
from data.canonicalize.convert_nmrexp import convert_record as convert_nmrexp_record
from data.canonicalize.convert_nmrgym import convert_record as convert_nmrgym_record
from data.canonicalize.convert_nmrsolver import (
    ChemicalMetadataConversionError,
    ChemicalMetadataRejectionReport,
    _convert_serialized_batch_in_worker,
    convert_record as convert_nmrsolver_record,
    iter_converted_records as iter_nmrsolver_records,
)
from data.canonicalize.convert_nmrtrans import convert_record as convert_nmrtrans_record
from data import CanonicalParquetDataset, CanonicalRecord, CarbonPeak, ProtonPeak
from model_benchmarks.extract_embeddings import build_dataset


class ConverterMappingTests(unittest.TestCase):
    def test_nmrexp_mapping_preserves_peak_fields(self):
        raw_record = {
            "smiles": "OCC",
            "smiles_canonical": "source-value-is-overwritten",
            "molecular_formula": "source-value-is-overwritten",
            "atoms": ["source-value-is-overwritten"],
            "coordinates": [[0.0, 0.0, 0.0]],
            "nmr_frequency": "400 MHz",
            "nmr_solvent": "CDCl3",
            "h_nmr_peaks": [
                {
                    "centroid": 1.20,
                    "rangeMin": 1.18,
                    "rangeMax": 1.22,
                    "category": "hept",
                    "nH": 1,
                    "j_values": "7.2_2.1_",
                }
            ],
            "c_nmr_peaks": [{"delta (ppm)": 18.3}],
        }

        record = convert_nmrexp_record(raw_record, "nmrexp-1", "fixture")
        peak = record.h_nmr_peaks[0]
        self.assertEqual(record.source, "NMRPeak-NMRexp")
        self.assertEqual(record.smiles, "OCC")
        self.assertEqual(record.smiles_canonical, "CCO")
        self.assertEqual(record.molecular_formula, "C2H6O")
        self.assertEqual(record.atoms, ("C", "C", "O"))
        self.assertFalse(hasattr(record, "coordinates"))
        self.assertEqual(peak.multiplicity_raw, "hept")
        self.assertEqual(peak.multiplicity, "sept")
        self.assertEqual(peak.j_values, (7.2, 2.1))
        self.assertAlmostEqual(peak.range_half_span, 0.02)
        self.assertIsNone(record.c_nmr_peaks[0].intensity)

    def test_mst_mapping_preserves_carbon_properties(self):
        raw_record = {
            "smiles": "C(C)O",
            "h_nmr_peaks": [
                {
                    "centroid": 3.2,
                    "category": "p",
                    "nH": 0,
                    "j_values": None,
                }
            ],
            "c_nmr_peaks": [
                {
                    "delta (ppm)": 42.5,
                    "integral": 1.5,
                    "intensity": 22.0,
                    "width (ppm)": 0.4,
                }
            ],
        }

        record = convert_mst_record(raw_record, "mst-1", "fixture")
        peak = record.h_nmr_peaks[0]
        carbon = record.c_nmr_peaks[0]
        self.assertEqual(record.source, "NMRPeak-MST-NMR")
        self.assertEqual(peak.integration, 0)
        self.assertEqual(peak.multiplicity, "quint")
        self.assertEqual(peak.j_values, ())
        self.assertEqual(peak.range_min, 3.2)
        self.assertEqual(peak.range_max, 3.2)
        self.assertEqual(carbon.integral, 1.5)
        self.assertEqual(carbon.intensity, 22.0)
        self.assertEqual(carbon.width, 0.4)

    def test_nmrexp_missing_j_values_become_empty_list(self):
        raw_record = {
            "smiles": "CCO",
            "h_nmr_peaks": [{"centroid": 1.2, "j_values": None}],
            "c_nmr_peaks": [],
        }

        record = convert_nmrexp_record(raw_record, "nmrexp-1", "fixture")

        self.assertEqual(record.h_nmr_peaks[0].j_values, ())

    def test_null_modalities_become_empty_lists(self):
        nmrexp_record = convert_nmrexp_record(
            {"smiles": "CCO", "h_nmr_peaks": None, "c_nmr_peaks": None},
            "nmrexp-null-modalities",
            "fixture",
        )
        nmrtrans_record = convert_nmrtrans_record(
            {
                "smiles": "CCO",
                "tokenized_input": json.dumps({"1HNMR": None}),
            },
            "nmrtrans-null-modalities",
            "fixture",
        )
        nmrgym_record = convert_nmrgym_record(
            {"smiles": "CCO", "h_shift": None, "c_shift": None},
            "nmrgym-null-modalities",
            "fixture",
        )

        for record in (nmrexp_record, nmrtrans_record, nmrgym_record):
            self.assertEqual(record.h_nmr_peaks, ())
            self.assertEqual(record.c_nmr_peaks, ())

    def test_nmrtrans_mapping_uses_half_span_and_empty_j_list(self):
        raw_record = {
            "id": 7,
            "smiles": "OCC",
            "molecular_formula": "source-value-is-overwritten",
            "tokenized_input": json.dumps(
                {
                    "1HNMR": [[1.25, 0.05, "brd", "3H", []]],
                    "13CNMR": [18.4, 58.3],
                }
            ),
        }

        record = convert_nmrtrans_record(raw_record, "nmrtrans-1", "fixture")
        peak = record.h_nmr_peaks[0]
        self.assertEqual(record.source, "NMRTrans-NMRSpec")
        self.assertEqual(record.smiles, "OCC")
        self.assertEqual(record.smiles_canonical, "CCO")
        self.assertEqual(record.molecular_formula, "C2H6O")
        self.assertEqual(record.atoms, ("C", "C", "O"))
        self.assertEqual(peak.multiplicity_raw, "brd")
        self.assertEqual(peak.multiplicity, "bd")
        self.assertEqual(peak.integration, 3)
        self.assertEqual(peak.j_values, ())
        self.assertAlmostEqual(peak.range_min, 1.20)
        self.assertAlmostEqual(peak.range_max, 1.30)

    def test_nmrtrans_missing_j_values_become_empty_list(self):
        raw_record = {
            "id": 8,
            "smiles": "CCO",
            "tokenized_input": json.dumps(
                {"1HNMR": [[1.25, 0.05, "s", "3H", None]], "13CNMR": []}
            ),
        }

        record = convert_nmrtrans_record(raw_record, "nmrtrans-2", "fixture")

        self.assertEqual(record.h_nmr_peaks[0].j_values, ())


    def test_nmrtrans_older_original_smiles_field_is_supported(self):
        raw_record = {
            "id": 9,
            "original_smiles": "OCC",
            "smiles": [1, 2, 3],
            "tokenized_input": json.dumps({"1HNMR": [], "13CNMR": []}),
        }

        record = convert_nmrtrans_record(raw_record, "nmrtrans-3", "fixture")

        self.assertEqual(record.smiles, "OCC")
        self.assertEqual(record.smiles_canonical, "CCO")
        self.assertEqual(record.molecular_formula, "C2H6O")
        self.assertEqual(record.atoms, ("C", "C", "O"))

    def test_nmrsolver_mapping_groups_exact_equivalence_classes(self):
        raw_record = {
            "smiles": "OCC",
            "canonical_smiles": "source-value-is-overwritten",
            "nmr_predict": [18.0, 20.0, 1.0, 1.2, 1.1],
            "atom_index": [6, 6, 1, 1, 1],
            "equi_class": [4, 4, 7, 7, 9],
        }

        record = convert_nmrsolver_record(raw_record, "solver-1", "fixture")

        self.assertEqual(record.source, "SimNMR-PubChem")
        self.assertEqual(record.smiles, "OCC")
        self.assertEqual(record.smiles_canonical, "CCO")
        self.assertEqual(record.molecular_formula, "C2H6O")
        self.assertEqual(record.atoms, ("C", "C", "O"))
        self.assertEqual(len(record.h_nmr_peaks), 2)
        self.assertAlmostEqual(record.h_nmr_peaks[0].shift, 1.1)
        self.assertEqual(record.h_nmr_peaks[0].integration, 2)
        self.assertEqual(record.h_nmr_peaks[0].equivalence_class, 7)
        self.assertEqual(record.h_nmr_peaks[0].member_shifts, (1.0, 1.2))
        self.assertIsNone(record.h_nmr_peaks[0].multiplicity)
        self.assertIsNone(record.h_nmr_peaks[0].j_values)
        self.assertIsNone(record.h_nmr_peaks[0].range_min)
        self.assertEqual(record.h_nmr_peaks[1].equivalence_class, 9)
        self.assertAlmostEqual(record.c_nmr_peaks[0].shift, 19.0)
        self.assertIsNone(record.c_nmr_peaks[0].integral)

    def test_nmrsolver_marks_unprocessable_chemical_metadata(self):
        raw_record = {
            "smiles": (
                "C=CC1CN2CC[C@H]1C[C@H]2[C@@H](C3=CC=[Al]C4=CC=CC=C34)O"
            ),
            "nmr_predict": [18.0, 1.0],
            "atom_index": [6, 1],
            "equi_class": [1, 2],
        }

        with self.assertRaisesRegex(
            ChemicalMetadataConversionError,
            "canonical SMILES that RDKit could not parse",
        ):
            convert_nmrsolver_record(raw_record, "solver-invalid", "fixture")

    def test_nmrsolver_skips_and_reports_unprocessable_metadata(self):
        invalid_record = {
            "smiles": (
                "C=CC1CN2CC[C@H]1C[C@H]2[C@@H](C3=CC=[Al]C4=CC=CC=C34)O"
            ),
            "nmr_predict": [18.0, 1.0],
            "atom_index": [6, 1],
            "equi_class": [1, 2],
        }
        valid_record = {
            "smiles": "CCO",
            "nmr_predict": [18.0, 1.0],
            "atom_index": [6, 1],
            "equi_class": [1, 2],
        }
        report_buffer = io.StringIO()
        rejection_report = ChemicalMetadataRejectionReport(report_buffer)

        with patch(
            "data.canonicalize.convert_nmrsolver.resolve_input",
            return_value=Path("fixture.lmdb"),
        ), patch(
            "data.canonicalize.convert_nmrsolver.iter_lmdb_records",
            return_value=[(b"invalid", invalid_record), (b"valid", valid_record)],
        ):
            records = list(
                iter_nmrsolver_records("fixture.lmdb", rejection_report)
            )

        self.assertEqual([record.record_id for record in records], [
            "nmrsolver-simnmr-pubchem:valid"
        ])
        self.assertEqual(rejection_report.count, 1)
        report_row = json.loads(report_buffer.getvalue())
        self.assertEqual(report_row["record_id"], "nmrsolver-simnmr-pubchem:invalid")
        self.assertEqual(report_row["source_smiles"], invalid_record["smiles"])
        self.assertIn("canonical SMILES", report_row["reason"])

    def test_nmrsolver_worker_unpickles_and_returns_arrow_ready_rows(self):
        raw_record = {
            "smiles": "CCO",
            "nmr_predict": [18.0, 1.0],
            "atom_index": [6, 1],
            "equi_class": [1, 2],
        }
        results = _convert_serialized_batch_in_worker(
            [(b"fixture", pickle.dumps(raw_record))],
            "fixture.lmdb",
        )

        row, rejection_reason, report_record = results[0]
        self.assertEqual(
            row["record_id"],
            "nmrsolver-simnmr-pubchem:fixture",
        )
        self.assertEqual(row["smiles_canonical"], "CCO")
        self.assertIsNone(rejection_reason)
        self.assertIsNone(report_record)

    def test_nmrsolver_parallel_conversion_preserves_order_and_rejections(self):
        invalid_record = {
            "smiles": (
                "C=CC1CN2CC[C@H]1C[C@H]2[C@@H](C3=CC=[Al]C4=CC=CC=C34)O"
            ),
            "nmr_predict": [18.0, 1.0],
            "atom_index": [6, 1],
            "equi_class": [1, 2],
        }
        first_record = {
            "smiles": "CCO",
            "nmr_predict": [18.0, 1.0],
            "atom_index": [6, 1],
            "equi_class": [1, 2],
        }
        last_record = {
            "smiles": "CC",
            "nmr_predict": [20.0, 1.2],
            "atom_index": [6, 1],
            "equi_class": [3, 4],
        }
        report_buffer = io.StringIO()
        rejection_report = ChemicalMetadataRejectionReport(report_buffer)

        with patch(
            "data.canonicalize.convert_nmrsolver.resolve_input",
            return_value=Path("fixture.lmdb"),
        ), patch(
            "data.canonicalize.convert_nmrsolver.iter_lmdb_records",
            return_value=[
                (b"first", first_record),
                (b"invalid", invalid_record),
                (b"last", last_record),
            ],
        ):
            records = list(
                iter_nmrsolver_records(
                    "fixture.lmdb",
                    rejection_report,
                    workers=2,
                    max_in_flight=4,
                    records_per_task=2,
                )
            )

        self.assertEqual(
            [record.record_id for record in records],
            [
                "nmrsolver-simnmr-pubchem:first",
                "nmrsolver-simnmr-pubchem:last",
            ],
        )
        self.assertEqual(rejection_report.count, 1)
        report_row = json.loads(report_buffer.getvalue())
        self.assertEqual(
            report_row["record_id"], "nmrsolver-simnmr-pubchem:invalid"
        )
        self.assertEqual(report_row["source_smiles"], invalid_record["smiles"])

    def test_missing_or_invalid_smiles_fails_conversion(self):
        with self.assertRaisesRegex(ConversionError, "non-empty SMILES"):
            convert_mst_record(
                {"h_nmr_peaks": [], "c_nmr_peaks": []},
                "mst-missing-smiles",
                "missing fixture",
            )

        with self.assertRaisesRegex(ConversionError, "could not be parsed"):
            convert_nmrexp_record(
                {
                    "smiles": "not a valid smiles",
                    "h_nmr_peaks": [],
                    "c_nmr_peaks": [],
                },
                "nmrexp-invalid-smiles",
                "invalid fixture",
            )

    def test_nmrexp_directory_uses_splits_for_stable_combined_ids(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            filenames = ["all.lmdb", "train.lmdb", "valid.lmdb", "test.lmdb"]
            for filename in filenames:
                (directory / filename).touch()

            inputs = resolve_lmdb_inputs(directory, prefer_all_file=False)

        self.assertEqual(
            [path.name for path in inputs],
            ["train.lmdb", "valid.lmdb", "test.lmdb"],
        )


@unittest.skipUnless(
    importlib.util.find_spec("pyarrow") is not None,
    "pyarrow is needed for the Parquet reader smoke test",
)
class ParquetDatasetTests(unittest.TestCase):
    def test_parquet_dataset_yields_canonical_records_in_order(self):
        records = [
            CanonicalRecord(
                record_id="parquet-1",
                source="fixture",
                h_nmr_peaks=(
                    ProtonPeak(
                        shift=1.2,
                        integration=3,
                        multiplicity="s",
                        j_values=(),
                        range_min=1.2,
                        range_max=1.2,
                        range_half_span=0.0,
                    ),
                ),
                c_nmr_peaks=(CarbonPeak(shift=18.4),),
            ),
            CanonicalRecord(
                record_id="parquet-2",
                source="fixture",
                h_nmr_peaks=None,
                c_nmr_peaks=None,
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "records.parquet"
            count = write_canonical_parquet(
                records,
                output,
                source_name="fixture",
                converter_name="test",
                row_group_size=1,
            )

            import pyarrow.parquet as parquet
            import rdkit

            stored_schema = parquet.ParquetFile(output).schema_arrow
            read_records = list(CanonicalParquetDataset(output, arrow_batch_size=1))

            limited_records = list(build_dataset(output, max_records=1))
        self.assertEqual(count, 2)
        self.assertNotIn("coordinates", stored_schema.names)
        self.assertEqual(
            stored_schema.metadata[b"rdkit_version"].decode(), rdkit.__version__
        )
        self.assertEqual(
            [record.record_id for record in read_records],
            ["parquet-1", "parquet-2"],
        )
        self.assertEqual(read_records[0].h_nmr_peaks[0].j_values, ())
        self.assertEqual(read_records[1].h_nmr_peaks, ())
        self.assertEqual(read_records[1].c_nmr_peaks, ())
        self.assertEqual(
            [record.record_id for record in limited_records],
            ["parquet-1"],
        )


if __name__ == "__main__":
    unittest.main()

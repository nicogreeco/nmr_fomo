from __future__ import annotations

import math
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
    from data.postprocess.filter_dataset import (
        clean_parquet,
        default_output_paths,
    )
except ModuleNotFoundError:
    pa = None
    parquet = None


def h_peak(
    shift: float,
    *,
    integration: int | None = 1,
    multiplicity: str | None = "s",
    j_values: list[float] | None = None,
    include_range: bool = True,
) -> dict[str, object]:
    return {
        "shift": shift,
        "integration": integration,
        "multiplicity_raw": multiplicity,
        "multiplicity": multiplicity,
        "j_values": [] if j_values is None else j_values,
        "range_min": shift if include_range else None,
        "range_max": shift if include_range else None,
        "range_half_span": 0.0 if include_range else None,
        "equivalence_class": None,
        "member_shifts": None,
    }


def c_peak(shift: float) -> dict[str, object]:
    return {
        "shift": shift,
        "integral": None,
        "intensity": None,
        "width": None,
    }


def record(
    record_id: str,
    smiles: str,
    h_peaks: list[dict[str, object]] | None,
    c_peaks: list[dict[str, object]] | None,
) -> dict[str, object]:
    return {
        "record_id": record_id,
        "source": "fixture",
        "smiles": smiles,
        "smiles_canonical": smiles,
        "molecular_formula": None,
        "nmr_frequency": None,
        "nmr_solvent": None,
        "atoms": None,
        "h_nmr_peaks": h_peaks,
        "c_nmr_peaks": c_peaks,
    }


@unittest.skipUnless(pa is not None, "requires Polars and PyArrow")
class FilterDatasetTest(unittest.TestCase):
    def write_fixture(self, path: Path) -> None:
        less_annotated_peak = h_peak(
            1.0,
            integration=None,
            multiplicity="<unk>",
            include_range=False,
        )
        less_annotated_peak["j_values"] = None

        records = [
            record("duplicate-less", "CC", [less_annotated_peak], [c_peak(20.0)]),
            record("duplicate-best", "CC", [h_peak(1.0)], [c_peak(20.0)]),
            record("different-spectrum", "CC", [h_peak(1.1)], [c_peak(20.0)]),
            record("h-only", "CCC", [h_peak(2.0)], None),
            record("c-only", "CCCC", None, [c_peak(30.0)]),
            record("empty", "CCCCC", None, []),
            record("bad-h-shift", "CCCCCC", [h_peak(21.0)], None),
            record("bad-c-shift", "CCCCCCC", None, [c_peak(301.0)]),
            record(
                "too-many-c",
                "CCCCCCCC",
                None,
                [c_peak(float(index)) for index in range(61)],
            ),
            record(
                "too-many-j",
                "CCCCCCCCC",
                [h_peak(3.0, j_values=[1.0] * 7)],
                None,
            ),
            record(
                "negative-j",
                "CCCCCCCCCC",
                [h_peak(3.0, j_values=[-1.0])],
                None,
            ),
            record(
                "non-finite-j",
                "CCCCCCCCCCC",
                [h_peak(3.0, j_values=[math.inf])],
                None,
            ),
            record("zero-integration", "N", [h_peak(3.0, integration=0)], None),
            record("zero-j-is-kept", "O", [h_peak(3.0, j_values=[0.0])], None),
            record("missing-integration", "P", [h_peak(3.0, integration=None)], None),
            record("multifragment", "CC.O", [h_peak(3.0)], None),
            record("non-finite-shift", "S", [h_peak(math.inf)], None),
            record("partial-a", "ClCCl", None, [c_peak(40.0)]),
            record("partial-b", "ClCCl", [], [c_peak(40.0)]),
        ]

        schema = canonical_parquet_schema().with_metadata(
            {
                b"canonical_schema_version": (
                    CANONICAL_PARQUET_SCHEMA_VERSION.encode("utf-8")
                ),
                b"fixture_metadata": b"preserved",
            }
        )
        table = pa.Table.from_pylist(records, schema=schema)
        parquet.write_table(table, path)

    def test_cleaning_filters_and_keeps_the_best_exact_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "fixture.parquet"
            self.write_fixture(input_path)

            result = clean_parquet(input_path, batch_size=4)
            cleaned_path, removed_path = default_output_paths(input_path)
            cleaned = parquet.read_table(cleaned_path)
            removed = parquet.read_table(removed_path).to_pylist()

        cleaned_ids = cleaned["record_id"].to_pylist()
        removed_by_id = {row["record_id"]: row for row in removed}

        self.assertEqual(result["input_rows"], 19)
        self.assertEqual(result["output_rows"], 7)
        self.assertEqual(result["removed_rows"], 12)

        self.assertIn("duplicate-best", cleaned_ids)
        self.assertNotIn("duplicate-less", cleaned_ids)
        self.assertEqual(
            removed_by_id["duplicate-less"],
            {
                "record_id": "duplicate-less",
                "removal_reason": "duplicate_shift_signature",
                "duplicate_of": "duplicate-best",
            },
        )

        self.assertIn("different-spectrum", cleaned_ids)
        self.assertIn("h-only", cleaned_ids)
        self.assertIn("c-only", cleaned_ids)
        self.assertIn("zero-j-is-kept", cleaned_ids)
        self.assertIn("missing-integration", cleaned_ids)

        self.assertIn("partial-a", cleaned_ids)
        self.assertNotIn("partial-b", cleaned_ids)
        self.assertEqual(removed_by_id["partial-b"]["duplicate_of"], "partial-a")

        self.assertEqual(
            removed_by_id["empty"]["removal_reason"],
            "no_nmr_peaks",
        )
        self.assertEqual(
            removed_by_id["non-finite-j"]["removal_reason"],
            "invalid_j_value",
        )
        self.assertEqual(
            removed_by_id["multifragment"]["removal_reason"],
            "multifragment_smiles",
        )

        cleaned_metadata = cleaned.schema.metadata or {}
        self.assertEqual(cleaned_metadata[b"fixture_metadata"], b"preserved")
        self.assertEqual(cleaned_metadata[b"cleaning_policy_version"], b"1")

        partial_a_index = cleaned_ids.index("partial-a")
        self.assertEqual(cleaned["h_nmr_peaks"][partial_a_index].as_py(), [])

        zero_j_index = cleaned_ids.index("zero-j-is-kept")
        zero_j_peaks = cleaned["h_nmr_peaks"][zero_j_index].as_py()
        self.assertEqual(zero_j_peaks[0]["j_values"], [0.0])
        self.assertTrue(
            cleaned.schema.remove_metadata().equals(canonical_parquet_schema())
        )

    def test_existing_outputs_require_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "fixture.parquet"
            self.write_fixture(input_path)
            cleaned_path, _ = default_output_paths(input_path)
            cleaned_path.write_text("existing output\n", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                clean_parquet(input_path)

            shared_output = Path(directory) / "shared.parquet"
            with self.assertRaises(ValueError):
                clean_parquet(
                    input_path,
                    output_path=shared_output,
                    removed_output_path=shared_output,
                )


if __name__ == "__main__":
    unittest.main()

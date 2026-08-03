"""One small processor smoke test for each supported model."""

import importlib.util
import unittest
from dataclasses import replace

from data import CanonicalRecord, CarbonPeak, IncompatibleRecordError, ProtonPeak
from model_benchmarks import build_processor


def example_record(
    record_id="example-1",
    integration=2,
    member_shifts=None,
):
    return CanonicalRecord(
        record_id=record_id,
        h_nmr_peaks=(
            ProtonPeak(
                shift=1.20,
                integration=integration,
                multiplicity_raw="p",
                multiplicity="quint",
                j_values=(7.2,),
                range_min=1.19,
                range_max=1.21,
                range_half_span=0.01,
                member_shifts=member_shifts,
            ),
        ),
        c_nmr_peaks=(CarbonPeak(shift=42.0),),
    )


class ProcessorSmokeTests(unittest.TestCase):
    def test_nmrpeak_processor(self):
        processor = build_processor("nmrpeak", mode="canonical")

        # Conversion does not import the model repository.
        processor.prepare_record(example_record())

        dependencies_available = (
            importlib.util.find_spec("nmrpeak") is not None
            and importlib.util.find_spec("unicore") is not None
        )
        if dependencies_available:
            batch = processor([example_record()])
            self.assertEqual(batch.record_ids, ["example-1"])
            self.assertEqual(batch.inputs["src_tokens"].ndim, 2)
            self.assertEqual(
                batch.inputs["src_tokens"].shape,
                batch.inputs["attention_mask"].shape,
            )

    def test_nmrtrans_processor_and_native_multiplicity(self):
        processor = build_processor("nmrtrans", mode="native")
        batch = processor([example_record()])

        self.assertEqual(batch.record_ids, ["example-1"])
        self.assertEqual(tuple(batch.inputs["h_nmr_features"].shape), (1, 60, 10))
        self.assertEqual(tuple(batch.inputs["c_nmr_peaks"].shape), (1, 60, 1))
        # Native "p" is the alias for canonical "quint" (index 14).
        self.assertEqual(batch.inputs["h_nmr_features"][0, 0, 2].item(), 14.0)
        self.assertAlmostEqual(
            batch.inputs["c_nmr_peaks"][0, 0, 0].item(),
            42.0 / 220.0,
            places=6,
        )

    def test_nmrtrans_rejects_a_missing_required_field(self):
        record = example_record()
        peak_without_j_values = replace(record.h_nmr_peaks[0], j_values=None)
        record = replace(record, h_nmr_peaks=(peak_without_j_values,))

        with self.assertRaisesRegex(IncompatibleRecordError, "j_values"):
            build_processor("nmrtrans").prepare_record(record)

    def test_ultranmr_processor(self):
        processor = build_processor("ultranmr", mode="native")
        second_record = example_record(
            record_id="example-2",
            integration=None,
            member_shifts=(1.18, 1.22, 1.23),
        )
        batch = processor([example_record(), second_record])

        self.assertEqual(tuple(batch.inputs["shifts"].shape), (2, 4))
        self.assertEqual(batch.inputs["types"][0, :3].tolist(), [0, 0, 1])
        self.assertTrue(batch.inputs["padding_mask"][0, 3].item())
        self.assertFalse(batch.inputs["padding_mask"][1].any().item())

    def test_nmrsolver_processor(self):
        batch = build_processor("nmrsolver")([example_record()])

        self.assertEqual(batch.record_ids, ["example-1"])
        self.assertEqual(batch.inputs["h_shifts"], [(1.2, 1.2)])
        self.assertEqual(batch.inputs["c_shifts"], [(42.0,)])


if __name__ == "__main__":
    unittest.main()

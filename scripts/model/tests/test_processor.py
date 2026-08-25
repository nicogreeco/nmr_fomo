"""Tests for the new foundation-model input processor."""

import unittest

import torch
from torch.utils.data import DataLoader

from data import CanonicalNMRDataset, CanonicalRecord, CarbonPeak, ProtonPeak
from data.limits import MAX_J_VALUES_PER_PEAK, MAX_PEAKS_PER_MODALITY
from data.validation import IncompatibleRecordError
from model import (
    FoundationNMRProcessor,
    H_AVAILABILITY_FIELDS,
    ID_TO_MULTIPLICITY,
    MAX_C_PEAKS,
    MAX_H_PEAKS,
    MAX_J_VALUES,
    MULTIPLICITY_TO_ID,
)


def h_peak(
    shift,
    *,
    integration=None,
    multiplicity=None,
    j_values=None,
    range_half_span=None,
    range_min=None,
    range_max=None,
):
    return ProtonPeak(
        shift=shift,
        integration=integration,
        multiplicity=multiplicity,
        j_values=j_values,
        range_half_span=range_half_span,
        range_min=range_min,
        range_max=range_max,
    )


class FoundationNMRProcessorTests(unittest.TestCase):
    def setUp(self):
        self.processor = FoundationNMRProcessor()

    def test_dataloader_collates_mixed_modalities_in_input_order(self):
        records = [
            CanonicalRecord(
                record_id="paired",
                h_nmr_peaks=(
                    h_peak(
                        1.2,
                        integration=3,
                        multiplicity="t",
                        j_values=(7.2, 0.0),
                        range_half_span=0.01,
                    ),
                    h_peak(
                        3.4,
                        integration=1,
                        multiplicity="<unk>",
                        j_values=(),
                        range_half_span=0.0,
                    ),
                ),
                c_nmr_peaks=(CarbonPeak(42.0),),
            ),
            CanonicalRecord(
                record_id="h-only",
                h_nmr_peaks=(
                    h_peak(
                        2.5,
                        integration=2,
                        multiplicity="d",
                        j_values=(8.4,),
                        range_half_span=0.02,
                    ),
                ),
            ),
            CanonicalRecord(
                record_id="c-only",
                c_nmr_peaks=(CarbonPeak(18.0), CarbonPeak(65.0)),
            ),
            CanonicalRecord(
                record_id="shift-only-h",
                h_nmr_peaks=(h_peak(5.5),),
            ),
        ]
        dataset = CanonicalNMRDataset(records)
        loader = DataLoader(dataset, batch_size=4, collate_fn=self.processor)

        batch = next(iter(loader))

        self.assertEqual(
            batch["record_ids"],
            ["paired", "h-only", "c-only", "shift-only-h"],
        )
        self.assertEqual(tuple(batch["h"]["shift"].shape), (4, 60))
        self.assertEqual(tuple(batch["h"]["integration"].shape), (4, 60))
        self.assertEqual(tuple(batch["h"]["multiplicity"].shape), (4, 60))
        self.assertEqual(tuple(batch["h"]["j_values"].shape), (4, 60, 6))
        self.assertEqual(tuple(batch["h"]["range_half_span"].shape), (4, 60))
        self.assertEqual(tuple(batch["h"]["peak_mask"].shape), (4, 60))
        self.assertEqual(tuple(batch["h"]["j_mask"].shape), (4, 60, 6))
        self.assertEqual(tuple(batch["h"]["availability"].shape), (4, 60, 4))
        self.assertEqual(tuple(batch["c"]["shift"].shape), (4, 60))
        self.assertEqual(tuple(batch["c"]["peak_mask"].shape), (4, 60))

        self.assertEqual(batch["h"]["shift"].dtype, torch.float32)
        self.assertEqual(batch["h"]["integration"].dtype, torch.float32)
        self.assertEqual(batch["h"]["multiplicity"].dtype, torch.long)
        self.assertEqual(batch["h"]["j_values"].dtype, torch.float32)
        self.assertEqual(batch["h"]["range_half_span"].dtype, torch.float32)
        self.assertEqual(batch["h"]["peak_mask"].dtype, torch.bool)
        self.assertEqual(batch["h"]["j_mask"].dtype, torch.bool)
        self.assertEqual(batch["h"]["availability"].dtype, torch.bool)
        self.assertEqual(batch["c"]["shift"].dtype, torch.float32)
        self.assertEqual(batch["c"]["peak_mask"].dtype, torch.bool)

        self.assertEqual(
            batch["h"]["peak_mask"][:, :2].tolist(),
            [
                [True, True],
                [True, False],
                [False, False],
                [True, False],
            ],
        )
        self.assertEqual(
            batch["c"]["peak_mask"][:, :2].tolist(),
            [
                [True, False],
                [False, False],
                [True, True],
                [False, False],
            ],
        )
        self.assertFalse(batch["h"]["peak_mask"][:, 2:].any().item())
        self.assertFalse(batch["c"]["peak_mask"][:, 2:].any().item())
        self.assertFalse(batch["h"]["shift"][:, 2:].any().item())
        self.assertFalse(batch["h"]["integration"][:, 2:].any().item())
        self.assertFalse(batch["h"]["multiplicity"][:, 2:].any().item())
        self.assertFalse(batch["h"]["range_half_span"][:, 2:].any().item())
        self.assertFalse(batch["h"]["j_values"][:, 2:].any().item())
        self.assertFalse(batch["c"]["shift"][:, 2:].any().item())
        self.assertFalse(batch["h"]["shift"][2].any().item())
        self.assertFalse(batch["c"]["shift"][1].any().item())
        self.assertFalse(batch["c"]["shift"][3].any().item())

    def test_values_masks_and_availability_preserve_canonical_semantics(self):
        record = CanonicalRecord(
            record_id="annotations",
            h_nmr_peaks=(
                h_peak(
                    1.2,
                    integration=3,
                    multiplicity="t",
                    j_values=(8.4, 0.0),
                    range_half_span=0.01,
                ),
                h_peak(
                    2.4,
                    integration=1,
                    multiplicity="<unk>",
                    j_values=(),
                    range_half_span=0.0,
                ),
                h_peak(
                    3.6,
                    range_min=3.5,
                    range_max=3.7,
                ),
            ),
            c_nmr_peaks=(CarbonPeak(42.0),),
        )

        batch = self.processor([record])
        h = batch["h"]

        self.assertTrue(
            torch.allclose(h["shift"][0, :3], torch.tensor([1.2, 2.4, 3.6]))
        )
        self.assertEqual(h["integration"][0, :3].tolist(), [3.0, 1.0, 0.0])
        self.assertEqual(
            h["multiplicity"][0, :3].tolist(),
            [MULTIPLICITY_TO_ID["t"], 1, 0],
        )
        self.assertTrue(
            torch.allclose(
                h["j_values"][0, 0],
                torch.tensor([8.4, 0.0, 0.0, 0.0, 0.0, 0.0]),
            )
        )
        self.assertEqual(
            h["j_mask"][0, 0].tolist(),
            [True, True, False, False, False, False],
        )
        self.assertFalse(h["j_mask"][0, 1].any().item())
        self.assertFalse(h["j_mask"][0, 2].any().item())
        self.assertEqual(
            h["availability"][0, :3].tolist(),
            [
                [True, True, True, True],
                [True, True, True, True],
                [False, False, False, False],
            ],
        )
        self.assertTrue(
            torch.allclose(
                h["range_half_span"][0, :3],
                torch.tensor([0.01, 0.0, 0.0]),
            )
        )
        self.assertFalse(h["availability"][0, 3:].any().item())
        self.assertFalse(h["j_mask"][0, 3:].any().item())
        self.assertEqual(batch["c"]["shift"][0, 0].item(), 42.0)

    def test_multiplicity_mapping_is_stable_and_reversible(self):
        expected_labels = (
            None,
            "<unk>",
            "m",
            "d",
            "s",
            "dd",
            "t",
            "ddd",
            "q",
            "dt",
            "td",
            "br",
            "ddt",
            "dq",
            "tt",
            "quint",
            "dddd",
            "qd",
            "sept",
            "ddp",
            "ddq",
            "bd",
            "dqd",
        )
        expected_to_id = {
            label: index for index, label in enumerate(expected_labels)
        }
        expected_from_id = {
            index: label for index, label in enumerate(expected_labels)
        }

        self.assertEqual(MULTIPLICITY_TO_ID, expected_to_id)
        self.assertEqual(ID_TO_MULTIPLICITY, expected_from_id)

    def test_limits_are_aliases_of_the_shared_data_contract(self):
        self.assertEqual(MAX_H_PEAKS, MAX_PEAKS_PER_MODALITY)
        self.assertEqual(MAX_C_PEAKS, MAX_PEAKS_PER_MODALITY)
        self.assertEqual(MAX_J_VALUES, MAX_J_VALUES_PER_PEAK)
        self.assertEqual(
            H_AVAILABILITY_FIELDS,
            ("integration", "multiplicity", "j_values", "range_half_span"),
        )

    def test_dictionary_fixture_is_accepted(self):
        batch = self.processor(
            [
                {
                    "record_id": "mapping",
                    "h_nmr_peaks": [{"shift": 1.0, "j_values": []}],
                    "c_nmr_peaks": [],
                }
            ]
        )

        self.assertEqual(batch["record_ids"], ["mapping"])
        self.assertTrue(batch["h"]["peak_mask"][0, 0].item())
        self.assertTrue(batch["h"]["availability"][0, 0, 2].item())

    def test_rejects_more_than_60_h_peaks(self):
        record = CanonicalRecord(
            record_id="too-many-h",
            h_nmr_peaks=tuple(h_peak(1.0) for _ in range(61)),
        )

        with self.assertRaisesRegex(
            IncompatibleRecordError, "contains 61 1H peaks; maximum is 60"
        ):
            self.processor([record])

    def test_rejects_more_than_60_c_peaks(self):
        record = CanonicalRecord(
            record_id="too-many-c",
            c_nmr_peaks=tuple(CarbonPeak(42.0) for _ in range(61)),
        )

        with self.assertRaisesRegex(
            IncompatibleRecordError, "contains 61 13C peaks; maximum is 60"
        ):
            self.processor([record])

    def test_rejects_more_than_6_j_values(self):
        record = CanonicalRecord(
            record_id="too-many-j",
            h_nmr_peaks=(h_peak(1.0, j_values=(1.0,) * 7),),
        )

        with self.assertRaisesRegex(
            IncompatibleRecordError, "contains 7 J values; maximum is 6"
        ):
            self.processor([record])

    def test_reuses_canonical_validation(self):
        record = CanonicalRecord(
            record_id="invalid",
            h_nmr_peaks=(h_peak(float("nan")),),
        )

        with self.assertRaisesRegex(IncompatibleRecordError, "finite JSON number"):
            self.processor([record])

    def test_rejects_empty_batch(self):
        with self.assertRaisesRegex(ValueError, "empty foundation NMR batch"):
            self.processor([])


if __name__ == "__main__":
    unittest.main()

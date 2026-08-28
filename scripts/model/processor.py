"""Collate canonical NMR records for the new foundation model."""

import torch

from data.limits import MAX_J_VALUES_PER_PEAK, MAX_PEAKS_PER_MODALITY
from data.schema import CANONICAL_MULTIPLICITIES, ensure_record
from data.validation import (
    IncompatibleRecordError,
    ValidationIssue,
    validate_canonical_record,
)

from .dataset import MORGAN_FP_BYTES, PairedFoundationRecord


MAX_H_PEAKS = MAX_PEAKS_PER_MODALITY
MAX_C_PEAKS = MAX_PEAKS_PER_MODALITY
MAX_J_VALUES = MAX_J_VALUES_PER_PEAK

H_AVAILABILITY_FIELDS = (
    "integration",
    "multiplicity",
    "j_values",
    "range_half_span",
)

_MULTIPLICITY_LABELS = (None, "<unk>", *CANONICAL_MULTIPLICITIES)
MULTIPLICITY_TO_ID = {
    label: index for index, label in enumerate(_MULTIPLICITY_LABELS)
}
ID_TO_MULTIPLICITY = {
    index: label for index, label in enumerate(_MULTIPLICITY_LABELS)
}


def _unpack_morgan_fingerprints(fingerprints):
    """Expand RDKit binary bytes into little-endian float32 fingerprint bits."""

    for index, fingerprint in enumerate(fingerprints):
        if len(fingerprint) != MORGAN_FP_BYTES:
            raise ValueError(
                f"fingerprint {index} has {len(fingerprint)} bytes; "
                f"expected {MORGAN_FP_BYTES}"
            )

    packed = torch.tensor(
        [list(fingerprint) for fingerprint in fingerprints],
        dtype=torch.uint8,
    )
    bit_offsets = torch.arange(8, dtype=torch.uint8)
    bits = (packed.unsqueeze(-1) >> bit_offsets) & 1
    return bits.reshape(len(fingerprints), -1).to(torch.float32)


class FoundationNMRProcessor:
    """Validate, pad, and collate canonical peak records into dense tensors."""

    def __call__(self, records):
        records = list(records)
        if not records:
            raise ValueError("cannot collate an empty foundation NMR batch")

        paired_flags = [
            isinstance(record, PairedFoundationRecord) for record in records
        ]
        if any(paired_flags) and not all(paired_flags):
            raise TypeError(
                "cannot mix paired and unpaired records in one foundation batch"
            )

        fingerprints = None
        if all(paired_flags):
            fingerprints = [record.morgan_fingerprint for record in records]
            records = [record.record for record in records]

        canonical_records = []
        for value in records:
            record = ensure_record(value)
            validation = validate_canonical_record(record)
            if not validation.is_valid:
                raise IncompatibleRecordError(record.record_id, validation.issues)

            h_peak_count = len(record.h_nmr_peaks)
            if h_peak_count > MAX_H_PEAKS:
                issue = ValidationIssue(
                    code="too_many_h_peaks",
                    path="h_nmr_peaks",
                    message=(
                        f"contains {h_peak_count} 1H peaks; maximum is {MAX_H_PEAKS}"
                    ),
                )
                raise IncompatibleRecordError(record.record_id, [issue])

            c_peak_count = len(record.c_nmr_peaks)
            if c_peak_count > MAX_C_PEAKS:
                issue = ValidationIssue(
                    code="too_many_c_peaks",
                    path="c_nmr_peaks",
                    message=(
                        f"contains {c_peak_count} 13C peaks; maximum is {MAX_C_PEAKS}"
                    ),
                )
                raise IncompatibleRecordError(record.record_id, [issue])

            for peak_index, peak in enumerate(record.h_nmr_peaks):
                if peak.j_values is not None and len(peak.j_values) > MAX_J_VALUES:
                    issue = ValidationIssue(
                        code="too_many_j_values",
                        path=f"h_nmr_peaks[{peak_index}].j_values",
                        message=(
                            f"contains {len(peak.j_values)} J values; "
                            f"maximum is {MAX_J_VALUES}"
                        ),
                    )
                    raise IncompatibleRecordError(record.record_id, [issue])

            canonical_records.append(record)

        batch_size = len(canonical_records)

        h_shift = torch.zeros(batch_size, MAX_H_PEAKS, dtype=torch.float32)
        h_integration = torch.zeros(batch_size, MAX_H_PEAKS, dtype=torch.float32)
        h_multiplicity = torch.zeros(batch_size, MAX_H_PEAKS, dtype=torch.long)
        h_j_values = torch.zeros(
            batch_size, MAX_H_PEAKS, MAX_J_VALUES, dtype=torch.float32
        )
        h_range_half_span = torch.zeros(
            batch_size, MAX_H_PEAKS, dtype=torch.float32
        )
        h_peak_mask = torch.zeros(batch_size, MAX_H_PEAKS, dtype=torch.bool)
        h_j_mask = torch.zeros(
            batch_size, MAX_H_PEAKS, MAX_J_VALUES, dtype=torch.bool
        )
        h_availability = torch.zeros(
            batch_size,
            MAX_H_PEAKS,
            len(H_AVAILABILITY_FIELDS),
            dtype=torch.bool,
        )

        c_shift = torch.zeros(batch_size, MAX_C_PEAKS, dtype=torch.float32)
        c_peak_mask = torch.zeros(batch_size, MAX_C_PEAKS, dtype=torch.bool)

        record_ids = []
        for batch_index, record in enumerate(canonical_records):
            record_ids.append(record.record_id)

            for peak_index, peak in enumerate(record.h_nmr_peaks):
                h_shift[batch_index, peak_index] = peak.shift
                h_peak_mask[batch_index, peak_index] = True

                if peak.integration is not None:
                    h_integration[batch_index, peak_index] = peak.integration
                    h_availability[batch_index, peak_index, 0] = True

                if peak.multiplicity is not None:
                    h_multiplicity[batch_index, peak_index] = MULTIPLICITY_TO_ID[
                        peak.multiplicity
                    ]
                    h_availability[batch_index, peak_index, 1] = True

                if peak.j_values is not None:
                    h_availability[batch_index, peak_index, 2] = True
                    for j_index, j_value in enumerate(peak.j_values):
                        h_j_values[batch_index, peak_index, j_index] = j_value
                        h_j_mask[batch_index, peak_index, j_index] = True

                if peak.range_half_span is not None:
                    h_range_half_span[batch_index, peak_index] = (
                        peak.range_half_span
                    )
                    h_availability[batch_index, peak_index, 3] = True

            for peak_index, peak in enumerate(record.c_nmr_peaks):
                c_shift[batch_index, peak_index] = peak.shift
                c_peak_mask[batch_index, peak_index] = True

        batch = {
            "record_ids": record_ids,
            "h": {
                "shift": h_shift,
                "integration": h_integration,
                "multiplicity": h_multiplicity,
                "j_values": h_j_values,
                "range_half_span": h_range_half_span,
                "peak_mask": h_peak_mask,
                "j_mask": h_j_mask,
                "availability": h_availability,
            },
            "c": {
                "shift": c_shift,
                "peak_mask": c_peak_mask,
            },
        }
        if fingerprints is not None:
            batch["fingerprints"] = _unpack_morgan_fingerprints(fingerprints)
        return batch


__all__ = [
    "FoundationNMRProcessor",
    "H_AVAILABILITY_FIELDS",
    "ID_TO_MULTIPLICITY",
    "MAX_C_PEAKS",
    "MAX_H_PEAKS",
    "MAX_J_VALUES",
    "MULTIPLICITY_TO_ID",
]

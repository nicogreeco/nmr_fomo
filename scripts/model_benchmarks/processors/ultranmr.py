"""Convert canonical records to padded UltraNMR tensor batches."""

from data.schema import CanonicalRecord, ProtonPeak
from data.validation import (
    IncompatibleRecordError,
    ValidationIssue,
    validate_canonical_record,
)

from . import ModelBatch, read_record


def _expanded_proton_shifts(peak: ProtonPeak) -> list[float]:
    if peak.member_shifts:
        return [float(number) for number in peak.member_shifts]
    return [float(peak.shift)] * int(peak.integration)


class UltraNMRProcessor:
    """Validate, convert, pad, and collate combined 1H + 13C records.

    ``mode`` is accepted for a consistent API but is a no-op because UltraNMR
    does not use multiplicity.
    """

    model_name = "ultranmr"

    def __init__(self, mode: str = "canonical", strict: bool = True):
        if mode not in {"canonical", "native"}:
            raise ValueError("mode must be 'canonical' or 'native'")
        self.mode = mode
        self.strict = strict

    def _model_issues(self, record: CanonicalRecord) -> list[ValidationIssue]:
        issues = []
        if not record.h_nmr_peaks:
            issues.append(
                ValidationIssue(
                    "missing_h_nmr", "h_nmr_peaks", "UltraNMR requires 1H peaks"
                )
            )
        if not record.c_nmr_peaks:
            issues.append(
                ValidationIssue(
                    "missing_c_nmr", "c_nmr_peaks", "UltraNMR requires 13C peaks"
                )
            )

        for index, peak in enumerate(record.h_nmr_peaks or ()):
            if peak.member_shifts is not None and len(peak.member_shifts) == 0:
                issues.append(
                    ValidationIssue(
                        "empty_member_shifts",
                        f"h_nmr_peaks[{index}].member_shifts",
                        "member_shifts must contain at least one shift",
                    )
                )
            elif peak.member_shifts is None and (
                not isinstance(peak.integration, int)
                or isinstance(peak.integration, bool)
                or peak.integration <= 0
            ):
                issues.append(
                    ValidationIssue(
                        "missing_proton_expansion",
                        f"h_nmr_peaks[{index}]",
                        "UltraNMR requires member_shifts or a positive "
                        "integer integration",
                    )
                )
        return issues

    def prepare_record(self, value: object) -> dict:
        record = read_record(value)
        issues = []
        if self.strict:
            issues.extend(validate_canonical_record(record).issues)
        issues.extend(self._model_issues(record))
        if issues:
            raise IncompatibleRecordError(record.record_id, issues)

        h_shifts = []
        for peak in record.h_nmr_peaks or ():
            h_shifts.extend(_expanded_proton_shifts(peak))
        c_shifts = [float(peak.shift) for peak in record.c_nmr_peaks or ()]
        return {
            "record_id": record.record_id,
            "h_shifts": h_shifts,
            "c_shifts": c_shifts,
        }

    def collate(self, records: list[dict]) -> ModelBatch:
        if not records:
            raise ValueError("cannot collate an empty UltraNMR batch")

        import torch

        max_length = max(
            len(record["h_shifts"]) + len(record["c_shifts"])
            for record in records
        )
        shifts = torch.zeros(len(records), max_length, dtype=torch.float32)
        counts = torch.zeros(len(records), max_length, dtype=torch.float32)
        types = torch.zeros(len(records), max_length, dtype=torch.long)
        padding_mask = torch.ones(len(records), max_length, dtype=torch.bool)

        for batch_index, record in enumerate(records):
            h_shifts = record["h_shifts"]
            c_shifts = record["c_shifts"]
            all_shifts = h_shifts + c_shifts
            sequence_length = len(all_shifts)
            shifts[batch_index, :sequence_length] = torch.tensor(all_shifts)
            counts[batch_index, : len(h_shifts)] = 1.0
            types[batch_index, len(h_shifts) : sequence_length] = 1
            padding_mask[batch_index, :sequence_length] = False

        return ModelBatch(
            record_ids=[record["record_id"] for record in records],
            inputs={
                "shifts": shifts,
                "counts": counts,
                "types": types,
                "padding_mask": padding_mask,
            },
            metadata={
                "model_name": self.model_name,
                "modality": "1H+13C",
                "processor_mode": self.mode,
                "strict": self.strict,
            },
        )

    def __call__(self, records) -> ModelBatch:
        prepared_records = [self.prepare_record(record) for record in records]
        return self.collate(prepared_records)


__all__ = ["UltraNMRProcessor"]


"""Convert canonical records to fixed-size NMRTrans tensor batches."""

import importlib.util
import json
from pathlib import Path

from data.schema import CanonicalRecord, ProtonPeak, normalize_multiplicity
from data.validation import (
    IncompatibleRecordError,
    ValidationIssue,
    validate_canonical_record,
)

from . import ModelBatch, read_record


# This order is copied from the released NMRTrans model.
MULTIPLICITY_TO_INDEX = {
    "<unk>": 0,
    "m": 1,
    "d": 2,
    "s": 3,
    "dd": 4,
    "t": 5,
    "ddd": 6,
    "q": 7,
    "dt": 8,
    "td": 9,
    "br": 10,
    "ddt": 11,
    "dq": 12,
    "tt": 13,
    "quint": 14,
    "dddd": 15,
    "qd": 16,
    "sept": 17,
    "ddp": 18,
    "ddq": 19,
    "bd": 20,
    "dqd": 21,
}


def _load_official_features(path: Path):
    """Load only the released feature utility, not the full NMRTrans model."""

    if not path.is_file():
        raise FileNotFoundError(f"NMRTrans feature utility not found: {path}")
    specification = importlib.util.spec_from_file_location(
        "_model_benchmarks_nmrtrans_features", path
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"could not load NMRTrans features from {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class NMRTransProcessor:
    """Validate, convert, pad, and collate combined 1H + 13C records."""

    model_name = "nmrtrans"
    max_peaks = 60

    def __init__(
        self,
        mode: str = "canonical",
        strict: bool = True,
        model_root: str | Path | None = None,
    ):
        if mode not in {"canonical", "native"}:
            raise ValueError("mode must be 'canonical' or 'native'")
        self.mode = mode
        self.strict = strict

        project_root = Path(__file__).resolve().parents[3]
        self.model_root = Path(model_root or project_root / "models" / "NMRTrans")
        self._official_features = None

    def _multiplicity(self, peak: ProtonPeak) -> str | None:
        if self.mode == "native" and peak.multiplicity_raw:
            return normalize_multiplicity(peak.multiplicity_raw)
        return peak.multiplicity

    def _model_issues(self, record: CanonicalRecord) -> list[ValidationIssue]:
        issues = []
        if not record.h_nmr_peaks:
            issues.append(
                ValidationIssue(
                    "missing_h_nmr", "h_nmr_peaks", "NMRTrans requires 1H peaks"
                )
            )
        if not record.c_nmr_peaks:
            issues.append(
                ValidationIssue(
                    "missing_c_nmr", "c_nmr_peaks", "NMRTrans requires 13C peaks"
                )
            )
        if record.h_nmr_peaks and len(record.h_nmr_peaks) > self.max_peaks:
            issues.append(
                ValidationIssue(
                    "too_many_h_peaks",
                    "h_nmr_peaks",
                    f"NMRTrans supports at most {self.max_peaks} 1H peaks",
                )
            )
        if record.c_nmr_peaks and len(record.c_nmr_peaks) > self.max_peaks:
            issues.append(
                ValidationIssue(
                    "too_many_c_peaks",
                    "c_nmr_peaks",
                    f"NMRTrans supports at most {self.max_peaks} 13C peaks",
                )
            )

        for index, peak in enumerate(record.h_nmr_peaks or ()):
            prefix = f"h_nmr_peaks[{index}]"
            if (
                not isinstance(peak.integration, int)
                or isinstance(peak.integration, bool)
                or peak.integration <= 0
            ):
                issues.append(
                    ValidationIssue(
                        "missing_integration",
                        f"{prefix}.integration",
                        "NMRTrans requires a positive integer integration",
                    )
                )
            if self._multiplicity(peak) is None:
                issues.append(
                    ValidationIssue(
                        "missing_multiplicity",
                        f"{prefix}.multiplicity",
                        "NMRTrans requires multiplicity; use '<unk>' when "
                        "reported but unsupported",
                    )
                )
            if peak.j_values is None:
                issues.append(
                    ValidationIssue(
                        "missing_j_values",
                        f"{prefix}.j_values",
                        "NMRTrans cannot distinguish unavailable J values "
                        "from padded zeros",
                    )
                )
            elif len(peak.j_values) > 6:
                issues.append(
                    ValidationIssue(
                        "too_many_j_values",
                        f"{prefix}.j_values",
                        "NMRTrans supports at most six J values per proton peak",
                    )
                )
            if peak.range_half_span is None and (
                peak.range_min is None or peak.range_max is None
            ):
                issues.append(
                    ValidationIssue(
                        "missing_peak_width",
                        f"{prefix}.range_half_span",
                        "NMRTrans requires a range half-span or both range endpoints",
                    )
                )
        return issues

    def prepare_record(self, value: object) -> dict:
        """Validate one record and build the released JSON-like input."""

        record = read_record(value)
        issues = []
        if self.strict:
            issues.extend(validate_canonical_record(record).issues)
        issues.extend(self._model_issues(record))
        if issues:
            raise IncompatibleRecordError(record.record_id, issues)

        h_peaks = []
        for peak in record.h_nmr_peaks or ():
            half_span = peak.range_half_span
            if half_span is None:
                half_span = (peak.range_max - peak.range_min) / 2.0
            h_peaks.append(
                [
                    float(peak.shift),
                    float(half_span),
                    self._multiplicity(peak),
                    f"{peak.integration}H",
                    [float(number) for number in peak.j_values or ()],
                ]
            )

        c_shifts = [float(peak.shift) for peak in record.c_nmr_peaks or ()]
        return {
            "record_id": record.record_id,
            "tokenized_input": json.dumps({"1HNMR": h_peaks, "13CNMR": c_shifts}),
            "h_peak_count": len(h_peaks),
            "c_shifts": c_shifts,
        }

    def collate(self, records: list[dict]) -> ModelBatch:
        if not records:
            raise ValueError("cannot collate an empty NMRTrans batch")

        import torch

        if self._official_features is None:
            self._official_features = _load_official_features(
                self.model_root / "src" / "features.py"
            )
        official = self._official_features

        batch_size = len(records)
        h_features = torch.zeros(batch_size, self.max_peaks, 10, dtype=torch.float32)
        h_mask = torch.zeros(batch_size, self.max_peaks, dtype=torch.long)
        c_peaks = torch.zeros(batch_size, self.max_peaks, 1, dtype=torch.float32)
        c_mask = torch.zeros(batch_size, self.max_peaks, dtype=torch.long)

        for batch_index, record in enumerate(records):
            record_h_features = official.prepare_h_nmr_features(
                record["tokenized_input"], official.DEFAULT_SPLIT_VOCAB
            )
            if len(record_h_features) != record["h_peak_count"]:
                raise RuntimeError(
                    "NMRTrans official preprocessing dropped peaks for "
                    f"record {record['record_id']!r}"
                )

            h_count = len(record_h_features)
            c_count = len(record["c_shifts"])
            h_features[batch_index, :h_count] = torch.tensor(record_h_features)
            h_mask[batch_index, :h_count] = 1

            # The released collator scales carbon shifts to [0, 1].
            normalized_c = [
                max(0.0, min(1.0, shift / 220.0)) for shift in record["c_shifts"]
            ]
            c_peaks[batch_index, :c_count, 0] = torch.tensor(normalized_c)
            c_mask[batch_index, :c_count] = 1

        return ModelBatch(
            record_ids=[record["record_id"] for record in records],
            inputs={
                "h_nmr_features": h_features,
                "h_nmr_mask": h_mask,
                "c_nmr_peaks": c_peaks,
                "c_nmr_mask": c_mask,
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


__all__ = ["MULTIPLICITY_TO_INDEX", "NMRTransProcessor"]


"""Convert canonical records to padded NMRPeak-R token batches."""

from argparse import Namespace
from pathlib import Path

from data.schema import CanonicalRecord, ProtonPeak
from data.validation import (
    IncompatibleRecordError,
    ValidationIssue,
    validate_canonical_record,
)

from . import ModelBatch, read_record


# NMRPeak uses these older names for three canonical categories.
CANONICAL_TO_NMRPEAK = {"quint": "p", "sept": "hept", "bd": "brd"}


class NMRPeakProcessor:
    """Validate, tokenize, pad, and collate combined 1H + 13C records."""

    model_name = "nmrpeak"

    def __init__(
        self,
        mode: str = "canonical",
        strict: bool = True,
        dictionary_path: str | Path | None = None,
        max_sequence_length: int = 256,
    ):
        if mode not in {"canonical", "native"}:
            raise ValueError("mode must be 'canonical' or 'native'")
        self.mode = mode
        self.strict = strict

        project_root = Path(__file__).resolve().parents[3]
        default_dictionary = (
            project_root / "models" / "NMRPeak" / "dict" / "bart" / "total_dict.txt"
        )
        self.dictionary_path = Path(dictionary_path or default_dictionary)
        self.max_sequence_length = max_sequence_length
        self._dictionary = None

    def _multiplicity(self, peak: ProtonPeak) -> str | None:
        if self.mode == "native" and peak.multiplicity_raw:
            return peak.multiplicity_raw
        if peak.multiplicity is None:
            return None
        return CANONICAL_TO_NMRPEAK.get(peak.multiplicity, peak.multiplicity)

    def _model_issues(self, record: CanonicalRecord) -> list[ValidationIssue]:
        issues = []
        if not record.h_nmr_peaks:
            issues.append(
                ValidationIssue(
                    "missing_h_nmr", "h_nmr_peaks", "NMRPeak-R requires 1H peaks"
                )
            )
        if not record.c_nmr_peaks:
            issues.append(
                ValidationIssue(
                    "missing_c_nmr", "c_nmr_peaks", "NMRPeak-R requires 13C peaks"
                )
            )

        for index, peak in enumerate(record.h_nmr_peaks or ()):
            prefix = f"h_nmr_peaks[{index}]"
            if (
                not isinstance(peak.integration, int)
                or isinstance(peak.integration, bool)
                or not 1 <= peak.integration <= 50
            ):
                issues.append(
                    ValidationIssue(
                        "invalid_integration",
                        f"{prefix}.integration",
                        "NMRPeak-R requires an integer integration from 1 to 50",
                    )
                )
            if self._multiplicity(peak) is None:
                issues.append(
                    ValidationIssue(
                        "missing_multiplicity",
                        f"{prefix}.multiplicity",
                        "NMRPeak-R requires multiplicity; use '<unk>' when "
                        "reported but unsupported",
                    )
                )
            if peak.j_values is None:
                issues.append(
                    ValidationIssue(
                        "missing_j_values",
                        f"{prefix}.j_values",
                        "NMRPeak-R requires an available J-value list; "
                        "an empty list is allowed",
                    )
                )
            if peak.range_min is None or peak.range_max is None:
                issues.append(
                    ValidationIssue(
                        "missing_shift_range",
                        f"{prefix}.range_min",
                        "NMRPeak-R requires both proton range endpoints",
                    )
                )
        return issues

    def prepare_record(self, value: object) -> dict:
        """Validate one record and convert it to NMRPeak's fields."""

        record = read_record(value)
        issues = []
        if self.strict:
            issues.extend(validate_canonical_record(record).issues)
        issues.extend(self._model_issues(record))
        if issues:
            raise IncompatibleRecordError(record.record_id, issues)

        h_peaks = []
        for peak in record.h_nmr_peaks or ():
            j_values = "_".join(str(float(number)) for number in peak.j_values or ())
            h_peaks.append(
                {
                    "centroid": float(peak.shift),
                    "rangeMin": float(peak.range_min),
                    "rangeMax": float(peak.range_max),
                    "category": self._multiplicity(peak),
                    "nH": int(peak.integration),
                    "j_values": j_values or "_",
                }
            )

        c_peaks = [
            {"delta (ppm)": float(peak.shift)} for peak in record.c_nmr_peaks or ()
        ]
        return {
            "record_id": record.record_id,
            "h_nmr_peaks": h_peaks,
            "c_nmr_peaks": c_peaks,
        }

    def _load_official_tools(self):
        # Imports stay lazy because NMRPeak has its own environment.
        try:
            from nmrpeak.data import ListDataset, get_spec_dataset
            from unicore.data import Dictionary
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "NMRPeakProcessor must run in the NMRPeak environment "
                "with its repository installed"
            ) from error

        if self._dictionary is None:
            if not self.dictionary_path.is_file():
                raise FileNotFoundError(
                    f"NMRPeak dictionary not found: {self.dictionary_path}"
                )
            self._dictionary = Dictionary.load(str(self.dictionary_path))
            self._dictionary.add_symbol("[MASK]", is_special=True)
        return ListDataset, get_spec_dataset, self._dictionary

    def collate(self, records: list[dict]) -> ModelBatch:
        if not records:
            raise ValueError("cannot collate an empty NMRPeak batch")

        ListDataset, get_spec_dataset, dictionary = self._load_official_tools()
        arguments = Namespace(
            use_cnmr=True,
            use_hnmr=True,
            use_h_nh=True,
            use_h_jvalue=True,
            use_h_category=True,
            use_range_shift=True,
            use_formula=False,
            max_seq_len=512,
        )
        official_records = [
            {
                "h_nmr_peaks": record["h_nmr_peaks"],
                "c_nmr_peaks": record["c_nmr_peaks"],
            }
            for record in records
        ]
        dataset = get_spec_dataset(ListDataset(official_records), arguments, dictionary)

        samples = []
        for index, record in enumerate(records):
            sample = dataset[index]
            sequence_length = int(sample["spec.src_tokens"].numel())
            if sequence_length > self.max_sequence_length:
                issue = ValidationIssue(
                    "sequence_too_long",
                    "h_nmr_peaks/c_nmr_peaks",
                    f"NMRPeak token sequence has {sequence_length} positions; "
                    f"maximum is {self.max_sequence_length}",
                )
                raise IncompatibleRecordError(record["record_id"], [issue])
            samples.append(sample)

        official_batch = dataset.collater(samples)
        src_tokens = official_batch["spec"]["src_tokens"]
        return ModelBatch(
            record_ids=[record["record_id"] for record in records],
            inputs={
                "src_tokens": src_tokens,
                "attention_mask": src_tokens.ne(dictionary.pad()),
            },
            metadata={
                "model_name": self.model_name,
                "modality": "1H+13C",
                "processor_mode": self.mode,
                "strict": self.strict,
                "dictionary": str(self.dictionary_path),
                "pad_idx": dictionary.pad(),
                "max_sequence_length": self.max_sequence_length,
            },
        )

    def __call__(self, records) -> ModelBatch:
        prepared_records = [self.prepare_record(record) for record in records]
        return self.collate(prepared_records)


__all__ = ["NMRPeakProcessor"]


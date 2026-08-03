"""Validation rules shared by every consumer of canonical NMR records."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .schema import (
    CANONICAL_MULTIPLICITIES,
    CanonicalRecord,
    CarbonPeak,
    ProtonPeak,
    ensure_record,
)


@dataclass
class ValidationIssue:
    code: str
    path: str
    message: str


@dataclass
class ValidationResult:
    record_id: str
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.issues


class IncompatibleRecordError(ValueError):
    """Raised when a canonical record cannot be used safely."""

    def __init__(self, record_id: str, issues: Sequence[ValidationIssue]):
        self.record_id = record_id
        self.issues = list(issues)

        details = "; ".join(
            f"{issue.path}: {issue.message}" for issue in self.issues
        )
        message = f"record {record_id!r} is incompatible"
        if details:
            message += f": {details}"
        super().__init__(message)


def _is_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, float) and math.isfinite(value)


def _check_number(
    value: object,
    path: str,
    issues: list[ValidationIssue],
    optional: bool = False,
) -> bool:
    if value is None and optional:
        return False
    if not _is_number(value):
        issues.append(
            ValidationIssue(
                code="invalid_number",
                path=path,
                message="must be a finite JSON number",
            )
        )
        return False
    return True


def _validate_proton_peak(
    peak: ProtonPeak,
    path: str,
    issues: list[ValidationIssue],
) -> None:
    shift_is_valid = _check_number(peak.shift, f"{path}.shift", issues)

    if peak.integration is not None and (
        not isinstance(peak.integration, int)
        or isinstance(peak.integration, bool)
        or peak.integration < 0
    ):
        issues.append(
            ValidationIssue(
                code="invalid_integration",
                path=f"{path}.integration",
                message="must be a non-negative integer or None",
            )
        )

    if peak.multiplicity_raw is not None and not isinstance(
        peak.multiplicity_raw, str
    ):
        issues.append(
            ValidationIssue(
                code="invalid_type",
                path=f"{path}.multiplicity_raw",
                message="must be a string or None",
            )
        )

    allowed_multiplicities = set(CANONICAL_MULTIPLICITIES) | {"<unk>"}
    if peak.multiplicity is not None and (
        not isinstance(peak.multiplicity, str)
        or peak.multiplicity not in allowed_multiplicities
    ):
        issues.append(
            ValidationIssue(
                code="invalid_multiplicity",
                path=f"{path}.multiplicity",
                message="must use the canonical vocabulary, <unk>, or None",
            )
        )

    if peak.j_values is not None:
        if not isinstance(peak.j_values, tuple):
            issues.append(
                ValidationIssue(
                    code="invalid_type",
                    path=f"{path}.j_values",
                    message="must be an array or None",
                )
            )
        else:
            for index, value in enumerate(peak.j_values):
                _check_number(value, f"{path}.j_values[{index}]", issues)

    range_min_is_valid = _check_number(
        peak.range_min, f"{path}.range_min", issues, optional=True
    )
    range_max_is_valid = _check_number(
        peak.range_max, f"{path}.range_max", issues, optional=True
    )
    half_span_is_valid = _check_number(
        peak.range_half_span,
        f"{path}.range_half_span",
        issues,
        optional=True,
    )

    if (peak.range_min is None) != (peak.range_max is None):
        issues.append(
            ValidationIssue(
                code="incomplete_range",
                path=path,
                message="range_min and range_max must be supplied together",
            )
        )

    if range_min_is_valid and range_max_is_valid:
        if peak.range_min > peak.range_max:
            issues.append(
                ValidationIssue(
                    code="invalid_range",
                    path=path,
                    message="range_min must be less than or equal to range_max",
                )
            )
        elif shift_is_valid and not peak.range_min <= peak.shift <= peak.range_max:
            issues.append(
                ValidationIssue(
                    code="shift_outside_range",
                    path=f"{path}.shift",
                    message="shift must lie between range_min and range_max",
                )
            )

    if half_span_is_valid and peak.range_half_span < 0:
        issues.append(
            ValidationIssue(
                code="invalid_range",
                path=f"{path}.range_half_span",
                message="must be non-negative",
            )
        )

    if range_min_is_valid and range_max_is_valid and half_span_is_valid:
        expected_half_span = (peak.range_max - peak.range_min) / 2
        if not math.isclose(
            peak.range_half_span,
            expected_half_span,
            rel_tol=1e-6,
            abs_tol=1e-8,
        ):
            issues.append(
                ValidationIssue(
                    code="inconsistent_range",
                    path=f"{path}.range_half_span",
                    message="must equal (range_max - range_min) / 2",
                )
            )

    if peak.equivalence_class is not None and (
        not isinstance(peak.equivalence_class, int)
        or isinstance(peak.equivalence_class, bool)
    ):
        issues.append(
            ValidationIssue(
                code="invalid_type",
                path=f"{path}.equivalence_class",
                message="must be an integer or None",
            )
        )

    if peak.member_shifts is not None:
        if not isinstance(peak.member_shifts, tuple):
            issues.append(
                ValidationIssue(
                    code="invalid_type",
                    path=f"{path}.member_shifts",
                    message="must be an array or None",
                )
            )
        else:
            for index, value in enumerate(peak.member_shifts):
                _check_number(value, f"{path}.member_shifts[{index}]", issues)


def _validate_carbon_peak(
    peak: CarbonPeak,
    path: str,
    issues: list[ValidationIssue],
) -> None:
    _check_number(peak.shift, f"{path}.shift", issues)
    for field_name in ("integral", "intensity", "width"):
        _check_number(
            getattr(peak, field_name),
            f"{path}.{field_name}",
            issues,
            optional=True,
        )


def validate_canonical_record(
    value: CanonicalRecord | Mapping[str, object],
) -> ValidationResult:
    """Check common schema rules without imposing model-specific requirements."""

    record = ensure_record(value)
    issues: list[ValidationIssue] = []

    if not isinstance(record.record_id, str) or not record.record_id.strip():
        issues.append(
            ValidationIssue(
                code="invalid_record_id",
                path="record_id",
                message="must be a non-empty string",
            )
        )

    optional_strings = (
        "source",
        "smiles",
        "smiles_canonical",
        "molecular_formula",
        "nmr_frequency",
        "nmr_solvent",
    )
    for field_name in optional_strings:
        field_value = getattr(record, field_name)
        if field_value is not None and not isinstance(field_value, str):
            issues.append(
                ValidationIssue(
                    code="invalid_type",
                    path=field_name,
                    message="must be a string or None",
                )
            )

    for field_name in ("atoms", "coordinates"):
        field_value = getattr(record, field_name)
        if field_value is not None and not isinstance(field_value, tuple):
            issues.append(
                ValidationIssue(
                    code="invalid_type",
                    path=field_name,
                    message="must be an array or None",
                )
            )

    if record.h_nmr_peaks is not None:
        if not isinstance(record.h_nmr_peaks, tuple):
            issues.append(
                ValidationIssue(
                    code="invalid_type",
                    path="h_nmr_peaks",
                    message="must be an array or None",
                )
            )
        else:
            for index, peak in enumerate(record.h_nmr_peaks):
                if not isinstance(peak, ProtonPeak):
                    issues.append(
                        ValidationIssue(
                            code="invalid_type",
                            path=f"h_nmr_peaks[{index}]",
                            message="must be a ProtonPeak",
                        )
                    )
                    continue
                _validate_proton_peak(peak, f"h_nmr_peaks[{index}]", issues)

    if record.c_nmr_peaks is not None:
        if not isinstance(record.c_nmr_peaks, tuple):
            issues.append(
                ValidationIssue(
                    code="invalid_type",
                    path="c_nmr_peaks",
                    message="must be an array or None",
                )
            )
        else:
            for index, peak in enumerate(record.c_nmr_peaks):
                if not isinstance(peak, CarbonPeak):
                    issues.append(
                        ValidationIssue(
                            code="invalid_type",
                            path=f"c_nmr_peaks[{index}]",
                            message="must be a CarbonPeak",
                        )
                    )
                    continue
                _validate_carbon_peak(peak, f"c_nmr_peaks[{index}]", issues)

    if isinstance(record.record_id, str):
        record_id = record.record_id
    else:
        record_id = "<unknown>"

    return ValidationResult(record_id=record_id, issues=issues)

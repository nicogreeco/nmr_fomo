"""Reusable canonical NMR records, validation, and dataset readers."""

from .dataset import (
    CanonicalNMRDataset,
    CanonicalParquetDataset,
    JsonlCanonicalReader,
)
from .schema import (
    CANONICAL_MULTIPLICITIES,
    CanonicalRecord,
    CarbonPeak,
    ProtonPeak,
    ensure_record,
    normalize_multiplicity,
)
from .validation import (
    IncompatibleRecordError,
    ValidationIssue,
    ValidationResult,
    validate_canonical_record,
)

__all__ = [
    "CANONICAL_MULTIPLICITIES",
    "CanonicalNMRDataset",
    "CanonicalParquetDataset",
    "CanonicalRecord",
    "CarbonPeak",
    "IncompatibleRecordError",
    "JsonlCanonicalReader",
    "ProtonPeak",
    "ValidationIssue",
    "ValidationResult",
    "ensure_record",
    "normalize_multiplicity",
    "validate_canonical_record",
]

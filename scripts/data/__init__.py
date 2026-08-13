"""Reusable canonical NMR records, validation, and dataset readers."""
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


def __getattr__(name: str):
    """Load PyTorch-backed dataset readers only when they are requested."""

    if name not in {
        "CanonicalNMRDataset",
        "CanonicalParquetDataset",
        "JsonlCanonicalReader",
    }:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from .dataset import (
        CanonicalNMRDataset,
        CanonicalParquetDataset,
        JsonlCanonicalReader,
    )

    readers = {
        "CanonicalNMRDataset": CanonicalNMRDataset,
        "CanonicalParquetDataset": CanonicalParquetDataset,
        "JsonlCanonicalReader": JsonlCanonicalReader,
    }
    globals().update(readers)
    return readers[name]

"""Input preparation for the new NMR foundation model."""

from .dataset import PairedFoundationDataset, PairedFoundationRecord
from .processor import (
    FoundationNMRProcessor,
    H_AVAILABILITY_FIELDS,
    ID_TO_MULTIPLICITY,
    MAX_C_PEAKS,
    MAX_H_PEAKS,
    MAX_J_VALUES,
    MULTIPLICITY_TO_ID,
)


__all__ = [
    "PairedFoundationDataset",
    "PairedFoundationRecord",
    "FoundationNMRProcessor",
    "H_AVAILABILITY_FIELDS",
    "ID_TO_MULTIPLICITY",
    "MAX_C_PEAKS",
    "MAX_H_PEAKS",
    "MAX_J_VALUES",
    "MULTIPLICITY_TO_ID",
]

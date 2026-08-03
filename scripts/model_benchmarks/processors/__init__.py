"""Shared batch container for the four model-specific processors."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from data.schema import CanonicalRecord, ensure_record
from data.validation import IncompatibleRecordError, ValidationIssue


@dataclass
class ModelBatch:
    """Model inputs and their ordered canonical record identifiers."""

    record_ids: list[str]
    inputs: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


def read_record(value: object) -> CanonicalRecord:
    """Convert one dataset item to a canonical record with a clear error."""

    try:
        return ensure_record(value)
    except (TypeError, ValueError) as error:
        record_id = "<unknown>"
        if isinstance(value, Mapping):
            record_id = str(value.get("record_id", record_id))
        issue = ValidationIssue("invalid_schema", "record", str(error))
        raise IncompatibleRecordError(record_id, [issue]) from error


__all__ = ["ModelBatch"]


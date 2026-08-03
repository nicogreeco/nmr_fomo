"""Model-independent Python representation of one canonical NMR record."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


CANONICAL_MULTIPLICITIES = (
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

MULTIPLICITY_ALIASES = {
    "p": "quint",
    "hept": "sept",
    "brd": "bd",
}


def normalize_multiplicity(value: str | None) -> str | None:
    """Return a canonical multiplicity without guessing unknown labels.

    ``None`` means that no annotation was available. ``<unk>`` means that an
    annotation was present but is not in the canonical vocabulary.
    """

    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("multiplicity must be a string or None")

    normalized = value.strip().lower()
    normalized = MULTIPLICITY_ALIASES.get(normalized, normalized)
    if normalized in CANONICAL_MULTIPLICITIES or normalized == "<unk>":
        return normalized
    return "<unk>"


@dataclass
class ProtonPeak:
    shift: float
    integration: int | None = None
    multiplicity_raw: str | None = None
    multiplicity: str | None = None
    j_values: tuple[float, ...] | None = None
    range_min: float | None = None
    range_max: float | None = None
    range_half_span: float | None = None
    equivalence_class: int | None = None
    member_shifts: tuple[float, ...] | None = None


@dataclass
class CarbonPeak:
    shift: float
    integral: float | None = None
    intensity: float | None = None
    width: float | None = None


@dataclass
class CanonicalRecord:
    record_id: str
    source: str | None = None
    smiles: str | None = None
    smiles_canonical: str | None = None
    molecular_formula: str | None = None
    nmr_frequency: str | None = None
    nmr_solvent: str | None = None
    atoms: tuple[Any, ...] | None = None
    coordinates: tuple[Any, ...] | None = None
    h_nmr_peaks: tuple[ProtonPeak, ...] | None = None
    c_nmr_peaks: tuple[CarbonPeak, ...] | None = None


def _optional_tuple(value: Any, field_name: str) -> tuple[Any, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field_name} must be an array or None")
    return tuple(value)


def _proton_peak(value: Any) -> ProtonPeak:
    if isinstance(value, ProtonPeak):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("each h_nmr_peaks item must be an object")

    peak_data = dict(value)
    if "j_values" in peak_data:
        peak_data["j_values"] = _optional_tuple(peak_data["j_values"], "j_values")
    if "member_shifts" in peak_data:
        peak_data["member_shifts"] = _optional_tuple(
            peak_data["member_shifts"], "member_shifts"
        )
    return ProtonPeak(**peak_data)


def _carbon_peak(value: Any) -> CarbonPeak:
    if isinstance(value, CarbonPeak):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("each c_nmr_peaks item must be an object")
    return CarbonPeak(**dict(value))


def ensure_record(value: CanonicalRecord | Mapping[str, Any]) -> CanonicalRecord:
    """Return a canonical record from a dataclass or JSON-like dictionary.

    JSON arrays are stored as tuples. Scientific values are copied as supplied;
    in particular, numeric strings are not silently converted to numbers.
    """

    if isinstance(value, CanonicalRecord):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("canonical record must be a CanonicalRecord or mapping")

    record_data = dict(value)

    if "h_nmr_peaks" in record_data:
        peaks = _optional_tuple(record_data["h_nmr_peaks"], "h_nmr_peaks")
        if peaks is not None:
            record_data["h_nmr_peaks"] = tuple(_proton_peak(peak) for peak in peaks)

    if "c_nmr_peaks" in record_data:
        peaks = _optional_tuple(record_data["c_nmr_peaks"], "c_nmr_peaks")
        if peaks is not None:
            record_data["c_nmr_peaks"] = tuple(_carbon_peak(peak) for peak in peaks)

    for field_name in ("atoms", "coordinates"):
        if field_name in record_data:
            record_data[field_name] = _optional_tuple(
                record_data[field_name], field_name
            )

    return CanonicalRecord(**record_data)

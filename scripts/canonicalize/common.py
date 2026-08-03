"""Small shared helpers for the raw-to-canonical converters."""

import math
import pickle
import re
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from data.schema import CanonicalRecord, ensure_record, normalize_multiplicity
from data.validation import validate_canonical_record


CANONICAL_PARQUET_SCHEMA_VERSION = "1"


class ConversionError(ValueError):
    """Raised when one source record cannot be represented canonically."""


def require_mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConversionError(f"{location} must be a dictionary")
    return value


def _to_python_value(value: Any) -> Any:
    """Turn NumPy-style values into ordinary Python values when needed."""

    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def finite_float(value: object, location: str) -> float:
    """Read one finite source number without accepting booleans."""

    value = _to_python_value(value)
    if isinstance(value, bool):
        raise ConversionError(f"{location} must be a finite number, not a boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ConversionError(f"{location} must be a finite number") from error
    if not math.isfinite(number):
        raise ConversionError(f"{location} must be a finite number")
    return number


def optional_float(value: object, location: str) -> float | None:
    if value is None:
        return None
    return finite_float(value, location)


def non_negative_integer(value: object, location: str) -> int | None:
    """Read an integration such as ``0``, ``2``, or ``2H``."""

    if value is None:
        return None
    value = _to_python_value(value)
    if isinstance(value, bool):
        raise ConversionError(f"{location} must be a non-negative integer or None")

    if isinstance(value, int):
        integration = value
    elif isinstance(value, float):
        if not value.is_integer():
            raise ConversionError(f"{location} must be a non-negative integer or None")
        integration = int(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        match = re.fullmatch(r"(\d+)(?:\.0+)?\s*[Hh]?", text)
        if match is None:
            raise ConversionError(
                f"{location} must look like a non-negative integer or '2H'"
            )
        integration = int(match.group(1))
    else:
        raise ConversionError(f"{location} must be a non-negative integer or None")

    if integration < 0:
        raise ConversionError(f"{location} must be a non-negative integer")
    return integration


def optional_text(value: object, location: str) -> str | None:
    if value is None:
        return None
    value = _to_python_value(value)
    if not isinstance(value, str):
        raise ConversionError(f"{location} must be text or None")
    text = value.strip()
    return text or None


def metadata_text(value: object, location: str) -> str | None:
    """Keep reported metadata as text, including numeric spectrometer values."""

    if value is None:
        return None
    value = _to_python_value(value)
    if isinstance(value, bool):
        raise ConversionError(f"{location} must be text or None")
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, (int, float)):
        return str(value)
    raise ConversionError(f"{location} must be text or None")


def atom_list(value: object, location: str) -> list[str] | None:
    """Store source atom labels in Arrow's simple list-of-text representation."""

    if value is None:
        return None
    values = _to_python_value(value)
    if not isinstance(values, (list, tuple)):
        raise ConversionError(f"{location} must be an array or None")

    atoms: list[str] = []
    for index, atom in enumerate(values):
        if atom is None:
            raise ConversionError(f"{location}[{index}] must not be None")
        atoms.append(str(atom))
    return atoms


def coordinate_list(value: object, location: str) -> list[list[float]] | None:
    if value is None:
        return None
    values = _to_python_value(value)
    if not isinstance(values, (list, tuple)):
        raise ConversionError(f"{location} must be an array or None")

    coordinates: list[list[float]] = []
    for atom_index, atom_coordinates in enumerate(values):
        atom_coordinates = _to_python_value(atom_coordinates)
        if not isinstance(atom_coordinates, (list, tuple)):
            raise ConversionError(f"{location}[{atom_index}] must be an array")
        coordinates.append(
            [
                finite_float(component, f"{location}[{atom_index}][{axis_index}]")
                for axis_index, component in enumerate(atom_coordinates)
            ]
        )
    return coordinates


def parse_nmrpeak_j_values(value: object, location: str) -> list[float] | None:
    """Parse NMRPeak's underscore/comma-delimited J-value field.

    An absent field remains ``None``. The release's ``_`` marker means the
    field was present but contained no numerical coupling values, so it becomes
    an empty list.
    """

    if value is None:
        return None
    values = _to_python_value(value)
    if isinstance(values, (list, tuple)):
        return [
            finite_float(item, f"{location}[{index}]")
            for index, item in enumerate(values)
        ]
    if isinstance(values, str):
        text = values.strip()
        if text in {"", "_"}:
            return []
        number_texts = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text)
        if not number_texts:
            raise ConversionError(f"{location} contains no numerical J values")
        return [
            finite_float(number_text, f"{location}[{index}]")
            for index, number_text in enumerate(number_texts)
        ]
    return [finite_float(values, location)]


def parse_nmrtrans_j_values(value: object, location: str) -> list[float] | None:
    """Read NMRTrans's already-list-valued J couplings without inventing data."""

    if value is None:
        return None
    values = _to_python_value(value)
    if not isinstance(values, (list, tuple)):
        raise ConversionError(f"{location} must be an array or None")
    return [
        finite_float(item, f"{location}[{index}]")
        for index, item in enumerate(values)
    ]


def nmrpeak_proton_peak(raw_peak: object, location: str) -> dict[str, object]:
    """Map one NMRPeak/NMRexp or NMRPeak/MST proton peak."""

    peak = require_mapping(raw_peak, location)
    range_min_raw = peak.get("rangeMin")
    range_max_raw = peak.get("rangeMax")

    if (range_min_raw is None) != (range_max_raw is None):
        raise ConversionError(
            f"{location} must contain both rangeMin and rangeMax or neither"
        )

    if range_min_raw is None:
        range_min = None
        range_max = None
    else:
        range_min = finite_float(range_min_raw, f"{location}.rangeMin")
        range_max = finite_float(range_max_raw, f"{location}.rangeMax")
        if range_min > range_max:
            raise ConversionError(f"{location}.rangeMin must not exceed rangeMax")

    shift_raw = peak.get("centroid")
    if shift_raw is None:
        shift_raw = peak.get("delta")
    if shift_raw is None:
        if range_min is None:
            raise ConversionError(
                f"{location} needs centroid, delta, or both range endpoints"
            )
        shift = (range_min + range_max) / 2.0
    else:
        shift = finite_float(shift_raw, f"{location}.centroid")

    if range_min is None:
        # NMRPeak's tokenizer uses a point range when only a single shift exists.
        range_min = shift
        range_max = shift
    range_half_span = (range_max - range_min) / 2.0

    multiplicity_raw = optional_text(peak.get("category"), f"{location}.category")
    return {
        "shift": shift,
        "integration": non_negative_integer(peak.get("nH"), f"{location}.nH"),
        "multiplicity_raw": multiplicity_raw,
        "multiplicity": normalize_multiplicity(multiplicity_raw),
        "j_values": parse_nmrpeak_j_values(peak.get("j_values"), f"{location}.j_values"),
        "range_min": range_min,
        "range_max": range_max,
        "range_half_span": range_half_span,
        "equivalence_class": None,
        "member_shifts": None,
    }


def nmrpeak_carbon_peak(
    raw_peak: object,
    location: str,
    *,
    keep_properties: bool,
) -> dict[str, float | None]:
    """Map one NMRPeak carbon peak, optionally keeping MST-only properties."""

    peak = require_mapping(raw_peak, location)
    if "delta (ppm)" not in peak:
        raise ConversionError(f"{location} is missing 'delta (ppm)'")

    if keep_properties:
        integral = optional_float(peak.get("integral"), f"{location}.integral")
        intensity = optional_float(peak.get("intensity"), f"{location}.intensity")
        width = optional_float(
            peak.get("width (ppm)"), f"{location}.width (ppm)"
        )
    else:
        integral = None
        intensity = None
        width = None

    return {
        "shift": finite_float(peak["delta (ppm)"], f"{location}.delta (ppm)"),
        "integral": integral,
        "intensity": intensity,
        "width": width,
    }


def nmrpeak_metadata(raw_record: object, location: str) -> dict[str, object]:
    """Copy the common NMRPeak structure and acquisition metadata."""

    record = require_mapping(raw_record, location)
    return {
        "smiles": optional_text(record.get("smiles"), f"{location}.smiles"),
        "smiles_canonical": optional_text(
            record.get("smiles_canonical"), f"{location}.smiles_canonical"
        ),
        "molecular_formula": optional_text(
            record.get("molecular_formula"), f"{location}.molecular_formula"
        ),
        "nmr_frequency": metadata_text(
            record.get("nmr_frequency"), f"{location}.nmr_frequency"
        ),
        "nmr_solvent": metadata_text(
            record.get("nmr_solvent"), f"{location}.nmr_solvent"
        ),
        "atoms": atom_list(record.get("atoms"), f"{location}.atoms"),
        "coordinates": coordinate_list(
            record.get("coordinates"), f"{location}.coordinates"
        ),
    }


def checked_canonical_record(
    record_data: Mapping[str, object], location: str
) -> CanonicalRecord:
    """Create and validate a canonical record before it is written."""

    try:
        record = ensure_record(record_data)
    except (TypeError, ValueError) as error:
        raise ConversionError(f"{location}: invalid canonical fields: {error}") from error

    result = validate_canonical_record(record)
    if result.is_valid:
        return record

    details = "; ".join(
        f"{issue.path}: {issue.message}" for issue in result.issues
    )
    raise ConversionError(f"{location}: canonical validation failed: {details}")


def canonical_record_to_row(record: CanonicalRecord) -> dict[str, object]:
    """Convert one validated data class to the Arrow nested-row representation."""

    return {
        "record_id": record.record_id,
        "source": record.source,
        "smiles": record.smiles,
        "smiles_canonical": record.smiles_canonical,
        "molecular_formula": record.molecular_formula,
        "nmr_frequency": record.nmr_frequency,
        "nmr_solvent": record.nmr_solvent,
        "atoms": list(record.atoms) if record.atoms is not None else None,
        "coordinates": (
            [list(coordinates) for coordinates in record.coordinates]
            if record.coordinates is not None
            else None
        ),
        "h_nmr_peaks": (
            [
                {
                    "shift": peak.shift,
                    "integration": peak.integration,
                    "multiplicity_raw": peak.multiplicity_raw,
                    "multiplicity": peak.multiplicity,
                    "j_values": list(peak.j_values)
                    if peak.j_values is not None
                    else None,
                    "range_min": peak.range_min,
                    "range_max": peak.range_max,
                    "range_half_span": peak.range_half_span,
                    "equivalence_class": peak.equivalence_class,
                    "member_shifts": list(peak.member_shifts)
                    if peak.member_shifts is not None
                    else None,
                }
                for peak in record.h_nmr_peaks
            ]
            if record.h_nmr_peaks is not None
            else None
        ),
        "c_nmr_peaks": (
            [
                {
                    "shift": peak.shift,
                    "integral": peak.integral,
                    "intensity": peak.intensity,
                    "width": peak.width,
                }
                for peak in record.c_nmr_peaks
            ]
            if record.c_nmr_peaks is not None
            else None
        ),
    }


def canonical_parquet_schema():
    """Return the Arrow schema shared by every converter output."""

    try:
        import pyarrow as pa
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "writing canonical Parquet requires pyarrow; install it in the "
            "environment used for conversion"
        ) from error

    proton_peak = pa.struct(
        [
            pa.field("shift", pa.float64(), nullable=False),
            pa.field("integration", pa.int64()),
            pa.field("multiplicity_raw", pa.string()),
            pa.field("multiplicity", pa.string()),
            pa.field("j_values", pa.list_(pa.float64())),
            pa.field("range_min", pa.float64()),
            pa.field("range_max", pa.float64()),
            pa.field("range_half_span", pa.float64()),
            pa.field("equivalence_class", pa.int64()),
            pa.field("member_shifts", pa.list_(pa.float64())),
        ]
    )
    carbon_peak = pa.struct(
        [
            pa.field("shift", pa.float64(), nullable=False),
            pa.field("integral", pa.float64()),
            pa.field("intensity", pa.float64()),
            pa.field("width", pa.float64()),
        ]
    )
    return pa.schema(
        [
            pa.field("record_id", pa.string(), nullable=False),
            pa.field("source", pa.string()),
            pa.field("smiles", pa.string()),
            pa.field("smiles_canonical", pa.string()),
            pa.field("molecular_formula", pa.string()),
            pa.field("nmr_frequency", pa.string()),
            pa.field("nmr_solvent", pa.string()),
            pa.field("atoms", pa.list_(pa.string())),
            pa.field("coordinates", pa.list_(pa.list_(pa.float64()))),
            pa.field("h_nmr_peaks", pa.list_(proton_peak)),
            pa.field("c_nmr_peaks", pa.list_(carbon_peak)),
        ]
    )


def write_canonical_parquet(
    records: Iterable[CanonicalRecord],
    output_path: str | Path,
    *,
    source_name: str,
    converter_name: str,
    row_group_size: int = 50000,
    overwrite: bool = False,
) -> int:
    """Write one physical Parquet file in bounded batches.

    Row groups are internal to one Parquet file; they do not create shards or
    split the dataset. A temporary file prevents a failed run from replacing a
    completed output.
    """

    if row_group_size < 1:
        raise ValueError("row_group_size must be at least 1")

    try:
        import pyarrow as pa
        import pyarrow.parquet as parquet
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "writing canonical Parquet requires pyarrow; install it in the "
            "environment used for conversion"
        ) from error

    output = Path(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite {output}; pass --overwrite to replace it"
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    temporary_output = output.with_name(f".{output.name}.partial")
    if temporary_output.exists():
        raise FileExistsError(
            f"temporary output already exists: {temporary_output}; remove it "
            "after checking the interrupted conversion"
        )

    metadata = {
        b"canonical_schema_version": CANONICAL_PARQUET_SCHEMA_VERSION.encode(),
        b"source_dataset": source_name.encode(),
        b"converter": converter_name.encode(),
    }
    schema = canonical_parquet_schema().with_metadata(metadata)
    writer = parquet.ParquetWriter(temporary_output, schema, compression="zstd")

    record_count = 0
    rows: list[dict[str, object]] = []
    try:
        for record in records:
            rows.append(canonical_record_to_row(record))
            if len(rows) >= row_group_size:
                table = pa.Table.from_pylist(rows, schema=schema)
                writer.write_table(table, row_group_size=row_group_size)
                record_count += len(rows)
                rows = []

        if rows:
            table = pa.Table.from_pylist(rows, schema=schema)
            writer.write_table(table, row_group_size=row_group_size)
            record_count += len(rows)
    except Exception:
        writer.close()
        raise

    writer.close()
    temporary_output.replace(output)
    return record_count


def iter_lmdb_records(
    path: str | Path, *, readahead: bool = False
) -> Iterator[tuple[bytes, Mapping[str, Any]]]:
    """Yield LMDB rows in key order, optionally enabling sequential readahead."""

    try:
        import lmdb
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "reading source LMDB files requires lmdb; install it in the "
            "environment used for conversion"
        ) from error

    lmdb_path = Path(path)
    if not lmdb_path.is_file():
        raise FileNotFoundError(f"LMDB input not found: {lmdb_path}")

    environment = lmdb.open(
        str(lmdb_path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=readahead,
        meminit=False,
        max_readers=256,
    )
    try:
        with environment.begin() as transaction:
            cursor = transaction.cursor()
            for key, value in cursor:
                key_bytes = bytes(key)
                try:
                    raw_record = pickle.loads(bytes(value))
                except Exception as error:
                    raise ConversionError(
                        f"could not unpickle {lmdb_path} key {key_bytes.hex()}"
                    ) from error
                yield key_bytes, require_mapping(
                    raw_record, f"{lmdb_path} key {key_bytes.hex()}"
                )
    finally:
        environment.close()


def lmdb_key_text(key: bytes) -> str:
    """Use an LMDB key in a stable text ``record_id`` component."""

    try:
        text = key.decode("ascii")
    except UnicodeDecodeError:
        text = key.hex()
    return text or key.hex()


def resolve_lmdb_inputs(
    input_path: str | Path,
    *,
    prefer_all_file: bool,
) -> list[Path]:
    """Accept an LMDB file, its directory, or the parent dataset directory."""

    path = Path(input_path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"LMDB input not found: {path}")

    lmdb_directory = path / "lmdb_dataset"
    if lmdb_directory.is_dir():
        path = lmdb_directory

    all_file = path / "all.lmdb"
    if prefer_all_file and all_file.is_file():
        return [all_file]

    split_files = [
        path / "train.lmdb",
        path / "valid.lmdb",
        path / "test.lmdb",
    ]
    split_files = [split_file for split_file in split_files if split_file.is_file()]
    if split_files:
        return split_files

    raise FileNotFoundError(
        f"no all.lmdb or train/valid/test LMDB files found under {path}"
    )


def iter_lz4_pickle_records(path: str | Path) -> Iterator[Mapping[str, Any]]:
    """Yield every raw record from NMRTrans's compressed pickle release."""

    try:
        import lz4.frame
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "reading NMRTrans .pkl.lz4 files requires lz4; use the NMRTrans "
            "environment or install lz4"
        ) from error

    input_file = Path(path)
    if not input_file.is_file():
        raise FileNotFoundError(f"NMRTrans input not found: {input_file}")

    with lz4.frame.open(input_file, "rb") as source_file:
        pickle_index = 0
        while True:
            try:
                loaded_value = pickle.load(source_file)
            except EOFError:
                break
            except Exception as error:
                raise ConversionError(
                    f"could not read pickle object {pickle_index} from {input_file}"
                ) from error

            if isinstance(loaded_value, (list, tuple)):
                for row_index, raw_record in enumerate(loaded_value):
                    yield require_mapping(
                        raw_record,
                        f"{input_file} pickle object {pickle_index} row {row_index}",
                    )
            else:
                yield require_mapping(
                    loaded_value, f"{input_file} pickle object {pickle_index}"
                )
            pickle_index += 1


def resolve_nmrtrans_inputs(input_path: str | Path) -> list[Path]:
    """Accept one NMRTrans split file or its directory of three split files."""

    path = Path(input_path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"NMRTrans input not found: {path}")

    split_files = [
        path / "train.pkl.lz4",
        path / "val.pkl.lz4",
        path / "test.pkl.lz4",
    ]
    missing_files = [str(split_file) for split_file in split_files if not split_file.is_file()]
    if missing_files:
        raise FileNotFoundError(
            "NMRTrans directory needs train.pkl.lz4, val.pkl.lz4, and "
            "test.pkl.lz4; missing: " + ", ".join(missing_files)
        )
    return split_files


def nmrtrans_file_label(path: str | Path) -> str:
    filename = Path(path).name
    suffix = ".pkl.lz4"
    if filename.endswith(suffix):
        return filename[: -len(suffix)]
    return Path(path).stem

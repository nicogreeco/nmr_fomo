#!/usr/bin/env python3
"""Compare canonical Parquet files with their raw splits and split IDs."""

import argparse
import json
from pathlib import Path

from data.canonicalize.analysis.analyze_parquet import analyze_parquet_file
from data.canonicalize.common import (
    finite_float,
    iter_lmdb_records,
    iter_lz4_pickle_records,
    non_negative_integer,
    parse_nmrpeak_j_values,
    parse_nmrtrans_j_values,
    require_mapping,
)


DATASET_CONFIGS = {
    "mst_nmr": {
        "format": "nmrpeak",
        "source_format": "LMDB containing pickled processed NMRPeak records",
        "splits": {
            "train": Path("models/NMRPeak/data/MST_NMR/lmdb_dataset/train.lmdb"),
            "val": Path("models/NMRPeak/data/MST_NMR/lmdb_dataset/valid.lmdb"),
            "test": Path("models/NMRPeak/data/MST_NMR/lmdb_dataset/test.lmdb"),
        },
    },
    "nmrexp": {
        "format": "nmrpeak",
        "source_format": "LMDB containing pickled processed NMRPeak records",
        "splits": {
            "train": Path("models/NMRPeak/data/NMRexp/lmdb_dataset/train.lmdb"),
            "val": Path("models/NMRPeak/data/NMRexp/lmdb_dataset/valid.lmdb"),
            "test": Path("models/NMRPeak/data/NMRexp/lmdb_dataset/test.lmdb"),
        },
    },
    "nmrtrans": {
        "format": "nmrtrans",
        "source_format": "LZ4-compressed pickle containing NMRTrans/NMRSpec rows",
        "splits": {
            "train": Path("models/NMRTrans/data/train.pkl.lz4"),
            "val": Path("models/NMRTrans/data/val.pkl.lz4"),
            "test": Path("models/NMRTrans/data/test.pkl.lz4"),
        },
    },
}


def write_report(report: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.partial")
    if temporary_path.exists():
        raise FileExistsError(f"temporary report already exists: {temporary_path}")
    temporary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def list_or_none(value: object, location: str) -> list | None:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{location} must be an array or None")
    return list(value)


def iter_raw_records(path: Path, source_format: str):
    if source_format == "nmrpeak":
        for _, record in iter_lmdb_records(path):
            yield record
    else:
        yield from iter_lz4_pickle_records(path)


def raw_spectra(record, source_format: str, location: str):
    if source_format == "nmrpeak":
        return (
            list_or_none(record.get("h_nmr_peaks"), f"{location}.h_nmr_peaks"),
            list_or_none(record.get("c_nmr_peaks"), f"{location}.c_nmr_peaks"),
        )

    tokenized_input = record.get("tokenized_input")
    if isinstance(tokenized_input, str):
        spectra = json.loads(tokenized_input)
    else:
        spectra = tokenized_input
    spectra = require_mapping(spectra, f"{location}.tokenized_input")
    return (
        list_or_none(spectra.get("1HNMR"), f"{location}.1HNMR"),
        list_or_none(spectra.get("13CNMR"), f"{location}.13CNMR"),
    )


def empty_totals() -> dict[str, int]:
    return {
        "rows": 0,
        "h_no_peak_records": 0,
        "c_no_peak_records": 0,
        "both_modality_usable_records": 0,
        "h_total_peaks": 0,
        "c_total_peaks": 0,
        "h_max_peaks": 0,
        "c_max_peaks": 0,
        "zero_integration_peaks": 0,
        "j_null_peaks": 0,
        "j_empty_peaks": 0,
        "j_nonempty_peaks": 0,
    }


def empty_observations() -> dict[str, int]:
    return {
        "raw_missing_j_peaks": 0,
        "zero_integration_records": 0,
        "h_shift_outlier_records": 0,
        "h_shift_outlier_peaks": 0,
        "c_shift_outlier_records": 0,
        "c_shift_outlier_peaks": 0,
        "records_over_60_h_peaks": 0,
        "records_over_60_c_peaks": 0,
    }


def analyze_raw_split(path: Path, source_format: str) -> tuple[dict, dict]:
    totals = empty_totals()
    observations = empty_observations()

    for row_index, record in enumerate(iter_raw_records(path, source_format)):
        location = f"{path} row {row_index}"
        h_peaks, c_peaks = raw_spectra(record, source_format, location)
        totals["rows"] += 1

        if not h_peaks:
            totals["h_no_peak_records"] += 1
        else:
            totals["h_total_peaks"] += len(h_peaks)
            totals["h_max_peaks"] = max(totals["h_max_peaks"], len(h_peaks))
        if not c_peaks:
            totals["c_no_peak_records"] += 1
        else:
            totals["c_total_peaks"] += len(c_peaks)
            totals["c_max_peaks"] = max(totals["c_max_peaks"], len(c_peaks))
        if h_peaks and c_peaks:
            totals["both_modality_usable_records"] += 1
        if h_peaks and len(h_peaks) > 60:
            observations["records_over_60_h_peaks"] += 1
        if c_peaks and len(c_peaks) > 60:
            observations["records_over_60_c_peaks"] += 1

        has_zero_integration = False
        has_h_outlier = False
        for peak_index, raw_peak in enumerate(h_peaks or []):
            peak_location = f"{location}.h_nmr_peaks[{peak_index}]"
            if source_format == "nmrpeak":
                peak = require_mapping(raw_peak, peak_location)
                integration_raw = peak.get("nH")
                j_raw = peak.get("j_values")
                j_values = parse_nmrpeak_j_values(
                    j_raw,
                    f"{peak_location}.j_values",
                )
            else:
                if not isinstance(raw_peak, (list, tuple)) or len(raw_peak) < 5:
                    raise ValueError(f"{peak_location} is not a complete peak")
                integration_raw = raw_peak[3]
                j_raw = raw_peak[4]
                j_values = parse_nmrtrans_j_values(j_raw, f"{peak_location}[4]")
                shift = finite_float(raw_peak[0], f"{peak_location}[0]")
                if not -5 <= shift <= 20:
                    observations["h_shift_outlier_peaks"] += 1
                    has_h_outlier = True

            integration = non_negative_integer(
                integration_raw,
                f"{peak_location}.integration",
            )
            if integration == 0:
                totals["zero_integration_peaks"] += 1
                has_zero_integration = True
            if j_raw is None:
                observations["raw_missing_j_peaks"] += 1
            if j_values:
                totals["j_nonempty_peaks"] += 1
            else:
                totals["j_empty_peaks"] += 1

        if has_zero_integration:
            observations["zero_integration_records"] += 1
        if has_h_outlier:
            observations["h_shift_outlier_records"] += 1

        has_c_outlier = False
        if source_format == "nmrtrans":
            for peak_index, raw_peak in enumerate(c_peaks or []):
                shift = finite_float(
                    raw_peak,
                    f"{location}.c_nmr_peaks[{peak_index}]",
                )
                if not -20 <= shift <= 300:
                    observations["c_shift_outlier_peaks"] += 1
                    has_c_outlier = True
        if has_c_outlier:
            observations["c_shift_outlier_records"] += 1

    return totals, observations


def add_totals(combined: dict, split: dict) -> None:
    for name, value in split.items():
        if name in {"h_max_peaks", "c_max_peaks"}:
            combined[name] = max(combined[name], value)
        else:
            combined[name] += value


def canonical_totals(analysis: dict) -> dict[str, int]:
    return {
        "rows": analysis["rows"],
        "h_no_peak_records": (
            analysis["h_nmr_peaks"]["null_record_count"]
            + analysis["h_nmr_peaks"]["empty_record_count"]
        ),
        "c_no_peak_records": (
            analysis["c_nmr_peaks"]["null_record_count"]
            + analysis["c_nmr_peaks"]["empty_record_count"]
        ),
        "both_modality_usable_records": analysis["both_modality_usable_count"],
        "h_total_peaks": analysis["h_nmr_peaks"]["total_peak_count"],
        "c_total_peaks": analysis["c_nmr_peaks"]["total_peak_count"],
        "h_max_peaks": analysis["h_nmr_peaks"]["maximum_peaks_per_record"],
        "c_max_peaks": analysis["c_nmr_peaks"]["maximum_peaks_per_record"],
        "zero_integration_peaks": analysis["proton_integration"]["zero_count"],
        "j_null_peaks": analysis["proton_j_values"]["null_count"],
        "j_empty_peaks": analysis["proton_j_values"]["empty_count"],
        "j_nonempty_peaks": analysis["proton_j_values"]["nonempty_count"],
    }


def read_record_ids(path: Path, arrow_batch_size: int) -> tuple[set[str], int]:
    import pyarrow.parquet as parquet

    identifiers = set()
    duplicate_count = 0
    parquet_file = parquet.ParquetFile(path)
    for batch in parquet_file.iter_batches(
        batch_size=arrow_batch_size,
        columns=["record_id"],
    ):
        for record_id in batch.column("record_id").to_pylist():
            if record_id in identifiers:
                duplicate_count += 1
            else:
                identifiers.add(record_id)
    return identifiers, duplicate_count


def analyze_split_consistency(
    dataset_directory: Path,
    arrow_batch_size: int,
) -> tuple[dict, int]:
    split_sets = {}
    split_counts = {}
    split_duplicate_count = 0
    for split_name in ("train", "val", "test"):
        identifiers, duplicate_count = read_record_ids(
            dataset_directory / f"{split_name}.parquet",
            arrow_batch_size,
        )
        split_sets[split_name] = identifiers
        split_counts[split_name] = len(identifiers) + duplicate_count
        split_duplicate_count += duplicate_count

    split_union = set()
    split_overlap_count = 0
    for split_name in ("train", "val", "test"):
        identifiers = split_sets[split_name]
        split_overlap_count += len(split_union.intersection(identifiers))
        split_union.update(identifiers)

    all_ids, all_duplicate_count = read_record_ids(
        dataset_directory / "all.parquet",
        arrow_batch_size,
    )
    report = {
        "split_counts": split_counts,
        "all_count": len(all_ids) + all_duplicate_count,
        "split_overlap_count": split_overlap_count,
        "all_equals_split_union": (
            split_duplicate_count == 0
            and all_duplicate_count == 0
            and all_ids == split_union
        ),
        "missing_from_all_count": len(split_union - all_ids),
        "extra_in_all_count": len(all_ids - split_union),
    }
    return report, all_duplicate_count


def build_source_observations(
    dataset_name: str,
    split_observations: dict[str, dict],
) -> dict:
    report = {
        "missing_j_policy": (
            "Raw missing J values become empty canonical lists for this dataset."
        )
    }

    observation_names = [
        "raw_missing_j_peaks",
        "zero_integration_records",
        "records_over_60_h_peaks",
        "records_over_60_c_peaks",
    ]
    if dataset_name == "nmrtrans":
        observation_names.extend(
            [
                "h_shift_outlier_records",
                "h_shift_outlier_peaks",
                "c_shift_outlier_records",
                "c_shift_outlier_peaks",
            ]
        )

    for name in observation_names:
        by_split = {
            split_name: values[name]
            for split_name, values in split_observations.items()
        }
        report[f"{name}_by_split"] = by_split
        report[name] = sum(by_split.values())

    if dataset_name == "nmrtrans":
        report["shift_outlier_thresholds_ppm"] = {
            "h": "outside [-5, 20]",
            "c": "outside [-20, 300]",
        }
    return report


def audit_dataset(
    dataset_name: str,
    arrow_batch_size: int,
) -> tuple[dict, dict]:
    config = DATASET_CONFIGS[dataset_name]
    dataset_directory = Path("datasets") / dataset_name

    split_report, all_duplicate_count = analyze_split_consistency(
        dataset_directory,
        arrow_batch_size,
    )
    all_analysis = analyze_parquet_file(
        dataset_directory / "all.parquet",
        arrow_batch_size=arrow_batch_size,
    )
    canonical = canonical_totals(all_analysis)

    source_totals = empty_totals()
    split_observations = {}
    source_splits = {}
    for split_name, source_path in config["splits"].items():
        split_totals, observations = analyze_raw_split(
            source_path,
            config["format"],
        )
        add_totals(source_totals, split_totals)
        split_observations[split_name] = observations
        source_splits[split_name] = {
            "path": str(source_path),
            "bytes": source_path.stat().st_size,
            "rows": split_totals["rows"],
        }

    matches = {
        name: canonical[name] == source_totals[name]
        for name in canonical
    }
    source_report = {
        "dataset": dataset_name,
        "source_format": config["source_format"],
        "source_splits": source_splits,
        "source_split_bytes_total": sum(
            split["bytes"] for split in source_splits.values()
        ),
        "source_totals": source_totals,
        "canonical_all_totals": canonical,
        "source_to_canonical_matches": matches,
        "all_source_statistics_match": all(matches.values()),
        "canonical_schema_matches": all_analysis["schema_matches_canonical"],
        "canonical_duplicate_record_ids": all_duplicate_count,
        "split_consistency": split_report,
        "source_observations": build_source_observations(
            dataset_name,
            split_observations,
        ),
    }
    return split_report, source_report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(DATASET_CONFIGS))
    parser.add_argument("--arrow-batch-size", type=int, default=50_000)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    split_report, source_report = audit_dataset(
        args.dataset,
        args.arrow_batch_size,
    )
    output_directory = Path("datasets") / args.dataset
    write_report(split_report, output_directory / "split_consistency.json")
    write_report(source_report, output_directory / "source_comparison.json")
    print(f"Wrote source and split reports for {args.dataset}")


if __name__ == "__main__":
    main()

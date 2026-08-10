#!/usr/bin/env python3
"""Generate tables and plots for the final cleaned NMR collection.

The script reproduces the molecular-property and peak analyses that were
developed in ``scripts/test_notebook.ipynb``. Summary tables use every record.
Violin plots use a deterministic sample and clip only their displayed values
to Q1 - 2.5*IQR through Q3 + 2.5*IQR.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

MAIN_DATASETS = ("train_val", "test_benchmark")
ADMET_ENDPOINTS = ("ames", "ld50_zhu", "solubility_aqsoldb")
ADMET_SPLITS = ("train_val", "test")

PROPERTY_COLUMNS = (
    "exact_molecular_weight",
    "calculated_logp",
    "tpsa",
    "hba",
    "hbd",
    "rotatable_bonds",
    "fraction_csp3",
    "aromatic_atom_fraction",
)
FUNCTIONAL_GROUP_COLUMNS = (
    "has_amine",
    "has_amide",
    "has_alcohol_or_phenol",
    "has_ester",
    "has_carboxylic_acid",
    "has_aldehyde_or_ketone",
    "has_nitrile",
    "has_halogenated_group",
    "has_heteroaromatic_ring",
)
CONTINUOUS_MOLECULAR_COLUMNS = ("num_atoms", *PROPERTY_COLUMNS)

SOURCE_DISPLAY_NAMES = {
    "NMRGym": "NMRGym",
    "NMRPeak-MST-NMR": "MST-NMR",
    "NMRPeak-NMRexp": "NMRexp",
    "NMRTrans-NMRSpec": "NMRTrans / NMRSpec",
    "SimNMR-PubChem": "SimNMR-PubChem",
}
ENDPOINT_DISPLAY_NAMES = {
    "ames": "Ames",
    "ld50_zhu": "LD50 Zhu",
    "solubility_aqsoldb": "AqSolDB solubility",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cleaned-root",
        type=Path,
        default=Path("datasets/cleaned"),
        help="final cleaned dataset directory (default: datasets/cleaned)",
    )
    parser.add_argument(
        "--sample-per-group",
        type=int,
        default=20_000,
        help="maximum records per source/endpoint used by plots (default: 20000)",
    )
    parser.add_argument(
        "--nmrsolver-parquet",
        type=Path,
        help=(
            "analyze one cleaned NMR-Solver Parquet and its adjacent "
            "_mol_properties.csv, then exit"
        ),
    )
    parser.add_argument(
        "--nmrgym-parquet",
        type=Path,
        help=(
            "analyze one cleaned NMRGym Parquet and its adjacent "
            "_mol_properties.csv as a shift-only dataset, then exit"
        ),
    )
    parser.add_argument(
        "--train-val-parquet",
        type=Path,
        help="custom cleaned train/validation Parquet",
    )
    parser.add_argument(
        "--test-benchmark-parquet",
        type=Path,
        help="custom cleaned benchmark-test Parquet",
    )
    parser.add_argument(
        "--analytics-dir",
        type=Path,
        help="custom output directory for generated analytics",
    )
    parser.add_argument(
        "--skip-admet",
        action="store_true",
        help="analyze only the two main datasets and skip unchanged ADMET cohorts",
    )
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"required input not found: {path}")


def display_name(group_column: str, value: str) -> str:
    if group_column == "source":
        return SOURCE_DISPLAY_NAMES.get(value, value)
    if group_column == "endpoint":
        return ENDPOINT_DISPLAY_NAMES.get(value, value)
    return value


def sample_by_group(
    frame: pl.LazyFrame,
    group_column: str,
    sample_per_group: int,
) -> pl.DataFrame:
    return (
        frame.group_by(group_column, maintain_order=True)
        .head(sample_per_group)
        .collect(engine="streaming")
    )


def first_rows_sample(
    frame: pl.LazyFrame,
    columns: tuple[str, ...],
    sample_size: int,
) -> pl.DataFrame:
    """Collect a small projected prefix without grouping the full input."""

    return frame.select(*columns).head(sample_size).collect(engine="streaming")


def numeric_summary_expressions(columns: tuple[str, ...]) -> list[pl.Expr]:
    expressions: list[pl.Expr] = []
    for column in columns:
        expressions.extend(
            [
                pl.col(column).mean().alias(f"{column}_mean"),
                pl.col(column).std().alias(f"{column}_std"),
                pl.col(column).min().alias(f"{column}_min"),
                pl.col(column).max().alias(f"{column}_max"),
            ]
        )
    return expressions


def record_statistics(records: pl.LazyFrame) -> pl.LazyFrame:
    return (
        records
        .with_columns(
            pl.col("atoms").list.len().alias("num_atoms"),
            pl.col("h_nmr_peaks").list.len().alias("n_h_peaks"),
            pl.col("c_nmr_peaks").list.len().alias("n_c_peaks"),
        )
        .with_columns(
            pl.when(pl.col("num_atoms") > 0)
            .then(pl.col("n_h_peaks") / pl.col("num_atoms"))
            .otherwise(None)
            .alias("h_peaks_per_atom"),
            pl.when(pl.col("num_atoms") > 0)
            .then(pl.col("n_c_peaks") / pl.col("num_atoms"))
            .otherwise(None)
            .alias("c_peaks_per_atom"),
        )
    )


def inventory_by_source(records: pl.LazyFrame, dataset_name: str) -> pl.DataFrame:
    stats = record_statistics(records)
    aggregations = [
        pl.len().alias("records"),
        pl.col("smiles_canonical").n_unique().alias("unique_canonical_smiles"),
        (pl.col("n_h_peaks") > 0).sum().alias("records_with_h"),
        (pl.col("n_c_peaks") > 0).sum().alias("records_with_c"),
        (
            (pl.col("n_h_peaks") > 0) & (pl.col("n_c_peaks") > 0)
        ).sum().alias("records_with_h_and_c"),
    ]
    by_source = (
        stats.group_by("source")
        .agg(aggregations)
        .with_columns(pl.lit(dataset_name).alias("dataset"))
        .select("dataset", "source", *[expr.meta.output_name() for expr in aggregations])
        .collect(engine="streaming")
    )
    total = (
        stats.select(aggregations)
        .with_columns(
            pl.lit(dataset_name).alias("dataset"),
            pl.lit("ALL").alias("source"),
        )
        .select(by_source.columns)
        .collect(engine="streaming")
    )
    return pl.concat([by_source, total]).sort("source")


def nmrsolver_inventory(records: pl.LazyFrame) -> pl.DataFrame:
    """Summarize NMR-Solver without an exact 105M-string distinct set."""

    stats = record_statistics(records)
    return (
        stats.group_by("source")
        .agg(
            pl.len().alias("records"),
            pl.col("smiles_canonical")
            .approx_n_unique()
            .alias("approx_unique_canonical_smiles"),
            (pl.col("n_h_peaks") > 0).sum().alias("records_with_h"),
            (pl.col("n_c_peaks") > 0).sum().alias("records_with_c"),
            (
                (pl.col("n_h_peaks") > 0) & (pl.col("n_c_peaks") > 0)
            ).sum().alias("records_with_h_and_c"),
        )
        .with_columns(pl.lit("nmrsolver").alias("dataset"))
        .select(
            "dataset",
            "source",
            "records",
            "approx_unique_canonical_smiles",
            "records_with_h",
            "records_with_c",
            "records_with_h_and_c",
        )
        .collect(engine="streaming")
    )


def molecular_frame_from_csv(
    records: pl.LazyFrame,
    properties_path: Path,
) -> pl.LazyFrame:
    require_file(properties_path)
    properties = pl.scan_csv(
        properties_path,
        empty_string_is_null=True,
    ).select(
        "record_id",
        "rdkit_status",
        *PROPERTY_COLUMNS,
        *FUNCTIONAL_GROUP_COLUMNS,
    )
    record_fields = records.select(
        "record_id",
        "source",
        pl.col("atoms").list.len().alias("num_atoms"),
    )
    return (
        properties
        .filter(pl.col("rdkit_status") == "ok")
        .join(record_fields, on="record_id", how="inner")
        .drop("rdkit_status")
    )


def molecular_frame_from_aligned_csv(
    records: pl.LazyFrame,
    properties_path: Path,
) -> pl.LazyFrame:
    """Read a very large property CSV without a record-ID hash join.

    calculate_mol_properties.py writes exactly one output row per input row
    and preserves input order, including when worker processes are used.
    """

    require_file(properties_path)
    record_fields = records.select(
        "record_id",
        "source",
        pl.col("atoms").list.len().alias("num_atoms"),
    )
    property_fields = pl.scan_csv(
        properties_path,
        empty_string_is_null=True,
    ).select(
        pl.col("record_id").alias("_properties_record_id"),
        "rdkit_status",
        *PROPERTY_COLUMNS,
        *FUNCTIONAL_GROUP_COLUMNS,
    )
    return (
        pl.concat([record_fields, property_fields], how="horizontal_extend")
        .filter(
            (pl.col("record_id") == pl.col("_properties_record_id"))
            & (pl.col("rdkit_status") == "ok")
        )
        .drop("_properties_record_id", "rdkit_status")
    )


def write_molecular_tables_and_plot(
    molecular: pl.LazyFrame,
    group_column: str,
    output_prefix: str,
    title: str,
    analytics_dir: Path,
    sample_per_group: int,
    *,
    first_rows_sample_only: bool = False,
) -> None:
    summary = (
        molecular.group_by(group_column)
        .agg(numeric_summary_expressions(CONTINUOUS_MOLECULAR_COLUMNS))
        .sort(group_column)
        .collect(engine="streaming")
    )
    prevalence = (
        molecular.group_by(group_column)
        .agg(
            [
                pl.col(column).mean().alias(column)
                for column in FUNCTIONAL_GROUP_COLUMNS
            ]
        )
        .sort(group_column)
        .collect(engine="streaming")
    )
    summary.write_csv(analytics_dir / f"{output_prefix}_molecular_property_summary.csv")
    prevalence.write_csv(
        analytics_dir / f"{output_prefix}_functional_group_prevalence.csv"
    )

    sample_columns = (group_column, *CONTINUOUS_MOLECULAR_COLUMNS)
    if first_rows_sample_only:
        sample = first_rows_sample(molecular, sample_columns, sample_per_group)
    else:
        sample = sample_by_group(
            molecular.select(*sample_columns),
            group_column,
            sample_per_group,
        )
    groups = sample.get_column(group_column).unique().sort().to_list()
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(groups)))

    n_columns = 3
    n_rows = math.ceil(len(CONTINUOUS_MOLECULAR_COLUMNS) / n_columns)
    figure = plt.figure(
        figsize=(16, 4.0 * n_rows + 3.0),
        layout="constrained",
    )
    grid = figure.add_gridspec(
        nrows=n_rows + 1,
        ncols=n_columns,
        height_ratios=[1] * n_rows + [0.8],
    )

    for index, column in enumerate(CONTINUOUS_MOLECULAR_COLUMNS):
        row, col = divmod(index, n_columns)
        axis = figure.add_subplot(grid[row, col])
        draw_source_violin(
            axis,
            sample,
            group_column,
            groups,
            colors,
            column,
            column.replace("_", " "),
            "",
        )

    prevalence = pl.DataFrame({group_column: groups}).join(
        prevalence,
        on=group_column,
        how="left",
    )
    axis = figure.add_subplot(grid[-1, :])
    prevalence_values = prevalence.select(FUNCTIONAL_GROUP_COLUMNS).to_numpy()
    image = axis.imshow(
        prevalence_values,
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        aspect="auto",
        origin="lower",
    )
    axis.set_title("Functional-group prevalence", fontsize=11)
    axis.set_yticks(
        range(len(groups)),
        [display_name(group_column, group) for group in groups],
        fontsize=8,
    )
    for label, color in zip(axis.get_yticklabels(), colors):
        label.set_color(color)
        label.set_fontweight("bold")
    axis.set_xticks(
        range(len(FUNCTIONAL_GROUP_COLUMNS)),
        [
            column.removeprefix("has_").replace("_", " ")
            for column in FUNCTIONAL_GROUP_COLUMNS
        ],
        rotation=25,
        ha="right",
        fontsize=8,
    )
    for row_index, row_values in enumerate(prevalence_values):
        for column_index, value in enumerate(row_values):
            axis.text(
                column_index,
                row_index,
                f"{value:.0%}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > 0.55 else "black",
            )
    colorbar = figure.colorbar(image, ax=axis, label="share of records")
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))

    figure.legend(
        handles=[
            Line2D([0], [0], color="black", lw=1, label="quartiles"),
            Line2D(
                [0],
                [0],
                color="black",
                marker="D",
                linestyle="None",
                markersize=5,
                label="mean",
            ),
        ],
        loc="upper center",
        ncols=2,
        bbox_to_anchor=(0.5, 1.03),
    )
    figure.suptitle(title, fontsize=15, y=1.06)
    figure.text(
        0.5,
        -0.01,
        "Violin display: Q1 - 2.5*IQR to Q3 + 2.5*IQR; tables use all values.",
        ha="center",
        fontsize=9,
    )
    figure.savefig(
        analytics_dir / f"{output_prefix}_molecular_property_distributions.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def peak_frames_for_group(
    records: pl.LazyFrame,
    group_column: str,
) -> tuple[pl.LazyFrame, ...]:
    stats = record_statistics(records)
    h_peaks = (
        stats.select(group_column, "h_nmr_peaks")
        .explode("h_nmr_peaks", empty_as_null=True)
        .filter(pl.col("h_nmr_peaks").is_not_null())
        .unnest("h_nmr_peaks")
    )
    c_peaks = (
        stats.select(group_column, "c_nmr_peaks")
        .explode("c_nmr_peaks", empty_as_null=True)
        .filter(pl.col("c_nmr_peaks").is_not_null())
        .unnest("c_nmr_peaks")
    )
    j_values = (
        h_peaks.select(group_column, "j_values")
        .explode("j_values", empty_as_null=True)
        .filter(pl.col("j_values").is_not_null())
        .rename({"j_values": "j_hz"})
    )
    return stats, h_peaks, c_peaks, j_values


def write_peak_tables_and_plot(
    records: pl.LazyFrame,
    group_column: str,
    output_prefix: str,
    title: str,
    analytics_dir: Path,
    sample_per_group: int,
    *,
    shift_only: bool = False,
    first_rows_sample_only: bool = False,
) -> None:
    stats, h_peaks, c_peaks, j_values = peak_frames_for_group(
        records,
        group_column,
    )
    record_columns = (
        "n_h_peaks",
        "n_c_peaks",
        "h_peaks_per_atom",
        "c_peaks_per_atom",
    )
    record_summary = (
        stats.group_by(group_column)
        .agg(
            pl.len().alias("records"),
            *numeric_summary_expressions(record_columns),
        )
        .collect(engine="streaming")
    )
    h_summary = (
        h_peaks.group_by(group_column)
        .agg(
            pl.len().alias("h_total_peaks"),
            pl.col("shift").mean().alias("h_shift_mean_ppm"),
            pl.col("shift").std().alias("h_shift_std_ppm"),
            pl.col("shift").min().alias("h_shift_min_ppm"),
            pl.col("shift").max().alias("h_shift_max_ppm"),
        )
        .collect(engine="streaming")
    )
    c_summary = (
        c_peaks.group_by(group_column)
        .agg(
            pl.len().alias("c_total_peaks"),
            pl.col("shift").mean().alias("c_shift_mean_ppm"),
            pl.col("shift").std().alias("c_shift_std_ppm"),
            pl.col("shift").min().alias("c_shift_min_ppm"),
            pl.col("shift").max().alias("c_shift_max_ppm"),
        )
        .collect(engine="streaming")
    )
    if shift_only:
        j_summary = record_summary.select(group_column).with_columns(
            pl.lit(None, dtype=pl.UInt64).alias("j_value_count"),
            pl.lit(None, dtype=pl.Float64).alias("j_mean_hz"),
            pl.lit(None, dtype=pl.Float64).alias("j_std_hz"),
            pl.lit(None, dtype=pl.Float64).alias("j_min_hz"),
            pl.lit(None, dtype=pl.Float64).alias("j_max_hz"),
        )
    else:
        j_summary = (
            j_values.group_by(group_column)
            .agg(
                pl.len().alias("j_value_count"),
                pl.col("j_hz").mean().alias("j_mean_hz"),
                pl.col("j_hz").std().alias("j_std_hz"),
                pl.col("j_hz").min().alias("j_min_hz"),
                pl.col("j_hz").max().alias("j_max_hz"),
            )
            .collect(engine="streaming")
        )
    peak_summary = (
        record_summary
        .join(h_summary, on=group_column, how="left")
        .join(c_summary, on=group_column, how="left")
        .join(j_summary, on=group_column, how="left")
        .sort(group_column)
    )
    peak_summary.write_csv(analytics_dir / f"{output_prefix}_peak_summary.csv")

    annotation_summary = (
        h_peaks.group_by(group_column)
        .agg(
            pl.len().alias("h_total_peaks"),
            pl.col("integration").is_null().sum().alias("integration_null"),
            (pl.col("integration") == 0).sum().alias("integration_zero"),
            pl.col("multiplicity").is_null().sum().alias("multiplicity_null"),
            pl.col("range_min").is_null().sum().alias("range_min_null"),
            pl.col("range_max").is_null().sum().alias("range_max_null"),
            pl.col("range_half_span").is_null().sum().alias("range_half_span_null"),
            pl.col("j_values").is_null().sum().alias("j_values_null"),
            (pl.col("j_values").list.len() == 0).sum().alias("j_values_empty"),
        )
        .sort(group_column)
        .collect(engine="streaming")
    )
    annotation_summary.write_csv(
        analytics_dir / f"{output_prefix}_proton_annotation_completeness.csv"
    )

    multiplicity = (
        h_peaks
        .with_columns(pl.col("multiplicity").fill_null("<missing>"))
        .group_by(group_column, "multiplicity")
        .len(name="n_peaks")
        .with_columns(
            (pl.col("n_peaks") / pl.col("n_peaks").sum().over(group_column))
            .alias("fraction")
        )
        .sort([group_column, "n_peaks"], descending=[False, True])
        .collect(engine="streaming")
    )
    multiplicity.write_csv(
        analytics_dir / f"{output_prefix}_multiplicity_distribution.csv"
    )

    sample_columns = (
        group_column,
        "n_h_peaks",
        "n_c_peaks",
        "h_peaks_per_atom",
        "c_peaks_per_atom",
        "h_nmr_peaks",
        "c_nmr_peaks",
    )
    if first_rows_sample_only:
        record_sample = first_rows_sample(stats, sample_columns, sample_per_group)
    else:
        record_sample = sample_by_group(
            stats.select(*sample_columns),
            group_column,
            sample_per_group,
        )
    h_sample = (
        record_sample.lazy()
        .select(group_column, "h_nmr_peaks")
        .explode("h_nmr_peaks", empty_as_null=True)
        .filter(pl.col("h_nmr_peaks").is_not_null())
        .unnest("h_nmr_peaks")
        .collect()
    )
    c_sample = (
        record_sample.lazy()
        .select(group_column, "c_nmr_peaks")
        .explode("c_nmr_peaks", empty_as_null=True)
        .filter(pl.col("c_nmr_peaks").is_not_null())
        .unnest("c_nmr_peaks")
        .collect()
    )
    if shift_only:
        j_sample = pl.DataFrame(
            schema={group_column: pl.String, "j_hz": pl.Float64}
        )
    else:
        j_sample = (
            h_sample.lazy()
            .select(group_column, "j_values")
            .explode("j_values", empty_as_null=True)
            .filter(pl.col("j_values").is_not_null())
            .rename({"j_values": "j_hz"})
            .collect()
        )

    groups = peak_summary.get_column(group_column).to_list()
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(groups)))
    positions = np.arange(len(groups))
    figure = plt.figure(figsize=(16, 15), layout="constrained")
    grid = figure.add_gridspec(3, 3, height_ratios=[1, 1, 1.15])
    draw_source_violin(
        figure.add_subplot(grid[0, 0]), record_sample, group_column, groups,
        colors, "n_h_peaks", "1H peaks per record", "number of peaks",
    )
    draw_source_violin(
        figure.add_subplot(grid[0, 1]), record_sample, group_column, groups,
        colors, "n_c_peaks", "13C peaks per record", "number of peaks",
    )
    draw_source_violin(
        figure.add_subplot(grid[0, 2]), record_sample, group_column, groups,
        colors, "h_peaks_per_atom", "1H peaks per atom", "peaks / atom",
    )
    draw_source_violin(
        figure.add_subplot(grid[1, 0]), record_sample, group_column, groups,
        colors, "c_peaks_per_atom", "13C peaks per atom", "peaks / atom",
    )
    draw_source_violin(
        figure.add_subplot(grid[1, 1]), h_sample, group_column, groups,
        colors, "shift", "1H chemical shifts", "shift (ppm)",
    )
    draw_source_violin(
        figure.add_subplot(grid[1, 2]), c_sample, group_column, groups,
        colors, "shift", "13C chemical shifts", "shift (ppm)",
    )
    draw_source_violin(
        figure.add_subplot(grid[2, 0]), j_sample, group_column, groups,
        colors, "j_hz", "1H J couplings", "J (Hz)",
    )

    top_multiplicities = (
        multiplicity.group_by("multiplicity")
        .agg(pl.col("n_peaks").sum().alias("total_peaks"))
        .sort("total_peaks", descending=True)
        .head(10)
        .get_column("multiplicity")
        .to_list()
    )
    multiplicity_plot = (
        multiplicity
        .with_columns(
            pl.when(pl.col("multiplicity").is_in(top_multiplicities))
            .then(pl.col("multiplicity"))
            .otherwise(pl.lit("other"))
            .alias("plot_multiplicity")
        )
        .group_by(group_column, "plot_multiplicity")
        .agg(pl.col("fraction").sum().alias("fraction"))
    )
    plot_multiplicities = [*top_multiplicities, "other"]
    multiplicity_wide = (
        pl.DataFrame({group_column: groups})
        .join(
            multiplicity_plot.pivot(
                on="plot_multiplicity",
                index=group_column,
                values="fraction",
            ),
            on=group_column,
            how="left",
        )
    )
    for column in plot_multiplicities:
        if column not in multiplicity_wide.columns:
            multiplicity_wide = multiplicity_wide.with_columns(
                pl.lit(0.0).alias(column)
            )
    multiplicity_wide = multiplicity_wide.with_columns(
        [pl.col(column).fill_null(0) for column in plot_multiplicities]
    )

    axis = figure.add_subplot(grid[2, 1:])
    left = np.zeros(len(groups))
    multiplicity_colors = plt.get_cmap("tab20")(
        np.linspace(0, 1, len(plot_multiplicities))
    )
    for multiplicity_name, color in zip(
        plot_multiplicities,
        multiplicity_colors,
    ):
        values = multiplicity_wide.get_column(multiplicity_name).to_numpy()
        axis.barh(
            positions,
            values,
            left=left,
            color=color,
            label=multiplicity_name,
        )
        left += values
    axis.set_yticks(
        positions,
        [display_name(group_column, group) for group in groups],
        fontsize=8,
    )
    for label, color in zip(axis.get_yticklabels(), colors):
        label.set_color(color)
        label.set_fontweight("bold")
    axis.set_xlim(0, 1)
    axis.set_xlabel("fraction of 1H peaks")
    axis.set_title("Canonical multiplicity distribution", fontsize=11)
    axis.legend(title="multiplicity", bbox_to_anchor=(1.02, 1), loc="upper left")

    figure.suptitle(title, fontsize=16, y=1.01)
    figure.text(
        0.5,
        -0.01,
        "Violin display: Q1 - 2.5*IQR to Q3 + 2.5*IQR; tables use all values.",
        ha="center",
        fontsize=9,
    )
    figure.savefig(
        analytics_dir / f"{output_prefix}_peak_distributions.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def draw_source_violin(
    axis,
    frame: pl.DataFrame,
    group_column: str,
    groups: list[str],
    colors,
    value_column: str,
    title: str,
    x_label: str,
) -> None:
    raw_data = []
    for position, (group, color) in enumerate(zip(groups, colors)):
        values = (
            frame.filter(pl.col(group_column) == group)
            .get_column(value_column)
            .drop_nulls()
            .to_numpy()
        )
        raw_data.append((position, color, values[np.isfinite(values)]))

    non_empty = [values for _, _, values in raw_data if len(values)]
    if not non_empty:
        axis.set_visible(False)
        return
    combined = np.concatenate(non_empty)
    q1, q3 = np.quantile(combined, [0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 2.5 * iqr
    upper = q3 + 2.5 * iqr

    distributions = []
    positions = []
    present_colors = []
    for position, color, values in raw_data:
        clipped = values[(values >= lower) & (values <= upper)]
        if len(clipped):
            distributions.append(clipped)
            positions.append(position)
            present_colors.append(color)

    violin = axis.violinplot(
        distributions,
        positions=positions,
        orientation="horizontal",
        showextrema=False,
        quantiles=[[0.25, 0.5, 0.75] for _ in distributions],
    )
    for body, color in zip(violin["bodies"], present_colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.65)
    violin["cquantiles"].set_color("black")
    violin["cquantiles"].set_linewidth(0.8)
    axis.scatter(
        [values.mean() for values in distributions],
        positions,
        color="black",
        marker="D",
        s=18,
        zorder=3,
    )
    axis.set_yticks(
        np.arange(len(groups)),
        [display_name(group_column, group) for group in groups],
        fontsize=8,
    )
    for label, color in zip(axis.get_yticklabels(), colors):
        label.set_color(color)
        label.set_fontweight("bold")
    axis.set_title(title, fontsize=11)
    axis.set_xlabel(x_label)
    if iqr > 0:
        axis.set_xlim(lower, upper)
    axis.grid(axis="x", alpha=0.25)


def main_records(path: Path) -> pl.LazyFrame:
    require_file(path)
    return pl.scan_parquet(path).select(
        "record_id",
        "source",
        "smiles_canonical",
        "atoms",
        "h_nmr_peaks",
        "c_nmr_peaks",
    )


def analyze_main_dataset(
    parquet_path: Path,
    dataset_name: str,
    analytics_dir: Path,
    sample_per_group: int,
) -> pl.DataFrame:
    print(f"Analyzing {dataset_name}", flush=True)
    dataset_analytics_dir = analytics_dir / dataset_name
    dataset_analytics_dir.mkdir(parents=True, exist_ok=True)
    records = main_records(parquet_path)
    inventory = inventory_by_source(records, dataset_name)
    inventory.write_csv(
        dataset_analytics_dir / f"{dataset_name}_source_inventory.csv"
    )

    molecular = molecular_frame_from_csv(
        records,
        parquet_path.with_name(f"{parquet_path.stem}_mol_properties.csv"),
    )
    write_molecular_tables_and_plot(
        molecular,
        "source",
        dataset_name,
        f"{dataset_name}: molecular properties by source",
        dataset_analytics_dir,
        sample_per_group,
        first_rows_sample_only=True,
    )
    write_peak_tables_and_plot(
        records,
        "source",
        dataset_name,
        f"{dataset_name}: NMR peak statistics by source",
        dataset_analytics_dir,
        sample_per_group,
    )
    return inventory


def analyze_nmrsolver_dataset(
    parquet_path: Path,
    sample_per_group: int,
) -> None:
    require_file(parquet_path)
    properties_path = parquet_path.with_name(
        f"{parquet_path.stem}_mol_properties.csv"
    )
    require_file(properties_path)
    analytics_dir = parquet_path.parent / "analytics" / "simnmr"
    analytics_dir.mkdir(parents=True, exist_ok=True)

    print(f"Analyzing NMR-Solver dataset: {parquet_path}", flush=True)
    records = main_records(parquet_path)
    nmrsolver_inventory(records).write_csv(
        analytics_dir / "nmrsolver_source_inventory.csv"
    )
    molecular = molecular_frame_from_aligned_csv(records, properties_path)
    write_molecular_tables_and_plot(
        molecular,
        "source",
        "nmrsolver",
        "NMR-Solver: molecular properties",
        analytics_dir,
        sample_per_group,
    )
    write_peak_tables_and_plot(
        records,
        "source",
        "nmrsolver",
        "NMR-Solver: NMR peak statistics",
        analytics_dir,
        sample_per_group,
        shift_only=True,
        # NMR-Solver has one source and over 100 million nested rows. Select
        # the plotting prefix before collecting, while summaries use all rows.
        first_rows_sample_only=True,
    )
    print(f"Wrote NMR-Solver analytics to {analytics_dir}", flush=True)


def analyze_nmrgym_dataset(
    parquet_path: Path,
    sample_per_group: int,
) -> None:
    require_file(parquet_path)
    properties_path = parquet_path.with_name(
        f"{parquet_path.stem}_mol_properties.csv"
    )
    require_file(properties_path)
    analytics_dir = parquet_path.parent / "analytics" / "nmrgym"
    analytics_dir.mkdir(parents=True, exist_ok=True)

    print(f"Analyzing NMRGym dataset: {parquet_path}", flush=True)
    records = main_records(parquet_path)
    inventory_by_source(records, "nmrgym").write_csv(
        analytics_dir / "nmrgym_source_inventory.csv"
    )
    molecular = molecular_frame_from_aligned_csv(records, properties_path)
    write_molecular_tables_and_plot(
        molecular,
        "source",
        "nmrgym",
        "NMRGym: molecular properties",
        analytics_dir,
        sample_per_group,
    )
    write_peak_tables_and_plot(
        records,
        "source",
        "nmrgym",
        "NMRGym: NMR peak statistics",
        analytics_dir,
        sample_per_group,
        shift_only=True,
        first_rows_sample_only=True,
    )
    print(f"Wrote NMRGym analytics to {analytics_dir}", flush=True)


def admet_records(cleaned_root: Path) -> pl.LazyFrame:
    frames = []
    for endpoint in ADMET_ENDPOINTS:
        for split in ADMET_SPLITS:
            path = cleaned_root / "admet" / endpoint / f"{split}.parquet"
            require_file(path)
            frames.append(
                pl.scan_parquet(path)
                .select(
                    "record_id",
                    "source",
                    "smiles_canonical",
                    "atoms",
                    "h_nmr_peaks",
                    "c_nmr_peaks",
                )
                .with_columns(
                    pl.lit(endpoint).alias("endpoint"),
                    pl.lit(split).alias("split"),
                )
            )
    return pl.concat(frames)


def admet_inventory(records: pl.LazyFrame) -> pl.DataFrame:
    stats = record_statistics(records)
    return (
        stats.group_by("endpoint", "split", "source")
        .agg(
            pl.len().alias("records"),
            pl.col("smiles_canonical").n_unique().alias("unique_canonical_smiles"),
            (pl.col("n_h_peaks") > 0).sum().alias("records_with_h"),
            (pl.col("n_c_peaks") > 0).sum().alias("records_with_c"),
            (
                (pl.col("n_h_peaks") > 0) & (pl.col("n_c_peaks") > 0)
            ).sum().alias("records_with_h_and_c"),
        )
        .sort("endpoint", "split", "source")
        .collect(engine="streaming")
    )


def admet_target_summary(cleaned_root: Path) -> pl.DataFrame:
    rows = []
    for endpoint in ADMET_ENDPOINTS:
        for split in ADMET_SPLITS:
            path = cleaned_root / "admet" / endpoint / f"{split}.csv"
            require_file(path)
            labels = pl.read_csv(path, columns=["record_id", "Drug", "Y"])
            values = labels.get_column("Y")
            rows.append(
                {
                    "endpoint": endpoint,
                    "task": "classification" if endpoint == "ames" else "regression",
                    "split": split,
                    "records": labels.height,
                    "unique_property_smiles": labels.get_column("Drug").n_unique(),
                    "target_mean": values.mean(),
                    "target_std": values.std(),
                    "target_min": values.min(),
                    "target_max": values.max(),
                    "positive_fraction": values.mean() if endpoint == "ames" else None,
                }
            )
    return pl.DataFrame(rows).sort("endpoint", "split")


def admet_molecular_frame(records: pl.LazyFrame) -> pl.LazyFrame:
    # ADMET is the only analysis path that calculates properties in-process.
    # Keep this import lazy so the main and shift-only analyses do not import
    # the full ``data`` package (including optional PyTorch dependencies).
    from data.postprocess.calculate_mol_properties import calculate_row

    unique_records = (
        records.select("record_id", "smiles_canonical")
        .unique(subset=["record_id"])
        .collect(engine="streaming")
    )
    descriptor_rows = []
    for row in unique_records.iter_rows(named=True):
        calculated = calculate_row(row["record_id"], row["smiles_canonical"])
        if calculated["rdkit_status"] != "ok":
            raise ValueError(
                f"RDKit property calculation failed for {row['record_id']}: "
                f"{calculated['rdkit_error']}"
            )
        descriptor_rows.append(
            {
                "record_id": row["record_id"],
                **{column: calculated[column] for column in PROPERTY_COLUMNS},
                **{
                    column: calculated[column]
                    for column in FUNCTIONAL_GROUP_COLUMNS
                },
            }
        )
    descriptors = pl.DataFrame(descriptor_rows).lazy()
    return (
        records.select(
            "record_id",
            "endpoint",
            pl.col("atoms").list.len().alias("num_atoms"),
        )
        .join(descriptors, on="record_id", how="left")
    )


def analyze_admet(
    cleaned_root: Path,
    analytics_dir: Path,
    sample_per_group: int,
) -> None:
    print("Analyzing ADMET cohorts", flush=True)
    admet_analytics_dir = analytics_dir / "admet"
    admet_analytics_dir.mkdir(parents=True, exist_ok=True)
    records = admet_records(cleaned_root)
    admet_inventory(records).write_csv(
        admet_analytics_dir / "admet_source_inventory.csv"
    )
    admet_target_summary(cleaned_root).write_csv(
        admet_analytics_dir / "admet_summary.csv"
    )
    molecular = admet_molecular_frame(records)
    write_molecular_tables_and_plot(
        molecular,
        "endpoint",
        "admet",
        "ADMET cohorts: molecular properties",
        admet_analytics_dir,
        sample_per_group,
    )
    write_peak_tables_and_plot(
        records,
        "endpoint",
        "admet",
        "ADMET cohorts: NMR peak statistics",
        admet_analytics_dir,
        sample_per_group,
    )


def main() -> None:
    args = parse_args()
    if args.sample_per_group < 1:
        raise ValueError("--sample-per-group must be at least 1")

    if (
        args.nmrsolver_parquet is not None
        and args.nmrgym_parquet is not None
    ):
        raise ValueError(
            "--nmrsolver-parquet and --nmrgym-parquet are mutually exclusive"
        )

    if args.nmrsolver_parquet is not None:
        analyze_nmrsolver_dataset(
            args.nmrsolver_parquet,
            args.sample_per_group,
        )
        return

    if args.nmrgym_parquet is not None:
        analyze_nmrgym_dataset(
            args.nmrgym_parquet,
            args.sample_per_group,
        )
        return

    cleaned_root = args.cleaned_root
    analytics_dir = args.analytics_dir or cleaned_root / "analytics"
    analytics_dir.mkdir(parents=True, exist_ok=True)

    main_dataset_paths = {
        "train_val": (
            args.train_val_parquet or cleaned_root / "train_val.parquet"
        ),
        "test_benchmark": (
            args.test_benchmark_parquet
            or cleaned_root / "test_benchmark.parquet"
        ),
    }
    inventories = [
        analyze_main_dataset(
            main_dataset_paths[dataset_name],
            dataset_name,
            analytics_dir,
            args.sample_per_group,
        )
        for dataset_name in MAIN_DATASETS
    ]
    collection_analytics_dir = analytics_dir / "collection"
    collection_analytics_dir.mkdir(parents=True, exist_ok=True)
    pl.concat(inventories).write_csv(
        collection_analytics_dir / "source_inventory.csv"
    )
    if not args.skip_admet:
        analyze_admet(cleaned_root, analytics_dir, args.sample_per_group)
    print(f"Wrote cleaned-dataset analytics to {analytics_dir}")


if __name__ == "__main__":
    main()

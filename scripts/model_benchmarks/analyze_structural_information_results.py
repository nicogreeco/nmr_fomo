#!/usr/bin/env python3
"""Aggregate saved structural-probe metrics; never load embeddings or models."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = {"functional": ("roc_auc", "average_precision"), "descriptors": ("r2", "mae")}
PAIRS = (
    ("post_minus_pre_shifts", "fomonmr-post-shifts", "fomonmr-pre-shifts"),
    ("rich_minus_shifts_standard", "fomonmr-post-rich", "fomonmr-post-shifts"),
    ("rich_minus_shifts_unimol", "fomonmr-post-unimol-rich", "fomonmr-post-unimol-shifts"),
    ("unimol_minus_standard_rich", "fomonmr-post-unimol-rich", "fomonmr-post-rich"),
    ("unimol_minus_standard_shifts", "fomonmr-post-unimol-shifts", "fomonmr-post-shifts"),
)


def summarize(values):
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
            "n_seeds": len(values), "min": float(values.min()), "max": float(values.max())}


def markdown_table(frame):
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def formatted(row):
    return f"{row['mean']:.4f}" if row['n_seeds'] == 1 else f"{row['mean']:.4f} ± {row['std']:.4f}"


def plot_summary(macro, output, n_test):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["nmrpeak", "nmrtrans", "ultranmr", "nmrsolver", "fomonmr-pre-shifts",
             "fomonmr-post-shifts", "fomonmr-post-rich", "fomonmr-post-unimol-shifts",
             "fomonmr-post-unimol-rich"]
    order = [name for name in order if name in set(macro.representation)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=True)
    for ax, metric, title in zip(axes, ("roc_auc", "average_precision", "r2"),
                                ("Functional groups: macro ROC-AUC", "Functional groups: macro AP", "Descriptors: macro R²")):
        for probe, offset, color, marker in [("linear", -0.12, "#777777", "o"), ("mlp", 0.12, "#007c91", "s")]:
            rows = macro[(macro.metric == metric) & (macro.probe == probe)].set_index("representation").loc[order]
            ax.errorbar(rows["mean"], np.arange(len(order)) + offset, xerr=rows["std"].fillna(0),
                        fmt=marker, color=color, capsize=3, label=probe, markersize=5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Higher is better")
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(np.arange(len(order)), order, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].legend(loc="best")
    fig.suptitle(f"Frozen NMR structural probes — common test cohort: {n_test:,} records")
    fig.text(0.5, 0.015, "Linear: seed 42. MLP: mean ± sample SD over seeds 13, 42, 73; error bars are often smaller than the markers.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    for extension in ("png", "svg"):
        fig.savefig(output / f"macro_comparison.{extension}", dpi=180)
    plt.close(fig)


def analyze(root, output):
    frames, runs, sources = [], [], {}
    for path in sorted(root.glob('*/run_manifest.json')):
        manifest = json.loads(path.read_text())
        representation = manifest['representation_id']
        if representation != path.parent.name:
            raise ValueError(f"Representation mismatch: {path}")
        metrics_path = path.parent / manifest['metrics_file']
        frame = pd.read_csv(metrics_path)
        if set(frame.model) != {manifest['model']} or set(frame.input_mode) != {manifest['input_mode']}:
            raise ValueError(f"Model/input mismatch: {metrics_path}")
        if frame.duplicated(['probe', 'seed', 'target']).any():
            raise ValueError(f"Duplicate rows: {metrics_path}")
        if set(frame.support) != {manifest['n_test']}:
            raise ValueError(f"Test support mismatch: {metrics_path}")
        for task, metrics in METRICS.items():
            selected = frame[frame[metrics[0]].notna()]
            expected_targets = 9 if task == 'functional' else 8
            if len(set(selected.target)) != expected_targets:
                raise ValueError(f"Incomplete targets: {metrics_path}, {task}")
            for probe, expected_seeds in [('linear', [42]), ('mlp', manifest['mlp_seeds'])]:
                rows = selected[selected.probe == probe]
                if set(rows.seed) != set(expected_seeds) or len(rows) != expected_targets * len(expected_seeds):
                    raise ValueError(f"Incomplete seeds: {metrics_path}, {task}, {probe}")
                if not np.isfinite(rows[list(metrics)].to_numpy()).all():
                    raise ValueError(f"Nonfinite metrics: {metrics_path}")
        frame['representation'] = representation
        frame['task'] = np.where(frame.roc_auc.notna(), 'functional', 'descriptors')
        frames.append(frame)
        runs.append({'representation': representation, **{key: manifest[key] for key in
                     ('model', 'input_mode', 'n_train', 'n_validation', 'n_test', 'common_cohort_manifest')},
                     'dimension': manifest['embedding_metadata']['dimension'],
                     'checkpoint': manifest['embedding_metadata']['checkpoint']})
        for source in (path, metrics_path):
            sources[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    raw = pd.concat(frames, ignore_index=True)
    run_table = pd.DataFrame(runs)
    for field in ('n_train', 'n_validation', 'n_test', 'common_cohort_manifest'):
        if run_table[field].nunique() != 1:
            raise ValueError(f"Runs disagree on {field}")
    for _, rows in raw.groupby(['task', 'target']):
        for field in ('support', 'positives', 'prevalence'):
            if rows[field].nunique() > 1:
                raise ValueError(f"Target support/prevalence mismatch: {field}")
    cohort_path = Path(runs[0]['common_cohort_manifest'])
    cohort = json.loads(cohort_path.read_text())
    ids = cohort['record_ids']
    for split, field in [('train', 'n_train'), ('validation', 'n_validation'), ('test', 'n_test')]:
        if len(ids[split]) != len(set(ids[split])) or len(ids[split]) != runs[0][field]:
            raise ValueError(f"Invalid current cohort: {split}")
    if any(set(ids[a]) & set(ids[b]) for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]):
        raise ValueError('Current cohort has overlapping record IDs')
    sources[str(cohort_path)] = hashlib.sha256(cohort_path.read_bytes()).hexdigest()
    aggregated, per_seed = [], []
    for (representation, probe, task, target), rows in raw.groupby(['representation', 'probe', 'task', 'target']):
        for metric in METRICS[task]:
            aggregated.append({'representation': representation, 'probe': probe, 'task': task, 'target': target,
                               'metric': metric, **summarize(rows[metric]), 'support': int(rows.support.iloc[0]),
                               'positives': rows.positives.iloc[0], 'prevalence': rows.prevalence.iloc[0]})
    targets = pd.DataFrame(aggregated)
    # Macro scores are formed within each seed before computing their SD.
    # MAEs have different units, so never average them across descriptors.
    for (representation, probe, task, seed), rows in raw.groupby(['representation', 'probe', 'task', 'seed']):
        for metric in METRICS[task]:
            if metric != 'mae':
                per_seed.append({'representation': representation, 'probe': probe, 'task': task, 'seed': seed,
                                 'metric': metric, 'value': rows[metric].mean()})
    seed_table = pd.DataFrame(per_seed)
    macro = []
    for (representation, probe, task, metric), rows in seed_table.groupby(['representation', 'probe', 'task', 'metric']):
        macro.append({'representation': representation, 'probe': probe, 'task': task, 'metric': metric,
                      **summarize(rows.value)})
    macro = pd.DataFrame(macro)
    deltas = []
    for label, left, right in PAIRS:
        a = seed_table[seed_table.representation == left]
        b = seed_table[seed_table.representation == right]
        paired = a.merge(b, on=['probe', 'task', 'metric', 'seed'], suffixes=('_a', '_b'), validate='one_to_one')
        for (probe, task, metric), rows in paired.groupby(['probe', 'task', 'metric']):
            deltas.append({'comparison': label, 'probe': probe, 'task': task, 'metric': metric,
                           **summarize(rows.value_a - rows.value_b)})
    validation = raw.groupby(['representation', 'probe', 'task', 'seed'], as_index=False).agg(
        validation_score=('validation_score', 'first'), weight_decay=('weight_decay', 'first'))
    output.mkdir(parents=True, exist_ok=True)
    for name, table in [('per_target', targets), ('macro_summary', macro), ('macro_per_seed', seed_table),
                        ('fomonmr_deltas', pd.DataFrame(deltas)), ('runs', run_table), ('validation_selection', validation)]:
        table.to_csv(output / f'{name}.csv', index=False)
    (output / 'input_sha256.json').write_text(json.dumps(sources, indent=2) + '\n')
    lines = ['# Tabelle finali: structural information', '',
             'MLP: media ± deviazione standard campionaria sui seed 13, 42, 73. Linear: solo seed 42; nessuna SD stimabile.', '',
             'Le macro-medie danno uguale peso ai 9 gruppi funzionali o agli 8 descrittori. La SD macro è calcolata sulle macro-medie dei singoli seed. Le MAE restano per target nelle rispettive unità.', '']
    for probe in ('linear', 'mlp'):
        lines += [f'## Sintesi {probe}', '']
        part = macro[macro.probe == probe].copy()
        part['score'] = part.apply(formatted, axis=1)
        lines += [markdown_table(part.pivot(index='representation', columns='metric', values='score').reset_index()), '']
        for task, metrics in METRICS.items():
            for metric in metrics:
                part = targets[(targets.probe == probe) & (targets.metric == metric)].copy()
                part['score'] = part.apply(formatted, axis=1)
                wide = part.pivot(index='target', columns='representation', values='score').reset_index()
                wide.to_csv(output / f'{probe}_{metric}_table.csv', index=False)
                lines += [f'### {task}: {metric}', '', markdown_table(wide), '']
    for representation in sorted(raw.representation.unique()):
        lines += [f'## {representation}: tutti i target', '']
        part = targets[targets.representation == representation].copy()
        part['column'] = part.probe + '_' + part.metric
        part['score'] = part.apply(formatted, axis=1)
        wide = part.pivot(index='target', columns='column', values='score').fillna('—').reset_index()
        lines += [markdown_table(wide), '']
    (output / 'tables.md').write_text('\n'.join(lines))
    plot_summary(macro, output, runs[0]["n_test"])
    print(f'Validated {len(runs)} runs, {len(raw)} metric rows; wrote {output}')
    print(macro.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=Path('results/structural_information'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/structural_information/analysis'))
    args = parser.parse_args()
    analyze(args.results_dir, args.output_dir)

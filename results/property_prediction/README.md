# Frozen ADMET probes

The five endpoint CSVs contain 17 representations and two probes each. Morgan
and UniMol2 are structure-only; names containing `+` are combined inputs.
FoMoNMR-only and other NMR-only representations use spectra at inference.

## Protocol refresh

The checked-in CSVs were refreshed on 2026-09-23 from a successful uniform CPU
rerun: 17 representations × 2 probes × 5 endpoints = 170 result rows. The MLP
groups full InChIKeys in both outer CV and its internal early-stopping holdout.
These replace the historical mixed-protocol scores. All cohort record and
molecule counts are unchanged; preflight also checked supervised train/test
molecular disjointness. The embedding caches and labels were not changed.

The run used code revision `bdfd555398773a96b0232ae95cdb9b0add4451c4`, with runner
SHA-256 `66e49c5af7f3cf497f3c843e2be563bbb2bb9ccd83896c584d03790e18d4be75`.
Recorded versions: Python 3.11.15, NumPy 2.4.6, scikit-learn 1.9.0,
PyTorch 2.6.0+cpu and Polars 1.43.1. The shell recipe limits grid-search workers
to four and numerical-library threads to one.

The largest absolute primary-score change from the previous CSVs is 0.000315
for linear probes and 0.230954 for MLPs. Historical and refreshed environments
are not a controlled split-only ablation, so not every numerical change can
be attributed solely to the internal holdout correction.

One logistic fit during Ames/NMRTrans emitted an 8,000-iteration convergence
warning. All rows and metrics are present and finite; the log does not identify
the candidate/fold, so it does not establish that every fit converged. No
hyperparameters were changed after reviewing test scores. The local log and
previous CSV copies remain under `results/property_prediction_rerun/`.

Activate the main environment without cuML and run from the repository root:

```bash
bash scripts/model_benchmarks/rerun_admet_probes.sh
```

This uses existing `embeddings/admet/` caches and `datasets/cleaned/admet/`
labels. It reruns both linear and MLP probes, with five CV folds, seed 42 and
at most 300 MLP epochs. All 17 representations use the same per-endpoint ID
intersection. Outputs go to the new, ignored `results/property_prediction_rerun/`
directory, including `rerun.log`. It refuses to overwrite an existing output
directory; pass a different path as its first argument for another run.

## Analysis

Run all cells of [the notebook](property_prediction_analysis.ipynb) in the main
environment. It reads only the five CSVs and writes `analysis/scores.csv`,
`analysis/mean_rank.csv` and `analysis/complementarity.csv` (derived, ignored
files). It displays endpoint tables,
standalone and combined comparisons, and FoMoNMR-only comparisons. The context
note uses `main_standalone_8` (six baselines and two rich FoMoNMR checkpoints)
and `main_15` (those eight plus seven concatenations); `all_17` also retains
the two shift-only conditions. Best-combination tables are explicitly
retrospective selections on test scores, not separately validated methods. Change
`RESULTS_DIR` to `results/property_prediction_rerun` to inspect the completed
rerun before adoption. Incomplete endpoint sets fail explicitly.

ROC-AUC is used for Ames and R² for regression. Cross-task summaries average
ranks, not unlike metrics. Ranking pools are named explicitly; single-run
differences are descriptive, not significance tests. Fine-tuning is analysed
separately because it uses a different cohort.

ADMET molecules are removed from rich pretraining only, not explicitly from
SimNMR, NMRGym or external encoders' pretraining corpora. Grouped supervised
splits therefore do not guarantee unseen molecules during pretraining.

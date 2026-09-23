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
PyTorch 2.6.0+cpu and Polars 1.43.1. The rerun limited grid-search workers to
four and numerical-library threads to one.

The largest absolute primary-score change from the previous CSVs is 0.000315
for linear probes and 0.230954 for MLPs. Historical and refreshed environments
are not a controlled split-only ablation, so not every numerical change can
be attributed solely to the internal holdout correction.

One logistic fit during Ames/NMRTrans emitted an 8,000-iteration convergence
warning. All rows and metrics are present and finite; the log does not identify
the candidate/fold, so it does not establish that every fit converged. No
hyperparameters were changed after reviewing test scores. The local log and
previous CSV copies remain under `results/property_prediction_rerun/`.

## Reproduce from canonical data

Materialize the five endpoint datasets first:

```bash
nmr-env main
dvc repro prepare_admet
```

Create the model-specific environments as described in
[`scripts/envs_scr/README.md`](../../scripts/envs_scr/README.md) and place the
published baseline checkpoints at their documented default paths. Download the
FoMoNMR release from [`niccogreek/fomonmr`](https://huggingface.co/niccogreek/fomonmr)
into `models_release/`; no MLflow server is required.

### Extract embeddings

For every endpoint, extract `train_val.parquet` and `test.parquet` with the
normal streaming extractor. Run one model family in its own environment. For
example:

```bash
nmr-env nmrpeak
PYTHONPATH=scripts python -m model_benchmarks.extract_embeddings \
  --model nmrpeak \
  --input datasets/cleaned/admet/ames/train_val.parquet \
  --output embeddings/admet_new/ames/nmrpeak/train.parquet \
  --device cuda --batch-size 32 \
  --on-incompatible skip \
  --rejections embeddings/admet_new/ames/nmrpeak/train_rejections.json
```

Repeat for the test split, all five endpoints, and the six base comparisons:
Morgan, NMRPeak-R, NMR-Solver, NMRTrans, UltraNMR, and UniMol2. Use the main
environment for Morgan and NMR-Solver, and the corresponding model environment
for each learned external encoder.

FoMoNMR also runs in the main environment. Give the local release checkpoint
and input mode, for example:

```bash
nmr-env main
PYTHONPATH=scripts python -m model_benchmarks.extract_embeddings \
  --model fomonmr \
  --checkpoint models_release/posttraining/fomonmr-posttrain.ckpt \
  --input-mode rich \
  --input datasets/cleaned/admet/ames/train_val.parquet \
  --output embeddings/admet_new/ames/fomonmr-post-rich/train.parquet \
  --device cuda --batch-size 32 \
  --on-incompatible skip \
  --rejections embeddings/admet_new/ames/fomonmr-post-rich/train_rejections.json
```

Extract four FoMoNMR caches: `fomonmr-pre-shifts`,
`fomonmr-post-shifts`, `fomonmr-post-rich`, and
`fomonmr-post-unimol-rich`. The seven `molecular+nmr` representations do not
need separate extraction: the property runner concatenates their base caches.

### Train frozen probes

Activate the main CPU environment without cuML and choose a new output
directory. The following reproduces all 17 conditions with one common
per-endpoint record intersection:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=4

PYTHONPATH=scripts python -m model_benchmarks.run_property_prediction \
  --models \
    morgan nmrpeak nmrsolver nmrtrans ultranmr unimol2 \
    unimol2+ultranmr unimol2+nmrpeak unimol2+nmrsolver \
    morgan+nmrpeak morgan+ultranmr morgan+nmrsolver \
    fomonmr-pre-shifts fomonmr-post-shifts fomonmr-post-rich \
    fomonmr-post-unimol-rich morgan+fomonmr-post-unimol-rich \
  --datasets ames ld50_zhu lipophilicity_astrazeneca \
    sangster_logp solubility_aqsoldb \
  --embeddings-dir embeddings/admet_new \
  --output-dir results/property_prediction_new \
  --cv-folds 5 --epochs 300 --device cpu
```

The runner trains both linear and MLP probes. It uses full InChIKey groups for
outer CV and for the MLP early-stopping holdout; seed 42 is fixed in the runner.

## Analysis

Copy [the notebook](property_prediction_analysis.ipynb) into a new result
directory, start Jupyter there, and run all cells in the main environment. It
reads only the five CSVs and writes `analysis/scores.csv`,
`analysis/mean_rank.csv` and `analysis/complementarity.csv` (derived, ignored
files). It displays endpoint tables,
standalone and combined comparisons, and FoMoNMR-only comparisons. The context
note uses `main_standalone_8` (six baselines and two rich FoMoNMR checkpoints)
and `main_15` (those eight plus seven concatenations); `all_17` also retains
the two shift-only conditions. Best-combination tables are explicitly
retrospective selections on test scores, not separately validated methods.
Incomplete endpoint sets fail explicitly.

ROC-AUC is used for Ames and R² for regression. Cross-task summaries average
ranks, not unlike metrics. Ranking pools are named explicitly; single-run
differences are descriptive, not significance tests. Fine-tuning is analysed
separately because it uses a different cohort.

ADMET molecules are removed from rich pretraining only, not explicitly from
SimNMR, NMRGym or external encoders' pretraining corpora. Grouped supervised
splits therefore do not guarantee unseen molecules during pretraining.

# Structural-information benchmark results

The checked-in `test_metrics.csv` files contain the nine frozen NMR
representations used in the structural-information comparison. Linear probes
use seed 42; MLP probes use seeds 13, 42 and 73.

## Reproduce from canonical data

First initialize the published-model repositories, create the project
environments, and materialize the required DVC inputs:

```bash
git submodule update --init --recursive
./scripts/envs_scr/setup_gpu_envs.sh /path/to/nmr-envs
source "${XDG_CONFIG_HOME:-${HOME}/.config}/nmr/envs.sh"
nmr-env main
dvc repro create_structural_probe_dataset
```

This stage runs its required upstream stages and deterministically recreates
the 50,000-molecule training split and 10,000-molecule validation split under
`datasets/downstream_structural_information/`. The published-model checkpoints
must then be placed in the locations documented below.

### Extract embeddings

Run the extractor once for `train.parquet`, `validation.parquet`, and the held-out
`datasets/cleaned/test_benchmark.parquet`. Use one model family per environment.
For example, NMRPeak train embeddings are extracted with:

```bash
nmr-env nmrpeak
PYTHONPATH=scripts python -m model_benchmarks.extract_embeddings \
  --model nmrpeak \
  --input datasets/downstream_structural_information/train.parquet \
  --output embeddings/structural_information_new/nmrpeak/train.parquet \
  --device cuda --batch-size 32
```

Use the equivalent `validation.parquet` input and output, then extract the test
cache from `datasets/cleaned/test_benchmark.parquet`. Repeat this in the
`nmrtrans`, `ultranmr`, and main environments for NMRTrans, UltraNMR, and
NMR-Solver.

For held-out-test extraction, add:

```bash
--on-incompatible skip \
--rejections embeddings/structural_information_new/REPRESENTATION/test_rejections.json
```

The four external NMR paths require non-empty H+C input and reject 20,106
single-modality test records, leaving the published common test cohort of
67,791 records. FoMoNMR can encode those records, but the probe runner later
uses the intersection shared by all nine representations.

FoMoNMR runs in the main environment and needs its checkpoint and input mode:

```bash
nmr-env main
PYTHONPATH=scripts python -m model_benchmarks.extract_embeddings \
  --model fomonmr \
  --checkpoint models_release/posttraining/fomonmr-posttrain.ckpt \
  --input-mode rich \
  --input datasets/downstream_structural_information/train.parquet \
  --output embeddings/structural_information_new/fomonmr-post-rich/train.parquet \
  --device cuda --batch-size 32
```

The published comparison extracts five FoMoNMR conditions: pretrain/shifts,
standard posttrain/shifts, standard posttrain/rich, UniMol2 posttrain/shifts,
and UniMol2 posttrain/rich. The extractor stores checkpoint and input-mode
metadata in every Parquet footer.

### Train probes

After all nine representations have train, validation, and test caches, run
the probe script once per representation. The first run builds the common
record intersection across all caches:

```bash
nmr-env main
PYTHONPATH=scripts python -m model_benchmarks.run_structural_information_probes \
  --model nmrpeak --representation-id nmrpeak \
  --embeddings-root embeddings/structural_information_new \
  --rebuild-common-cohort \
  --output-dir results/structural_information_new \
  --device cuda --no-mlflow
```

Subsequent representations reuse the same cohort manifest and omit
`--rebuild-common-cohort`. FoMoNMR runs must also pass the same
`--checkpoint-path` and `--input-mode` used during extraction. The defaults are
linear seed 42, MLP seeds 13/42/73, and 300 epochs.

### Analyse results

Copy
`structural_information_analysis.ipynb` into that directory, start Jupyter
there, and run all cells. The notebook reads only the sibling
`*/test_metrics.csv` files and writes `analysis/per_target.csv`,
`analysis/macro_summary.csv`, and `analysis/mean_rank.csv`.

## Required environments and checkpoints

Create the five environments with
`scripts/envs_scr/setup_cpu_envs.sh` or `setup_gpu_envs.sh`. The setup installs
the `NMR_VENV_ROOT` configuration used by the wrappers.

The external checkpoints must exist at the extractor defaults:

- NMRPeak-R: the combined H+C retrieval `checkpoint_best.pt` below
  `models/NMRPeak/weights/retrieval/all_weights/.../CH/`;
- NMRTrans: `models/NMRTrans/model/nmrtrans-c-h-nmr.ckpt`;
- UltraNMR: `models/UltraNMR/model_checkpoint/checkpoints_nce/model_epoch_1.pth`.

Their upstream download instructions are in the corresponding `models/*/README.md`.
Alternative files can be supplied with `NMRPEAK_CHECKPOINT`,
`NMRTRANS_CHECKPOINT`, or `ULTRANMR_CHECKPOINT`.

Download the FoMoNMR release from
[`niccogreek/fomonmr`](https://huggingface.co/niccogreek/fomonmr) into
`models_release/`, or set `FOMONMR_RELEASE_ROOT`. The wrapper uses the released
pretraining, standard posttraining and UniMol2-relational Lightning
checkpoints. It never requires access to the original MLflow server.

```bash
hf download niccogreek/fomonmr --local-dir models_release
```

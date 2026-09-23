# Project scripts

The code under `scripts/` is divided by responsibility. Start here to choose a
workflow, then use the README inside the relevant package for commands and
Python examples.

## Directory map

| Directory | Responsibility | Documentation |
| --- | --- | --- |
| `data/` | Canonical schema, readers, DVC data preparation and analytics | [`data/README.md`](data/README.md) |
| `model/` | FoMoNMR processor, datasets, architecture and training | [`model/README.md`](model/README.md) |
| `model_benchmarks/` | Published-model adapters, embedding extraction, downstream probes and FoMoNMR fine-tuning | [`model_benchmarks/README.md`](model_benchmarks/README.md) |
| `envs_scr/` | Main and model-specific Python environments | [`envs_scr/README.md`](envs_scr/README.md) |

The main boundary is:

```text
canonical dataset -> model-specific processor -> native batch -> embedder
```

Canonical data code does not tokenize for a model. Benchmark processors do not
parse raw sources. FoMoNMR training remains separate from adapters for published
models.

## Main workflows

Run commands from the repository root. The main environment is sufficient for
data preparation, FoMoNMR, downstream probes, and analysis:

```bash
nmr-env main
```

### Build the datasets

The root DVC graph is the maintained raw-to-cleaned pipeline:

```bash
dvc dag
dvc repro
```

Use `dvc repro STAGE` for a targeted stage. See
[`data/README.md`](data/README.md) before changing the DAG or canonical schema.

### Train FoMoNMR

```bash
PYTHONPATH=scripts python -m model.train \
  --stage pretrain --batch-size 128
```

Continued pretraining, configuration files, checkpoints, MLflow, and loading
examples are documented in [`model/README.md`](model/README.md).

### Extract and evaluate representations

```bash
PYTHONPATH=scripts python -m model_benchmarks.extract_embeddings \
  --model nmrpeak \
  --input datasets/cleaned/test_benchmark.parquet \
  --output embeddings/nmrpeak.parquet \
  --device cuda
```

Published encoders use their own environments. The benchmark README shows how
to use processors and embedders directly, extract Parquet caches, run the
structural and ADMET probes, and fine-tune FoMoNMR.

## Where changes belong

- Shared records, validation, and readers: `data/`.
- Source-specific conversion: `data/canonicalize/`.
- Derived datasets and filtering: `data/postprocess/`.
- FoMoNMR architecture and training: `model/`.
- Published-model input conversion and pooling: `model_benchmarks/`.
- Environment pins and setup: `envs_scr/`.

Small fixture-based tests live beside each package. The repository-level test
commands are listed in the corresponding README files and `AGENTS.md`.

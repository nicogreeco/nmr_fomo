# NMR latent-space project

This private repository contains the code, notes, and reproducible model
references used to compare representations from existing one-dimensional NMR
models. It also contains the small canonical-data layer that will later support
training a new NMR foundation model.

## Clone the project

The existing-model repositories are Git submodules pinned to the versions used
by this project. Clone them at the same time:

```bash
git clone --recurse-submodules git@github.com:nicogreeco/nmr_fomo.git
cd nmr_fomo
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

The currently pinned upstream repositories are:

| Path | Upstream repository | Role |
|---|---|---|
| `models/NMRPeak` | `Colin-Jay/NMRPeak` | rich peak-token baseline |
| `models/NMRTrans` | `little1d/NMRTrans` | experimental peak-set baseline |
| `models/NMR-Solver` | `YongqiJin/NMR-Solver` | fixed Gaussian shift featurizer |
| `models/UltraNMR` | `wuycM/UltraNMR` | large-scale shift-only encoder |

Do not casually run `git pull` inside a submodule. Update one deliberately,
test the corresponding processor/embedder, then commit the changed submodule
pointer in this repository.

## Local environments

The models have incompatible dependencies, so use one environment per model
family. The setup scripts create environments on local VM storage while code
and large data remain on the shared filesystem.

For a CPU machine:

```bash
./scripts/setup_cpu_envs.sh /path/on/local/disk/nmr
```

For a GPU machine:

```bash
./scripts/setup_gpu_envs.sh /path/on/local/disk/nmr
```

After setup, activate one environment with the installed helper:

```bash
nmr-env main
nmr-env nmrpeak
nmr-env nmrtrans
nmr-env ultranmr
```

Read [scripts/envs_scr/README.md](scripts/envs_scr/README.md) before setting up
a new VM. PyArrow is installed in every environment because it is used to
stream canonical Parquet files and write incremental embedding outputs.

## Project layout

```text
contex/                     project diary, paper notes, design decisions, analyses
contex/DL Methods/          one note per relevant model or dataset paper
datasets/                   local canonical Parquet files and analysis reports (ignored)
models/                     pinned upstream model repositories (submodules)
scripts/
  data/                     canonical schema, validation, and dataset readers
  canonicalize/             source-to-canonical conversion and analysis tools
  model_benchmarks/         model processors, embedders, extraction CLI, smoke tests
  envs_scr/                 machine-local environment setup scripts and notes
  notebooks/                small exploratory notebooks
```

The core data flow is:

```text
CanonicalParquetDataset
    -> model-specific processor/collator
    -> model-specific batch
    -> lightweight embedder
    -> fixed-size embeddings with record IDs
```

The canonical dataset layer is model-independent. It does not tokenize, pad,
normalise, or know how a source release was originally stored.

## Common tasks

### Extract embeddings

Activate the matching model environment, then run from the repository root:

```bash
PYTHONPATH=scripts python scripts/extract_embeddings.py \
  --model nmrpeak \
  --input datasets/mst_nmr/test.parquet \
  --output embeddings/mst_nmr_nmrpeak.parquet \
  --device cuda \
  --batch-size 64 \
  --mode canonical
```

The extractor reads input incrementally and writes one Parquet row group per
embedding batch. Add `--on-incompatible skip --rejections rejections.json` for
bulk extraction where some records do not meet a model's requirements.

### Use the reusable dataset in Python

```python
from torch.utils.data import DataLoader

from data import CanonicalParquetDataset
from model_benchmarks import build_embedder, build_processor

dataset = CanonicalParquetDataset("datasets/mst_nmr/test.parquet")
processor = build_processor("nmrpeak", mode="canonical", strict=True)
embedder = build_embedder("nmrpeak", device="cpu")

loader = DataLoader(dataset, batch_size=64, collate_fn=processor)
for batch in loader:
    result = embedder.encode(batch)
    break
```

Only combined `1H + 13C` input is supported by the current benchmark bridge.

## Where to read and edit

- [Embedding Pipeline Architecture](contex/Embedding_Pipeline_Architecture.md):
  data flow, embedding definitions, extraction behaviour.
- [Canonicalization Implementation Notes](contex/Canonicalization_Implementation_Notes.md):
  canonical fields and converter rules.
- [Dataset Analysis](contex/Dataset%20Analysis.md): measured canonical dataset
  statistics and processor compatibility.
- [Dataset Filtering and Processing](contex/Dataset_Filtering_and_Processing.md):
  how paper-level filtering explains distribution differences.
- [scripts/README.md](scripts/README.md): the practical map of the Python code.

For an implementation change, keep common schema logic in `scripts/data/`;
change one model path directly in
`scripts/model_benchmarks/processors/<model>.py` and
`scripts/model_benchmarks/embedders/<model>.py`; keep source parsing only in
`scripts/canonicalize/`.

## Private-repository hygiene

Before committing or pushing, inspect changes:

```bash
git status
git diff --cached
git submodule status
```

Never add credentials, SSH keys, tokens, private URLs, raw PDFs with restricted
distribution, datasets, or checkpoints. If an ignored large file appears in a
submodule's own `git status`, it is still local, but should not be committed in
that submodule.

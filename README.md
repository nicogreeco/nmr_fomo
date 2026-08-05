# NMR FoMo

This is the private working repository for the NMR foundation-model project
described in [Internship – NMR Project Plan](<contex/Internship%20%E2%80%93%20NMR%20Project%20Plan.md>).
It contains the project notes, the canonical NMR-data code, data-preparation
tools, and a small bridge for comparing this future model with published NMR
encoders.

The central goal is to learn useful representations from combined `1H + 13C`
NMR spectra. The project plan is the source of truth for scope, milestones, and
deliverables; this README only explains how the repository is arranged.

## Clone

The published model repositories are included as Git submodules. Clone them
with the project:

```bash
git clone --recurse-submodules git@github.com:nicogreeco/nmr_fomo.git
cd nmr_fomo
```

If you already cloned the repository, initialise them with:

```bash
git submodule update --init --recursive
```

The submodules are pinned to specific revisions so that the benchmark code is
reproducible. Do not update one casually: test the relevant benchmark path and
then commit the new submodule pointer in this repository.

## Published-model benchmarks

`models/` contains code from published repositories, used here only to extract
latent representations from their existing encoders. Those embeddings are a
benchmark for the future foundation model; the upstream models are not part of
the new model itself.

| Local path | Published repository | Role here |
|---|---|---|
| `models/NMRPeak` | NMRPeak | peak-token encoder baseline |
| `models/NMRTrans` | NMRTrans | paired H/C peak-set encoder baseline |
| `models/NMR-Solver` | NMR-Solver | fixed Gaussian-spectrum featurizer |
| `models/UltraNMR` | UltraNMR | shift-only encoder baseline |

Each checkpoint, dataset release, and model asset must be downloaded according
to the instructions and licence in that model's own upstream README. They are
large local assets and are deliberately not versioned here.

## Environments and local assets

The four published projects have incompatible dependencies, so the project
uses one Python environment per model family. Set them up on the VM's local
disk, while keeping code and large assets on the shared filesystem:

```bash
./scripts/envs_scr/setup_cpu_envs.sh /path/on/local/disk/nmr
# or, on a prepared GPU VM:
./scripts/envs_scr/setup_gpu_envs.sh /path/on/local/disk/nmr
```

After setup, use `nmr-env main`, `nmr-env nmrpeak`, `nmr-env nmrtrans`, or
`nmr-env ultranmr`. See [scripts/envs_scr/README.md](scripts/envs_scr/README.md)
for the exact environment setup and repair commands.

## Repository map

```text
contex/                     project diary, literature notes, decisions, and analyses
  NMR/DL Methods/           notes on the relevant papers and repositories
datasets/                   local raw/canonical data and generated reports (ignored)
models/                     pinned upstream repositories (Git submodules)
scripts/
  data/                     reusable canonical schema, validation, and Parquet readers
  canonicalize/             source-specific conversion and dataset-analysis tools
  model_benchmarks/         processors and embedders for published-model comparison
  envs_scr/                 model-specific environment setup material
  postprocess/              derived-data and benchmark-preparation utilities
  test_notebook.ipynb       exploratory work
```

The reusable data layer remains independent of the published models:

```text
CanonicalParquetDataset -> processor/collator -> model batch -> embedder -> embeddings
```

Canonical Parquet schema version 2 stores source SMILES plus RDKit-derived
canonical SMILES, molecular formula, and atom symbols. It deliberately omits
3D coordinates; no current spectral benchmark consumes them. Conversion and
schema details are in `scripts/canonicalize/README.md` and the canonicalization
implementation note linked below.

## How to navigate the project

Start with the [Internship – NMR Project Plan](<contex/Internship%20%E2%80%93%20NMR%20Project%20Plan.md>), then use `contex/` as the project diary:

- [NMR foundations and AI](<contex/NMR%20foundations%20and%20AI.md>) and
  [DL Methods](<contex/NMR/DL%20Methods/>) collect the literature context.
- [Datasets](<contex/Datasets.md>), [Dataset Analysis](<contex/Dataset%20Analysis.md>),
  and [Dataset Filtering and Processing](<contex/Dataset_Filtering_and_Processing.md>) explain the data choices and observed distributions.
- [Embedding Pipeline Architecture](<contex/Embedding_Pipeline_Architecture.md>) and
  [Canonicalization Implementation Notes](<contex/Canonicalization_Implementation_Notes.md>) describe the current implementation decisions.

Each `scripts/` subfolder has a short README with the practical details for
that part of the code. In particular, start from `scripts/README.md` for the
Python layout, and from `scripts/canonicalize/README.md` for conversion or
canonical-dataset analysis.

Before a push, check `git status`, `git diff --cached`, and
`git submodule status`.

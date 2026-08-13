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
datasets/
  raw/                      DVC-pinned upstream releases
  canonical/                reproducible source conversions
  intermediate/             reproducible merge, overlap, and audit outputs
  cleaned/                  final datasets, molecular properties, and analytics
models/                     pinned upstream repositories (Git submodules)
scripts/
  data/                     canonical data code and data-preparation utilities
    canonicalize/           source-specific conversion and dataset-analysis tools
    postprocess/            derived-data and benchmark-preparation utilities
    download_raw_datasets.sh  pinned public-source downloader
    generate_dvc_pipeline.sh  generator for the root DVC DAG
  model_benchmarks/         processors and embedders for published-model comparison
  envs_scr/                 model-specific environment setup material
  test_notebook.ipynb       exploratory work
```

The reusable data layer remains independent of the published models:

```text
CanonicalParquetDataset -> processor/collator -> model batch -> embedder -> embeddings
```

Canonical Parquet schema version 2 stores source SMILES plus RDKit-derived
canonical SMILES, molecular formula, and atom symbols. It deliberately omits
3D coordinates; no current spectral benchmark consumes them. Conversion and schema details are in
`scripts/data/canonicalize/README.md` and the canonicalization
implementation note linked below.

## Dataset management

Git versions code and small control files; DVC versions dataset contents. The
data directories have deliberately different storage policies:

| Directory | Role | Versioning and remote policy |
|---|---|---|
| `datasets/raw/` | Original inputs to this pipeline | One committed `.dvc` pointer per source; public users download the pinned upstream releases |
| `datasets/canonical/` | Schema-v2 conversion outputs | DVC pipeline outputs with `push: false`; reproducible from raw inputs |
| `datasets/intermediate/` | Merges, overlap outputs, filtering inputs, removal audits | DVC pipeline outputs with `push: false`; reproducible and not uploaded normally |
| `datasets/cleaned/` | Final train/test, shift-only pools, ADMET cohorts, molecular properties, analytics | Public release artifacts published on Hugging Face |
| `*_report.json` | Deterministic stage counts and useful breakdowns | Small ordinary DVC outputs retained with the run |

The raw pointers pin exact byte hashes. Their human-readable provenance and
role are in [sources.yaml](datasets/raw/sources.yaml), while shared batch,
worker, and analytics settings are in [params.yaml](params.yaml).

The project has two deliberately separate distribution channels:

- [Hugging Face](https://huggingface.co/datasets/niccogreek/nmr-canonical-cleaned)
  is the public release for the final cleaned collection.
- Nebius Object Storage is a private maintainer cache. It accelerates work on
  trusted project machines, but its credentials and objects are not part of
  the public reproduction procedure.

The `.dvc` files still pin the exact raw bytes expected by the pipeline. Public
users obtain those bytes from their upstream releases with
[`download_raw_datasets.sh`](scripts/data/download_raw_datasets.sh), then DVC
checks dependencies and reproduces the graph locally.

Canonical and intermediate data still participate in the DVC cache and are
recorded in `dvc.lock`, but `push: false` prevents a normal maintainer
`dvc push` from uploading those large rebuildable files. The local cache uses
hardlinks to avoid a second local copy where the filesystem supports them.

The maintained transformation order is:

```text
raw MST-NMR, NMRexp, NMRTrans, NMRGym, SimNMR-PubChem
  -> schema-v2 canonical Parquets
  -> rich train/validation and source-test merges
  -> move rich test spectra whose molecules occur in SimNMR into rich train
  -> remove residual test overlap with extended rich train and NMRGym
  -> common filtering of rich train, rich test, SimNMR, and NMRGym
  -> prepare ADMET cohorts and remove matched records from rich pretraining
  -> calculate aligned RDKit molecular-property sidecars
  -> generate final analytics
```

All NMR-to-NMR overlap uses exact equality of converter-produced
`smiles_canonical`. External ADMET structures are matched with exact full RDKit
InChIKeys. Structured reports record outcomes such as row counts, filtering
reasons, overlap removals, and cohort sizes without duplicating DVC hashes or
environment information already captured elsewhere.

The public cleaned release is available at
[nmr-canonical-cleaned](https://huggingface.co/datasets/niccogreek/nmr-canonical-cleaned).
Its files can be used directly without reproducing the raw-to-cleaned pipeline.

## Reproducing the dataset pipeline

### 1. Prepare the environment

From the repository root, activate the main environment:

```bash
source ~/.bashrc
nmr-env main
```

The main requirements include RDKit, PyArrow, DVC, the Hugging Face CLI, and
the YAML dependency used by the pipeline generator. DVC's S3 extra supports
the optional private maintainer cache. See
[scripts/envs_scr/README.md](scripts/envs_scr/README.md) to create or repair it.

### 2. Download the raw inputs

```bash
scripts/data/download_raw_datasets.sh all
```

The script downloads fixed upstream revisions of ADMET, MST-NMR, NMRexp,
NMRTrans, NMRGym, and the roughly 400 GB SimNMR-PubChem LMDB into the exact
paths expected by `dvc.yaml`. It verifies the selected directories against the
committed `.dvc` pointers by default. Downloads can also be requested
individually; run it without arguments to see the available targets.

The raw licences and provenance remain those of the upstream projects. Review
[sources.yaml](datasets/raw/sources.yaml) before downloading or redistributing
them. Do not rename or copy raw files into the model submodules.

### 3. Inspect configuration

```bash
dvc dag
cat params.yaml
```

Edit `params.yaml` for batch sizes, worker counts, or plot sampling. Regenerate
`dvc.yaml` only when a stage command, dependency, output, or topology changes:

```bash
scripts/data/generate_dvc_pipeline.sh
```

The generator calls `dvc stage add` and validates the DAG. It does not execute
any stage and does not create `dvc.lock`.

### 4. Reproduce

Run the complete pipeline:

```bash
dvc repro
```

Or target a final component while allowing DVC to run stale prerequisites:

```bash
dvc repro filter_test
dvc repro analyze_cleaned_collection
```

The full run includes conversion and filtering of the 106-million-record
SimNMR corpus. Plan disk, time, and CPU capacity before starting it. DVC invokes
the scripts with `--quiet`, so the combined log retains one progress bar per
stage plus concise start/completion lines and real errors.

### 5. Review and release a successful run

```bash
git status --short
git add dvc.yaml dvc.lock params.yaml datasets/raw/*.dvc
```

Review the processing reports and commit the relevant code, documentation, and
`dvc.lock` together. The lock file records exact commands, parameters,
dependencies, and output hashes. Maintainers publish the reviewed cleaned
directory to Hugging Face with:

```bash
hf upload niccogreek/nmr-canonical-cleaned datasets/cleaned . \
  --repo-type dataset \
  --exclude .gitignore \
  --exclude "*_processor_compatibility.json"
```

Users who only need the released collection can instead run:

```bash
hf download niccogreek/nmr-canonical-cleaned \
  --repo-type dataset \
  --local-dir datasets/cleaned
```

### Private maintainer cache

The committed DVC configuration points to the private Nebius cache. Authorized
maintainers may configure machine-local AWS credentials and run `dvc push` or
`dvc pull`; public users neither need nor receive those credentials. A DVC
remote failure therefore does not prevent public reproduction from upstream
raw releases, nor direct use of the Hugging Face cleaned release.

More detail is in [datasets/README.md](datasets/README.md),
[scripts/data/README.md](scripts/data/README.md), and
[scripts/data/postprocess/README.md](scripts/data/postprocess/README.md).

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
Python layout, and from `scripts/data/canonicalize/README.md` for conversion or
canonical-dataset analysis.

Before a push, check `git status`, `git diff --cached`, and
`git submodule status`.

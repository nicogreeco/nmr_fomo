# Dataset workspace

This directory is the local data root for the project. Large data files are not
stored in Git: they are downloaded from the public release or materialized by
DVC. A fresh clone therefore contains several apparently empty directories.
Their `.gitignore` files preserve the paths expected by scripts without
committing generated Parquets, LMDBs, CSVs, or plots.

## Directory ownership

| Path | Contents |
|---|---|
| `raw/` | Original source releases pinned by `.dvc` files |
| `canonical/` | Source-specific conversions to canonical schema v2 |
| `intermediate/` | Temporary merge, overlap, filtering, and audit outputs |
| `cleaned/` | Published final datasets, molecular sidecars, and analytics |
| `train_splits/` | Local molecule-safe train/validation files for FoMoNMR |
| `downstream_structural_information/` | Dataset generated for structural probes |

`datasets_old/` is an ignored local archive and is not a dependency of the DVC
DAG. Do not use it as an implicit input or copy its counts into reports for a
new run.

The tracked [`cleaned/README.md`](cleaned/README.md) is the Hugging Face data
card and [`cleaned/ANALYTICS.md`](cleaned/ANALYTICS.md) indexes the generated
analytics. They remain useful even when the data files have not been
downloaded, so they should not be removed with the empty placeholders.

## Use the published data

Most users should download the processed release directly:

```bash
hf download niccogreek/nmr-canonical-cleaned \
  --repo-type dataset \
  --local-dir datasets/cleaned
```

Create the training-only splits when needed:

```bash
PYTHONPATH=scripts python -m data.postprocess.split_foundation_datasets
PYTHONPATH=scripts python -m data.postprocess.split_maccs_probe \
  datasets/train_splits/rich_val.parquet \
  datasets/train_splits/rich_val_mol_properties.parquet
```

The first command creates aligned NMR and molecular-property files under
`train_splits/`; the second creates the fixed MACCS diagnostic split. Running
the utilities directly keeps the committed full-build DVC lock unchanged.

## Rebuild from source

To reproduce the complete collection, install the main environment and run:

```bash
scripts/data/download_raw_datasets.sh all
dvc repro
```

The downloader obtains the pinned public inputs in the exact paths consumed by
the pipeline and verifies them against the committed DVC pointers. The full
rebuild includes the roughly 400 GB SimNMR source; use `dvc repro STAGE` when a
single downstream product is sufficient. `dvc dag` shows the available stages.

The root [`dvc.yaml`](../dvc.yaml) is generated from
[`scripts/data/generate_dvc_pipeline.sh`](../scripts/data/generate_dvc_pipeline.sh).
Normal use does not require regenerating it. Change `params.yaml` for exposed
pipeline parameters; change the generator only when adding or restructuring a
stage.

## Main outputs

A successful full run produces:

```text
cleaned/
├── README.md
├── ANALYTICS.md
├── rich.parquet
├── test_benchmark.parquet
├── simnmr.parquet
├── nmrgym.parquet
├── {rich,test_benchmark,simnmr,nmrgym}_mol_properties.parquet
├── *_report.json
├── admet/
│   ├── preparation_report.json
│   └── {ames,ld50_zhu,solubility_aqsoldb,lipophilicity_astrazeneca,sangster_logp}/{train_val,test}.{csv,parquet}
└── analytics/
```

Each maintained transformation writes a compact JSON report with input, output,
and row counts. `dvc.lock` is authoritative for the commands, parameters, and
hashes of the current materialized pipeline.

Do not edit generated canonical or derived files by hand. Change the relevant
script or parameter and rerun its DVC stage.

The Python schema and processing commands are documented in
[`scripts/data/README.md`](../scripts/data/README.md). Source provenance,
released counts, schema fields, and generated dataset audits are documented in
[`cleaned/README.md`](cleaned/README.md) and
[`cleaned/ANALYTICS.md`](cleaned/ANALYTICS.md). Exact transformation commands
are described in
[`scripts/data/postprocess/README.md`](../scripts/data/postprocess/README.md).

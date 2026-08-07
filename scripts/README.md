# Scripts

This directory contains the small Python tools used to prepare canonical NMR
records and compare representations from the existing models. The reusable data
code is deliberately separate from model-specific benchmarking code.

## Layout

```text
scripts/
├── data/                 canonical data code and preparation utilities
│   ├── canonicalize/    source-specific conversion programs
│   └── postprocess/     derived merges, audits, and benchmark preparation
├── model_benchmarks/     processors and embedders for existing NMR models
├── envs_scr/             environment setup notes and requirement lists
├── test_notebook.ipynb    exploratory notebook
└── extract_embeddings.py thin command-line entry point
```

The modules directly under `data` form the reusable core. They define canonical
records and read already-canonical Parquet files without depending on published
model families. The `canonicalize` and `postprocess` subpackages group the
one-time and derived-data tools alongside that core without making those tools
part of its public API.

`model_benchmarks` contains the bridge to those existing repositories. A
processor converts canonical records into one model's batch format. An
embedder loads that model and returns a fixed-size representation.

`data/canonicalize` is a separate, one-time data-preparation area. Nothing in
the dataset, processor, embedder, or extraction command converts or rewrites
the input dataset.

The notebook remains useful for interactive inspection. The reproducible final
collection analysis lives in `data/postprocess/analyze_cleaned_datasets.py`.

The current canonical storage format is schema version 2. Raw-data converters
derive canonical SMILES, formula, and atom symbols with RDKit and do not store
molecular coordinates. See `data/canonicalize/README.md` for the exact policy
and conversion commands.

## Embedding flow

```text
CanonicalParquetDataset
    -> model-specific processor/collator
    -> model-specific batch
    -> lightweight embedder
    -> embeddings and ordered record IDs
```

Run commands from the project root with `scripts` on `PYTHONPATH`:

```bash
PYTHONPATH=scripts python scripts/extract_embeddings.py \
  --model nmrpeak \
  --input datasets/mst_nmr/test.parquet \
  --output embeddings/mst_nmr_nmrpeak.parquet \
  --device cpu \
  --batch-size 64 \
  --mode canonical
```

Use the Python environment for the selected model family. Imports are lazy, so
loading `data` or `model_benchmarks` does not import all four model repositories.

## Where to make changes

- Change the common record definition or reader only in `data/`.
- Change canonical-to-model conversion in
  `model_benchmarks/processors/<model>.py`.
- Change checkpoint loading, encoder execution, or pooling in
  `model_benchmarks/embedders/<model>.py`.
- Change model selection in `model_benchmarks/factory.py`.
- Keep exploratory code in `test_notebook.ipynb` and derived-data utilities in
  `data/postprocess/`, not in the reusable core modules.

The concise architecture and canonical field conventions are documented in
`contex/Embedding_Pipeline_Architecture.md` and
`contex/Canonicalization_Implementation_Notes.md`.

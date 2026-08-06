# Canonicalization tools

This folder contains source-specific scripts that create or inspect canonical
NMR records. It is separate from the reusable `scripts/data/` package: the
latter only reads records that are already canonical.

Use the converter that matches the original release, then validate the result
with the streaming analysis tool. These commands write a new output file; they
do not modify the source dataset.

## Canonical schema version 2

Run converters in the main project environment (`nmr-env main`). They require
RDKit as well as PyArrow. For every converted row, the converter preserves the
selected source SMILES in `smiles` and uses RDKit to overwrite three derived
fields:

- `smiles_canonical` is the isomeric canonical SMILES;
- `molecular_formula` is calculated from the parsed molecule;
- `atoms` is the atom-symbol sequence obtained after reparsing that canonical
  SMILES.

The `atoms` list contains atoms explicit in the SMILES graph, normally heavy
atoms; implicit hydrogens still contribute to `molecular_formula`. Reparsing
the canonical string makes atom order consistent with the emitted canonical
SMILES. This is serialization consistency under the recorded RDKit version,
not tautomer, salt, charge, or protonation standardization.

The v2 schema does not contain molecular coordinates. Source coordinate arrays
are deliberately ignored because they are large, no current spectral processor
uses them, and a task that needs a conformer can generate one from the stored
structure. A missing or invalid source SMILES stops conversion with the source
location instead of silently retaining stale source metadata. The NMR-Solver
converter is the explicit exception: it skips only records for which RDKit
cannot derive and reparse canonical chemical metadata, writing their IDs,
source SMILES, and reasons to a JSONL rejection report beside the output. Each
Parquet footer records both `canonical_schema_version=2` and the installed
RDKit version.

The NMR-Solver converter code follows the same v2 rules, but its existing
105-million-row Parquet has intentionally not been regenerated yet and remains
a legacy schema-v1 artifact until that separate run is performed.

```bash
PYTHONPATH=scripts python scripts/canonicalize/analysis/analyze_parquet.py \
  datasets/<source>/all.parquet \
  --output datasets/<source>/canonical_analysis.json
```

The normal analysis uses Arrow's column operations and has bounded memory. It
checks the stored Arrow schema but deliberately does not retain every ID or
decode every record into Python. For a small file, add
`--check-duplicate-record-ids` for an exact duplicate-ID count and
`--validate-records` for full record-by-record validation. Both options are
slow and use substantially more memory on a very large file.

For reproducible source/split and processor-compatibility audits, run:

```bash
PYTHONPATH=scripts python scripts/canonicalize/analysis/compare_source_and_splits.py \
  mst_nmr

PYTHONPATH=scripts python \
  scripts/canonicalize/analysis/analyze_processor_compatibility.py \
  datasets/mst_nmr/all.parquet \
  --dataset-name mst_nmr \
  --output datasets/mst_nmr/processor_compatibility.json
```

The same commands accept `nmrexp` or `nmrtrans`. The source/split command
writes `source_comparison.json` and `split_consistency.json` beside the
dataset.

Run `scripts/postprocess/merge_datasets.py` only after every input has been
regenerated. The merger rejects legacy or structurally different schemas and
inputs produced by different RDKit versions instead of labeling them as v2.

For the 106-million-row SimNMR-PubChem conversion, `convert_nmrsolver.py`
can run RDKit record conversion in separate processes while one parent process
keeps LMDB reading and Parquet/JSONL writing ordered and bounded in memory:

```bash
PYTHONPATH=scripts python scripts/canonicalize/convert_nmrsolver.py \
  models/NMR-Solver/database/metadata/PubChem_merged_id.lmdb \
  datasets/nmrsolver/all.parquet --overwrite --workers 4
```

Start with four workers and increase only after confirming available CPU, RAM,
and storage throughput. Each worker task converts 256 records by default,
which reduces process-pool scheduling overhead; adjust `--records-per-task`
only after measuring the machine. `--max-in-flight` bounds queued source
records (default: two tasks per worker). The generated record and
rejection-report order remains source-key order.

The canonical fields and the rules to preserve during future conversions are
in [Canonicalization Implementation Notes](../../contex/Canonicalization_Implementation_Notes.md).
For the reason different source releases need different treatment, see
[Dataset Filtering and Processing](../../contex/Dataset_Filtering_and_Processing.md).

Do not import a converter from training or embedding code. A model benchmark
should consume the resulting canonical Parquet file through `scripts/data/`.

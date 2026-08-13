# Canonicalization tools

This folder contains source-specific scripts that create or inspect canonical
NMR records. It is separate from the reusable `scripts/data/` package: the
latter only reads records that are already canonical.

Use the converter that matches the original release, then validate the result
with the streaming analysis tool. The production invocations are the
`canonicalize_*` stages in the root `dvc.yaml`; their inputs come from
`datasets/raw/`, and their outputs go to `datasets/canonical/`. These commands
write new files and never modify the source datasets.

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

The top-level `h_nmr_peaks` and `c_nmr_peaks` fields are always lists; a
modality with no usable peaks is written as `[]`. In rich peak-table sources,
every proton peak also has a J list, with `[]` representing no retained or
reported numerical coupling. Shift-only sources use `j_values = null` because
J is structurally unavailable.

A previously completed SimNMR-PubChem conversion following these v2 rules
contained 105,764,812 records from 105,764,875 candidate rows. The 63 rejected
rows could not provide coherent RDKit chemical metadata and were written to the
adjacent JSONL rejection report. These are measured historical results; the
next DVC reproduction writes its own counts to
`datasets/canonical/simnmr/all_report.json`.

Each converter also writes a compact deterministic processing JSON beside its
Parquet (`<output>_report.json`). It records the stage, input, output, and
record counts; use `--report-output` when a DVC stage needs a different path.
SimNMR keeps its separate JSONL rejection audit because that file identifies
the individual skipped source rows.

```bash
PYTHONPATH=scripts python scripts/data/canonicalize/analysis/analyze_parquet.py \
  datasets/canonical/<source>/all.parquet \
  --output datasets/canonical/<source>/canonical_analysis.json
```

The normal analysis uses Arrow's column operations and has bounded memory. It
checks the stored Arrow schema but deliberately does not retain every ID or
decode every record into Python. For a small file, add
`--check-duplicate-record-ids` for an exact duplicate-ID count and
`--validate-records` for full record-by-record validation. Both options are
slow and use substantially more memory on a very large file.

For reproducible source/split and processor-compatibility audits, run:

```bash
PYTHONPATH=scripts python scripts/data/canonicalize/analysis/compare_source_and_splits.py \
  mst_nmr

PYTHONPATH=scripts python \
  scripts/data/canonicalize/analysis/analyze_processor_compatibility.py \
  datasets/canonical/mst_nmr/train.parquet \
  --dataset-name mst_nmr \
  --output datasets/canonical/mst_nmr/processor_compatibility.json
```

The source/split command supports the published split sources `mst_nmr`,
`nmrexp`, and `nmrtrans`; it writes `source_comparison.json` and
`split_consistency.json` beside the dataset. `analyze_parquet.py` can inspect
every canonical source, including the unsplit NMRGym and SimNMR-PubChem files.

Run `scripts/data/postprocess/merge_datasets.py` only after every input has been
regenerated. The merger rejects legacy or structurally different schemas and
inputs produced by different RDKit versions instead of labeling them as v2.

For the 106-million-row SimNMR-PubChem conversion, `convert_nmrsolver.py`
can run RDKit record conversion in separate processes while one parent process
keeps LMDB reading and Parquet/JSONL writing ordered and bounded in memory.
Serialized LMDB values are unpickled in the workers, which return Arrow-ready
rows; the result is still one physical Parquet file:

```bash
PYTHONPATH=scripts python scripts/data/canonicalize/convert_nmrsolver.py \
  datasets/raw/simnmr_pubchem/metadata/PubChem_merged_id.lmdb \
  datasets/canonical/simnmr/all.parquet --workers 4
```

The DVC pipeline uses four workers and 256 records per task; local
profiling found no benefit from eight workers for this parent-written single
Parquet design. `--max-in-flight` bounds queued source records (default: two
tasks per worker). The generated record and rejection-report order remains
source-key order. Every converter also supports `--quiet` and `--no-progress`;
quiet mode keeps the progress bar and real errors.

NMRGym is a separate shift-only source. It reads its released pickle split
files, writes a schema-v2 combined file, and represents unavailable proton
annotations with null fields while keeping modalities themselves as lists:

```bash
PYTHONPATH=scripts python scripts/data/canonicalize/convert_nmrgym.py \
  datasets/raw/nmrgym datasets/canonical/nmrgym/all.parquet
```

The canonical fields and the rules to preserve during future conversions are
in [Canonicalization Implementation Notes](../../../contex/Canonicalization_Implementation_Notes.md).
For the reason different source releases need different treatment, see
[Dataset Filtering and Processing](../../../contex/Dataset_Filtering_and_Processing.md).

Converter parameters are centralized in `params.yaml`. Do not hand-edit
`dvc.yaml`; change the Bash generator when stage structure changes.

Do not import a converter from training or embedding code. A model benchmark
should consume the resulting canonical Parquet file through `scripts/data/`.

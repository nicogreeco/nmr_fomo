# Canonicalization tools

This folder contains source-specific scripts that create or inspect canonical
NMR records. It is separate from the reusable `scripts/data/` package: the
latter only reads records that are already canonical.

Use the converter that matches the original release. Production invocations
are recorded in the `canonicalize_*` stages of the root `dvc.yaml`; converters
write new files and never modify their sources.

| Script | Accepted source |
| --- | --- |
| `convert_mst_nmr.py` | MST-NMR LMDB file or directory |
| `convert_nmrexp.py` | NMRexp LMDB file or directory |
| `convert_nmrtrans.py` | NMRTrans compressed pickle file or directory |
| `convert_nmrgym.py` | released NMRGym split directory |
| `convert_nmrsolver.py` | SimNMR/NMR-Solver PubChem LMDB |

All converters use the same basic interface:

```bash
PYTHONPATH=scripts python -m data.canonicalize.CONVERTER \
  INPUT OUTPUT.parquet
```

Use `--help` for source-specific options. Common options include
`--row-group-size`, `--report-output`, `--quiet`, and `--no-progress`.

## Canonical schema version 2

Run converters in the main project environment. They preserve source SMILES
and use RDKit to derive:

- `smiles_canonical` is the isomeric canonical SMILES;
- `molecular_formula` is calculated from the parsed molecule;
- `atoms` is the atom-symbol sequence obtained after reparsing that canonical
  SMILES.

The output has no coordinate column. H and C modalities are always lists;
unavailable peak annotations stay null. Invalid structures fail with source
context, except for the large SimNMR converter, which writes rejected rows to
an adjacent JSONL audit and continues. Parquet metadata records schema version
2 and the RDKit version. See [`../README.md`](../README.md) for the Python
objects and
[Canonicalization Implementation Notes](../../../contex/Canonicalization_Implementation_Notes.md)
for the exact semantic rules.

Each converter also writes a compact deterministic processing JSON beside its
Parquet (`<output>_report.json`). It records the stage, input, output, and
record counts; use `--report-output` when a DVC stage needs a different path.
SimNMR keeps its separate JSONL rejection audit because that file identifies
the individual skipped source rows.

```bash
PYTHONPATH=scripts python -m data.canonicalize.analysis.analyze_parquet \
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
PYTHONPATH=scripts python -m data.canonicalize.analysis.compare_source_and_splits \
  mst_nmr

PYTHONPATH=scripts python -m \
  data.canonicalize.analysis.analyze_processor_compatibility \
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

For the large SimNMR-PubChem source, `convert_nmrsolver.py` can run RDKit
conversion in worker processes while the parent keeps LMDB reading and output
writing ordered:

```bash
PYTHONPATH=scripts python -m data.canonicalize.convert_nmrsolver \
  datasets/raw/simnmr_pubchem/metadata/PubChem_merged_id.lmdb \
  datasets/canonical/simnmr/all.parquet --workers 4
```

`--max-in-flight` bounds queued work. Output and rejection-report order remains
source-key order.

NMRGym is a separate shift-only source. It reads its released pickle split
files, writes a schema-v2 combined file, and represents unavailable proton
annotations with null fields while keeping modalities themselves as lists:

```bash
PYTHONPATH=scripts python -m data.canonicalize.convert_nmrgym \
  datasets/raw/nmrgym datasets/canonical/nmrgym/all.parquet
```

The canonical fields and the rules to preserve during future conversions are
in [Canonicalization Implementation Notes](../../../contex/Canonicalization_Implementation_Notes.md).
For the reason different source releases need different treatment, see
[Dataset Filtering and Processing](../../../contex/Dataset_Filtering_and_Processing.md).

Converter parameters used by the maintained pipeline are in `params.yaml`.
Run the converter directly for a local experiment; use `dvc repro
canonicalize_<source>` to reproduce a maintained output.

Do not import a converter from training or embedding code. A model benchmark
should consume the resulting canonical Parquet file through `scripts/data/`.

# Canonicalization tools

This folder contains source-specific scripts that create or inspect canonical
NMR records. It is separate from the reusable `scripts/data/` package: the
latter only reads records that are already canonical.

Use the converter that matches the original release, then validate the result
with the streaming analysis tool. These commands write a new output file; they
do not modify the source dataset.

```bash
PYTHONPATH=scripts python scripts/canonicalize/analyze_parquet.py \
  datasets/<source>/all.parquet \
  --output datasets/<source>/canonical_analysis.json
```

The normal analysis uses Arrow's column operations and has bounded memory. It
checks the stored Arrow schema but deliberately does not retain every ID or
decode every record into Python. For a small file, add
`--check-duplicate-record-ids` for an exact duplicate-ID count and
`--validate-records` for full record-by-record validation. Both options are
slow and use substantially more memory on a very large file.

The canonical fields and the rules to preserve during future conversions are
in [Canonicalization Implementation Notes](../../contex/Canonicalization_Implementation_Notes.md).
For the reason different source releases need different treatment, see
[Dataset Filtering and Processing](../../contex/Dataset_Filtering_and_Processing.md).

Do not import a converter from training or embedding code. A model benchmark
should consume the resulting canonical Parquet file through `scripts/data/`.

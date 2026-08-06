# Post-processing tools

These tools operate on already-canonical datasets or on derived audit outputs.
They are not source converters, do not define the canonical schema, and should
not be used as a substitute for source-level curation.

Run the commands from the repository root with `scripts` on `PYTHONPATH`.

## `merge_datasets.py`

Concatenates compatible canonical Parquet files into one v2 file. It requires
the current Arrow schema, canonical schema version, and a common RDKit version.
It preserves input rows in the requested order and does not deduplicate
molecules, assign splits, or remove benchmark overlap.

## `disjoin_benchmark_from_train.py`

This is a deliberately narrow, ad hoc benchmark-preparation helper. Its
intended use is:

1. pass `merged_train_val_all.parquet` as the first input;
2. pass one or more benchmark/test Parquet files after it;
3. write a derived train/validation file with records overlapping the benchmark
   removed from the first input.

The script compares connectivity InChIKeys and removes every record in the
first input whose connectivity key appears in a later input. This conservative
rule is intentional for this benchmark split, but it can group stereoisomers
with the same connectivity. It must not be used for exact property matching,
general dataset deduplication, or canonical source conversion. Those workflows
use their own identity and provenance policies.

The output is a derived benchmark artifact, not a newly canonicalized source
dataset. The script does not modify any input file.

## `extract_annotated_peaks.py`

Uses the ADMET audit's exact full-InChIKey matches to create endpoint-specific
annotated outputs and a merged dataset with those matched record IDs removed.
This is the property-benchmark path; it is distinct from the connectivity-based
benchmark disjoining helper above.

## `calculate_mol_properties.py`

Streams one canonical Parquet file and writes a same-directory CSV containing
one RDKit molecular-property row per canonical `record_id`. It reads only
`record_id` and `smiles_canonical`, processes bounded Arrow batches, and can
split each batch across several RDKit worker processes. The default output for
`datasets/merged/merged_train_val_all.parquet` is
`datasets/merged/merged_train_val_all_mol_properties.csv`.

```bash
PYTHONPATH=scripts python scripts/data/postprocess/calculate_mol_properties.py \
  datasets/merged/merged_train_val_all.parquet \
  --workers 8
```

The CSV contains the following columns in addition to `record_id` and
`smiles_canonical`:

- `exact_molecular_weight`, `calculated_logp`, `tpsa`, `hba`, `hbd`,
  `rotatable_bonds`, `fraction_csp3`, and `aromatic_atom_fraction`;
- one multi-label `has_*` column for each SMARTS-defined group: amine, amide,
  alcohol/phenol, ester, carboxylic acid, aldehyde/ketone, nitrile,
  carbon-halogen bond, and heteroaromatic ring;
- `morgan_ecfp4_2048_hex`: radius-2, 2,048-bit Morgan/ECFP4 fingerprint,
  encoded as reversible hexadecimal RDKit binary bytes (512 characters);
- `maccs_keys_166_bits`: the 166 usable RDKit MACCS keys. RDKit internally
  exposes 167 positions; the unused position 0 is omitted here.

Rows with missing or invalid SMILES are retained and marked through
`rdkit_status` and `rdkit_error`, instead of being silently dropped. Output is
written to `.<name>.partial` and promoted only on success; an existing CSV is
protected unless `--overwrite` is passed.

For larger machines, increase `--workers`; `--batch-size` bounds the rows held
by the parent at once, while `--records-per-task` bounds each serialized worker
task. The defaults are deliberately conservative (`4096`, `512`, and up to
four workers).

## `admet_overlap_audit.ipynb`

Downloads or reuses the configured property releases, audits their overlap with
the merged NMR dataset, and writes reports under `datasets/properties/`. It does
not rewrite the merged canonical input.

All post-processing outputs are disposable derived artifacts. Keep the source
canonical Parquet files unchanged and record the command and identity policy

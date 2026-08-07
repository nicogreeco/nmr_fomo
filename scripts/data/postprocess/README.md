# Post-processing tools

These tools operate on already-canonical datasets or on derived audit outputs.


Run the commands from the repository root with `scripts` on `PYTHONPATH`.

## `merge_datasets.py`

Concatenates compatible canonical Parquet files into one v2 file. It requires
the current Arrow schema, canonical schema version, and a common RDKit version.
It preserves input rows in the requested order and does not deduplicate
molecules, assign splits, or remove benchmark overlap.

## `disjoin_benchmark_from_train.py`

This is a deliberately narrow, ad hoc benchmark-preparation helper. The first
Parquet is always the dataset being filtered; every later Parquet is only a
comparison dataset and remains unchanged. Input order therefore determines the
direction of removal. The current benchmark-test preparation command is:

```bash
PYTHONPATH=scripts python \
  scripts/data/postprocess/disjoin_benchmark_from_train.py \
  datasets/merged/merged_benchmark_test.parquet \
  datasets/merged/merged_train_val_all.parquet \
  --output datasets/merged/merged_benchmark_test_disjoint.parquet \
  --overwrite
```

This removes benchmark-test records whose molecular connectivity is present in
the train/validation collection. Reversing the first two inputs would instead
remove overlapping train/validation records.

The script compares connectivity InChIKeys and removes every record in the
first input whose connectivity key appears in a later input. This conservative
rule is intentional for this benchmark split, but it can group stereoisomers
with the same connectivity.

The script does not modify any input file. It preserves all Parquet
footer metadata from the first input and adds the comparison inputs, identity
rule, script path, and removed-record count. Output is written to a hidden
`.partial` file and promoted only after a successful close.

## `extract_annotated_peaks.py`

Creates endpoint-specific annotated outputs and a merged dataset with those
matched record IDs removed. It independently calculates RDKit full InChIKeys
from the TDC `Drug` SMILES column and from the merged canonical SMILES, then
keeps only exact full-key equality. The audit notebook is useful for exploration
and reporting, but is not an input to this production extraction step. This is
the property-benchmark path; it is distinct from the connectivity-based
benchmark disjoining helper above.

```bash
PYTHONPATH=scripts python scripts/data/postprocess/extract_annotated_peaks.py \
  --merged datasets/merged/merged_train_val_all.parquet \
  --overwrite
```

The script reads the configured local TDC release, writes the pre-cleaning
endpoint files under `datasets/properties/annonated_peaks/`, and creates
`merged_train_val_all_disjoint.parquet`. The final cleaned endpoint subsets are
then placed under `datasets/cleaned/admet/` with their paired label CSVs.

## `filter_dataset.py`

Creates a cleaned derived copy of one canonical schema-v2 Parquet file. The
input is not modified. Output names are automatic: `records.parquet` produces
`records_cleaned.parquet` and the minimal audit `records_removed.parquet`.

```bash
PYTHONPATH=scripts python scripts/data/postprocess/filter_dataset.py \
  datasets/merged/merged_train_val_all_disjoint.parquet
```

The filter keeps one or both spectral modalities, requires finite shifts in
-5..20 ppm for H and -50..300 ppm for C, limits each modality to 60 peaks,
limits each proton peak to six J values, rejects non-finite or negative J
values and non-positive supplied proton integrations, and removes
multi-fragment structures. `J=0` and missing integration are retained.
Molecular-property and drug-likeness thresholds are not used.

Exact duplicates use canonical SMILES plus sorted exact H and C shift lists;
a modality without usable peaks contributes its canonical empty list. Hash
matches are verified with the actual key values. The best-annotated row is retained,
with `record_id` used as a deterministic final tie-breaker. Same-molecule rows
with different shift lists remain separate records, and output records retain
the canonical schema without extra columns.

For large collections such as NMR-Solver, duplicate detection is deliberately
two-stage: it first finds repeated hashes, then groups only those candidates by
the complete SMILES-and-shift-list key. The hash is therefore only a scalable
prefilter.

Existing outputs are protected unless `--overwrite` is passed. Both files are
first written to hidden `.partial` paths and promoted only after successful
completion.

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

## `analyze_cleaned_datasets.py`

Generates the final data-card analytics for `train_val`,
`test_benchmark`, and the three ADMET cohorts under `datasets/cleaned`.
It writes source inventories, molecular-property summaries,
functional-group prevalence, peak and multiplicity statistics, proton
annotation completeness, ADMET target summaries, and six distribution plots.

Summary tables use every eligible record. Plots use a deterministic sample of
at most 20,000 records per source or endpoint and limit only the displayed
violin values to `Q1 - 2.5*IQR` through `Q3 + 2.5*IQR`.

```bash
PYTHONPATH=scripts python scripts/data/postprocess/analyze_cleaned_datasets.py
```

The default output directory is `datasets/cleaned/analytics`, with one
subdirectory each for `train_val`, `test_benchmark`, `admet`, and collection-
level summaries. Use `--cleaned-root` for another final collection and
`--sample-per-group` to change the plotting sample size.

For the separately stored shift-only SimNMR-PubChem collection, pass its
cleaned Parquet directly. The script derives the adjacent molecular-property
CSV path and writes the results under `datasets/cleaned/analytics/simnmr`:

```bash
PYTHONPATH=scripts python scripts/data/postprocess/analyze_cleaned_datasets.py \
  --nmrsolver-parquet datasets/cleaned/simnmr.parquet
```

The NMR-Solver path makes three dataset-specific choices. It treats the source
as shift-only, so J-coupling summaries and the J violin are intentionally left
empty rather than inferred from unavailable annotations. Its property CSV is
aligned by row order with the Parquet file: `calculate_mol_properties.py`
preserves that order and both files retain one row per input record, avoiding a
memory-heavy 100-million-row join. Finally, numerical summaries still use every
record, while plots use the first projected `--sample-per-group` rows. Selecting
that small prefix before nested peak lists are collected avoids materialising the
whole NMR-Solver dataset solely to make a figure.

## `admet_overlap_audit.ipynb`

Downloads or reuses the configured property releases, audits their overlap with
the merged NMR dataset, and writes reports under `datasets/properties/`. It does
not rewrite the merged canonical input.

Intermediate post-processing outputs are reproducible derived artifacts. The
files under `datasets/cleaned/` are the curated final release; keep their
source canonical Parquet inputs unchanged and record the command and identity
policy used to regenerate them.

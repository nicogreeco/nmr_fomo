# Post-processing tools

These commands operate on canonical schema-v2 Parquet files. They do not
modify canonical source files in place. Run them from the repository root with
`scripts` on `PYTHONPATH` and use the main NMR environment.

```bash
source ~/.bashrc
nmr-env main
```

The production invocations are explicit stages in the root `dvc.yaml`.
Commands below explain individual tools; use `dvc repro` for the maintained
end-to-end workflow and `params.yaml` for shared batch and worker settings.

## Final collection workflow

The maintained preparation order is:

```text
merge rich train/validation inputs
merge rich benchmark-test inputs
        |
        v
move benchmark molecules found in SimNMR into train/validation
        |
        v
remove residual benchmark molecules found in:
  - the extended rich train/validation pool
  - NMRGym
        |
        v
apply the common filter to the final train, test, SimNMR, and NMRGym pools
        |
        v
prepare ADMET cohorts from the cleaned extended train/validation pool
and write the cleaned ADMET-disjoint train/validation dataset
        |
        v
calculate molecular-property sidecars for final Parquet files
        |
        v
generate final analytics
```

All NMR-to-NMR overlap operations use exact equality of the existing
`smiles_canonical` strings. The canonical converters generate these strings
with RDKit and record the RDKit version; overlap scripts require compatible
input versions. This avoids reparsing very large comparison datasets and keeps
stereoisomers distinct.

ADMET is different because its property SMILES come from an external release.
`prepare_admet_datasets.py` calculates full RDKit InChIKeys for both sides and
requires exact full-key equality. Connectivity-only matching is not part of
the maintained pipeline.

Intermediate names should describe their current role. Final collection files
should use stable names such as `train_val.parquet`, `test_benchmark.parquet`,
`simnmr.parquet`, and `nmrgym.parquet`; historical names such as `double` and
`extended` are not needed in the final release.

## Common command conventions

Transformation commands use the same conventions where applicable:

- input Parquet paths are positional;
- outputs use `--output`, or explicit role names when there are two outputs;
- `--batch-size` bounds each Arrow batch and defaults to 50,000 rows;
- `--report-output` selects the processing JSON path;
- `--overwrite` permits replacement only after temporary outputs are complete;
- `--quiet` suppresses non-essential warnings and intermediate messages but keeps the progress bar and errors;
- `--no-progress` disables only the progress bar.

Canonical Parquet outputs are written to hidden `.partial` files and promoted
only after a successful close. The scripts preserve the input schema and add
small footer fields describing the post-processing step, repository-relative
script path, inputs, identity policy, and relevant row counts.

By default, every command writes a compact report beside its primary output.
Reports use the same `stage`, `inputs`, `outputs`, and `counts` fields,
with an optional `details` section only for useful breakdowns. These JSONs are
ordinary cached DVC outputs: they record transformation results without
repeating the command, Git revision, DVC hashes, or environment already
captured by the pipeline. They are not configured as no-cache DVC metrics.

`common.py` contains shared Parquet validation, temporary-output, exact-SMILES
indexing, and record-ID filtering helpers. `scripts/data/console.py` provides the
common CLI flags, warning suppression, and progress bars. Both are internal
modules, not commands.

## `merge_datasets.py`

Concatenates compatible canonical Parquet files in the requested order. It
does not deduplicate, assign splits, filter records, or remove overlap.

```bash
PYTHONPATH=scripts python scripts/data/postprocess/merge_datasets.py \
  datasets/canonical/mst_nmr/train.parquet \
  datasets/canonical/mst_nmr/val.parquet \
  datasets/canonical/nmrexp/train.parquet \
  datasets/canonical/nmrexp/val.parquet \
  datasets/canonical/nmrtrans/train.parquet \
  datasets/canonical/nmrtrans/val.parquet \
  --output datasets/intermediate/rich_train_val.parquet
```

Call the same command separately for the published source-test files. Input
order is retained, which is useful for reproducible source inventories but
must not be mistaken for randomization.

## `move_benchmark_overlaps_to_train.py`

Streams one large reference dataset, normally SimNMR-PubChem, and finds exact
canonical-SMILES matches in the benchmark. Matching rich benchmark records are
removed from the test output and appended unchanged to the train/validation
output. Only the benchmark SMILES-to-record-ID index is retained in memory.

```bash
PYTHONPATH=scripts python \
  scripts/data/postprocess/move_benchmark_overlaps_to_train.py \
  datasets/intermediate/rich_test.parquet \
  datasets/canonical/simnmr/all.parquet \
  datasets/intermediate/rich_train_val.parquet \
  --test-output datasets/intermediate/test_after_simnmr.parquet \
  --train-output datasets/intermediate/train_val_extended.parquet
```

Run this before the general benchmark disjoint. This ordering recovers every
rich benchmark spectrum whose molecule occurs in SimNMR, including molecules
that may also occur in another training resource.

## `remove_benchmark_overlaps.py`

Removes rows only from the first input when its exact `smiles_canonical` occurs
in any later input. Comparison files are never modified. The first dataset is
indexed; later datasets are streamed one at a time.

```bash
PYTHONPATH=scripts python \
  scripts/data/postprocess/remove_benchmark_overlaps.py \
  datasets/intermediate/test_after_simnmr.parquet \
  datasets/intermediate/train_val_extended.parquet \
  datasets/canonical/nmrgym/all.parquet \
  --output datasets/intermediate/test_benchmark_preclean.parquet
```

The command is directional and reusable: the first positional input is always
the file being filtered, while every later positional input is a comparison.

## `filter_dataset.py`

Applies the common cleaning and exact-shift deduplication policy to one
canonical Parquet. It removes records with no NMR peaks, non-finite or
out-of-range shifts, more than 60 peaks per modality, more than six J values
per proton peak, non-finite or negative J values, non-positive supplied proton
integration, or a multi-fragment canonical structure.

Deduplication then uses:

```text
smiles_canonical + sorted exact 1H shifts + sorted exact 13C shifts
```

No shift rounding is applied. The most completely annotated record wins, with
`record_id` as the final deterministic tie-breaker. Same-molecule records with
different shift lists remain separate.

```bash
PYTHONPATH=scripts python scripts/data/postprocess/filter_dataset.py \
  datasets/intermediate/train_val_extended.parquet \
  --output datasets/intermediate/train_val_filtered.parquet \
  --removed-output datasets/intermediate/audits/train_val_removed.parquet
```

If output arguments are omitted, the command uses `<input>_cleaned.parquet`
and `<input>_removed.parquet` beside the input.

## `prepare_admet_datasets.py`

Runs after rich train/validation has been extended and filtered. It reads the
small TDC endpoint files first, calculates their full InChIKeys, then streams
the large cleaned NMR input once and retains only requested matches in memory.
It creates the endpoint `train_val` and `test` label/Parquet pairs and removes
the union of all matched records from the pretraining output.

Repeated or cross-split property identities are handled during the same step:

- agreeing numerical labels within a split are collapsed to one label per NMR
  `record_id`;
- discordant labels within a split are excluded from the supervised cohort;
- a full InChIKey present in both train/validation and test is excluded from
  both supervised cohorts;
- all matched identities excluded by either rule remain excluded from the
  pretraining output to prevent label leakage.

```bash
PYTHONPATH=scripts python scripts/data/postprocess/prepare_admet_datasets.py \
  datasets/intermediate/train_val_filtered.parquet \
  --tdc-root datasets/raw/admet \
  --output-root datasets/cleaned/admet \
  --train-output datasets/cleaned/train_val.parquet
```

Because the input has already passed the common filter, the resulting ADMET
Parquets and ADMET-disjoint train pool do not require another filtering pass.
The command writes `preparation_report.json` beside the endpoint directories.
It follows the common report schema and retains only aggregate matching and
cohort-cleanup counts.

## `calculate_mol_properties.py`

Streams one final canonical Parquet and writes a same-order
`*_mol_properties.csv`. It reads only `record_id` and `smiles_canonical`, uses
bounded batches, and parallelizes the CPU-bound RDKit calculations with a
small process pool.

```bash
PYTHONPATH=scripts python scripts/data/postprocess/calculate_mol_properties.py \
  datasets/cleaned/train_val.parquet \
  --workers 8
```

Function, CLI, and `params.yaml` defaults are identical: 50,000 rows per Arrow
batch, 1,000 records per worker task, and up to eight workers. A bounded queue
keeps workers occupied across Arrow-batch boundaries, and workers serialize
their result rows before returning them to the writer. Local profiling showed
eight workers outperforming four, although RDKit calculation remains the main
cost and machine-specific scaling should still be measured.

The output contains exact molecular weight, Crippen logP, TPSA, HBA/HBD,
rotatable bonds, fraction Csp3, aromatic heavy-atom fraction, nine binary
SMARTS functional-group indicators, Morgan/ECFP4, and MACCS. Invalid structures
remain represented by a row with `rdkit_status` and `rdkit_error` rather than
being silently dropped. The complete column definitions are in
`contex/Dataset Analysis.md`.

## `analyze_cleaned_datasets.py`

Generates source inventories, molecular-property summaries, functional-group
prevalence, peak and multiplicity statistics, proton annotation completeness,
ADMET target summaries, and plots for the final collection.

```bash
PYTHONPATH=scripts python scripts/data/postprocess/analyze_cleaned_datasets.py \
  --cleaned-root datasets/cleaned
```

Before analyzing a molecular-property sidecar, the script verifies that it has
the same number, order, and sequence of `record_id` values as its Parquet. A
stale or manually trimmed sidecar is an error rather than a partial analysis.

Summary tables use all eligible records. Main-dataset plots take a deterministic
sample of at most `--sample-per-group` records from every source, so block order
from the merge cannot hide later sources. The first-row optimization is used
only by single-source shift-only modes such as SimNMR and NMRGym.

Use `--nmrsolver-parquet` or `--nmrgym-parquet` to analyze those independent
shift-only components. For candidate collections with different filenames,
use `--train-val-parquet`, `--test-benchmark-parquet`, `--analytics-dir`, and
optionally `--skip-admet`.
By default analytics writes `processing_report.json` inside its analytics
directory. The DVC stage supplies three explicit report paths outside that
directory—main collection, SimNMR, and NMRGym—while keeping all figures and
tables under `datasets/cleaned/analytics/`.

## `admet_overlap_audit.ipynb`

This notebook remains an exploratory audit. It can compare property releases
and report alternative identity intersections, but it is not an input to the
production ADMET preparation command.

# Post-processing tools

These commands operate on canonical schema-v2 Parquet files. They do not
modify canonical source files in place. Run them from the repository root with
`scripts` on `PYTHONPATH` after activating the main project environment.

The production invocations are stages in the root `dvc.yaml`. Run a script
directly while developing or use `dvc repro STAGE` to reproduce a maintained
output.

| Script | Purpose |
| --- | --- |
| `merge_datasets.py` | concatenate compatible canonical Parquets |
| `move_benchmark_overlaps_to_train.py` | move benchmark rows whose molecules occur in a reference set |
| `remove_benchmark_overlaps.py` | remove benchmark rows matching one or more comparison sets |
| `filter_dataset.py` | apply common validity filters and exact-shift deduplication |
| `prepare_admet_datasets.py` | match property labels and create ADMET-disjoint outputs |
| `calculate_mol_properties.py` | create aligned RDKit descriptor/fingerprint sidecars |
| `split_foundation_datasets.py` | create molecule-safe FoMoNMR train/validation files |
| `split_maccs_probe.py` | create the fixed representation-diagnostic split |
| `analyze_cleaned_datasets.py` | generate final summary tables and plots |

## Maintained order

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
create molecule-safe foundation-model train/validation pairs
        |
        v
generate final analytics
```

NMR-to-NMR overlap commands compare exact `smiles_canonical`. External ADMET
labels are matched with full RDKit InChIKeys. The released source inventory,
identity rules, and filtering summary are documented in the
[`datasets/cleaned` data card](../../../datasets/cleaned/README.md).

## Common command conventions

Transformation commands use the same conventions where applicable:

- input Parquet paths are positional;
- outputs use `--output`, or explicit role names when there are two outputs;
- `--batch-size` bounds each Arrow batch and defaults to 50,000 rows;
- `--report-output` selects the processing JSON path;
- `--overwrite` permits replacement only after temporary outputs are complete;
- `--quiet` suppresses non-essential warnings and intermediate messages but keeps the progress bar and errors;
- `--no-progress` disables only the progress bar.

Outputs are first written to `.partial` files and promoted after a successful
close. Commands also write a compact processing report beside their main
output.

`common.py` contains shared Parquet validation, temporary-output, exact-SMILES
indexing, and record-ID filtering helpers. `scripts/data/console.py` provides the
common CLI flags, warning suppression, and progress bars. Both are internal
modules, not commands.

## `merge_datasets.py`

Concatenates compatible canonical Parquet files in the requested order. It
does not deduplicate, assign splits, filter records, or remove overlap.

```bash
PYTHONPATH=scripts python -m data.postprocess.merge_datasets \
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
PYTHONPATH=scripts python -m \
  data.postprocess.move_benchmark_overlaps_to_train \
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
PYTHONPATH=scripts python -m data.postprocess.remove_benchmark_overlaps \
  datasets/intermediate/test_after_simnmr.parquet \
  datasets/intermediate/train_val_extended.parquet \
  datasets/canonical/nmrgym/all.parquet \
  --output datasets/intermediate/test_benchmark_preclean.parquet
```

The command is directional and reusable: the first positional input is always
the file being filtered, while every later positional input is a comparison.

## `filter_dataset.py`

Applies the maintained validity filters and exact-shift deduplication policy to
one canonical Parquet. It can also retain rejected rows for auditing.

```bash
PYTHONPATH=scripts python -m data.postprocess.filter_dataset \
  datasets/intermediate/train_val_extended.parquet \
  --output datasets/intermediate/train_val_filtered.parquet \
  --removed-output datasets/intermediate/audits/train_val_removed.parquet
```

If output arguments are omitted, the command uses `<input>_cleaned.parquet`
and `<input>_removed.parquet` beside the input.

## `prepare_admet_datasets.py`

Matches the selected TDC and Sangster property labels to the cleaned rich NMR
pool. It writes endpoint-specific `train_val` and `test` pairs and a rich
training output with all matched ADMET identities removed.

```bash
PYTHONPATH=scripts python -m data.postprocess.prepare_admet_datasets \
  datasets/intermediate/train_val_filtered.parquet \
  --tdc-root datasets/raw/admet \
  --sangster-workbook datasets/raw/sangster_logp/Datasets.xlsx \
  --output-root datasets/cleaned/admet \
  --train-output datasets/cleaned/rich.parquet
```

The command writes `preparation_report.json` beside the endpoint directories.

## `calculate_mol_properties.py`

Writes a typed, same-order `*_mol_properties.parquet` containing RDKit
descriptors, Morgan/ECFP4, and MACCS. It preserves source row groups so the NMR
file and sidecar can be streamed together.

```bash
PYTHONPATH=scripts python -m data.postprocess.calculate_mol_properties \
  datasets/cleaned/rich.parquet \
  --workers 8
```

Morgan is stored as 256 fingerprint bytes and expanded to 2,048 bits by the
FoMoNMR processor. The complete released schema is documented in the
[`datasets/cleaned` data card](../../../datasets/cleaned/README.md).

## `split_foundation_datasets.py`

Creates aligned physical train/validation pairs for SimNMR, rich, and NMRGym
under `datasets/train_splits/`. Molecules are assigned consistently across all
three sources.

```bash
PYTHONPATH=scripts python -m data.postprocess.split_foundation_datasets
```

The NMR and molecular-property outputs retain matching rows and row groups.

## `split_maccs_probe.py`

Creates fixed, molecule-disjoint MACCS linear-probe train and evaluation pairs
from rich validation.

```bash
PYTHONPATH=scripts python -m data.postprocess.split_maccs_probe \
  datasets/train_splits/rich_val.parquet \
  datasets/train_splits/rich_val_mol_properties.parquet
```

The aligned outputs live under `datasets/train_splits/maccs_probe/`.

## `analyze_cleaned_datasets.py`

Generates the source, molecular-property, peak, annotation, and ADMET summary
tables and plots used by the released dataset card.

```bash
PYTHONPATH=scripts python -m data.postprocess.analyze_cleaned_datasets \
  --cleaned-root datasets/cleaned
```

The script verifies that each molecular sidecar is aligned before analysis.
Use `--help` for alternate paths and single-source modes.

## `admet_overlap_audit.ipynb`

This notebook remains an exploratory audit. It can compare property releases
and report alternative identity intersections, but it is not an input to the
production ADMET preparation command.

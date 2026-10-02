# Canonical data package

`scripts/data` is the model-independent data layer. It defines the Python
representation of one spectrum, validates canonical records, streams canonical
Parquets, and contains the commands that build derived datasets. It does not
tokenize, pad, mask, or otherwise adapt a record for a neural network.

| Path | Purpose |
| --- | --- |
| `schema.py` | `CanonicalRecord`, `ProtonPeak`, `CarbonPeak`, and multiplicity normalization |
| `validation.py` | common schema checks without model-specific requirements |
| `dataset.py` | in-memory, JSONL, and streaming Parquet readers |
| `canonicalize/` | source-specific raw-to-canonical converters |
| `postprocess/` | merge, filtering, splitting, sidecar, ADMET, and analytics tools |
| `reporting.py` | compact processing reports shared by commands |
| `download_raw_datasets.sh` | downloader for the pinned public source releases |
| `generate_dvc_pipeline.sh` | generator for the root `dvc.yaml` |

Run Python commands from the repository root with `scripts` on `PYTHONPATH`.

## Reproduce the data pipeline

The root [`dvc.yaml`](../../dvc.yaml) connects the scripts in this package:

```text
raw source releases
  -> canonical source/split Parquets
  -> rich train/test merges
  -> move SimNMR overlaps from rich test into rich train
  -> remove residual rich-test overlap with extended train and NMRGym
  -> common filtering of train, test, SimNMR, and NMRGym
  -> ADMET cohort preparation and leakage removal from rich train
  -> final molecular-property sidecars
  -> molecule-safe foundation-model train/validation splits
  -> fixed molecule-safe MACCS probe split from rich validation
  -> final collection analytics
```

For the full build, download the pinned sources and let DVC run the required
commands:

```bash
scripts/data/download_raw_datasets.sh all
dvc repro
```

Inspect the graph or reproduce only one target with:

```bash
dvc dag
dvc repro STAGE
```

Pipeline parameters live in `params.yaml`; `dvc.lock` records the commands,
parameters, dependencies, and hashes of the current run. Regenerate
`dvc.yaml` with `scripts/data/generate_dvc_pipeline.sh` only when a stage or
dependency changes. The generator updates stage definitions but does not
process data.

For individual command examples, see
[`canonicalize/README.md`](canonicalize/README.md) and
[`postprocess/README.md`](postprocess/README.md). Dataset download and directory
layout are covered by [`datasets/README.md`](../../datasets/README.md).

The package stops at canonical records and derived files. Training-only
pairing, padding, masking, and model tensors are documented in
[`../model/README.md`](../model/README.md).

## Canonical schema

Canonical schema v2 represents one source spectrum as a `CanonicalRecord`.

| Object | Main fields |
| --- | --- |
| `CanonicalRecord` | `record_id`, provenance, source/canonical SMILES, formula, atoms, acquisition metadata, H peaks, C peaks |
| `ProtonPeak` | shift plus optional integration, raw/normalized multiplicity, J values, range, and equivalence metadata |
| `CarbonPeak` | shift plus optional integral, intensity, and width |

Structure fields are metadata, not automatic encoder inputs. Converters derive
`smiles_canonical`, `molecular_formula`, and `atoms` with RDKit. Coordinates,
padding, tokens, split labels, and training targets are intentionally absent.

`atoms` is a list of element symbols in the order obtained by parsing the
stored canonical SMILES. It normally excludes implicit hydrogens, while the
molecular formula includes them. The general Python schema keeps structural
metadata nullable so it can represent small spectrum-only fixtures and future
sources; the implemented raw-data converters enforce non-empty, RDKit-valid
SMILES and populate all four structure fields.

The two modality fields are never nullable in the Python representation. A
missing or null source modality is normalized to an empty tuple, and canonical
Parquet writers serialize it as `[]`. Optional peak annotations remain nullable.

The Parquet reader requires `record_id` and projects the current dataclass
fields that are present; omitted optional fields receive their normal defaults.
All currently maintained source and final-release Parquet files use schema v2.
The reader still ignores extra physical columns so that small fixtures and
older external files can be inspected without changing the in-memory model.

`None` means that an annotation was unavailable. An empty H or C tuple means
that the record has no peaks for that modality. Normalized multiplicity
`<unk>` means an annotation was observed but is outside the supported
vocabulary; `multiplicity_raw` still preserves its source value.

### Construct a record in Python

This is useful in notebooks, tests, or for running a processor on a few spectra:

```python
from data import CanonicalRecord, CarbonPeak, ProtonPeak

record = CanonicalRecord(
    record_id="example-001",
    source="local",
    smiles="CCO",
    smiles_canonical="CCO",
    molecular_formula="C2H6O",
    atoms=("C", "C", "O"),
    h_nmr_peaks=(
        ProtonPeak(
            shift=1.18,
            integration=3,
            multiplicity_raw="triplet",
            multiplicity="t",
            j_values=(7.0,),
        ),
        ProtonPeak(shift=3.65, integration=2, multiplicity="q"),
    ),
    c_nmr_peaks=(CarbonPeak(shift=18.3), CarbonPeak(shift=58.1)),
)
```

The schema accepts nullable structural metadata for small fixtures, while the
maintained converters require valid source SMILES and populate the derived
structure fields.

### Stream canonical Parquets

`CanonicalParquetDataset` is an `IterableDataset`; it yields one validated
`CanonicalRecord` at a time and does not load the complete file into memory:

```python
from data import CanonicalParquetDataset

dataset = CanonicalParquetDataset("datasets/cleaned/test_benchmark.parquet")
record = next(iter(dataset))

print(record.record_id)
print([peak.shift for peak in record.h_nmr_peaks])
```

The reader can receive one path or a list of paths. Worker sharding happens by
file, so use `num_workers=0` when a `DataLoader` reads one large Parquet. The
reader ignores extra physical columns and fills omitted optional fields with
their dataclass defaults, but it never parses a raw source or creates
model-specific tensors.

Use `CanonicalNMRDataset([record, ...])` for a small in-memory collection and
`JsonlCanonicalReader` for canonical JSONL. FoMoNMR batching is described in
[`../model/README.md`](../model/README.md); published-model processors live in
[`../model_benchmarks/processors/`](../model_benchmarks/processors/).

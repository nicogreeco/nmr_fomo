# Canonical data

This package contains the reusable, model-independent data layer. It can be
used by the existing-model benchmarks and later by training code for a new
model.

- `schema.py` defines canonical records and multiplicity normalization.
- `validation.py` checks common schema rules without imposing model-specific
  requirements.
- `dataset.py` provides an in-memory dataset, a small JSONL reader, and a
  streaming Parquet dataset.
- `canonicalize/` contains source-specific conversion and dataset-analysis tools.
- `postprocess/` contains derived-data and benchmark-preparation utilities.

The post-processing commands, including benchmark disjoining, common
filtering, same-order RDKit molecular-property CSV generation, and final
analytics modes, are documented in
[`postprocess/README.md`](postprocess/README.md). Those CSV descriptors and
fingerprints are derived sidecars; they are not canonical schema fields or
filter acceptance criteria.

Canonical schema version 2 stores the record identifier and provenance,
source and canonical SMILES, molecular formula, acquisition metadata, an
atom-symbol list, and the two nested resonance lists. It deliberately has no
coordinate column. Structure fields are not encoder inputs: source converters
derive canonical SMILES, formula, and atoms with RDKit before these readers see
the data.

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

The readers only consume records that are already canonical. They do not
tokenize spectra, adapt fields for a model, or convert a raw dataset.

```python
from data import CanonicalParquetDataset

dataset = CanonicalParquetDataset("../datasets/mst_nmr/test.parquet")
```

Add shared canonical fields or validation rules here. Model-specific input
requirements belong in `scripts/model_benchmarks/processors/`.

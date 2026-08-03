# Canonical data

This package contains the reusable, model-independent data layer. It can be
used by the existing-model benchmarks and later by training code for a new
model.

- `schema.py` defines canonical records and multiplicity normalization.
- `validation.py` checks common schema rules without imposing model-specific
  requirements.
- `dataset.py` provides an in-memory dataset, a small JSONL reader, and a
  streaming Parquet dataset.

The readers only consume records that are already canonical. They do not
tokenize spectra, adapt fields for a model, or convert a raw dataset.

```python
from data import CanonicalParquetDataset

dataset = CanonicalParquetDataset("../datasets/mst_nmr/test.parquet")
```

Add shared canonical fields or validation rules here. Model-specific input
requirements belong in `scripts/model_benchmarks/processors/`.

# Existing-model benchmarks

This package adapts canonical records to the four NMR model families and the
structure-only UniMol2 baseline used for latent-space comparison.

Inputs come from the DVC-managed canonical or cleaned collection. This package
does not belong to the raw-to-cleaned DAG and must never convert, filter, split,
or rewrite those datasets.

The main pieces are:

- `factory.py`: readable, lazy `build_processor()` and `build_embedder()`
  selection;
- `processors/`: validation, model-specific conversion, tensor preparation,
  padding, and batch collation;
- `embedders/`: checkpoint loading, encoder execution, and fixed-size pooling;
- `extract_embeddings.py`: batched extraction with incremental Parquet output;
- `tests/`: short processor, checkpoint, and embedding smoke tests.

Typical use is:

```python
from torch.utils.data import DataLoader

from data import CanonicalParquetDataset
from model_benchmarks import build_embedder, build_processor

dataset = CanonicalParquetDataset("datasets/cleaned/test_benchmark.parquet")
processor = build_processor("nmrpeak", mode="canonical", strict=True)
embedder = build_embedder("nmrpeak", device="cpu")

loader = DataLoader(dataset, batch_size=64, collate_fn=processor)
for batch in loader:
    result = embedder.encode(batch)
    break
```

The NMR paths support combined `1H + 13C` input. UniMol2 instead reads
`smiles_canonical`; its default batch size is one, and `--model-size` selects
`84M` or `164M`. Run one model family per process in its corresponding
environment. Use the documented incompatibility handling for records missing
the fields required by the selected path.

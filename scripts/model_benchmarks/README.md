# Existing-model benchmarks

This package adapts canonical NMR records to the four existing model families
used for latent-space comparison.

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

Only combined `1H + 13C` input is supported. Run one model family per process
in its corresponding environment. The cleaned benchmark may also contain
single-modality NMRexp rows; extraction must therefore use the documented
incompatibility handling when processing the complete file.

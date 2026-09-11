# Existing-model benchmarks

This package adapts canonical records to published NMR representations,
FoMoNMR, and two structure-only molecular comparisons: UniMol2 and Morgan ECFP4.

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

The NMR paths support combined `1H + 13C` input. UniMol2 and Morgan instead
read `smiles_canonical`. UniMol2 defaults to batch size one and `--model-size`
selects `84M` or `164M`; Morgan is the fixed 2,048-bit radius-2 ECFP4
baseline. Run one model family per process in its corresponding environment. Use the documented incompatibility handling for records missing
the fields required by the selected path.

## Property-probe validation

The frozen MLP uses molecular groups (full InChIKey) for both outer
cross-validation and its internal early-stopping holdout. Regression uses
`GroupShuffleSplit`; classification uses one `StratifiedGroupKFold` fold to
approximate the validation fraction. The per-record groups are passed through
GridSearchCV to the MLP in each training fold and in the final refit. Direct
calls to `TorchMLP.fit` must also supply `groups`.

This fixes the earlier record-level internal holdout. Existing result CSVs
were not regenerated; their MLP scores still describe the earlier protocol.
The official train/test assignments and linear probes are unchanged.


## Fine-tuning a local FoMoNMR checkpoint

Use either `--run-id` for an MLflow model or `--checkpoint-path` for a local
Lightning checkpoint, including UniMol2 relational posttraining checkpoints:

```bash
PYTHONPATH=scripts python scripts/model_benchmarks/finetune_fomonmr_property_prediction.py \
  --checkpoint-path runs/fomonmr/RUN/checkpoints/best/MODEL.ckpt \
  --datasets ames --device cuda
```

For the standard `RUN/checkpoints/{best,latest}/` layout, results use the run
name. Other paths use the checkpoint filename without its extension. Fine-tuning
runs for `--epochs` epochs and restores the weights with the best validation
loss before testing; it does not currently implement patience-based stopping.

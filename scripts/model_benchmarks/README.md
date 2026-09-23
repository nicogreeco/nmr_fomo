# Model benchmarks

This package converts canonical NMR records into the native inputs expected by
published encoders, FoMoNMR, UniMol2, and Morgan ECFP4. It also contains the
downstream structural-information and ADMET runners.

It never parses raw sources or changes canonical datasets:

```text
CanonicalRecord
    -> processor
    -> model-native batch
    -> embedder
    -> one embedding per record
```

## Main components

| Component | Purpose |
| --- | --- |
| `processors/` | Validate required fields and create model-native batches |
| `embedders/` | Load checkpoints, run encoders, and apply the documented pooling rule |
| `factory.py` | Lazily construct one processor or embedder without importing every model family |
| `extract_embeddings.py` | Stream a canonical Parquet into an embedding Parquet |
| `run_structural_information_probes.py` | Train frozen structural-information probes |
| `run_property_prediction.py` | Train grouped frozen ADMET probes |
| `finetune_fomonmr_property_prediction.py` | End-to-end FoMoNMR property fine-tuning |

Use one model family per process in its environment. Environment creation and
activation are documented in [`../envs_scr/README.md`](../envs_scr/README.md).

## Reproduce a benchmark

All frozen-embedding analyses follow the same small workflow:

1. Reproduce the benchmark dataset with its DVC stage.
2. Run `model_benchmarks.extract_embeddings` once for every representation and
   dataset split. Keep the checkpoint and input mode fixed across splits, use
   one model environment per process, and give each representation a stable
   directory name.
3. Pass the resulting Parquets to the relevant runner. The runner aligns
   records before fitting the same downstream heads; `model_a+model_b`
   conditions are concatenated there and do not need another extraction.
4. Run a copy of the analysis notebook inside the new result directory to
   regenerate the summary CSVs and tables.

The exact DVC stage, splits, representation names, runner parameters, and
notebook command for each maintained analysis are documented in the
[structural-information](../../results/structural_information/README.md) and
[ADMET property-prediction](../../results/property_prediction/README.md)
result directories.

## Use a processor and embedder directly

`CanonicalParquetDataset` streams model-independent `CanonicalRecord` objects.
A processor can be used alone to inspect exactly what a model receives. This
example uses NMR-Solver because it is a fixed CPU featurizer and needs no
checkpoint:

```python
from data import CanonicalParquetDataset
from model_benchmarks import build_embedder, build_processor

dataset = CanonicalParquetDataset(
    "datasets/downstream_structural_information/train.parquet"
)
record = next(iter(dataset))

processor = build_processor("nmrsolver", mode="canonical", strict=True)
batch = processor([record])

print(batch.record_ids)
print(batch.inputs.keys())       # h_shifts, c_shifts

embedder = build_embedder("nmrsolver", device="cpu")
result = embedder.encode(batch)

print(result.embeddings.shape)  # torch.Size([1, 256])
print(result.record_ids)
print(result.pooling)
```

The same pattern works with a list of `CanonicalRecord` objects constructed in
Python. Processors do not accept arbitrary raw-source formats: local data must
already follow the canonical schema.

For batched extraction, use a processor as the `DataLoader` collator:

```python
from torch.utils.data import DataLoader

from data import CanonicalParquetDataset
from model_benchmarks import build_embedder, build_processor

dataset = CanonicalParquetDataset("datasets/cleaned/test_benchmark.parquet")
processor = build_processor("nmrpeak", mode="canonical", strict=True)
embedder = build_embedder("nmrpeak", device="cuda")

loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=0,
    collate_fn=processor,
)

for batch in loader:
    result = embedder.encode(batch)
    embeddings = result.embeddings       # float32 [batch, 768]
    record_ids = result.record_ids        # same order as the rows
    break
```

For FoMoNMR, give either a local Lightning checkpoint or an MLflow run and
declare the input policy:

```python
processor = build_processor("fomonmr")
embedder = build_embedder(
    "fomonmr",
    checkpoint_path="models_release/posttraining/fomonmr-posttrain.ckpt",
    input_mode="rich",
    device="cpu",
)
result = embedder.encode(processor([record]))
```

`input_mode="shifts"` masks optional proton annotations before inference.
`input_mode="rich"` requires a compatible posttraining checkpoint.

## Write embedding Parquets

The CLI performs the same flow incrementally without holding the full dataset
in memory:

```bash
PYTHONPATH=scripts python -m model_benchmarks.extract_embeddings \
  --model nmrpeak \
  --input datasets/cleaned/test_benchmark.parquet \
  --output embeddings/nmrpeak/test.parquet \
  --device cuda --batch-size 32 \
  --on-incompatible skip \
  --rejections embeddings/nmrpeak/test_rejections.json
```

The output contains `record_id` and a fixed-size float32 `embedding`. Its
Parquet footer records the model, checkpoint, dimension, pooling rule, accepted
records, and rejected records. A `.partial` file is used until writing
finishes. Existing outputs are not replaced unless `--overwrite` is passed.

NMRPeak, NMRTrans, UltraNMR, NMR-Solver, and FoMoNMR consume spectral inputs.
UniMol2 and Morgan consume `smiles_canonical` and are structure-only
comparisons. Their dimensions and exact pooling rules are listed in
[`embedders/README.md`](embedders/README.md); field conversion is documented in
[`processors/README.md`](processors/README.md).

## Downstream runners

The runners consume saved embedding Parquets and never invoke an encoder.

### Structural information

The benchmark is run with `run_structural_information_probes` and uses a fixed
molecule-disjoint train/validation split, one common embedding intersection,
linear seed 42, and MLP seeds 13, 42, and 73. Complete
commands from dataset preparation through notebook analysis are in
[`results/structural_information/README.md`](../../results/structural_information/README.md).

### Frozen ADMET probes

`run_property_prediction.py` evaluates standalone embeddings and `+`-joined
representations on the same record intersection. Combined representations are
concatenated by the runner and do not require separate cache files. Outer CV
and the MLP early-stopping split group records by full InChIKey.

Complete extraction, runner, and analysis commands are in
[`results/property_prediction/README.md`](../../results/property_prediction/README.md).

## Fine-tune FoMoNMR on ADMET

The fine-tuner loads one FoMoNMR checkpoint, adds a fresh MLP property head,
and updates the embedding and Transformer backbone. It creates a
molecule-grouped train/validation split, restores the lowest-validation-loss
state, and evaluates the official test split.

```bash
nmr-env main
PYTHONPATH=scripts python -m model_benchmarks.finetune_fomonmr_property_prediction \
  --checkpoint-path models_release/posttraining/fomonmr-posttrain.ckpt \
  --datasets ames solubility_aqsoldb \
  --output-dir results/fomonmr_finetune_new \
  --device cuda
```

Use `--run-id` instead of `--checkpoint-path` only when the original MLflow
server is available. Fine-tuning uses a larger FoMoNMR-compatible cohort than
the all-model frozen intersection, so the two result tables are not a controlled
estimate of fine-tuning alone. See
[`results/fomonmr_finetune/README.md`](../../results/fomonmr_finetune/README.md).


## Tests

Shared fixture tests run in the main environment:

```bash
PYTHONPATH=scripts python -m unittest discover \
  -s scripts/model_benchmarks/tests -v
```

Checkpoint-specific tests should be run in the corresponding model environment
and skip when their local asset is unavailable. See
[`tests/README.md`](tests/README.md).

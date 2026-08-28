# Foundation-model input pipeline

This package contains the input path for the new NMR foundation model. It is
separate from `model_benchmarks`, which adapts canonical records to published
checkpoints.

The paired training flow is:

```text
final canonical NMR Parquet
    + aligned *_mol_properties.parquet
    -> PairedFoundationDataset
    -> DataLoader(collate_fn=FoundationNMRProcessor())
    -> padded NMR tensors + Morgan fingerprints
```

## Basic DataLoader

Run Python from the repository root with `scripts` on `PYTHONPATH`, for example
after activating `nmr-env main`:

```python
from torch.utils.data import DataLoader

from model import FoundationNMRProcessor, PairedFoundationDataset

dataset = PairedFoundationDataset(
    "datasets/cleaned/train_val.parquet",
    "datasets/cleaned/train_val_mol_properties.parquet",
    arrow_batch_size=2048,
    shuffle=True,
    shuffle_buffer_size=8192,
    seed=42,
)

loader = DataLoader(
    dataset,
    batch_size=128,
    num_workers=4,
    collate_fn=FoundationNMRProcessor(),
    pin_memory=True,
    persistent_workers=True,
)

batch = next(iter(loader))
print(batch["fingerprints"].shape)  # torch.Size([128, 2048])
print(batch["fingerprints"].dtype)  # torch.float32
```

Start a script with:

```bash
PYTHONPATH=scripts python path/to/train.py
```

In a notebook started from the repository root, add `scripts` once if it is not
already importable:

```python
import sys
from pathlib import Path

scripts_path = str(Path.cwd() / "scripts")
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)
```

## Batch structure

The processor returns:

```text
batch
├── record_ids: list[str]
├── h
│   ├── shift, integration, multiplicity, j_values, range_half_span
│   └── peak_mask, j_mask, availability
├── c
│   └── shift, peak_mask
└── fingerprints: float32 tensor [B, 2048]
```

Only `record_ids` are retained as metadata. The dataset keeps each Morgan
fingerprint compact as 256 bytes while reading; the processor expands those
bytes to bits only for the current batch.

The same processor also accepts ordinary canonical records. In that case the
batch has no `fingerprints` key. Paired and unpaired records cannot be mixed in
one batch.

## Streaming and workers

`PairedFoundationDataset` is an `IterableDataset`. It validates that its two
files have the same number of rows and identical row-group sizes. Each worker
then receives a disjoint subset of complete row groups, avoiding duplicate
records and full-file materialization.

Do not set `shuffle=True` on the DataLoader and do not attach a sampler:
iterable datasets do not support those map-style options. Instead,
`PairedFoundationDataset` shuffles internally by default. Each worker first
changes the order of its assigned row groups and then uses a bounded record
buffer. This gives useful mixing without loading the complete dataset in RAM.

The random-generator state advances whenever the loader is iterated again, so
ordinary successive training epochs receive a different order. This also works
with persistent workers. `seed` makes the sequence reproducible; for exact
multi-worker reproducibility, also pass a seeded `torch.Generator` to the
DataLoader.

Use `shuffle=False` for validation, testing, or debugging when exact Parquet
order is useful. With multiple workers, all row groups are still covered once,
but worker interleaving means their global output order is not guaranteed.

Use `persistent_workers=True` only when `num_workers > 0`. `pin_memory=True` is
useful when training on a GPU. `drop_last=True` can be added when the model
requires constant batch size.

## Lightning

The DataLoader can be returned directly by a `LightningDataModule`:

```python
import lightning as L
from torch.utils.data import DataLoader

from model import FoundationNMRProcessor, PairedFoundationDataset


class NMRDataModule(L.LightningDataModule):
    def __init__(self, batch_size=128, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_data = PairedFoundationDataset(
            "datasets/cleaned/train_val.parquet",
            "datasets/cleaned/train_val_mol_properties.parquet",
            shuffle=True,
            seed=42,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=FoundationNMRProcessor(),
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
```

Lightning recursively moves tensors inside the nested `h` and `c` dictionaries,
including `fingerprints`, to the selected device before `training_step`.
`record_ids` remain a Python list of strings.

## Pretraining and continued pretraining

`FoMoNMR` uses one explicit `stage` from
[`configs/default.yaml`](configs/default.yaml):

- `pretrain` hides every rich H annotation and trains masked H/C shift
  reconstruction plus weak Morgan-similarity classification;
- `posttrain` keeps available rich inputs and additionally reconstructs masked
  integration, multiplicity, J couplings, and range half-span.

The model never changes stage based on its epoch. A normal pretraining run is:

```python
import lightning as L
import yaml

from model.FoMoNMR import FoMoNMR, ModelConfig

with open("scripts/model/configs/default.yaml") as file:
    config = ModelConfig(**yaml.safe_load(file))

model = FoMoNMR(config)
trainer = L.Trainer(max_epochs=10)
trainer.fit(model, datamodule=data_module)
```

The configured chemical-shift ranges are H `[-5.5, 20.0]` ppm with `0.02` ppm
bins and C `[-40.0, 300.0]` ppm with `0.2` ppm bins. The same values configure
Fourier frequencies, augmentation boundaries, prediction heads, and soft-label
centers.

For continued pretraining, load the pretraining weights into a model constructed
with `stage="posttrain"`:

```python
import lightning as L
import yaml

from model.FoMoNMR import FoMoNMR, ModelConfig

with open("scripts/model/configs/default.yaml") as file:
    config_values = yaml.safe_load(file)
config_values["stage"] = "posttrain"
posttrain_config = ModelConfig(**config_values)

model = FoMoNMR.load_from_checkpoint(
    "checkpoints/pretrain.ckpt",
    config=posttrain_config,
)

trainer = L.Trainer(max_epochs=10)
trainer.fit(model, datamodule=rich_data_module)
```

Do not pass the pretraining checkpoint as `ckpt_path` to this second
`trainer.fit` call. Starting a fresh fit rebuilds Adam after the rich embedder
and annotation heads have been unfrozen.

## Validation and current limitations

Construction checks the sidecar schema version, Morgan radius and bit count,
required fields, row counts, and row-group boundaries. Iteration checks IDs,
RDKit status, and the 256-byte fingerprint for every row. Invalid input raises
a descriptive error; records are never silently discarded.

`include_unimol` defaults to `False`. Passing `include_unimol=True` currently
raises `NotImplementedError`; the argument reserves a simple future extension
without pretending that UniMol collation is already available.

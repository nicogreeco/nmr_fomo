# FoMoNMR model and training

This package contains the FoMoNMR datasets, batch processor, architecture,
objectives, callbacks, and Lightning training entry point. It is separate from
`model_benchmarks`, which provides a common extraction interface for FoMoNMR
and published encoders.

The paired training flow is:

```text
final canonical NMR Parquet
    + aligned *_mol_properties.parquet
    -> PairedFoundationDataset
    -> DataLoader(collate_fn=FoundationNMRProcessor())
    -> padded NMR tensors + Morgan fingerprints
```

FoMoNMR applies separate H and C input adapters, concatenates the peak tokens,
processes them with a shared permutation-equivariant Transformer, and returns a
masked mean-pooled spectrum embedding together with the peak states. Optional
proton annotations are gated by availability masks.

| File | Responsibility |
| --- | --- |
| `dataset.py` | paired streaming datasets and weighted source mixture |
| `processor.py` | validation, padding, masks, and fingerprint expansion |
| `embedding.py` | H/C adapters and peak feature embeddings |
| `FoMoNMR.py` | Lightning module, Transformer, heads, and train/validation steps |
| `losses.py`, `utils/corruption.py` | objectives and masked-input corruption |
| `maccs_probe.py` | frozen-encoder diagnostic callback |
| `train.py` | CLI, loaders, Lightning trainer, logging, and checkpoints |
| `configs/` | reusable YAML configurations |

## Dataset classes

The three dataset classes have different roles:

| Class | Use |
| --- | --- |
| `CanonicalParquetDataset` from `data` | stream model-independent canonical records for inference or benchmarks |
| `PairedFoundationDataset` | stream a canonical NMR file with its same-order molecular-property sidecar for training |
| `MixedFoundationDataset` | sample indefinitely from several paired datasets with configured source proportions |

`CanonicalParquetDataset` has no fingerprints or training flags.
`PairedFoundationDataset` validates row count, IDs, and row-group alignment with
the sidecar. `MixedFoundationDataset` is the weighted streaming mixture used by
`train.py`; because it is effectively infinite, `max_steps` is the training
budget.

## Build a training DataLoader

Run Python from the repository root with `scripts` on `PYTHONPATH` after
activating the main project environment:

```python
from torch.utils.data import DataLoader

from model import FoundationNMRProcessor, PairedFoundationDataset

dataset = PairedFoundationDataset(
    "datasets/cleaned/rich.parquet",
    "datasets/cleaned/rich_mol_properties.parquet",
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

## Batch structure

The processor returns:

```text
batch
├── record_ids: list[str]
├── shift_only: bool tensor [B]
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

## Load a checkpoint and extract embeddings

Released Lightning checkpoints load directly with `FoMoNMR.load_from_checkpoint`.
For inference, canonical records and `FoundationNMRProcessor` are sufficient;
molecular-property sidecars are training inputs and are not required.

```python
import torch
from torch.utils.data import DataLoader

from data import CanonicalParquetDataset
from model import FoundationNMRProcessor
from model.FoMoNMR import FoMoNMR

dataset = CanonicalParquetDataset("datasets/cleaned/test_benchmark.parquet")
loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=0,
    collate_fn=FoundationNMRProcessor(),
)

model = FoMoNMR.load_from_checkpoint(
    "/path/to/fomonmr-posttrain.ckpt",
    map_location="cpu",
    weights_only=True,
)
model.eval()

batch = next(iter(loader))
with torch.inference_mode():
    embeddings, peak_embeddings, peak_mask = model(batch)

print(embeddings.shape)       # [batch, d_model]
print(peak_embeddings.shape) # [batch, peaks, d_model]
```

The rich posttraining checkpoints use optional proton annotations whenever
their availability masks are true. For shift-only evaluation, use the FoMoNMR
benchmark adapter with `input_mode="shifts"`; it applies the masking policy
consistently. See [`../model_benchmarks/README.md`](../model_benchmarks/README.md).

## Streaming and workers

`PairedFoundationDataset` is an `IterableDataset`. It validates aligned files
and shards complete row groups across DataLoader workers and DDP ranks. If a
small source has fewer row groups than global workers, use
`replicate_when_too_small=True`. Use `shard_across_ranks=False` when every rank
must see the complete validation set.

Do not set `shuffle=True` on the DataLoader and do not attach a sampler:
iterable datasets do not support those map-style options. Instead,
`PairedFoundationDataset` shuffles internally by default. Each worker first
changes the order of its assigned row groups and then uses a bounded record
buffer. This gives useful mixing without loading the complete dataset in RAM.

Use `shuffle=False` for validation, testing, or debugging when exact Parquet
but worker interleaving means their global output order is not guaranteed.

Use `persistent_workers=True` only when `num_workers > 0`. `pin_memory=True` is
useful when training on a GPU. `drop_last=True` can be added when the model
requires constant batch size.

## Controlled dataset mixtures

The DVC stage `split_foundation_datasets` creates molecule-safe physical splits
under `datasets/train_splits/`. A molecule selected for validation in any of
SimNMR, rich `train_val`, or NMRGym is excluded from training in all three.
The benchmark test is not changed.

Build the pretraining mixture with SimNMR as the primary dataset:

```python
from model import MixedFoundationDataset, PairedFoundationDataset


def paired(name, split, shuffle, replicate=False, shard_ranks=True):
    root = "datasets/train_splits"
    return PairedFoundationDataset(
        f"{root}/{name}_{split}.parquet",
        f"{root}/{name}_{split}_mol_properties.parquet",
        shuffle=shuffle,
        seed=42,
        source_name=name,
        replicate_when_too_small=replicate,
        shard_across_ranks=shard_ranks,
    )


pretrain_data = MixedFoundationDataset(
    datasets=[
        paired("simnmr", "train", True),
        paired("rich", "train", True, replicate=True),
        paired("nmrgym", "train", True, replicate=True),
    ],
    proportions=[0.90, 0.09, 0.01],
    shift_only=[True, False, True],
    seed=42,
)
```

The two `shift_only` arguments serve the same record flag at different levels:

- `PairedFoundationDataset(shift_only=...)` is convenient when one dataset is
  used by itself;
- `MixedFoundationDataset(shift_only=[...])` assigns the policy per source and
  overrides the child value.

When using a mixture, set the policy only on `MixedFoundationDataset`, as in
the example above. With model stage `pretrain`, FoMoNMR masks rich annotations
for every row, so the mixture flags do not alter pretraining inputs. With stage
`posttrain`, rich rows can use their annotations while replay sources remain
shift-only:

```python
posttrain_data = MixedFoundationDataset(
    datasets=[
        paired("rich", "train", True),
        paired("simnmr", "train", True, replicate=True),
        paired("nmrgym", "train", True, replicate=True),
    ],
    proportions=[0.90, 0.05, 0.05],
    shift_only=[False, True, True],
    seed=42,
)
```

In posttraining, the processor places these flags in `batch["shift_only"]`.
`FoMoNMR.corrupt_batch()` then hides rich inputs only for replayed SimNMR and
NMRGym rows. Loss computation needs no source-specific logic.

Validation does not use source proportions: `train.py` builds one unshuffled
loader per validation source and logs it through `source_name`. Training source
selection and per-source shuffling are seeded; exhausted source iterators are
restarted. `max_steps` remains the authoritative stopping condition.

## Lightning

Lightning recursively moves every tensor in the nested batch, including
`fingerprints`, to the selected device before `training_step`. `record_ids`
remain a Python list of strings. The project keeps loader construction directly
in `train.py`; a separate DataModule is not needed for the current workflow.

## Pretraining and continued pretraining

`FoMoNMR` uses one explicit `stage`. When `--config` is omitted, `train.py`
automatically loads [`configs/pretrain.yaml`](configs/pretrain.yaml) or
[`configs/posttrain.yaml`](configs/posttrain.yaml):

- `pretrain` hides every rich H annotation and trains masked H/C shift
  reconstruction plus weak Morgan-similarity classification;
- `posttrain` keeps available rich inputs and additionally reconstructs masked
  integration, multiplicity, J couplings, and range half-span.

The model never changes stage based on its epoch. Start pretraining from the
repository root with:

```bash
PYTHONPATH=scripts python -m model.train \
    --stage pretrain --batch-size 128
```

For a short local test without an MLflow server:

```bash
PYTHONPATH=scripts python -m model.train \
    --stage pretrain --batch-size 8 --num-workers 0 \
    --max-steps 100 --validation-interval 50 --no-mlflow
```

To stop a real trial after exactly 10,000 optimizer steps, pass
`--max-steps 10000`.

MLflow uses `fomonmr-<stage>` as the default experiment name. Pass, for
example, `--experiment-name fp_ablation_pretrain` to select another experiment.

`--precision` is passed to Lightning; choose a supported mode such as
`bf16-mixed`, `16-mixed`, or `32-true` for the available hardware.
`--batch-size` is per process and `--accumulate-grad-batches` increases the
effective batch without increasing the physical batch.

For continued pretraining, initialize a fresh posttrain run from the best
pretraining checkpoint:

```bash
PYTHONPATH=scripts python -m model.train \
    --stage posttrain --batch-size 128 \
    --pretrained-checkpoint runs/fomonmr/PRETRAIN_RUN/checkpoints/best/BEST.ckpt
```

This constructs the model with `stage="posttrain"`, loads the learned pretrain
weights, and starts a fresh optimizer with the rich modules unfrozen.

To resume an interrupted run instead:

```bash
PYTHONPATH=scripts python -m model.train \
    --stage pretrain --run-name PRETRAIN_RUN \
    --resume runs/fomonmr/PRETRAIN_RUN/checkpoints/latest/last.ckpt
```

`--resume` restores model, optimizer, scheduler, and current step. It is
different from `--pretrained-checkpoint`, which starts a new training stage.

The `configs/` directory contains reusable defaults and the `final_*.yaml`
configurations associated with released checkpoints. Published run provenance
and final hyperparameters belong to the
[model card](https://huggingface.co/niccogreek/fomonmr) and project report.

## Checkpoints and run outputs

Optimization, scheduling, early stopping, validation intervals, and objective
weights come from the selected YAML and can be overridden by launcher flags.
Validation and checkpoint intervals count optimizer steps.

The training script keeps:

- the latest three periodic checkpoints under `checkpoints/latest/`;
- `last.ckpt` as a link to the newest one for easy recovery;
- the best two checkpoints under `checkpoints/best/`.

The best checkpoint is selected by the lowest `val/loss`. Use
`--checkpoint-interval` and `--validation-interval` to control their frequency.
Validation artifacts are written below the run directory.

## MLflow

The script reads `MLFLOW_TRACKING_URI` from `mlflow.env`, unless overridden with
`--tracking-uri`. It logs configuration, learning rate, losses, validation
metrics, and CUDA peak memory. Use `--no-mlflow` for local tests. Checkpoints
remain under `runs/fomonmr/<run-name>/`; selected checkpoints can be published
separately.

## MACCS linear probe

The MACCS callback periodically freezes FoMoNMR, extracts clean shift-only
embeddings from a fixed molecule-safe split, fits a fresh linear classifier,
and logs `probe/maccs_macro_auroc`. It never contributes gradients or affects
checkpoint selection. Prepare its files with `dvc repro split_maccs_probe` or
disable it with `--no-maccs-probe`.

## Validation and current limitations

Construction checks the sidecar schema version, Morgan radius and bit count,
required fields, row counts, and row-group boundaries. Iteration checks IDs,
RDKit status, and the 256-byte fingerprint for every row. Invalid input raises
a descriptive error; records are never silently discarded.

`include_unimol=True` reads aligned, precomputed UniMol2 teacher embeddings
from a separate sidecar. It is needed only to reproduce UniMol2 relational
posttraining, not to load or use a released checkpoint.


## UniMol2 relational posttraining

Set `molecular_target: unimol` to train against pairwise cosine similarities of
fixed UniMol2 teacher embeddings. This path requires separately prepared,
same-order teacher sidecars for rich train and validation, including compatible
provenance metadata. Teacher-sidecar generation is currently outside the
public DVC pipeline, so this training recipe is not presented as a turnkey
reproduction from the released dataset.

Released relational checkpoints still use the ordinary checkpoint-loading and
canonical inference example above. No UniMol2 installation or teacher sidecar
is needed for inference. The exact training recipe and provenance are recorded
in the model card and project notes.

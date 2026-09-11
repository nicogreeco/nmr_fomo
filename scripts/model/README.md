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

## Streaming and workers

`PairedFoundationDataset` is an `IterableDataset`. It validates that its two
files have the same number of rows and identical row-group sizes. Each worker
then receives a disjoint subset of complete row groups, avoiding duplicate
records and full-file materialization. DDP ranks are included in this sharding.
If there are fewer row groups than global workers, the default is to fail with
a clear error. Small auxiliary training sources can instead opt into
`replicate_when_too_small=True`, which gives every worker a complete copy of
their row groups. Validation and rank-zero-only diagnostics use
`shard_across_ranks=False` so every participating rank sees the complete
dataset.

Do not set `shuffle=True` on the DataLoader and do not attach a sampler:
iterable datasets do not support those map-style options. Instead,
`PairedFoundationDataset` shuffles internally by default. Each worker first
changes the order of its assigned row groups and then uses a bounded record
buffer. This gives useful mixing without loading the complete dataset in RAM.

The random-generator state advances whenever a source iterator is restarted,
so repeated passes receive a different order. This also works with persistent
workers. `seed` makes the sequence reproducible; for exact
multi-worker reproducibility, also pass a seeded `torch.Generator` to the
DataLoader.

Use `shuffle=False` for validation, testing, or debugging when exact Parquet
order is useful. With multiple workers, all row groups are still covered once,
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


def paired(name, split, shuffle, shift_only=False, replicate=False, shard_ranks=True):
    root = "datasets/train_splits"
    return PairedFoundationDataset(
        f"{root}/{name}_{split}.parquet",
        f"{root}/{name}_{split}_mol_properties.parquet",
        shuffle=shuffle,
        seed=42,
        shift_only=shift_only,
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

`stage="pretrain"` hides rich annotations for every row, so the flags do not
change pretraining behavior. Continued pretraining keeps rich data first as its
primary source:

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

Validation does not use source proportions. Keep each validation dataset in a
separate loader. The model reads `source_name` from those loaders, so the metric
groups automatically follow whichever datasets are passed:

```python
validation_data = [
    paired("rich", "val", False, shift_only=False, shard_ranks=False),
]
if stage == "pretrain":
    validation_data.append(
        paired("simnmr", "val", False, shift_only=True, shard_ranks=False)
    )
validation_data.append(
    paired("nmrgym", "val", False, shift_only=True, shard_ranks=False)
)

val_loaders = [
    DataLoader(
        dataset,
        batch_size=128,
        num_workers=0,
        collate_fn=FoundationNMRProcessor(),
        pin_memory=True,
        drop_last=False,
    )
    for dataset in validation_data
]
```

Pass the training mixture to the ordinary `DataLoader` shown above. Training
source selection is reproducible, underlying datasets retain their existing
bounded shuffle, and every source iterator is restarted when exhausted. The
mixture is intentionally unsized and effectively infinite. The training
script's `max_steps` setting is the authoritative stopping condition.

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
PYTHONPATH=scripts python scripts/model/train.py \
    --stage pretrain --batch-size 128
```

For a short local test without an MLflow server:

```bash
PYTHONPATH=scripts python scripts/model/train.py \
    --stage pretrain --batch-size 8 --num-workers 0 \
    --max-steps 100 --validation-interval 50 --no-mlflow
```

To stop a real trial after exactly 10,000 optimizer steps, pass
`--max-steps 10000`.

MLflow uses `fomonmr-<stage>` as the default experiment name. Pass, for
example, `--experiment-name fp_ablation_pretrain` to select another experiment.

The default `--precision bf16-mixed` is appropriate for recent GPUs such as H200:
bfloat16 has a wide numerical range and is normally more stable than float16.
`--precision 16-mixed` uses float16 where safe, while keeping model weights and
some sensitive operations in float32. Use `--precision 32-true` for full float32.
Choose the physical batch size after a hardware benchmark.
If the desired batch does not fit, `--batch-size 128` together with
`--accumulate-grad-batches 4` gives an effective batch of 512 using four
forward and backward passes per optimizer step.

The configured chemical-shift ranges are H `[-5.5, 20.0]` ppm with `0.02` ppm
bins and C `[-40.0, 300.0]` ppm with `0.2` ppm bins. The same values configure
Fourier frequencies, augmentation boundaries, prediction heads, and soft-label
centers.

For continued pretraining, initialize a fresh posttrain run from the best
pretraining checkpoint:

```bash
PYTHONPATH=scripts python scripts/model/train.py \
    --stage posttrain --batch-size 128 \
    --pretrained-checkpoint runs/fomonmr/PRETRAIN_RUN/checkpoints/best/BEST.ckpt
```

This constructs the model with `stage="posttrain"`, loads the learned pretrain
weights, and starts a fresh optimizer with the rich modules unfrozen.

To resume an interrupted run instead:

```bash
PYTHONPATH=scripts python scripts/model/train.py \
    --stage pretrain --run-name PRETRAIN_RUN \
    --resume runs/fomonmr/PRETRAIN_RUN/checkpoints/latest/last.ckpt
```

`--resume` restores model, optimizer, scheduler, and current step. It is
different from `--pretrained-checkpoint`, which starts a new training stage.

## Optimization and checkpoints

The default optimizer is AdamW with `lr=1e-4`, weight decay `1e-4`, 5,000
linear warmup steps, and gradient norm clipping at `1.0`. After warmup,
`ReduceLROnPlateau` monitors `val/loss`. When validation stops improving,
it halves the learning rate after a patience of two checks, down to `1e-6`.

Early stopping uses the same validation metric with patience eight. The default
one million `max_steps` is therefore only a safety limit: a normal run can stop
earlier when validation has stopped improving. Both patience values count
validation checks, not optimizer steps or natural dataset epochs.

Pretraining validates on Rich, SimNMR, and NMRGym. Posttraining validates on
Rich and NMRGym. In either stage, `val/loss` is the normal mean over every
record in the validation loaders, so source names and dataset sizes do not
receive special weighting.

The default `--logging minimal` mode reports the global total, shift, H/C
shift, and fingerprint losses, H/C MAE in ppm, fingerprint MAE, and the
annotation loss during posttraining. Per-source logging contains only
`val/datasets/<source>/loss` for every available validation loader. Fingerprint
validation also reports
`val/fp_macro_mae` across five fixed Tanimoto ranges. Predictions use the
expected similarity under the classifier probabilities.

Each validation writes the five range MAEs, counts, and mean predicted
similarities under
`runs/fomonmr/<run-name>/artifacts/fingerprint_validation/` and, when enabled,
uploads the CSV to the MLflow artifact path `fingerprint_validation`.
It also writes H and C shift MAEs, counts, and mean predictions over 20 uniform
ranges under the corresponding `shift_validation` paths.

`--logging complete` keeps the same global metrics and adds the complete loss
and MAE breakdown under each `val/datasets/<source>/...` group. Annotation
components are included only for non-shift-only batches during posttraining.
Training logging is unchanged.

The training script keeps:

- the latest three periodic checkpoints under `checkpoints/latest/`;
- `last.ckpt` as a link to the newest one for easy recovery;
- the best two checkpoints under `checkpoints/best/`.

The best checkpoint is selected by the lowest `val/loss`. Change
`--checkpoint-interval` to control periodic saves and
`--validation-interval` to control both validation and best-model evaluation.

## MLflow

The script reads `MLFLOW_TRACKING_URI` from `mlflow.env`, unless it is overridden
with `--tracking-uri`. It logs parameters, learning rate, train/validation
losses, and two peak GPU memory metrics at the end of each training epoch:

- `memory/gpu_peak_allocated_mb`;
- `memory/gpu_peak_reserved_mb`.

Peak statistics are reset at the beginning of every epoch. The metrics are
present only when training on CUDA and MLflow logging is enabled.

Checkpoints remain local under `runs/fomonmr/<run-name>/`; `log_model=False`
avoids uploading every large rolling checkpoint. If the MLflow server is later
configured with an S3 artifact store, selected best checkpoints can be uploaded
explicitly without changing the training loop.

## MACCS linear probe

[DreaMS](https://pmc.ncbi.nlm.nih.gov/articles/PMC13090125/) periodically freezes
its encoder and trains a small linear classifier on a fixed molecule-safe set to
measure how well fingerprint bits are recoverable from the learned embedding.

The DVC stage `split_maccs_probe` derives fixed, molecule-disjoint probe files
from rich validation under `datasets/train_splits/maccs_probe/`. It retains
about 20,000 train and 10,000 evaluation records while preserving the rich
source proportions in both splits.

After every validation, a separate callback extracts clean shift-only pooled
embeddings with FoMoNMR in evaluation/no-gradient mode. It trains a fresh
`Linear(d_model, 166)` layer with `BCEWithLogitsLoss` and logs
`probe/maccs_macro_auroc`. The probe never contributes to the FoMoNMR loss or
its gradients, checkpoint selection, scheduling, or early stopping. Pass
`--no-maccs-probe` to disable this diagnostic.

The probe uses MACCS rather than the Morgan similarity target used during
training. It is a representation diagnostic, not an independent downstream
benchmark.

## Validation and current limitations

Construction checks the sidecar schema version, Morgan radius and bit count,
required fields, row counts, and row-group boundaries. Iteration checks IDs,
RDKit status, and the 256-byte fingerprint for every row. Invalid input raises
a descriptive error; records are never silently discarded.

`include_unimol=True` reads aligned, precomputed UniMol2 teacher embeddings
from a separate properties sidecar. See the posttraining instructions below.


## UniMol2 relational posttraining

Set `molecular_target: unimol` to replace the Morgan similarity objective with
MSE between pairwise cosines of pooled NMR embeddings and fixed UniMol2 teacher
embeddings. No projection or prediction head is added. Only Rich records with
teachers contribute; SimNMR/NMRGym replay keeps its shift loss. `lambda_fp`
weights distillation. `balanced_fp_pairs` caps training pairs per cosine bin;
validation uses all teacher pairs and aggregates their metrics by pair count.
The default `molecular_target: morgan` retains the existing training behavior.
Direct Morgan-bit prediction is not supported. The saved
`fingerprint_objective: similarity` field is retained for checkpoint compatibility.

Use the existing teacher sidecars in `datasets/experimental/unimol2_rich`, or
point `unimol_sidecar_dir` at equivalent prepared sidecars named
`rich_{train,val}_mol_properties.parquet`. They must align with the original
`datasets/train_splits/rich_{train,val}.parquet`, including IDs and row groups.
UniMol training uses these original Rich files with the existing streaming
shuffle; Morgan posttraining continues to use `rich_shuffle_train.parquet`.
Teachers must be 768 float32 values, extracted with batch size 1, centered on
the Rich training mean and L2-normalized. Both splits must carry the same
`unimol_center_sha256`; the dataset checks the teacher schema and provenance.
Teacher extraction remains a separate preparation step; training and inference
do not import or run UniMol2. Existing canonical data and sidecars are unchanged.

```bash
nmr-env main
PYTHONPATH=scripts python scripts/model/train.py \
    --stage posttrain \
    --config scripts/model/configs/final_posttrain_unimol_relational.yaml \
    --pretrained-checkpoint runs/fomonmr/PRETRAIN_RUN/checkpoints/best/BEST.ckpt \
    --run-name unimol-relational-posttrain
```

This recipe uses the run's distillation weight of 0.25 and disables annotation
reconstruction. Pretrained transfer discards only the Morgan similarity head
and strictly loads all remaining weights. Resume requires the same UniMol
configuration via `--config` and the checkpoint via `--resume`.
Metrics use `train/unimol_loss`, `val/unimol_loss`, `val/unimol_cosine_mae`, and
`val/unimol_macro_cosine_mae`; cosine-range CSVs go to `artifacts/unimol_validation`.

Existing relational checkpoints load directly, without teacher sidecars:

```python
import torch
from model.FoMoNMR import FoMoNMR

model = FoMoNMR.load_from_checkpoint(checkpoint_path, map_location="cpu")
model.eval()
# batch = FoundationNMRProcessor()(canonical_records)
with torch.inference_mode():
    embeddings, peak_embeddings, peak_mask = model(batch)
```

The ordinary canonical processor and forward API are unchanged. No checkpoint
conversion or relaxed state-dict loading is needed.

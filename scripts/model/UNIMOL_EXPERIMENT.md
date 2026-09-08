# Experimental UniMol2 relational distillation

Select `molecular_target: unimol` for posttraining only. The default remains
`morgan`, with the existing `fingerprint_objective: similarity` or `bits`.
UniMol requires `fingerprint_objective: similarity`; it uses a cosine MSE,
not the Morgan classifier or its Tanimoto loss.

Only Rich records have UniMol teachers. The posttraining mixture remains
90% Rich, 5% SimNMR, 5% NMRGym; only Rich–Rich pairs contribute to distillation.
The processor passes `unimol_embeddings` and `unimol_mask` instead of unpacked
Morgan targets. MACCS remains available for the existing external probe.

## Teacher extraction

Run in the existing UniMol2 environment, using the local 84M checkpoint:

```bash
nmr-env unimol2
PYTHONPATH=scripts python scripts/model/prepare_unimol_sidecars.py \
  --checkpoint /path/to/unimol2/84M/checkpoint.pt \
  --num-workers 4
```

The existing processor generates conformers with seed 42. Workers prepare
inputs, but **each GPU inference receives exactly one molecule**. No records
are silently skipped: incompatible structures stop extraction with their ID.

For a valence error introduced during conformer preparation, the experimental
extractor retries with the original canonical graph and the generated coordinates.
It checks atom order/isotopes before copying coordinates and logs the repaired
record ID. This handles a verified boron case where four single bonds became
aromatic during upstream conformer setup. Successful original preprocessing is
unchanged, so previously completed raw row groups remain reusable.

The output directory defaults to `datasets/experimental/unimol2_rich/`:

- `raw_parts/{train,val}/`: raw teacher vectors, cached per original row group.
- `rich_train_center.npy`: mean raw embedding over Rich train records only.
- `rich_{train,val}_mol_properties.parquet`: copies of the original sidecars,
  with an additional fixed-size 768-float `unimol_embedding` column.

The final column contains `(raw_embedding - train_mean) / norm`, using the
same train mean for validation. Original columns, IDs, row order and row-group
boundaries are preserved. Train contains 1,891,692 records; validation 50,155.
No existing Parquet or DVC stage is changed. These files are experimental
outputs, not newly registered DVC outputs.

Rerun the same command after interruption to reuse completed raw row groups.
An interrupted row group is recomputed. Output/cache provenance includes the
checkpoint SHA256 and input paths, sizes and modification times; incompatible
caches require a different output directory. Final files are published only
after successful completion, via a `.partial` file. Training requires both
final sidecars and checks that their teacher centers match.

## Objective

For valid Rich pairs `i < j`, the target is the cosine between their fixed,
train-centered UniMol embeddings. The prediction is the cosine between their
pooled FoMoNMR embeddings from the existing corrupted training forward pass.
The loss is the mean squared difference. There is no projection or learned
pair head. Teacher gradients are detached; cosine/MSE arithmetic uses float32
even with mixed precision. Fewer than two Rich records gives a differentiable
zero auxiliary loss.

With `balanced_fp_pairs: true`, training reuses the Morgan sampler: at most
1,024 random pairs per bin. Cosines are mapped from [-1, 1] to [0, 1] for
bin selection only. `fp_sim_bin_size: 0.05` therefore means cosine bins of
width 0.1. This caps dense bins; it does not oversample rare bins. Validation
uses every natural Rich pair, with pair-weighted aggregation across batches.

`lambda_fp` weights this MSE in the total objective. Its numerical scale is
different from the Morgan focal loss, so equal weights do not imply equal
gradient strength. The initial experimental configuration uses 0.1.
Metrics are `train/unimol_loss`, `val/unimol_loss` and
`val/unimol_cosine_mae`. Morgan range CSVs are disabled; shift logging and
the MACCS probe remain active.

## Initial numerical checks (2026-09-05)

On 160 training molecules sampled across all 40 Rich row groups, repeated
singleton extraction was identical. Putting a checked molecule in a two-record
batch changed its embedding (maximum absolute component difference 174.7).

Raw teacher cosine quantiles were approximately 0.990–1.000; predicting cosine
1 everywhere already gave MSE 0.00000734. Centering with a train-only calibration
subset of 120 records spread pilot cosines to roughly -0.88–0.95. Constant
cosine 1 then gave MSE 1.125. This motivates fitting the final center on the
full training set, with no validation fitting.

Optimizing independent 64-D test vectors against the pilot teacher geometry
reduced the capped-pair MSE from 0.150 to 0.00037 in 200 updates. This is a
numerical check, not a trained FoMoNMR result or evidence of property gains.

A separate 16-record real-data check with the step-30,000 no-fp checkpoint and
bfloat16 produced finite losses/gradients: shift loss 5.77, relational MSE 0.50.
With weight 0.1, the relational Transformer gradient norm was about 2.5% of
the shift gradient norm. This small-batch check uses a fixture-fitted center;
it supports a conservative starting weight, not a globally calibrated one.
The complete Rich atom-count audit found maxima of 84 train / 65 validation,
below the existing processor limit of 128.

## Posttraining

After both final sidecars exist:

```bash
nmr-env main
PYTHONPATH=scripts python scripts/model/train.py \
  --stage posttrain \
  --config scripts/model/configs/ablation/unimol_relational.yaml \
  --pretrained-checkpoint runs/fomonmr/fp-ablation-pretrain-no-fp/checkpoints/latest/step=30000.ckpt \
  --experiment-name unimol_ablation_posttrain \
  --run-name unimol-relational-posttrain \
  --accelerator gpu --devices 1 --precision bf16-mixed \
  --batch-size 2048 --num-workers 4 --max-steps 30000 \
  --validation-interval 1000 --checkpoint-interval 1000
```

Pretrained loading removes only old Morgan head parameters and strictly checks
all remaining weights. Normal resume uses a checkpoint with the same objective.
Use the same starting checkpoint, steps and Rich input/annotation settings for
the no-auxiliary-loss control. Downstream benefits remain untested.

```bash
PYTHONPATH=scripts python -m unittest scripts.model.tests.test_unimol_distillation -v
```

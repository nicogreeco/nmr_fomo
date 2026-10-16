# Ablation MLflow exports

The ablation training runs were logged in MLflow. The CSV files tracked here
were exported from those runs with `scripts/model/download_ablation_mlflow.py`.
They are local snapshots of the logged results.

The CSV files in each experiment directory contain the complete MLflow metric
history and the last logged value for every metric. The three notebooks in this
directory load these local CSV files and make simple comparison tables and
learning curves.

Run the notebooks in this directory to regenerate the consolidated tables and
training curves from the tracked exports.

## Reading the metrics

`train/` metrics come from training batches; `val/` metrics come from validation.
`val/datasets/<source>/...` reports the same validation metric for one source
(for example, `simnmr`, `nmrgym`, or `rich`). The plain `val/...` value aggregates
across the validation data. Lower is better for losses and MAE; higher is better
for AUROC.

| Metric | Meaning |
| --- | --- |
| `train/loss`, `val/loss` | Total training objective: shift loss plus any enabled, weighted auxiliary losses. |
| `train/shift`, `val/shift`, `val/shift_h`, `val/shift_c` | Masked shift-reconstruction loss (soft-label cross-entropy), overall or separately for proton and carbon peaks. Training may log the components as `train/shift/h` and `train/shift/c`. |
| `val/shift/h_mae_ppm`, `val/shift/c_mae_ppm` | Mean absolute error on masked proton/carbon shifts, in ppm. |
| `train/fp_loss`, `val/fp_loss`, `val/fp_mae`, `val/fp_macro_mae` | Molecular fingerprint-similarity objective and its validation errors. `fp_mae` averages over pairs; `fp_macro_mae` first averages within Tanimoto-similarity ranges, then across ranges. |
| `train/rich`, `val/annotation` | Reconstruction loss for available rich proton annotations when that objective is enabled. |
| `val/unimol_loss`, `val/unimol_cosine_mae` | UniMol teacher objective and cosine-similarity error, present only in the UniMol run. |
| `probe/maccs_macro_auroc` | Mean AUROC across eligible MACCS bits from a fresh linear probe on frozen embeddings; the probe does not train the encoder. |
| `lr-AdamW`, `memory/gpu_peak_*_mb` | Learning rate and peak GPU memory (allocated/reserved). |

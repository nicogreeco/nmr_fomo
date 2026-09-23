# FoMoNMR fine-tuning diagnostic

Run [the notebook](fomonmr_finetune_analysis.ipynb) in the main environment.
It reads the five endpoint CSVs in each of the three checkpoint directories,
preserving the recorded run IDs and names. No model is trained or loaded.

It writes `analysis/scores.csv` and `analysis/mean_rank.csv` (derived, ignored
files), and displays the endpoint scores and cohort sizes. Ames uses ROC-AUC;
the four regression endpoints use R². Mean ranks compare only these three
fine-tuned checkpoints, separately within each endpoint before averaging.

These results use larger cohorts than the frozen all-model intersection and
must not be ranked against frozen probes. There is one run per configuration,
not a replicated estimate. ADMET exclusion from rich pretraining does not
establish absence from SimNMR, NMRGym or other pretraining sources.

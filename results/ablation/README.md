# Ablation MLflow exports

The CSV files in each experiment directory contain the complete MLflow metric
history and the last logged value for every metric. The three notebooks in this
directory load these local CSV files and make simple comparison tables and
learning curves.

The export includes all runs found in the experiment. Runs not launched by the
three ablation shell scripts are marked as `extra_run_in_experiment` in
`runs.csv` and `last_metrics.csv`.

Run the notebooks in this directory to regenerate the consolidated tables and
training curves from the tracked exports.

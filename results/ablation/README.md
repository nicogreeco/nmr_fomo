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

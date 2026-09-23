#!/usr/bin/env bash
set -euo pipefail

output_dir="${1:-results/property_prediction_rerun}"
if [[ -e "$output_dir" ]]; then
    echo "Refusing to overwrite existing output: $output_dir" >&2
    exit 1
fi

mkdir -p "$output_dir"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT=4

models=(
    morgan nmrpeak nmrsolver nmrtrans ultranmr unimol2
    unimol2+ultranmr unimol2+nmrpeak unimol2+nmrsolver
    morgan+nmrpeak morgan+ultranmr morgan+nmrsolver
    fomonmr-pre-shifts fomonmr-post-shifts fomonmr-post-rich
    fomonmr-post-unimol-rich morgan+fomonmr-post-unimol-rich
)
datasets=(
    ames ld50_zhu lipophilicity_astrazeneca
    sangster_logp solubility_aqsoldb
)

{
    python --version
    git rev-parse HEAD
    sha256sum scripts/model_benchmarks/run_property_prediction.py
    python - <<'PY'
import numpy
import polars
import sklearn
import torch

print({
    "numpy": numpy.__version__,
    "sklearn": sklearn.__version__,
    "torch": torch.__version__,
    "polars": polars.__version__,
})
PY
    python scripts/model_benchmarks/run_property_prediction.py \
        --models "${models[@]}" \
        --datasets "${datasets[@]}" \
        --output-dir "$output_dir" \
        --cv-folds 5 \
        --epochs 300 \
        --device cpu
} 2>&1 | tee "$output_dir/rerun.log"

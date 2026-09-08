#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root. The three runs execute sequentially on one GPU.
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="runs/fomonmr/fp_ablation_pretrain_logs/$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$LOG_DIR"

run_training() {
    local config="$1"
    local run_name="$2"
    local log_file="$LOG_DIR/${run_name}.log"

    {
        echo "Starting $run_name; log: $log_file"
        PYTHONPATH=scripts "$PYTHON_BIN" scripts/model/train.py \
            --stage pretrain \
            --config "$config" \
            --experiment-name fp_ablation_pretrain \
            --accelerator gpu \
            --devices 1 \
            --precision bf16-mixed \
            --batch-size 2048 \
            --accumulate-grad-batches 1 \
            --num-workers 6 \
            --max-steps 30000 \
            --validation-interval 1000 \
            --checkpoint-interval 1000 \
            --maccs-probe-every-n-validations 1 \
            --run-name "$run_name"
    } 2>&1 | tee "$log_file"
}

run_training scripts/model/configs/ablation/no_fp.yaml fp-ablation-pretrain-no-fp
run_training scripts/model/configs/ablation/weighted_fp.yaml fp-ablation-pretrain-weighted-fp
run_training scripts/model/configs/ablation/balanced_fp.yaml fp-ablation-pretrain-balanced-fp

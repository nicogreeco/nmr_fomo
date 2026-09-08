#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root after activating the main GPU environment.
# All four runs execute sequentially on one GPU.
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="runs/fomonmr/shift_encoder_ablation_logs/$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "${LOG_DIR}"

BATCH_SIZE="${BATCH_SIZE:-2048}"
NUM_WORKERS="${NUM_WORKERS:-6}"
MAX_STEPS="${MAX_STEPS:-30000}"
VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-2500}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5000}"

run_training() {
    local config="$1"
    local run_name="$2"
    local log_file="${LOG_DIR}/${run_name}.log"

    {
        echo "Starting ${run_name}; log: ${log_file}"
        PYTHONPATH=scripts "${PYTHON_BIN}" scripts/model/train.py \
            --stage pretrain \
            --config "${config}" \
            --experiment-name shift_encoder_ablation \
            --accelerator gpu \
            --devices 1 \
            --precision bf16-mixed \
            --batch-size "${BATCH_SIZE}" \
            --accumulate-grad-batches 1 \
            --num-workers "${NUM_WORKERS}" \
            --max-steps "${MAX_STEPS}" \
            --validation-interval "${VALIDATION_INTERVAL}" \
            --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
            --logging complete \
            --maccs-probe-every-n-validations 1 \
            --run-name "${run_name}"
    } 2>&1 | tee "${log_file}"
}

run_training \
    scripts/model/configs/ablation/fourier_log_256.yaml \
    shift-ablation-log-256
run_training \
    scripts/model/configs/ablation/fourier_log_128.yaml \
    shift-ablation-log-128
run_training \
    scripts/model/configs/ablation/fourier_log_512.yaml \
    shift-ablation-log-512
run_training \
    scripts/model/configs/ablation/fourier_linear_256.yaml \
    shift-ablation-linear-256

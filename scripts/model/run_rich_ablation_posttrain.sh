#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root. The four runs execute sequentially on one GPU.
PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT="${CHECKPOINT:-runs/fomonmr/gpu-test-fp-classes-fp-weights/checkpoints/best/best-step=100000.ckpt}"
LOG_DIR="runs/fomonmr/rich_ablation_posttrain_logs/$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$LOG_DIR"

run_training() {
    local config="$1"
    local run_name="$2"
    local log_file="$LOG_DIR/${run_name}.log"

    {
        echo "Starting $run_name; log: $log_file"
        PYTHONPATH=scripts "$PYTHON_BIN" scripts/model/train.py \
            --stage posttrain \
            --config "$config" \
            --pretrained-checkpoint "$CHECKPOINT" \
            --experiment-name rich_ablation_posttrain \
            --accelerator gpu \
            --devices 1 \
            --precision bf16-mixed \
            --batch-size 2048 \
            --accumulate-grad-batches 2 \
            --num-workers 6 \
            --max-steps 15000 \
            --validation-interval 500 \
            --checkpoint-interval 1000 \
            --maccs-probe-every-n-validations 1 \
            --run-name "$run_name"
    } 2>&1 | tee "$log_file"
}

if [[ "${SKIP_FULL:-0}" != "1" ]]; then
    run_training scripts/model/configs/ablation/rich_full.yaml \
        rich-ablation-posttrain-full
fi
run_training scripts/model/configs/ablation/rich_target_only.yaml \
    rich-ablation-posttrain-target-only
run_training scripts/model/configs/ablation/rich_input_only.yaml \
    rich-ablation-posttrain-input-only
run_training scripts/model/configs/ablation/rich_disabled.yaml \
    rich-ablation-posttrain-disabled

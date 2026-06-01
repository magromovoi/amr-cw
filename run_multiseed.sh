#!/usr/bin/env bash
set -euo pipefail

PYTHON=".venv/bin/python"
DATASET="bbcsport"
CT="weighted_frequent_subgraphs"
SEEDS=(42 123 456 789 1024)

CONFIG_BASELINE="config_baseline.yaml"
CONFIG_EXP="config.yaml"

RUNS_DIR="data/output/runs"

run_one() {
    local config="$1"
    local config_label="$2"
    local seed="$3"

    local run_id="${DATASET}_${config_label}_seed${seed}"
    local run_dir="${RUNS_DIR}/${run_id}"

    echo "Run: ${run_id}"

    mkdir -p "${run_dir}"

    # concept construction
    $PYTHON main.py --seed "$seed" --config "$config" --dataset "$DATASET" \
        --operation graph_concept_construction --concept_type "$CT" \
        --run_dir "$run_dir" 2>&1 | tee -a "${run_dir}/log.txt"

    # CW training
    $PYTHON main.py --seed "$seed" --config "$config" --dataset "$DATASET" \
        --operation graph_concept_whitening --mode train --concept_type "$CT" \
        --run_dir "$run_dir" 2>&1 | tee -a "${run_dir}/log.txt"

    # doc-CAP evaluation
    echo "EVAL ${run_id}:" | tee -a "${run_dir}/log.txt"
    $PYTHON main.py --seed "$seed" --config "$config" --dataset "$DATASET" \
        --operation evaluation --mode doc_cap --concept_type "$CT" \
        --run_dir "$run_dir" 2>&1 | tee -a "${run_dir}/log.txt"

    echo "" | tee -a "${run_dir}/log.txt"
}

echo "Multi-seed evaluation: baseline vs exp"
echo "Seeds: ${SEEDS[*]}"
echo "Output: ${RUNS_DIR}/"
echo ""

for SEED in "${SEEDS[@]}"; do
    run_one "$CONFIG_BASELINE" "baseline" "$SEED"
    run_one "$CONFIG_EXP" "exp" "$SEED"
done

echo "All seeds complete"
echo ""
echo "Run directories:"
ls -d ${RUNS_DIR}/${DATASET}_* 2>/dev/null | sort

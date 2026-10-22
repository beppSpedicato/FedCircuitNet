#!/bin/bash

# Seed sweep over the 20261012 client-count runs: {iid, FH} x K = 5 / 10 / 20
# x seeds 0 / 1 / 2 (K = 50 is excluded).  Train + test only, no augmented
# metrics.  Configs live in ./sweep_seed_config/ and are copies of
# 20261012_configs/ where only the seed (runtime.seed plus the partitioner's
# seed / random_state), experiment, tag and work_dirs paths change.
#
# Usage:
#     ./run_sweep_seed.sh                       # all 18 runs
#     ./run_sweep_seed.sh FH_c5_seed0 iid_c20_seed2   # just the named IDs

set -e

CONFIG_PREFIX='fedavg'
SEEDS=(0 1 2)
CLIENTS=(5 10 20)
SCHEMES=('iid' 'FH')

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
CONFIG_DIR="${SCRIPT_DIR}/sweep_seed_config"

if [ "$#" -gt 0 ]; then
    IDS=("$@")
else
    # Seed outermost: one full seed (all schemes/K) finishes before the next.
    IDS=()
    for SEED in "${SEEDS[@]}"; do
        for K in "${CLIENTS[@]}"; do
            for SCHEME in "${SCHEMES[@]}"; do
                IDS+=("${SCHEME}_c${K}_seed${SEED}")
            done
        done
    done
fi

MISSING=0
for ID in "${IDS[@]}"; do
    for PHASE in train test; do
        CFG="${CONFIG_DIR}/${CONFIG_PREFIX}_${ID}_${PHASE}.yaml"
        if [ ! -s "${CFG}" ]; then
            echo "ERROR: missing or empty config ${CFG}" >&2
            MISSING=1
        fi
    done
done
if [ "${MISSING}" -ne 0 ]; then
    echo "Aborting: resolve the missing config(s) above." >&2
    exit 1
fi

echo "Starting seed sweep: ${#IDS[@]} run(s). Logs will be recorded by Aim."

for ID in "${IDS[@]}"; do
    echo "=========================================================="
    echo "Running ${ID}"
    echo "=========================================================="

    echo "--> Training ${ID}"
    python train_aim.py --config-path="${CONFIG_DIR}" --config-name="${CONFIG_PREFIX}_${ID}_train"

    sleep 2

    echo "--> Testing ${ID}"
    python test_aim.py --config-path="${CONFIG_DIR}" --config-name="${CONFIG_PREFIX}_${ID}_test"

    echo "${ID} completed."
    echo ""
done

echo "Seed sweep finished (${#IDS[@]} run(s))."

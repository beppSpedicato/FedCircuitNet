#!/bin/bash

# Sequentially runs train_aim.py + test_aim.py for each Hydra config pair
# under ./config/, mirroring
# code_examples/CircuitNet/drc_prediction/run_all_experiments.sh.
#
# Each entry is treated as an experiment suffix and resolves to the
# config pair:
#     ./config/${CONFIG_PREFIX}_train_${ID}.yaml
#     ./config/${CONFIG_PREFIX}_test_${ID}.yaml
#
# Two lists, kept separate because they vary along different axes:
#   BASELINE_CONFIGS  -- partitioning scheme, model fixed at RouteNet
#   GROUPNORM_CONFIGS -- same partitioning, model swapped to
#                        RouteNetGroupNorm.  Each entry here is a
#                        controlled A/B against the same-named baseline
#                        (identical seed / E / eta / B / T), so it is only
#                        interpretable if that baseline has also been run.
#
# Usage:
#     ./run_all_experiments.sh                      # everything, both lists
#     ./run_all_experiments.sh kmeans_groupnorm     # just the named IDs
#     ./run_all_experiments.sh iid kmeans
#
# Aim records every run under the experiment tag set inside each config.

set -e

CONFIG_PREFIX='fedavg'
BASELINE_CONFIGS=('iid' 'kmeans' 'dirichlet')
GROUPNORM_CONFIGS=()

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# CLI args select a subset; no args runs both lists in full.
if [ "$#" -gt 0 ]; then
    CONFIGS=("$@")
else
    CONFIGS=("${BASELINE_CONFIGS[@]}" "${GROUPNORM_CONFIGS[@]}")
fi

# Fail before burning a single GPU-hour if any pair is missing or misnamed.
MISSING=0
for ID in "${CONFIGS[@]}"; do
    for PHASE in train test; do
        CFG="./config/${CONFIG_PREFIX}_${PHASE}_${ID}.yaml"
        if [ ! -f "${CFG}" ]; then
            echo "ERROR: missing config ${CFG}" >&2
            MISSING=1
        fi
    done
done
if [ "${MISSING}" -ne 0 ]; then
    echo "Aborting: resolve the missing config(s) above." >&2
    exit 1
fi

echo "Starting sequential execution of ${#CONFIGS[@]} FedCircuitNet configuration(s)..."
echo "Logs will be recorded by Aim."

for ID in "${CONFIGS[@]}"; do
    TRAIN_CFG="${CONFIG_PREFIX}_train_${ID}"
    TEST_CFG="${CONFIG_PREFIX}_test_${ID}"
    LABEL="${ID}"

    echo "=========================================================="
    echo "Running Configuration ID: ${LABEL}"
    echo "=========================================================="

    echo "--> Training Config ${LABEL} (${TRAIN_CFG})"
    python train_aim.py --config-name="${TRAIN_CFG}"

    sleep 2

    echo "--> Testing Config ${LABEL} (${TEST_CFG})"
    python test_aim.py --config-name="${TEST_CFG}"

    echo "Configuration ${LABEL} completed successfully."
    echo ""
done

echo "All ${#CONFIGS[@]} configuration(s) have been trained and tested!"

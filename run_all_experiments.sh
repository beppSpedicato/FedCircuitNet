#!/bin/bash

# Sequentially runs train_aim.py + test_aim.py + drc_augmented_metrics.py
# for each Hydra config triple under ./20261007_configs/, mirroring
# run_sweep_clients_exp.sh.
#
# Each entry is a run ID and resolves to the config triple:
#     ./20261007_configs/${CONFIG_PREFIX}_${ID}_train.yaml
#     ./20261007_configs/${CONFIG_PREFIX}_${ID}_test.yaml
#     ./20261007_configs/${CONFIG_PREFIX}_${ID}_aug.yaml
#
# Every run uses RouteNetGroupNorm and matches
# config/fedavg_train_{iid,feature_hierarchical}_groupnorm.yaml except for
# the fields below.  Two lists, kept separate because they vary along
# different axes:
#   PARTICIPATION_CONFIGS -- {iid, FH} x K = 5 / 10 / 20 clients with full
#                            participation (round_clients = n_partitions),
#                            T = 200, E = 20.  The local-step budget
#                            T * E * m grows with K: 20k / 40k / 80k.
#   ROUNDS_CONFIGS        -- FH at the base K = 20 / m = 10 with T = 400
#                            (80k steps, the same budget as FH_c20).
#
# The entry scripts default to config_path=./config, so the folder is
# handed to Hydra with --config-path.
#
# Usage:
#     ./run_all_experiments.sh                  # everything, both lists
#     ./run_all_experiments.sh FH_c5            # just the named IDs
#     ./run_all_experiments.sh iid_c20 FH_c20
#
# Aim records the runs under the fedavg_drc_{train,test,augmented_metrics}
# _20261007 experiments, each tagged with its run ID.

set -e

CONFIG_PREFIX='fedavg'
# Smallest K first, so each client count is finished for both schemes
# before the next (and longer) one starts.
PARTICIPATION_CONFIGS=('iid_c5' 'FH_c5' 'iid_c10' 'FH_c10' 'iid_c20' 'FH_c20')
ROUNDS_CONFIGS=('FH_400r')

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
CONFIG_DIR="${SCRIPT_DIR}/20261007_configs"

# CLI args select a subset; no args runs both lists in full.
if [ "$#" -gt 0 ]; then
    CONFIGS=("$@")
else
    CONFIGS=("${PARTICIPATION_CONFIGS[@]}" "${ROUNDS_CONFIGS[@]}")
fi

# Fail before burning a single GPU-hour if any triple is missing or misnamed.
MISSING=0
for ID in "${CONFIGS[@]}"; do
    for PHASE in train test aug; do
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

echo "Starting sequential execution of ${#CONFIGS[@]} FedCircuitNet configuration(s)..."
echo "Logs will be recorded by Aim."

for ID in "${CONFIGS[@]}"; do
    TRAIN_CFG="${CONFIG_PREFIX}_${ID}_train"
    TEST_CFG="${CONFIG_PREFIX}_${ID}_test"
    METRICS_CFG="${CONFIG_PREFIX}_${ID}_aug"
    LABEL="${ID}"

    echo "=========================================================="
    echo "Running Configuration ID: ${LABEL}"
    echo "=========================================================="

    echo "--> Training Config ${LABEL} (${TRAIN_CFG})"
    python train_aim.py --config-path="${CONFIG_DIR}" --config-name="${TRAIN_CFG}"

    sleep 2

    echo "--> Testing Config ${LABEL} (${TEST_CFG})"
    python test_aim.py --config-path="${CONFIG_DIR}" --config-name="${TEST_CFG}"

    echo "--> Augmented metrics Config ${LABEL} (${METRICS_CFG})"
    python drc_augmented_metrics.py --config-path="${CONFIG_DIR}" --config-name="${METRICS_CFG}"

    echo "Configuration ${LABEL} completed successfully."
    echo ""
done

echo "All ${#CONFIGS[@]} configuration(s) have been trained and tested!"

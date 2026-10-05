#!/bin/bash

# Client-count sweep: sequentially runs train_aim.py + test_aim.py +
# drc_augmented_metrics.py for each Hydra config triple under
# ./sweep_clients_config/, mirroring run_all_experiments.sh.
#
# Each entry is an experiment ID of the form ${SCHEME}_groupnorm_c${K} and
# resolves to the config triple:
#     ./sweep_clients_config/${CONFIG_PREFIX}_train_${ID}.yaml
#     ./sweep_clients_config/${CONFIG_PREFIX}_test_${ID}.yaml
#     ./sweep_clients_config/${CONFIG_PREFIX}_augmented_metrics_${ID}.yaml
#
# The sweep is SCHEMES x CLIENTS with the model fixed at RouteNetGroupNorm
# and participation fixed at 50 % (round_clients = ceil(K / 2)); every other
# field matches config/fedavg_train_${SCHEME}_groupnorm.yaml.  K = 20 is not
# in the list: those existing runs (round_clients = 10) are already the
# K = 20 point of the sweep.
#
# The entry scripts default to config_path=./config, so the sweep folder is
# handed to Hydra with --config-path.
#
# Usage:
#     ./run_sweep_clients_exp.sh                          # full 4 x 4 sweep
#     ./run_sweep_clients_exp.sh iid_groupnorm_c5         # just the named IDs
#     ./run_sweep_clients_exp.sh kmeans_groupnorm_c50 kmeans_groupnorm_c100
#
# Aim records the runs under the fedavg_drc_{train,test,augmented_metrics}
# _sweep_clients experiments, kept apart from the K = 20 runs.

set -e

CONFIG_PREFIX='fedavg'
SCHEMES=('iid' 'dirichlet' 'kmeans' 'feature_hierarchical')
CLIENTS=(5 10 50 100)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
CONFIG_DIR="${SCRIPT_DIR}/sweep_clients_config"

# CLI args select a subset; no args runs the full sweep, smallest K first so
# each client count is finished across all schemes before the next starts.
if [ "$#" -gt 0 ]; then
    CONFIGS=("$@")
else
    CONFIGS=()
    for K in "${CLIENTS[@]}"; do
        for SCHEME in "${SCHEMES[@]}"; do
            CONFIGS+=("${SCHEME}_groupnorm_c${K}")
        done
    done
fi

# Fail before burning a single GPU-hour if any triple is missing or misnamed.
MISSING=0
for ID in "${CONFIGS[@]}"; do
    for PHASE in train test augmented_metrics; do
        CFG="${CONFIG_DIR}/${CONFIG_PREFIX}_${PHASE}_${ID}.yaml"
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

echo "Starting sequential execution of ${#CONFIGS[@]} client-sweep configuration(s)..."
echo "Logs will be recorded by Aim."

for ID in "${CONFIGS[@]}"; do
    TRAIN_CFG="${CONFIG_PREFIX}_train_${ID}"
    TEST_CFG="${CONFIG_PREFIX}_test_${ID}"
    METRICS_CFG="${CONFIG_PREFIX}_augmented_metrics_${ID}"
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

echo "All ${#CONFIGS[@]} client-sweep configuration(s) have been trained and tested!"

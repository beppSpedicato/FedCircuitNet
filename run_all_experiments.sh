#!/bin/bash

# Sequentially runs train_aim.py + test_aim.py for each Hydra config pair
# under ./config/, mirroring
# code_examples/CircuitNet/drc_prediction/run_all_experiments.sh.
#
# Each entry in CONFIGS is treated as an experiment suffix and resolves to
# the config pair:
#     ./config/${CONFIG_PREFIX}_train_${ID}.yaml
#     ./config/${CONFIG_PREFIX}_test_${ID}.yaml
# An empty ID (`''`) selects the unsuffixed base configs
# (`fedavg_train.yaml` / `fedavg_test.yaml`), which is the layout currently
# shipped with the repo.
#
# Aim records every run under the experiment tag set inside each config.

set -e

CONFIG_PREFIX='fedavg'
CONFIGS=('')

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "Starting sequential execution of ${#CONFIGS[@]} FedCircuitNet configuration(s)..."
echo "Logs will be recorded by Aim."

for ID in "${CONFIGS[@]}"; do
    if [ -z "${ID}" ]; then
        TRAIN_CFG="${CONFIG_PREFIX}_train"
        TEST_CFG="${CONFIG_PREFIX}_test"
        LABEL='<base>'
    else
        TRAIN_CFG="${CONFIG_PREFIX}_train_${ID}"
        TEST_CFG="${CONFIG_PREFIX}_test_${ID}"
        LABEL="${ID}"
    fi

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

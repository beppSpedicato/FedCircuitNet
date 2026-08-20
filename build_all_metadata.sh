#!/bin/bash

# Translate every CircuitNet ann-file CSV under
# code_examples/CircuitNet/drc_prediction/files/ into a FedCircuitNet
# metadata CSV of the same basename under main_code/FedCircuitNet/files/.
#
# Schema (N28 vs N14) is auto-detected per file by create_metadata_files.py.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SRC_DIR="${REPO_ROOT}/code_examples/CircuitNet/drc_prediction/files"
DST_DIR="${SCRIPT_DIR}/files"
TRANSLATOR="${SCRIPT_DIR}/create_metadata_files.py"

if [ ! -d "${SRC_DIR}" ]; then
    echo "Source directory not found: ${SRC_DIR}" >&2
    exit 1
fi
if [ ! -f "${TRANSLATOR}" ]; then
    echo "Translator script not found: ${TRANSLATOR}" >&2
    exit 1
fi

mkdir -p "${DST_DIR}"

shopt -s nullglob
csv_files=("${SRC_DIR}"/*.csv)
shopt -u nullglob

if [ "${#csv_files[@]}" -eq 0 ]; then
    echo "No CSV files under ${SRC_DIR}" >&2
    exit 1
fi

echo "Translating ${#csv_files[@]} CSV file(s) into ${DST_DIR}"

for src in "${csv_files[@]}"; do
    base="$(basename "${src}")"
    dst="${DST_DIR}/${base}"
    echo "--> ${base}"
    python "${TRANSLATOR}" --input="${src}" --output="${dst}"
done

echo "Done. Wrote ${#csv_files[@]} file(s) to ${DST_DIR}."

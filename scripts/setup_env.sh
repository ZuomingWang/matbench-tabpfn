#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="matbench-tabpfn"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda env update -f "${PROJECT_DIR}/environment.yml" --prune
else
  conda env create -f "${PROJECT_DIR}/environment.yml"
fi

conda run -n "${ENV_NAME}" python -m pip install --no-deps matbench==0.6
conda run -n "${ENV_NAME}" python "${PROJECT_DIR}/scripts/check_setup.py"

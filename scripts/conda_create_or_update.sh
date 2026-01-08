#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/conda/env.dlp.yml"
ENV_NAME="dlp"

# env 존재 여부 확인
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[conda] Updating existing env: ${ENV_NAME}"
  conda env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune
else
  echo "[conda] Creating env: ${ENV_NAME}"
  conda env create -f "${ENV_FILE}"
fi

echo "[conda] Done. Activate with: conda activate ${ENV_NAME}"

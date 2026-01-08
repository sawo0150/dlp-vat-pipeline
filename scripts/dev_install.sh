#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# (중요) conda activate dlp 상태에서 실행해야 함
python -V
pip -V

echo "[pip] Editable install: common, pipeline, ml"
pip install -e ./common
pip install -e ./pipeline
pip install -e ./ml

echo "[ok] Installed. Try:"
echo "  ./scripts/run_pipeline.sh"
echo "  ./scripts/run_ml.sh"

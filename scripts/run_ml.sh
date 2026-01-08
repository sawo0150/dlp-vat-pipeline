#!/usr/bin/env bash
set -euo pipefail
cd ml
python3 -m dlp_ml.train "$@"

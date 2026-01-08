#!/usr/bin/env bash
set -euo pipefail
cd pipeline
python3 -m src.main "$@"

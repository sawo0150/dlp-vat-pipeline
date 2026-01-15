# src/dlp_ml/main.py

"""Backward-compatible entrypoint.

Prefer:
    python -m dlp_ml.cli ...

This file keeps `dlp_ml.train` working for older scripts.
"""

from dlp_ml.cli import main

if __name__ == "__main__":
    main()

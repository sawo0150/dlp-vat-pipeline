# src/dlp_ml/utils/reproducibility.py

import os
import random
from typing import Optional

import numpy as np

try:
    import torch
except Exception:
    torch = None


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy, and torch (if available)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

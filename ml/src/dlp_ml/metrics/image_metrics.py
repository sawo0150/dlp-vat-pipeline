# src/dlp_ml/metrics/image_metrics.py

import math
import torch


@torch.no_grad()
def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean((pred - target) ** 2).item())


@torch.no_grad()
def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    m = torch.mean((pred - target) ** 2).item()
    if m <= 1e-12:
        return 99.0
    return 10.0 * math.log10((data_range ** 2) / m)

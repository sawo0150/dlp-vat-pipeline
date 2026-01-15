# src/dlp_ml/losses/recon.py

import torch
import torch.nn as nn


class L1Loss(nn.Module):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = float(weight)
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.weight * self.l1(pred, target)


class MSELoss(nn.Module):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = float(weight)
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.weight * self.mse(pred, target)

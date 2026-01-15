# ml/src/dlp_ml/models/pix2pix_wrapper.py
import torch
import torch.nn as nn

class Pix2PixModel(nn.Module):
    """
    Wraps (G, D) so Hydra can instantiate a single model object.
    forward(x) returns G(x).
    """
    def __init__(self, generator: nn.Module, discriminator: nn.Module):
        super().__init__()
        self.G = generator
        self.D = discriminator

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.G(x)

# ml/src/dlp_ml/trainers/pix2pix_trainer.py
import os
from typing import Optional, Callable

import torch
import torch.nn as nn
from tqdm import tqdm

from dlp_ml.utils.io import ensure_dir, save_json

def _to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out

class Pix2PixTrainer:
    """
    Trains Pix2PixModel (expects model.G and model.D).
    Uses:
      lossG = gan_w * BCE(D(x, G(x)), 1) + l1_w * L1(G(x), y)
      lossD = 0.5*(BCE(D(x,y),1) + BCE(D(x,G(x).detach()),0))
    """

    def __init__(
        self,
        device: torch.device,
        max_epochs: int,
        lr: float,
        weight_decay: float = 0.0,
        log_every: int = 50,
        grad_clip: float = 0.0,
        amp: bool = True,
        gan_weight: float = 1.0,
        l1_weight: float = 100.0,
        betas=(0.5, 0.999),
    ):
        self.device = device
        self.max_epochs = int(max_epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.log_every = int(log_every)
        self.grad_clip = float(grad_clip)
        self.amp = bool(amp)
        self.gan_weight = float(gan_weight)
        self.l1_weight = float(l1_weight)
        self.betas = betas

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()

    def fit(self, model, train_loader, val_loader, logger, run_dir: str, metrics_fn: Optional[Callable] = None) -> str:
        assert hasattr(model, "G") and hasattr(model, "D"), "Pix2PixTrainer expects model.G and model.D"

        G = model.G
        D = model.D

        optG = torch.optim.Adam(G.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.weight_decay)
        optD = torch.optim.Adam(D.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.weight_decay)

        ensure_dir(os.path.join(run_dir, "checkpoints"))
        best_path = os.path.join(run_dir, "checkpoints", "best.pt")
        best_val = float("inf")

        global_step = 0
        for epoch in range(self.max_epochs):
            G.train(); D.train()
            pbar = tqdm(train_loader, desc=f"train epoch {epoch}", leave=False)

            for batch in pbar:
                batch = _to_device(batch, self.device)
                x, y = batch["x"], batch["y"]

                # ---- Train D ----
                optD.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=self.amp):
                    y_hat_det = G(x).detach()
                    logits_real = D(x, y)
                    logits_fake = D(x, y_hat_det)
                    lossD = 0.5 * (
                        self.bce(logits_real, torch.ones_like(logits_real)) +
                        self.bce(logits_fake, torch.zeros_like(logits_fake))
                    )
                self.scaler.scale(lossD).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(optD)
                    torch.nn.utils.clip_grad_norm_(D.parameters(), self.grad_clip)
                self.scaler.step(optD)

                # ---- Train G ----
                optG.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=self.amp):
                    y_hat = G(x)
                    logits_fake_for_g = D(x, y_hat)
                    lossG_gan = self.bce(logits_fake_for_g, torch.ones_like(logits_fake_for_g))
                    lossG_l1 = self.l1(y_hat, y)
                    lossG = self.gan_weight * lossG_gan + self.l1_weight * lossG_l1

                self.scaler.scale(lossG).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(optG)
                    torch.nn.utils.clip_grad_norm_(G.parameters(), self.grad_clip)
                self.scaler.step(optG)
                self.scaler.update()

                if global_step % self.log_every == 0:
                    logger.log_metrics({
                        "train/lossD": float(lossD.item()),
                        "train/lossG": float(lossG.item()),
                        "train/lossG_gan": float(lossG_gan.item()),
                        "train/lossG_l1": float(lossG_l1.item()),
                        "epoch": float(epoch),
                    }, step=global_step)

                pbar.set_postfix(lossG=float(lossG.item()), lossD=float(lossD.item()))
                global_step += 1

            # ---- Val (L1 only for selection) ----
            val_l1 = self.validate(G, val_loader, logger, global_step, metrics_fn=metrics_fn)
            logger.log_metrics({"val/lossG_l1": float(val_l1), "epoch": float(epoch)}, step=global_step)

            if val_l1 < best_val:
                best_val = val_l1
                torch.save({
                    "G": G.state_dict(),
                    "D": D.state_dict(),
                    "epoch": epoch,
                    "val_l1": best_val
                }, best_path)

        last_path = os.path.join(run_dir, "checkpoints", "last.pt")
        torch.save({"G": G.state_dict(), "D": D.state_dict(), "epoch": self.max_epochs - 1}, last_path)

        save_json(os.path.join(run_dir, "summary.json"), {"best_val_l1": best_val, "best_ckpt": best_path})
        return best_path

    @torch.no_grad()
    def validate(self, G, val_loader, logger, step: int, metrics_fn=None) -> float:
        G.eval()
        losses = []
        for batch in val_loader:
            batch = _to_device(batch, self.device)
            x, y = batch["x"], batch["y"]
            y_hat = G(x)
            loss = self.l1(y_hat, y)
            losses.append(float(loss.item()))
            if metrics_fn is not None:
                m = metrics_fn(y_hat, y)
                if m:
                    logger.log_metrics({f"val/{k}": float(v) for k, v in m.items()}, step=step)
        return sum(losses) / max(1, len(losses))

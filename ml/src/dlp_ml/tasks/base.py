# src/dlp_ml/tasks/base.py

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch.optim import Adam
from tqdm import tqdm

from dlp_ml.loggers.base import BaseLogger
from dlp_ml.utils.io import ensure_dir, save_json


def _to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


class BaseTask:
    def __init__(self, cfg, run_dir: str, logger: BaseLogger):
        self.cfg = cfg
        self.run_dir = run_dir
        self.logger = logger

    def run(self) -> None:
        raise NotImplementedError


class SimpleTrainer:
    """A tiny trainer to keep deps light (Hydra + Torch only)."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn,
        device: torch.device,
        max_epochs: int,
        log_every: int = 50,
        grad_clip: float = 0.0,
        amp: bool = True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.max_epochs = int(max_epochs)
        self.log_every = int(log_every)
        self.grad_clip = float(grad_clip)
        self.amp = bool(amp)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

    def fit(self, train_loader, val_loader, logger: BaseLogger, run_dir: str, metrics_fn=None) -> str:
        ensure_dir(os.path.join(run_dir, "checkpoints"))
        best_path = os.path.join(run_dir, "checkpoints", "best.pt")
        best_val = float("inf")

        global_step = 0
        for epoch in range(self.max_epochs):
            self.model.train()
            pbar = tqdm(train_loader, desc=f"train epoch {epoch}", leave=False)
            for batch in pbar:
                batch = _to_device(batch, self.device)
                x, y = batch["x"], batch["y"]

                self.optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=self.amp):
                    pred = self.model(x)
                    loss = self.loss_fn(pred, y)

                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                if global_step % self.log_every == 0:
                    logger.log_metrics({"train/loss": float(loss.item()), "epoch": float(epoch)}, step=global_step)
                pbar.set_postfix(loss=float(loss.item()))
                global_step += 1

            # val
            val_loss = self.validate(val_loader, logger, global_step, metrics_fn=metrics_fn)
            logger.log_metrics({"val/loss": float(val_loss), "epoch": float(epoch)}, step=global_step)

            if val_loss < best_val:
                best_val = val_loss
                torch.save({"model": self.model.state_dict(), "epoch": epoch, "val_loss": best_val}, best_path)

        # save last
        last_path = os.path.join(run_dir, "checkpoints", "last.pt")
        torch.save({"model": self.model.state_dict(), "epoch": self.max_epochs - 1}, last_path)

        save_json(os.path.join(run_dir, "summary.json"), {"best_val_loss": best_val, "best_ckpt": best_path})
        return best_path

    @torch.no_grad()
    def validate(self, val_loader, logger: BaseLogger, step: int, metrics_fn=None) -> float:
        self.model.eval()
        losses = []
        for batch in val_loader:
            batch = _to_device(batch, self.device)
            x, y = batch["x"], batch["y"]
            pred = self.model(x)
            loss = self.loss_fn(pred, y)
            losses.append(float(loss.item()))
            if metrics_fn is not None:
                m = metrics_fn(pred, y)
                if m:
                    logger.log_metrics({f"val/{k}": float(v) for k, v in m.items()}, step=step)
        return sum(losses) / max(1, len(losses))

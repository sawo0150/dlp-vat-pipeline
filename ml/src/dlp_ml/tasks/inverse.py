# src/dlp_ml/tasks/inverse.py

import os
from typing import Any, Dict, Optional

import numpy as np
import torch

from dlp_ml.data.datamodule import build_loaders
from dlp_ml.loggers.none_logger import NoneLogger
from dlp_ml.loggers.wandb_logger import WandbLogger
from dlp_ml.metrics.image_metrics import psnr
from dlp_ml.tasks.base import BaseTask, SimpleTrainer
from dlp_ml.utils.io import ensure_dir

from dlp_ml.utils.instantiate import instantiate_from_config
from dlp_ml.trainers.pix2pix_trainer import Pix2PixTrainer
from dlp_ml.losses.recon import L1Loss, MSELoss


def _make_logger(cfg) -> Any:
    if cfg.logger.name == "none":
        return NoneLogger()
    if cfg.logger.name == "wandb":
        return WandbLogger(
            project=cfg.logger.project,
            name=cfg.logger.run_name,
            entity=cfg.logger.entity,
            tags=list(cfg.logger.tags) if cfg.logger.tags is not None else None,
            mode=cfg.logger.mode,
            config=None,
        )
    raise ValueError(f"Unknown logger: {cfg.logger.name}")


class InverseTask(BaseTask):
    """Baseline inverse task: learn image-to-image mapping with UNet + L1/MSE.

    This is intentionally minimal to get you running fast.
    Later you can swap model/loss/metrics via Hydra groups.
    """

    def __init__(self, cfg, run_dir: str):
        self.cfg = cfg
        self.run_dir = run_dir
        self.logger = _make_logger(cfg)

    def run(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() and self.cfg.trainer.use_cuda else "cpu")

        # Dataloaders
        train_loader, val_loader = build_loaders(
            manifest_path=self.cfg.data.manifest_path,
            base_dir=self.cfg.data.base_dir,
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            val_ratio=self.cfg.data.val_ratio,
            seed=int(self.cfg.seed),
            input_key=self.cfg.data.input_key,
            target_key=self.cfg.data.target_key,
            input_mode=self.cfg.data.input_mode,
            target_mode=self.cfg.data.target_mode,
            normalize=bool(self.cfg.data.normalize),
            group_split=bool(getattr(self.cfg.data, "group_split", True)),
        )

        # Model (Hydra instantiate)
        # cfg.model must contain _target_
        model = instantiate_from_config(self.cfg.model).to(device) 

        def metrics_fn(pred, target):
            # pred/target are [B,1,H,W] in [0,1]
            return {"psnr": psnr(pred, target)}

        # Trainer 선택:
        # - Pix2PixModel이면 Pix2PixTrainer 사용 (model.G / model.D 필요)
        # - 그 외는 기존 SimpleTrainer (recon) 사용
        if hasattr(model, "G") and hasattr(model, "D"):
            # pix2pix-style
            trainer = Pix2PixTrainer(
                device=device,
                max_epochs=int(self.cfg.trainer.max_epochs),
                lr=float(self.cfg.trainer.lr),
                weight_decay=float(self.cfg.trainer.weight_decay),
                log_every=int(self.cfg.trainer.log_every),
                grad_clip=float(self.cfg.trainer.grad_clip),
                amp=bool(self.cfg.trainer.amp),
                gan_weight=float(getattr(self.cfg.loss, "gan_weight", 1.0)),
                l1_weight=float(getattr(self.cfg.loss, "l1_weight", 100.0)),
            )
            best_ckpt = trainer.fit(model, train_loader, val_loader, logger=self.logger, run_dir=self.run_dir, metrics_fn=metrics_fn)
        else:
            # recon-style
            if self.cfg.loss.name == "l1":
                loss_fn = L1Loss(weight=float(self.cfg.loss.weight))
            elif self.cfg.loss.name == "mse":
                loss_fn = MSELoss(weight=float(self.cfg.loss.weight))
            else:
                raise ValueError(f"Unknown loss: {self.cfg.loss.name}")

            opt = torch.optim.Adam(model.parameters(), lr=float(self.cfg.trainer.lr), weight_decay=float(self.cfg.trainer.weight_decay))
            trainer = SimpleTrainer(
                model=model,
                optimizer=opt,
                loss_fn=loss_fn,
                device=device,
                max_epochs=int(self.cfg.trainer.max_epochs),
                log_every=int(self.cfg.trainer.log_every),
                grad_clip=float(self.cfg.trainer.grad_clip),
                amp=bool(self.cfg.trainer.amp),
            )
            best_ckpt = trainer.fit(train_loader, val_loader, logger=self.logger, run_dir=self.run_dir, metrics_fn=metrics_fn)
 
        # Finish
        self.logger.log_text("run/best_ckpt", best_ckpt, step=int(self.cfg.trainer.max_epochs))
        self.logger.finish()

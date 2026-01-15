# src/dlp_ml/loggers/wandb_logger.py

import os
from typing import Any, Dict, Optional

from dlp_ml.loggers.base import BaseLogger

try:
    import wandb
except Exception:
    wandb = None


class WandbLogger(BaseLogger):
    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        entity: Optional[str] = None,
        tags: Optional[list] = None,
        config: Optional[dict] = None,
        mode: str = "online",
    ) -> None:
        if wandb is None:
            raise ImportError("wandb is not installed. Install it or switch logger=none.")
        self._run = wandb.init(project=project, name=name, entity=entity, tags=tags, config=config, mode=mode)

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        wandb.log(metrics, step=step)

    def log_images(self, images: Dict[str, Any], step: int) -> None:
        # images: dict[str, HxWxC uint8 or wandb.Image]
        payload = {}
        for k, v in images.items():
            if isinstance(v, wandb.Image):
                payload[k] = v
            else:
                payload[k] = wandb.Image(v)
        wandb.log(payload, step=step)

    def log_text(self, key: str, text: str, step: int) -> None:
        wandb.log({key: text}, step=step)

    def finish(self) -> None:
        wandb.finish()

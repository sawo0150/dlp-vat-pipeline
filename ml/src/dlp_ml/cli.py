# src/dlp_ml/cli.py

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

from dlp_ml.utils.reproducibility import seed_everything
from dlp_ml.utils.io import ensure_dir, save_text
from dlp_ml.utils.instantiate import instantiate_from_config

# -----------------------------------------------------------------------------
# CLI Entry
# -----------------------------------------------------------------------------
# Usage examples:
#   python -m dlp_ml.cli task=inverse model=unet data=manifest_pair logger=wandb
#   python -m dlp_ml.cli task=inverse data.dataset_id=B4_Mix_10k trainer.max_epochs=50
#
# Hydra will create a new run directory by default; we also save resolved config.
# -----------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Reproducibility
    seed_everything(int(cfg.seed))

    # Save resolved config in the run dir
    run_dir = os.getcwd()
    ensure_dir(run_dir)
    save_text(os.path.join(run_dir, "config_resolved.yaml"), OmegaConf.to_yaml(cfg))

    # Instantiate task (trainer loop lives inside task)
    task = instantiate_from_config(cfg.task, cfg=cfg, run_dir=run_dir)
    task.run()

if __name__ == "__main__":
    main()

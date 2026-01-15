# src/dlp_ml/utils/instantiate.py

from typing import Any, Dict, Optional

import hydra
from omegaconf import DictConfig, OmegaConf


def instantiate_from_config(node: DictConfig, **kwargs) -> Any:
    """Hydra instantiate with optional injected kwargs.

    Example config:
        task:
          _target_: dlp_ml.tasks.inverse.InverseTask
          ...
    """
    if "_target_" not in node:
        raise ValueError("Config node must contain _target_. Got:\n" + OmegaConf.to_yaml(node))
    return hydra.utils.instantiate(node, **kwargs)

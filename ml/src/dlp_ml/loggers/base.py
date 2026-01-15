# src/dlp_ml/loggers/base.py

from typing import Any, Dict, Optional


class BaseLogger:
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        pass

    def log_images(self, images: Dict[str, Any], step: int) -> None:
        pass

    def log_text(self, key: str, text: str, step: int) -> None:
        pass

    def finish(self) -> None:
        pass

# src/dlp_ml/data/manifest_dataset.py

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
except Exception as e:
    raise ImportError("This dataset requires PyTorch installed.") from e


def _read_image(path: str, mode: str = "L") -> np.ndarray:
    # mode: "L" grayscale, "RGB"
    img = Image.open(path).convert(mode)
    arr = np.array(img)
    return arr


class ManifestPairDataset(Dataset):
    """Generic (input, target) dataset from a jsonl manifest.

    Manifest format expectation (per line JSON):
      {
        "id": "...",
        "input_path": "relative/or/absolute.png",
        "target_path": "relative/or/absolute.png",
        "meta": {... optional ...}
      }

    If your pipeline uses different keys (e.g., mask_path/light_path),
    set `input_key` / `target_key` accordingly.

    Paths:
      - If a path is relative, it is resolved against `base_dir`.
    """

    def __init__(
        self,
        manifest_path: str,
        base_dir: Optional[str] = None,
        input_key: str = "input_path",
        target_key: str = "target_path",
        input_mode: str = "L",
        target_mode: str = "L",
        normalize: bool = True,
    ) -> None:
        self.manifest_path = manifest_path
        self.base_dir = base_dir or os.path.dirname(manifest_path)
        self.input_key = input_key
        self.target_key = target_key
        self.input_mode = input_mode
        self.target_mode = target_mode
        self.normalize = normalize

        self.records: List[Dict[str, Any]] = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.records.append(json.loads(line))

        if len(self.records) == 0:
            raise ValueError(f"Manifest is empty: {manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def _resolve(self, p: str) -> str:
        if os.path.isabs(p):
            return p
        return os.path.join(self.base_dir, p)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        x_path = self._resolve(rec[self.input_key])
        y_path = self._resolve(rec[self.target_key])

        x = _read_image(x_path, mode=self.input_mode)
        y = _read_image(y_path, mode=self.target_mode)

        # to float32 tensor in [0,1]
        if self.normalize:
            x = x.astype(np.float32) / 255.0
            y = y.astype(np.float32) / 255.0
        else:
            x = x.astype(np.float32)
            y = y.astype(np.float32)

        # shape: (1,H,W) for grayscale, (3,H,W) for RGB
        if x.ndim == 2:
            x = x[None, ...]
        else:
            x = np.transpose(x, (2, 0, 1))

        if y.ndim == 2:
            y = y[None, ...]
        else:
            y = np.transpose(y, (2, 0, 1))

        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y)

        meta = rec.get("meta", {})
        sample_id = rec.get("id", str(idx))
        return {"id": sample_id, "x": x_t, "y": y_t, "meta": meta}

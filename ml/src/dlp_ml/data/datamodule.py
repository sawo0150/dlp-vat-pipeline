# src/dlp_ml/data/datamodule.py

import math
from typing import Optional, Tuple, List, Dict

import random
from torch.utils.data import DataLoader, Subset

from dlp_ml.data.manifest_dataset import ManifestPairDataset


def _group_indices_by_id(ds: ManifestPairDataset) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for i, rec in enumerate(ds.records):
        gid = str(rec.get("id", i))
        groups.setdefault(gid, []).append(i)
    return groups

def build_loaders(
    manifest_path: str,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    seed: int,
    base_dir: Optional[str] = None,
    input_key: str = "input_path",
    target_key: str = "target_path",
    input_mode: str = "L",
    target_mode: str = "L",
    normalize: bool = True,
    group_split: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    ds = ManifestPairDataset(
        manifest_path=manifest_path,
        base_dir=base_dir,
        input_key=input_key,
        target_key=target_key,
        input_mode=input_mode,
        target_mode=target_mode,
        normalize=normalize,
    )
    
    n = len(ds)
    if n < 2:
        train_ds = ds
        val_ds = ds
    elif group_split:
        groups = _group_indices_by_id(ds)  # id == mask id
        group_ids = list(groups.keys())
        rng = random.Random(int(seed))
        rng.shuffle(group_ids)
        n_val_g = max(1, int(math.floor(len(group_ids) * float(val_ratio))))
        val_g = set(group_ids[:n_val_g])
        train_idx: List[int] = []
        val_idx: List[int] = []
        for gid, idxs in groups.items():
            (val_idx if gid in val_g else train_idx).extend(idxs)
        train_ds = Subset(ds, train_idx)
        val_ds = Subset(ds, val_idx)
    else:
        # fallback: sample-wise split (can leak for augmented thr images)
        n_val = max(1, int(math.floor(n * val_ratio)))
        indices = list(range(n))
        rng = random.Random(int(seed))
        rng.shuffle(indices)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        train_ds = Subset(ds, train_idx)
        val_ds = Subset(ds, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract TrainPack / MiniPack from DLP pipeline outputs.

Input (per dataset):
  <dataset>/interim/processed/index.csv
  <dataset>/interim/processed/qc.csv (optional)
  <dataset>/interim/processed/meta/*.json (optional)
  <dataset>/interim/processed/{mode}_thr_1280_random/*.png (optional)
  <dataset>/interim/processed/{mode}_thr_1280_Txx/*.png (optional, if fixed-thr enabled)

Output (pack):
  <dst_root>/<pack_name>/
    images/{mode}/mask_1280/
    images/{mode}/ld_1280/
    thr/{mode}/random/ (optional)
    thr/{mode}/fixed_Txx/ (optional)
    meta/ (optional)
    splits/{train,val,test}.txt
    manifest.csv
    dataset_card.json

Notes:
- Default uses symlink to save disk (link_mode=symlink). For resized outputs, it writes new PNGs.
- Splits are deterministic with seed.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as _dt
import glob
import hashlib
import json
import os
import random
import re
import shutil
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None


# -------------------------
# Utilities
# -------------------------
def _now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _relpath(path: str, start: str) -> str:
    return os.path.relpath(path, start).replace("\\", "/")


def _safe_name(s: str) -> str:
    # filesystem safe id
    s = s.replace(":", "__").replace("/", "__").replace("\\", "__")
    s = re.sub(r"[^0-9A-Za-z._\-]+", "_", s)
    return s


def _sha1_of_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _link_or_copy(src: str, dst: str, mode: str) -> str:
    _ensure_dir(os.path.dirname(dst))
    if os.path.abspath(src) == os.path.abspath(dst):
        return "same"
    if os.path.lexists(dst):
        os.remove(dst)
    try:
        if mode == "symlink":
            os.symlink(src, dst)
            return "symlink"
        if mode == "hardlink":
            os.link(src, dst)
            return "hardlink"
        shutil.copy2(src, dst)
        return "copy"
    except Exception:
        shutil.copy2(src, dst)
        return "copy"


def _load_image_gray_u8(path: str):
    if cv2 is None:
        raise RuntimeError("OpenCV(cv2) is required for resize. Please install opencv-python.")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _write_png(path: str, img_u8):
    if cv2 is None:
        raise RuntimeError("OpenCV(cv2) is required for resize. Please install opencv-python.")
    _ensure_dir(os.path.dirname(path))
    ok = cv2.imwrite(path, img_u8)
    if not ok:
        raise IOError(f"Failed to write: {path}")


# -------------------------
# QC filter parsing (simple)
# -------------------------
_QC_EXPR_RE = re.compile(
    r"^\s*([A-Za-z0-9_]+)\s*(>=|<=|==|!=|>|<)\s*([-+]?\d+(?:\.\d+)?)\s*$"
)


def _apply_qc_filter(df: pd.DataFrame, expr: str) -> pd.DataFrame:
    """
    expr example:
      ncc_mask1280_ld1280>0.2
      iou_T30>=0.6
    """
    m = _QC_EXPR_RE.match(expr or "")
    if not m:
        raise ValueError(f"Invalid qc_filter expression: {expr!r}")
    col, op, val_s = m.group(1), m.group(2), m.group(3)
    val = float(val_s)

    if col not in df.columns:
        raise ValueError(f"qc_filter column not found: {col}")

    s = pd.to_numeric(df[col], errors="coerce")
    if op == ">":
        mask = s > val
    elif op == "<":
        mask = s < val
    elif op == ">=":
        mask = s >= val
    elif op == "<=":
        mask = s <= val
    elif op == "==":
        mask = s == val
    elif op == "!=":
        mask = s != val
    else:
        raise ValueError(f"Unknown operator: {op}")
    return df[mask.fillna(False)].copy()


# -------------------------
# Data structures
# -------------------------

@dataclasses.dataclass
class PackConfig:
    src_root: str
    datasets: List[str]
    dst_root: str
    pack_name: str
    modes: List[str]              # ["binary","gray"]
    split: Tuple[float, float, float]
    seed: int
    link_mode: str                # "symlink|hardlink|copy"
    include_meta: bool
    qc_filter: Optional[str]
    mini: bool
    max_per_dataset: int          # interpreted as max per (dataset, mode)
    resize: int                   # 1280 means no resize (link allowed)
    include_masks_128_160: bool
    include_raw_ld_1600: bool
    include_thr_random: bool
    include_thr_fixed: bool
    thr_fixed_values: Optional[List[int]]  # if None and include_thr_fixed: auto-discover

    # ✅ mask-only
    mask_only: bool
    mask_subdir: str
    pad_each: int
    upsample_factor: int

# -------------------------
# Core
# -------------------------
def _discover_datasets(src_root: str) -> List[str]:
    # list directories directly under src_root
    out = []
    if not os.path.isdir(src_root):
        raise FileNotFoundError(f"src_root not found: {src_root}")
    for name in sorted(os.listdir(src_root)):
        p = os.path.join(src_root, name)
        if os.path.isdir(p):
            out.append(name)
    return out


def _load_processed_index(dataset_path: str) -> pd.DataFrame:
    idx = os.path.join(dataset_path, "interim", "processed", "index.csv")
    if not os.path.exists(idx):
        raise FileNotFoundError(f"index.csv not found: {idx}")
    df = pd.read_csv(idx)
    # expected columns: dataset_id, mode, mask_stem, mask_128, mask_160, mask_1280, ld_1280, meta
    for col in ["mode", "mask_stem", "mask_1280", "ld_1280"]:
        if col not in df.columns:
            raise ValueError(f"index.csv missing required column: {col}")
    return df


def _load_qc(dataset_path: str) -> Optional[pd.DataFrame]:
    qc = os.path.join(dataset_path, "interim", "processed", "qc.csv")
    if not os.path.exists(qc):
        return None
    df = pd.read_csv(qc)
    # expected columns include: mode, mask_stem, ncc_mask1280_ld1280, iou_Txx, dice_Txx...
    if "mode" not in df.columns or "mask_stem" not in df.columns:
        return None
    return df

def _find_file_by_stem_prefix(dir_path: str, stem: str) -> Optional[str]:
    """
    Find a file in dir_path whose filename starts with `stem`.
    Used for locating raw LD in {mode}_ld_1600_raw because index.csv stores only aligned LD.
    """
    if not os.path.isdir(dir_path):
        return None
    for fn in os.listdir(dir_path):
        if fn.startswith(stem):
            p = os.path.join(dir_path, fn)
            if os.path.isfile(p):
                return p
    return None


def _discover_thr_fixed_values(processed_root: str, mode: str, prefix: str = "T") -> List[int]:
    """
    Discover fixed threshold folders like: {mode}_thr_1280_T30, {mode}_thr_1280_T40, ...
    """
    out: List[int] = []
    patt = os.path.join(processed_root, f"{mode}_thr_1280_{prefix}*")
    for d in glob.glob(patt):
        if not os.path.isdir(d):
            continue
        base = os.path.basename(d)
        m = re.search(rf"{re.escape(mode)}_thr_1280_{re.escape(prefix)}(\d+)$", base)
        if m:
            out.append(int(m.group(1)))
    return sorted(set(out))

def _split_assign(n: int, split: Tuple[float, float, float], seed: int) -> List[str]:
    a, b, c = split
    if abs((a + b + c) - 1.0) > 1e-6:
        raise ValueError(f"split must sum to 1.0, got: {split}")
    idxs = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    n_train = int(round(a * n))
    n_val = int(round(b * n))
    # remainder to test
    n_test = n - n_train - n_val
    out = [""] * n
    for k, i in enumerate(idxs):
        if k < n_train:
            out[i] = "train"
        elif k < n_train + n_val:
            out[i] = "val"
        else:
            out[i] = "test"
    return out


def _maybe_resize_or_link(src_abs: str, dst_abs: str, link_mode: str, resize: int) -> Dict[str, str]:
    """
    If resize==1280: link/copy
    else: read, resize to (resize, resize), write PNG
    """
    if resize == 1280:
        op = _link_or_copy(src_abs, dst_abs, link_mode)
        return {"method": op, "src": src_abs, "dst": dst_abs, "resized": "false"}
    # resized: always write new file (not link)
    img = _load_image_gray_u8(src_abs)
    if img.shape[0] != resize or img.shape[1] != resize:
        img = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_AREA)
    _write_png(dst_abs, img)
    return {"method": "resize_write", "src": src_abs, "dst": dst_abs, "resized": "true"}


def _collect_thr_random(processed_root: str, mode: str, mask_stem: str) -> List[str]:
    d = os.path.join(processed_root, f"{mode}_thr_1280_random")
    if not os.path.isdir(d):
        return []
    # filenames: {stem}_r00_XX.png
    patt = os.path.join(d, f"{mask_stem}_r*.png")
    return sorted(glob.glob(patt))


def _collect_thr_fixed(processed_root: str, mode: str, mask_stem: str, t: int, prefix: str = "T") -> Optional[str]:
    # directory naming in your preprocessor: f"{mode}_thr_1280_{prefix}{t}"
    d = os.path.join(processed_root, f"{mode}_thr_1280_{prefix}{t}")
    if not os.path.isdir(d):
        return None
    cand = os.path.join(d, f"{mask_stem}.png")
    if os.path.exists(cand):
        return cand
    # fallback: any extension
    c2 = glob.glob(os.path.join(d, f"{mask_stem}.*"))
    return c2[0] if c2 else None


def build_pack(cfg: PackConfig) -> str:
    pack_root = os.path.join(cfg.dst_root, cfg.pack_name)
    _ensure_dir(pack_root)

    # output dirs
    img_root = _ensure_dir(os.path.join(pack_root, "images"))
    thr_root = _ensure_dir(os.path.join(pack_root, "thr"))
    meta_root = _ensure_dir(os.path.join(pack_root, "meta"))
    splits_root = _ensure_dir(os.path.join(pack_root, "splits"))

    # prepare unified manifest rows (ONE row per physical sample)
    rows: List[Dict[str, str]] = []
    warnings: List[str] = []

    for dname in tqdm(cfg.datasets, desc="Datasets", unit="ds"):
        qc_df = None  # ✅ FIX: avoid UnboundLocalError in mask_only branch
        dataset_path = os.path.join(cfg.src_root, dname)
        # processed_root는 non-mask_only에서만 의미가 있지만,
        # 공통 변수로 두되 mask_only에서는 사용하지 않도록 아래에서 가드 처리함.
        processed_root = os.path.join(dataset_path, "interim", "processed")

        if cfg.mask_only:
            # ✅ scan raw masks
            mask_dir = os.path.join(dataset_path, cfg.mask_subdir)
            if not os.path.isdir(mask_dir):
                warnings.append(f"[missing] {dname} mask_only mask_dir not found: {mask_dir}")
                continue
            mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith((".png",".jpg",".jpeg",".bmp",".tif",".tiff"))])
            if not mask_files:
                warnings.append(f"[empty] {dname} mask_only no masks in: {mask_dir}")
                continue

            # create pseudo table similar to index_df with required columns
            merged = pd.DataFrame({
                "mode": ["binary"] * len(mask_files),
                "mask_stem": [os.path.splitext(f)[0] for f in mask_files],
                "mask_128_path_abs": [os.path.join(mask_dir, f) for f in mask_files],
            })
        else:
            index_df = _load_processed_index(dataset_path)
            qc_df = _load_qc(dataset_path)
            if qc_df is not None:
                merged = pd.merge(index_df, qc_df, on=["mode","mask_stem"], how="left", suffixes=("", "_qc"))
            else:
                merged = index_df.copy()

        # filter modes
        merged = merged[merged["mode"].isin(cfg.modes)].reset_index(drop=True)

        # qc filter
        if cfg.qc_filter:
            try:
                merged = _apply_qc_filter(merged, cfg.qc_filter).reset_index(drop=True)
            except Exception as e:
                raise RuntimeError(f"QC filter failed for dataset {dname}: {e}")

        # mini sampling (max per (dataset, mode) to keep balance)
        if cfg.mini and cfg.max_per_dataset > 0:
            parts = []
            for mm in cfg.modes:
                sub = merged[merged["mode"] == mm].reset_index(drop=True)
                if len(sub) == 0:
                    continue
                rng = random.Random(cfg.seed + (hash(dname) % 100000) + (hash(mm) % 10000))
                idxs = list(range(len(sub)))
                rng.shuffle(idxs)
                idxs = idxs[: min(cfg.max_per_dataset, len(idxs))]
                parts.append(sub.iloc[sorted(idxs)].copy())
            merged = pd.concat(parts, axis=0).reset_index(drop=True) if parts else merged.iloc[0:0].copy()

        it = merged.iterrows()
        # tqdm needs total for speed/ETA
        it = tqdm(it, total=len(merged), desc=f"{dname}", unit="sample", leave=False)
        for _, r in it:
            mode = str(r["mode"])
            stem = str(r["mask_stem"])

            if cfg.mask_only:
                mask128_abs = str(r["mask_128_path_abs"])
                if not os.path.exists(mask128_abs):
                    warnings.append(f"[missing] {dname} mask_only {stem} -> mask_128 not found")
                    continue
            else:
                mask1280_rel = str(r["mask_1280"])
                ld1280_rel = str(r["ld_1280"])
                mask1280_abs = os.path.join(dataset_path, mask1280_rel)
                ld1280_abs = os.path.join(dataset_path, ld1280_rel)
                if not os.path.exists(mask1280_abs) or not os.path.exists(ld1280_abs):
                    warnings.append(f"[missing] {dname} {mode} {stem} -> mask/ld not found")
                    continue

            # output base names
            sample_key = _safe_name(f"{dname}__{mode}__{stem}")

            # output dirs (keep original naming for simplicity; resize can rewrite)
            out_mask128_dir = _ensure_dir(os.path.join(img_root, mode, "mask_128"))
            out_mask160_dir = _ensure_dir(os.path.join(img_root, mode, "mask_160"))
            out_mask1280_dir = _ensure_dir(os.path.join(img_root, mode, "mask_1280"))
            out_ld1280_dir = _ensure_dir(os.path.join(img_root, mode, "ld_1280_aligned"))
            out_ld1600_dir = _ensure_dir(os.path.join(img_root, mode, "ld_1600_raw"))
            out_mask1280 = os.path.join(out_mask1280_dir, f"{sample_key}.png")
            out_ld1280 = os.path.join(out_ld1280_dir, f"{sample_key}.png")

            # ✅ IMPORTANT: out_mask128/out_mask160는 mask_only에서 채운 값을 유지해야 함
            out_mask128: str = ""
            out_mask160: str = ""
 

            if cfg.mask_only:
                # ✅ materialize mask_128, mask_160, mask_1280 from raw mask_128
                out_mask128 = os.path.join(out_mask128_dir, f"{sample_key}.png")
                if not os.path.exists(out_mask128):
                    _maybe_resize_or_link(
                        mask128_abs,
                        out_mask128,
                        cfg.link_mode,
                        1280 if cfg.resize == 1280 else cfg.resize,
                    )
                # build 160/1280 (always write, because derived)
                m128 = _load_image_gray_u8(mask128_abs)
                # pad to 160
                m160 = cv2.copyMakeBorder(m128, cfg.pad_each, cfg.pad_each, cfg.pad_each, cfg.pad_each,
                                          borderType=cv2.BORDER_CONSTANT, value=0)
                out_mask160 = os.path.join(out_mask160_dir, f"{sample_key}.png")
                if not os.path.exists(out_mask160):
                    _write_png(out_mask160, m160)
                # upsample to 1280
                m1280 = cv2.resize(m160, (m160.shape[1]*cfg.upsample_factor, m160.shape[0]*cfg.upsample_factor),
                                   interpolation=cv2.INTER_NEAREST)
                if not os.path.exists(out_mask1280):
                    _write_png(out_mask1280, m1280)

                # no LD in mask_only
                out_ld1280 = ""
            else:
                # existing behavior: link/copy canonical mask1280 & ld1280
                if not os.path.exists(out_mask1280):
                    _maybe_resize_or_link(mask1280_abs, out_mask1280, cfg.link_mode, cfg.resize)
                if not os.path.exists(out_ld1280):
                    _maybe_resize_or_link(ld1280_abs, out_ld1280, cfg.link_mode, cfg.resize)

            # Optional: include mask_128 / mask_160 if present in index.csv (non-mask_only only)
            # ✅ BUGFIX: mask_only에서 out_mask128/out_mask160를 만들었는데,
            # 아래에서 다시 ""로 초기화해 manifest가 비어버리던 문제를 제거함.
            if (not cfg.mask_only) and cfg.include_masks_128_160:
                if "mask_128" in r and isinstance(r["mask_128"], str) and r["mask_128"].strip():
                    mask128_abs = os.path.join(dataset_path, str(r["mask_128"]))
                    if os.path.exists(mask128_abs):
                        out_mask128 = os.path.join(out_mask128_dir, f"{sample_key}.png")
                        if not os.path.exists(out_mask128):
                            _maybe_resize_or_link(mask128_abs, out_mask128, cfg.link_mode, cfg.resize)
                if "mask_160" in r and isinstance(r["mask_160"], str) and r["mask_160"].strip():
                    mask160_abs = os.path.join(dataset_path, str(r["mask_160"]))
                    if os.path.exists(mask160_abs):
                        out_mask160 = os.path.join(out_mask160_dir, f"{sample_key}.png")
                        if not os.path.exists(out_mask160):
                            _maybe_resize_or_link(mask160_abs, out_mask160, cfg.link_mode, cfg.resize)

            # Optional: include raw LD 1600 (stem-prefix match in processed/{mode}_ld_1600_raw)
            out_ld1600 = ""
            if (not cfg.mask_only) and cfg.include_raw_ld_1600:
                raw_dir = os.path.join(processed_root, f"{mode}_ld_1600_raw")
                raw_abs = _find_file_by_stem_prefix(raw_dir, stem)
                if raw_abs and os.path.exists(raw_abs):
                    # keep original raw filename-ish but normalize to sample_key
                    out_ld1600 = os.path.join(out_ld1600_dir, f"{sample_key}.png")
                    if not os.path.exists(out_ld1600):
                        _maybe_resize_or_link(raw_abs, out_ld1600, cfg.link_mode, cfg.resize)
 
            # meta (optional)
            out_meta = ""
            if (not cfg.mask_only) and cfg.include_meta and "meta" in r and isinstance(r["meta"], str) and r["meta"].strip():
                meta_abs = os.path.join(dataset_path, str(r["meta"]))
                if os.path.exists(meta_abs):
                    out_meta = os.path.join(meta_root, f"{sample_key}.json")
                    if not os.path.exists(out_meta):
                        _link_or_copy(meta_abs, out_meta, cfg.link_mode)

            # -------- thresholds: materialize ALL that exist (random + fixed) --------
            thr_random_dir_rel = ""
            thr_random_count = 0
            if (not cfg.mask_only) and cfg.include_thr_random:
                thr_list = _collect_thr_random(processed_root, mode, stem)
                if thr_list:
                    out_thr_dir = _ensure_dir(os.path.join(thr_root, mode, "random"))
                    for thr_abs in thr_list:
                        bn = os.path.basename(thr_abs)
                        m = re.search(r"_r(\d+)_([0-9]+)\.png$", bn)
                        thr_value = m.group(2) if m else ""
                        rid = m.group(1) if m else "00"
                        thr_id = f"{sample_key}__R{rid}__{thr_value}" if thr_value else f"{sample_key}__R{rid}"
                        out_thr = os.path.join(out_thr_dir, f"{thr_id}.png")
                        if not os.path.exists(out_thr):
                            _maybe_resize_or_link(thr_abs, out_thr, cfg.link_mode, cfg.resize)
                        thr_random_count += 1
                    thr_random_dir_rel = _relpath(out_thr_dir, pack_root)

            thr_fixed_map: Dict[str, str] = {}
            if (not cfg.mask_only) and cfg.include_thr_fixed:
                values = cfg.thr_fixed_values
                if values is None:
                    values = _discover_thr_fixed_values(processed_root, mode, prefix="T")
                for thr_t in values:
                    thr_abs = _collect_thr_fixed(processed_root, mode, stem, int(thr_t), prefix="T")
                    if not thr_abs:
                        continue
                    out_thr_dir = _ensure_dir(os.path.join(thr_root, mode, f"fixed_T{int(thr_t)}"))
                    out_thr = os.path.join(out_thr_dir, f"{sample_key}__T{int(thr_t)}.png")
                    if not os.path.exists(out_thr):
                        _maybe_resize_or_link(thr_abs, out_thr, cfg.link_mode, cfg.resize)
                    thr_fixed_map[str(int(thr_t))] = _relpath(out_thr, pack_root)

            # -------- QC columns (if present) --------
            qc_cols: Dict[str, str] = {}
            for c in merged.columns:
                if c in ("dataset_id", "dataset", "mode", "mask_stem", "mask_128", "mask_160", "mask_1280", "ld_1280", "meta"):
                    continue
                # keep numeric / qc-like columns; store as string for csv safety
                if c.startswith(("ncc", "iou", "dice")) or c.endswith(("_qc",)):
                    v = r.get(c, "")
                    qc_cols[f"qc__{c}"] = "" if pd.isna(v) else str(v)

            # unified availability flags
            has_fwd = int(bool(out_mask1280) and bool(out_ld1280)) if not cfg.mask_only else 0
            has_inv_random = int(thr_random_count > 0)
            has_inv_fixed = int(len(thr_fixed_map) > 0)

            # -------- ONE manifest row per physical sample --------
            row_out: Dict[str, str] = {
                "sample_key": sample_key,
                "dataset": dname,
                "mode": mode,
                "mask_stem": stem,
                "mask_128_path": _relpath(out_mask128, pack_root) if out_mask128 else "",
                "mask_160_path": _relpath(out_mask160, pack_root) if out_mask160 else "",
                "mask_1280_path": _relpath(out_mask1280, pack_root),
                "ld_1280_aligned_path": _relpath(out_ld1280, pack_root) if out_ld1280 else "",
                "ld_1600_raw_path": _relpath(out_ld1600, pack_root) if out_ld1600 else "",
                "thr_random_dir": thr_random_dir_rel,
                "thr_random_count": str(thr_random_count),
                "thr_fixed_map_json": json.dumps(thr_fixed_map, ensure_ascii=False),
                "meta_path": _relpath(out_meta, pack_root) if out_meta else "",
                "has_fwd": str(has_fwd),
                "has_inv_random": str(has_inv_random),
                "has_inv_fixed": str(has_inv_fixed),
            }
            row_out.update(qc_cols)
            rows.append(row_out)
 

    if not rows:
        raise RuntimeError("No samples collected. Check src_root/datasets/modes/tasks/qc_filter.")

    # assign splits (global shuffle, ONCE per sample)
    split_tags = _split_assign(len(rows), cfg.split, cfg.seed)
    for rr, sp in zip(rows, split_tags):
        rr["split"] = sp

    # write manifest.csv
    manifest_path = os.path.join(pack_root, "manifest.csv")
    fieldnames = list(rows[0].keys())
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # write split files
    by_split: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    for r in rows:
        by_split[r["split"]].append(r["sample_key"])

    for sp, ids in by_split.items():
        with open(os.path.join(splits_root, f"{sp}.txt"), "w", encoding="utf-8") as f:
            for sid in ids:
                f.write(sid + "\n")

    # dataset_card.json
    card = {
        "pack_name": cfg.pack_name,
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "src_root": os.path.abspath(cfg.src_root),
        "datasets": cfg.datasets,
        "modes": cfg.modes,
        "split": {"train": cfg.split[0], "val": cfg.split[1], "test": cfg.split[2]},
        "seed": cfg.seed,
        "link_mode": cfg.link_mode,
        "include_meta": cfg.include_meta,
        "qc_filter": cfg.qc_filter or "",
        "mini": cfg.mini,
        "max_per_dataset_per_mode": cfg.max_per_dataset if cfg.mini else 0,
        "resize": cfg.resize,
        "include_masks_128_160": cfg.include_masks_128_160,
        "include_raw_ld_1600": cfg.include_raw_ld_1600,
        "include_thr_random": cfg.include_thr_random,
        "include_thr_fixed": cfg.include_thr_fixed,
        "thr_fixed_values": cfg.thr_fixed_values if cfg.include_thr_fixed else [],
        "num_samples": len(rows),
        "warnings_count": len(warnings),
    }
    # try to include registration_params.json hashes per dataset (traceability)
    reg_hashes = {}
    for dname in cfg.datasets:
        rp = os.path.join(cfg.src_root, dname, "interim", "processed", "registration_params.json")
        if os.path.exists(rp):
            try:
                reg_hashes[dname] = {"path": _relpath(rp, cfg.src_root), "sha1": _sha1_of_file(rp)}
            except Exception:
                pass
    if reg_hashes:
        card["registration_params"] = reg_hashes

    with open(os.path.join(pack_root, "dataset_card.json"), "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)

    # warnings
    if warnings:
        with open(os.path.join(pack_root, "warnings.txt"), "w", encoding="utf-8") as f:
            for w in warnings:
                f.write(w + "\n")

    return pack_root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract TrainPack/MiniPack from DLP pipeline datasets (interim/processed)."
    )
    p.add_argument("--src_root", required=True, help="Root containing datasets (e.g., ~/Desktop/26-1_UROP/raw_datasets)")
    p.add_argument("--datasets", default="all",
                   help="Comma-separated dataset names, or 'all' to use every directory under src_root")
    p.add_argument("--dst_root", required=True, help="Output root folder for packs")
    p.add_argument("--pack_name", default="", help="Output pack folder name (default: TrainPack_<timestamp>)")

    p.add_argument("--modes", default="both", choices=["binary", "gray", "both"],
                   help="Which modes to include")

    p.add_argument("--split", default="0.9,0.05,0.05",
                   help="Train/Val/Test split ratios, comma-separated, sum to 1.0")
    p.add_argument("--seed", type=int, default=1234)

    p.add_argument("--link_mode", default="symlink", choices=["symlink", "hardlink", "copy"],
                   help="How to materialize files in pack. symlink saves disk space.")
    p.add_argument("--include_meta", action="store_true", help="Also include per-sample meta JSON if available")

    p.add_argument("--qc_filter", default="",
                   help="Optional QC filter expression, e.g. 'ncc_mask1280_ld1280>0.2'")

    p.add_argument("--mini", action="store_true", help="Enable MiniPack sampling")
    p.add_argument("--max_per_dataset", type=int, default=200,
                   help="If --mini: max samples per (dataset, mode) AFTER qc filter (balanced)")
    p.add_argument("--resize", type=int, default=1280,
                   help="If !=1280, outputs are resized square PNG (requires cv2). Example: 512")

    # include toggles (default: include everything that exists in processed/)
    p.add_argument("--no_masks_128_160", action="store_true", help="Do not include mask_128/mask_160 in pack")
    p.add_argument("--no_raw_ld_1600", action="store_true", help="Do not include ld_1600_raw in pack")
    p.add_argument("--no_thr_random", action="store_true", help="Do not include thr random images in pack")
    p.add_argument("--thr_fixed", default="auto",
                   help="Include fixed thresholds if present. 'auto' to discover, or comma list like '30,40', or 'none'")
 
    # ✅ NEW: mask-only pack
    p.add_argument("--mask_only", action="store_true",
                   help="Build pack from raw binary masks only (no LD/index.csv required).")
    p.add_argument("--mask_subdir", default="mask_input",
                   help="If --mask_only: subdir under each dataset that contains binary masks (e.g., mask_input).")
    return p.parse_args()


def main():
    args = parse_args()

    src_root = os.path.expanduser(args.src_root)
    dst_root = os.path.expanduser(args.dst_root)
    _ensure_dir(dst_root)

    if args.datasets.strip().lower() == "all":
        datasets = _discover_datasets(src_root)
    else:
        datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]

    if not datasets:
        raise RuntimeError("No datasets selected.")

    pack_name = args.pack_name.strip() or f"TrainPack_{_now_stamp()}"

    if args.modes == "both":
        modes = ["binary", "gray"]
    else:
        modes = [args.modes]

    split = tuple(float(x.strip()) for x in args.split.split(","))
    if len(split) != 3:
        raise ValueError("--split must be like '0.9,0.05,0.05'")

    qc_filter = args.qc_filter.strip() or None

    if args.resize != 1280:
        if cv2 is None:
            raise RuntimeError("Resize requested but cv2 not available. Install opencv-python.")

    # fixed threshold policy
    thr_fixed_values: Optional[List[int]] = None
    include_thr_fixed = True
    thr_fixed_arg = (args.thr_fixed or "auto").strip().lower()
    if thr_fixed_arg == "none":
        include_thr_fixed = False
        thr_fixed_values = []
    elif thr_fixed_arg == "auto":
        thr_fixed_values = None  # discover per dataset/mode
    else:
        # comma list
        thr_fixed_values = [int(x.strip()) for x in thr_fixed_arg.split(",") if x.strip()]
        include_thr_fixed = True

    cfg = PackConfig(
        src_root=src_root,
        datasets=datasets,
        dst_root=dst_root,
        pack_name=pack_name,
        modes=modes,
        split=(split[0], split[1], split[2]),
        seed=int(args.seed),
        link_mode=args.link_mode,
        include_meta=bool(args.include_meta),
        qc_filter=qc_filter,
        mini=bool(args.mini),
        max_per_dataset=int(args.max_per_dataset),
        resize=int(args.resize),
        include_masks_128_160=not bool(args.no_masks_128_160),
        include_raw_ld_1600=not bool(args.no_raw_ld_1600),
        include_thr_random=not bool(args.no_thr_random),
        include_thr_fixed=bool(include_thr_fixed),
        thr_fixed_values=thr_fixed_values,

        mask_only=bool(args.mask_only),
        mask_subdir=str(args.mask_subdir),
        pad_each=16,
        upsample_factor=8,
    )

    out = build_pack(cfg)
    print("\n========================================")
    print("✅ Pack created")
    print(f" - path: {out}")
    print(" - files: manifest.csv, dataset_card.json, splits/*.txt")
    print(" - unified manifest: ONE row per sample_key (use has_fwd/has_inv_* in dataloader)")
    print("========================================\n")


if __name__ == "__main__":
    main()

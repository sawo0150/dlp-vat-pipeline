# pipeline/src/dlp_pipeline/preprocessor.py

import os
import json
import math
import shutil
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from dlp_pipeline.utils import ensure_dir

log = logging.getLogger(__name__)


# -----------------------------
# Small utilities
# -----------------------------
def _interp_from_str(s: str) -> int:
    s = str(s).lower()
    if s in ("nearest", "nn"):
        return cv2.INTER_NEAREST
    if s in ("linear", "bilinear"):
        return cv2.INTER_LINEAR
    if s in ("area",):
        return cv2.INTER_AREA
    if s in ("cubic", "bicubic"):
        return cv2.INTER_CUBIC
    return cv2.INTER_LINEAR


def _safe_link_or_copy(src: str, dst: str, mode: str = "copy") -> str:
    ensure_dir(os.path.dirname(dst))
    if os.path.abspath(src) == os.path.abspath(dst):
        return "same"
    if os.path.exists(dst):
        os.remove(dst)
    try:
        if mode == "hardlink":
            os.link(src, dst)
            return "hardlink"
        if mode == "symlink":
            os.symlink(src, dst)
            return "symlink"
        shutil.copy2(src, dst)
        return "copy"
    except Exception as e:
        log.warning(f"[fallback copy] {e} | {src} -> {dst}")
        shutil.copy2(src, dst)
        return "copy"


def _imread_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def _write_png(path: str, img: np.ndarray):
    ensure_dir(os.path.dirname(path))
    ok = cv2.imwrite(path, img)
    if not ok:
        raise IOError(f"Failed to write image: {path}")


def _to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def _binarize(img_u8: np.ndarray, thr: int) -> np.ndarray:
    _, out = cv2.threshold(img_u8, int(thr), 255, cv2.THRESH_BINARY)
    return out


def _iou(a_bin: np.ndarray, b_bin: np.ndarray) -> float:
    a = (a_bin > 0)
    b = (b_bin > 0)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union + 1e-9)


def _dice(a_bin: np.ndarray, b_bin: np.ndarray) -> float:
    a = (a_bin > 0)
    b = (b_bin > 0)
    inter = np.logical_and(a, b).sum()
    s = a.sum() + b.sum()
    return float(2.0 * inter) / float(s + 1e-9)


def _ncc(a_u8: np.ndarray, b_u8: np.ndarray) -> float:
    # normalized cross correlation (Pearson)
    a = a_u8.astype(np.float32).ravel()
    b = b_u8.astype(np.float32).ravel()
    a -= a.mean()
    b -= b.mean()
    denom = (a.std() * b.std()) + 1e-9
    return float((a * b).mean() / denom)


# -----------------------------
# Core transforms (S4 + S3 style)
# -----------------------------
def make_mask_160(mask_128: np.ndarray, pad_each: int = 16) -> np.ndarray:
    # constant 0 padding
    return cv2.copyMakeBorder(mask_128, pad_each, pad_each, pad_each, pad_each,
                              borderType=cv2.BORDER_CONSTANT, value=0)


def make_mask_1280_from_160(mask_160: np.ndarray, upsample_factor: int = 8, interp=cv2.INTER_NEAREST) -> np.ndarray:
    # 160 * 8 = 1280
    return cv2.resize(mask_160, (mask_160.shape[1] * upsample_factor, mask_160.shape[0] * upsample_factor),
                      interpolation=interp)


def align_ld_to_1280(
    ld_raw: np.ndarray,
    angle: float,
    scale: float,
    tx: float,
    ty: float,
    transpose: bool = True,
    crop_size: int = 1280,
    warp_interp=cv2.INTER_LINEAR,
    border_value: int = 0
) -> np.ndarray:
    # MATLAB logic: transpose first
    if transpose:
        ld_raw = cv2.transpose(ld_raw)

    h, w = ld_raw.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, float(angle), float(scale))
    M[0, 2] += float(tx)
    M[1, 2] += float(ty)

    warped = cv2.warpAffine(ld_raw, M, (w, h), flags=warp_interp, borderValue=border_value)

    start_x = (w - crop_size) // 2
    start_y = (h - crop_size) // 2

    # robust crop (pad if needed)
    if start_x < 0 or start_y < 0:
        pad_w = max(0, -start_x)
        pad_h = max(0, -start_y)
        warped = cv2.copyMakeBorder(warped, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=border_value)
        start_x += pad_w
        start_y += pad_h

    cropped = warped[start_y:start_y + crop_size, start_x:start_x + crop_size]
    return cropped


# -----------------------------
# Preprocessor
# -----------------------------
@dataclass
class SampleResult:
    dataset_id: str
    mode: str
    mask_name: str
    ld_src_name: str
    mask_128: str
    mask_160: str
    mask_1280: str
    ld_1280: str
    thr_outputs: Dict[str, str]
    meta_path: str


class Preprocessor:
    """
    Input: pairing/pairs.csv
    Output: interim/processed/{...}
    """

    def __init__(self, cfg, ds):
        self.cfg = cfg
        self.ds = ds

        self.dataset_id = os.path.basename(ds.path)
        self.dataset_path = ds.path

        self.processed_root = ds.dirs["processed"]
        self.meta_root = ds.dirs.get("processed_meta", ensure_dir(os.path.join(self.processed_root, "meta")))

        # load preprocess config block
        self.pcfg = cfg.preprocess

        # debug controls
        self.debug_enable = bool(getattr(cfg.debug, "enable", False))
        self.debug_every = int(getattr(cfg.debug, "sample_every", 50))
        self.debug_max = int(getattr(cfg.debug, "max_images", 200))
        self.debug_dir = ds.dirs.get("debug", ensure_dir(os.path.join(ds.path, "interim", "debug")))

        # outputs (strict to your desired structure)
        self.out_dirs = self._build_output_dirs()

        # pairs csv
        self.pairs_csv = ds.dirs.get("pairs_csv", os.path.join(ds.path, "pairing", "pairs.csv"))

        # threshold config
        self.thr_cfg = getattr(self.pcfg, "threshold", None)

        # registration config
        self.reg_cfg = getattr(self.pcfg, "registration", None)

        # deterministic rng for threshold random
        seed = int(getattr(getattr(self.pcfg, "threshold", {}), "random", {}).get("seed", getattr(cfg, "seed", 1234))) \
            if isinstance(getattr(self.pcfg, "threshold", None), (dict,)) else int(getattr(cfg, "seed", 1234))
        self.rng = np.random.default_rng(seed)

    def _build_output_dirs(self) -> Dict[str, str]:
        # You specified exact layout; we create them here.
        od = {}
        # binary
        od["binary_mask_128"] = ensure_dir(os.path.join(self.processed_root, "binary_mask_128"))
        od["binary_mask_160"] = ensure_dir(os.path.join(self.processed_root, "binary_mask_160"))
        od["binary_mask_1280"] = ensure_dir(os.path.join(self.processed_root, "binary_mask_1280"))
        od["binary_ld_1600_raw"] = ensure_dir(os.path.join(self.processed_root, "binary_ld_1600_raw"))
        od["binary_ld_1280_aligned"] = ensure_dir(os.path.join(self.processed_root, "binary_ld_1280_aligned"))
        # gray
        od["gray_mask_128"] = ensure_dir(os.path.join(self.processed_root, "gray_mask_128"))
        od["gray_mask_160"] = ensure_dir(os.path.join(self.processed_root, "gray_mask_160"))
        od["gray_mask_1280"] = ensure_dir(os.path.join(self.processed_root, "gray_mask_1280"))
        od["gray_ld_1600_raw"] = ensure_dir(os.path.join(self.processed_root, "gray_ld_1600_raw"))
        od["gray_ld_1280_aligned"] = ensure_dir(os.path.join(self.processed_root, "gray_ld_1280_aligned"))
        # threshold roots (created on demand)
        # meta / index / qc
        od["meta"] = ensure_dir(os.path.join(self.processed_root, "meta"))
        return od

    def _load_pairs(self) -> pd.DataFrame:
        if not os.path.exists(self.pairs_csv):
            raise FileNotFoundError(f"pairs.csv not found: {self.pairs_csv}")
        df = pd.read_csv(self.pairs_csv)
        df = df[(df["status"] == "OK") & (df["mode"].isin(["binary", "gray"]))].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError("No OK pairs found.")
        return df

    def _load_registration_params(self) -> Dict[str, Dict[str, float]]:
        """
        returns:
          {"binary": {"angle":..,"scale":..,"tx":..,"ty":..},
           "gray":   {...}}
        """
        reg = self.reg_cfg
        if reg is None:
            raise ValueError("cfg.preprocess.registration is missing")

        source = str(getattr(reg, "source", "yaml")).lower()

        if source == "file":
            pattern = str(getattr(reg, "file_pattern", "interim/processed/registration_params_{mode}.json"))
            out = {}
            for mode in ["binary", "gray"]:
                fpath = pattern.format(mode=mode)
                if not os.path.isabs(fpath):
                    fpath = os.path.join(self.dataset_path, fpath)
                if not os.path.exists(fpath):
                    raise FileNotFoundError(f"registration file not found for {mode}: {fpath}")
                with open(fpath, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                # accept both your manual tool format and a plain dict format
                if "params" in obj:
                    p = obj["params"]
                    # manual tool keys
                    if "rotation_degree" in p:
                        out[mode] = {
                            "angle": float(p["rotation_degree"]),
                            "scale": float(p["scale"]),
                            "tx": float(p["translation_x"]),
                            "ty": float(p["translation_y"]),
                        }
                    else:
                        out[mode] = {k: float(v) for k, v in p.items()}
                else:
                    out[mode] = {k: float(v) for k, v in obj.items()}
            return out

        # default: yaml
        params = getattr(reg, "params", None)
        if params is None:
            raise ValueError("registration.source=yaml but registration.params missing")
        out = {}
        for mode in ["binary", "gray"]:
            mp = getattr(params, mode, None)
            if mp is None:
                raise ValueError(f"registration.params.{mode} missing")
            out[mode] = {
                "angle": float(getattr(mp, "angle")),
                "scale": float(getattr(mp, "scale")),
                "tx": float(getattr(mp, "tx")),
                "ty": float(getattr(mp, "ty")),
            }
        return out

    def _debug_should_save(self, k: int, saved: int) -> bool:
        if not self.debug_enable:
            return False
        if saved >= self.debug_max:
            return False
        return (k % self.debug_every) == 0

    def run(self):
        df = self._load_pairs()
        reg_params = self._load_registration_params()

        # write final registration_params.json for traceability
        reg_out = {
            "dataset_id": self.dataset_id,
            "created_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            "source": str(getattr(self.reg_cfg, "source", "yaml")),
            "params": reg_params,
        }
        reg_path = os.path.join(self.processed_root, "registration_params.json")
        with open(reg_path, "w", encoding="utf-8") as f:
            json.dump(reg_out, f, ensure_ascii=False, indent=2)
        log.info(f"[Preprocess] saved: {reg_path}")

        # config values
        mask_cfg = getattr(self.pcfg, "mask", None)
        ld_cfg = getattr(self.pcfg, "ld", None)
        out_cfg = getattr(self.pcfg, "output", None)
        qc_cfg = getattr(self.pcfg, "qc", None)

        pad_each = int(getattr(mask_cfg, "pad_each", 16))
        up_factor = int(getattr(mask_cfg, "upsample_factor", 8))
        mask_interp = _interp_from_str(getattr(mask_cfg, "interp", "nearest"))

        transpose = bool(getattr(ld_cfg, "transpose", True))
        crop_size = int(getattr(ld_cfg, "crop_size", 1280))
        warp_interp = _interp_from_str(getattr(ld_cfg, "warp_interp", "linear"))
        border_value = int(getattr(ld_cfg, "border_value", 0))

        save_raw_ld = bool(getattr(out_cfg, "save_raw_ld_copy", False))
        raw_ld_mode = str(getattr(out_cfg, "raw_ld_copy_mode", "symlink"))
        image_ext = str(getattr(out_cfg, "image_ext", ".png"))

        thr_enable = bool(getattr(getattr(self.pcfg, "threshold", None), "enable", False))
        thr_fixed_enable = bool(getattr(getattr(getattr(self.pcfg, "threshold", None), "fixed", None), "enable", False))
        
        # [FIX] getattr(..., "values")는 DictConfig의 .values() 메서드를 반환하므로 .get() 사용
        _thr_root = getattr(self.pcfg, "threshold", {}) or {}
        _fixed_node = getattr(_thr_root, "fixed", {}) or {}
        thr_values = list(_fixed_node.get("values", [])) if thr_enable else []

        thr_prefix = str(getattr(getattr(getattr(self.pcfg, "threshold", None), "fixed", None), "prefix", "T"))
 
        thr_rand_enable = bool(getattr(getattr(getattr(self.pcfg, "threshold", None), "random", None), "enable", False))
        thr_rand_num = int(getattr(getattr(getattr(self.pcfg, "threshold", None), "random", None), "num", 0))
        thr_rand_low = int(getattr(getattr(getattr(self.pcfg, "threshold", None), "random", None), "low", 10))
        thr_rand_high = int(getattr(getattr(getattr(self.pcfg, "threshold", None), "random", None), "high", 80))

        qc_enable = bool(getattr(qc_cfg, "enable", True))
        qc_metrics = list(getattr(qc_cfg, "metrics", ["iou", "dice", "ncc"]))

        results: List[SampleResult] = []
        index_rows: List[dict] = []
        qc_rows: List[dict] = []

        debug_saved = 0

        for k in tqdm(range(len(df)), desc="Preprocess"):
            row = df.iloc[k]
            mode = str(row["mode"])
            mask_name = str(row["mask_name"])
            mask_stem = os.path.splitext(os.path.basename(mask_name))[0]

            # input paths (pairing outputs)
            pairing_root = os.path.join(self.dataset_path, "pairing")
            if mode == "binary":
                in_mask = os.path.join(pairing_root, "binary_mask_128", mask_name)
                in_ld = self._find_ld_by_stem(os.path.join(pairing_root, "binary_rawLD_1600"), mask_stem, fallback=row.get("src_ld_file", ""))
            else:
                in_mask = os.path.join(pairing_root, "gray_mask_128", mask_name)
                in_ld = self._find_ld_by_stem(os.path.join(pairing_root, "gray_rawLD_1600"), mask_stem, fallback=row.get("src_ld_file", ""))

            # outputs
            out_mask_128 = os.path.join(self.out_dirs[f"{mode}_mask_128"], f"{mask_stem}{image_ext}")
            out_mask_160 = os.path.join(self.out_dirs[f"{mode}_mask_160"], f"{mask_stem}{image_ext}")
            out_mask_1280 = os.path.join(self.out_dirs[f"{mode}_mask_1280"], f"{mask_stem}{image_ext}")
            out_ld_1280 = os.path.join(self.out_dirs[f"{mode}_ld_1280_aligned"], f"{mask_stem}{image_ext}")

            # optional raw ld copy
            out_ld_raw = os.path.join(self.out_dirs[f"{mode}_ld_1600_raw"], os.path.basename(in_ld))

            # read
            mask_128 = _imread_gray(in_mask)
            ld_1600 = _imread_gray(in_ld)

            # mask derivations
            mask_160 = make_mask_160(mask_128, pad_each=pad_each)
            mask_1280 = make_mask_1280_from_160(mask_160, upsample_factor=up_factor, interp=mask_interp)

            # ld align
            rp = reg_params[mode]
            ld_1280 = align_ld_to_1280(
                ld_1600,
                angle=rp["angle"], scale=rp["scale"], tx=rp["tx"], ty=rp["ty"],
                transpose=transpose, crop_size=crop_size,
                warp_interp=warp_interp, border_value=border_value
            )

            # write outputs
            _write_png(out_mask_128, _to_uint8(mask_128))
            _write_png(out_mask_160, _to_uint8(mask_160))
            _write_png(out_mask_1280, _to_uint8(mask_1280))
            _write_png(out_ld_1280, _to_uint8(ld_1280))
            if save_raw_ld:
                _safe_link_or_copy(in_ld, out_ld_raw, mode=raw_ld_mode)

            # threshold derivations
            thr_outputs: Dict[str, str] = {}
            if thr_enable:
                if thr_fixed_enable:
                    for t in thr_values:
                        t = int(t)
                        thr_dir = ensure_dir(os.path.join(self.processed_root, f"{mode}_thr_1280_{thr_prefix}{t}"))
                        out_thr = os.path.join(thr_dir, f"{mask_stem}{image_ext}")
                        thr_img = _binarize(_to_uint8(ld_1280), t)
                        _write_png(out_thr, thr_img)
                        thr_outputs[f"{thr_prefix}{t}"] = out_thr

                if thr_rand_enable and thr_rand_num > 0:
                    thr_dir = ensure_dir(os.path.join(self.processed_root, f"{mode}_thr_1280_random"))
                    # draw random thresholds
                    ts = self.rng.integers(low=thr_rand_low, high=thr_rand_high + 1, size=thr_rand_num)
                    for i, t in enumerate(ts.tolist()):
                        out_thr = os.path.join(thr_dir, f"{mask_stem}_r{i:02d}_{t}{image_ext}")
                        thr_img = _binarize(_to_uint8(ld_1280), int(t))
                        _write_png(out_thr, thr_img)
                        thr_outputs[f"R{i:02d}_{t}"] = out_thr

            # meta json per sample
            meta_obj = {
                "dataset_id": self.dataset_id,
                "mode": mode,
                "mask_stem": mask_stem,
                "inputs": {
                    "pair_mask_128": in_mask,
                    "pair_ld_1600": in_ld,
                },
                "outputs": {
                    "mask_128": out_mask_128,
                    "mask_160": out_mask_160,
                    "mask_1280": out_mask_1280,
                    "ld_1280_aligned": out_ld_1280,
                    "ld_1600_raw_copy": out_ld_raw if save_raw_ld else None,
                    "thresholds": thr_outputs,
                },
                "registration_params": reg_params[mode],
                "pair_row": {c: (row[c].item() if hasattr(row[c], "item") else row[c]) for c in df.columns},
            }
            meta_path = os.path.join(self.out_dirs["meta"], f"{mask_stem}.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_obj, f, ensure_ascii=False, indent=2)

            # qc metrics
            if qc_enable:
                qc_row = {
                    "dataset_id": self.dataset_id,
                    "mode": mode,
                    "mask_stem": mask_stem,
                }
                # always compute NCC between mask_1280 and aligned ld (as intensity)
                if "ncc" in qc_metrics:
                    qc_row["ncc_mask1280_ld1280"] = _ncc(_to_uint8(mask_1280), _to_uint8(ld_1280))

                # IoU/Dice against fixed threshold(s)
                if thr_enable and thr_fixed_enable and len(thr_values) > 0:
                    for t in thr_values:
                        t = int(t)
                        ld_bin = _binarize(_to_uint8(ld_1280), t)
                        if "iou" in qc_metrics:
                            qc_row[f"iou_T{t}"] = _iou(_to_uint8(mask_1280), ld_bin)
                        if "dice" in qc_metrics:
                            qc_row[f"dice_T{t}"] = _dice(_to_uint8(mask_1280), ld_bin)

                qc_rows.append(qc_row)

            # index row (global)
            index_rows.append({
                "dataset_id": self.dataset_id,
                "mode": mode,
                "mask_stem": mask_stem,
                "mask_128": os.path.relpath(out_mask_128, self.dataset_path),
                "mask_160": os.path.relpath(out_mask_160, self.dataset_path),
                "mask_1280": os.path.relpath(out_mask_1280, self.dataset_path),
                "ld_1280": os.path.relpath(out_ld_1280, self.dataset_path),
                "meta": os.path.relpath(meta_path, self.dataset_path),
            })

            # debug samples
            if self._debug_should_save(k, debug_saved):
                dbg = ensure_dir(os.path.join(self.debug_dir, "preprocess"))
                _write_png(os.path.join(dbg, f"{mode}_{mask_stem}_mask1280.png"), _to_uint8(mask_1280))
                _write_png(os.path.join(dbg, f"{mode}_{mask_stem}_ld1280.png"), _to_uint8(ld_1280))
                debug_saved += 2

        # write index.csv and qc.csv
        index_df = pd.DataFrame(index_rows).sort_values(["mode", "mask_stem"], kind="stable").reset_index(drop=True)
        index_path = os.path.join(self.processed_root, "index.csv")
        index_df.to_csv(index_path, index=False)
        log.info(f"[Preprocess] saved: {index_path} ({len(index_df)})")

        if qc_enable and len(qc_rows) > 0:
            qc_df = pd.DataFrame(qc_rows).sort_values(["mode", "mask_stem"], kind="stable").reset_index(drop=True)
            qc_path = os.path.join(self.processed_root, "qc.csv")
            qc_df.to_csv(qc_path, index=False)
            log.info(f"[Preprocess] saved: {qc_path} ({len(qc_df)})")

        log.info("[Preprocess] Done.")

    def _find_ld_by_stem(self, ld_dir: str, stem: str, fallback: str = "") -> str:
        # pairing task already tried to name LD as stem.* in many cases.
        if os.path.isdir(ld_dir):
            for f in os.listdir(ld_dir):
                if f.startswith(stem):
                    return os.path.join(ld_dir, f)
        # fallback to csv column
        if fallback:
            cand = os.path.join(ld_dir, str(fallback))
            if os.path.exists(cand):
                return cand
        raise FileNotFoundError(f"LD file not found for stem={stem} in {ld_dir}")

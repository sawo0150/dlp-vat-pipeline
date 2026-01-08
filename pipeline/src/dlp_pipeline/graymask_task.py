# pipeline/src/dlp_pipeline/graymask_task.py
from __future__ import annotations
import os
import json
import logging
import glob
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf

from dlp_pipeline.utils import save_image, should_dump_debug
from dlp_pipeline.graymask_synth import GrayMaskSynthesizer
from dlp_pipeline.projector_interface import ProjectorWindow

log = logging.getLogger(__name__)


class GrayMaskTask:
    def __init__(self, cfg, ds_manager):
        self.cfg = cfg
        self.ds = ds_manager
        self.synth = GrayMaskSynthesizer(cfg, base_seed=int(getattr(cfg, "seed", 0) or 0))
        self.proj = ProjectorWindow(cfg)

    def run(self):
        # 이 task는 기존 dataset에 들어가서 작업하는 게 자연스러움
        if not self.cfg.dataset.load_id:
            raise ValueError("GrayMask task requires dataset.load_id (existing dataset).")

        # ------------------------------------------------------------
        # [NEW] rig 설정 스냅샷 저장 (dataset/rig/ 아래)
        # - 실험 재현성/추적을 위해 반드시 남기는 게 좋음
        # ------------------------------------------------------------
        rig_name = str(getattr(self.cfg.rig, "name", "rig"))
        rig_out_path = os.path.join(self.ds.dirs["rig"], f"{rig_name}.yaml")
        try:
            with open(rig_out_path, "w", encoding="utf-8") as f:
                f.write(OmegaConf.to_yaml(self.cfg.rig))
            log.info(f"Saved rig snapshot: {rig_out_path}")
        except Exception as e:
            log.warning(f"Failed to save rig snapshot: {e}")

        source_mode = getattr(self.cfg.task, "source", "manifest")
        overwrite = bool(getattr(self.cfg.task, "overwrite", False))

        if source_mode == "manifest":
            df = self.ds.manifest
            if df is None or df.empty:
                raise ValueError("Manifest is empty. GrayMask task expects existing manifest with mask_path.")
            rows = df.to_dict("records")
            items = []
            for r in rows:
                mp = r.get("mask_path", None)
                sid = r.get("sample_id", None)
                if sid is None or mp is None or (isinstance(mp, float) and pd.isna(mp)):
                    continue
                items.append((sid, mp))
        else:
            # scan mode: raw/mask_input 폴더 스캔
            pattern = os.path.join(self.ds.dirs["mask_input"], "*.png")
            files = sorted(glob.glob(pattern))
            items = []
            for f in files:
                # 파일명에서 sample_id 추정: sample_00000_mask.png -> sample_00000
                stem = Path(f).stem
                if stem.endswith("_mask"):
                    sid = stem.replace("_mask", "")
                else:
                    sid = stem
                items.append((sid, os.path.basename(f)))

        log.info(f"GrayMaskTask: processing {len(items)} masks (source={source_mode}, overwrite={overwrite})")

        updates = []
        debug_saved = 0

        for idx, (sample_id, mask_fname) in enumerate(tqdm(items, desc="GrayMask")):
            src_mask_path = os.path.join(self.ds.dirs["mask_input"], mask_fname)
            bin_img = cv2.imread(src_mask_path, cv2.IMREAD_GRAYSCALE)
            if bin_img is None:
                log.warning(f"Failed to read mask: {src_mask_path}")
                continue

            gray_name = f"{sample_id}_mask_gray.png"
            band_name = f"{sample_id}_band.png"
            win_gray_name = f"{sample_id}_window_gray.png"
            dst_gray_path = os.path.join(self.ds.dirs["mask_gray"], gray_name)
            dst_band_path = os.path.join(self.ds.dirs["mask_band"], band_name)
            dst_win_gray_path = os.path.join(self.ds.dirs["window_gray"], win_gray_name)

            if (not overwrite) and (os.path.exists(dst_gray_path) and os.path.exists(dst_band_path) and os.path.exists(dst_win_gray_path)):
                # 이미 존재하면 스킵하지만 manifest는 맞춰둘 수 있음
                meta = {"skipped": True}
                updates.append({
                    "sample_id": sample_id,
                    "mask_gray_path": gray_name,
                    "band_path": band_name,
                    "window_gray_path": win_gray_name,
                    "graymask_meta": json.dumps(meta, ensure_ascii=False),
                })
                continue

            gray_img, band_img, meta = self.synth.synthesize(bin_img, sample_id=sample_id)

            save_image(dst_gray_path, gray_img)
            save_image(dst_band_path, band_img)

            # ------------------------------------------------------------
            # [NEW] gray mask를 projector window(1080p)에 삽입해서 저장
            # rig.projector.insert_x/y, width/height를 그대로 사용
            # ------------------------------------------------------------
            win_gray = self.proj.insert_mask(gray_img)
            save_image(dst_win_gray_path, win_gray)

            # debug dump
            if should_dump_debug(idx, self.cfg) and debug_saved < int(self.cfg.debug.max_images):
                dbg_dir = self.ds.dirs["debug"]
                save_image(os.path.join(dbg_dir, f"{sample_id}_dbg_bin.png"), bin_img)
                save_image(os.path.join(dbg_dir, f"{sample_id}_dbg_gray.png"), gray_img)
                save_image(os.path.join(dbg_dir, f"{sample_id}_dbg_band.png"), band_img)
                save_image(os.path.join(dbg_dir, f"{sample_id}_dbg_window_gray.png"), win_gray)

                debug_saved += 1

            updates.append({
                "sample_id": sample_id,
                "mask_gray_path": gray_name,
                "band_path": band_name,
                "window_gray_path": win_gray_name,
                "graymask_meta": json.dumps(meta, ensure_ascii=False),
            })

        # manifest update (merge)
        if updates:
            df_new = pd.DataFrame(updates)
            self.ds.manifest = pd.merge(self.ds.manifest, df_new, on="sample_id", how="left", suffixes=("", "_new"))

            # _new 우선 반영
            for col in ["mask_gray_path", "band_path", "window_gray_path", "graymask_meta"]:
                if f"{col}_new" in self.ds.manifest.columns:
                    self.ds.manifest[col] = self.ds.manifest[f"{col}_new"].fillna(self.ds.manifest.get(col))
                    self.ds.manifest.drop(columns=[f"{col}_new"], inplace=True)

            self.ds.manifest.to_csv(self.ds.manifest_path, index=False)

        log.info(f"GrayMaskTask complete. Updated {len(updates)} records.")

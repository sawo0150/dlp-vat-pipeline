# pipeline/src/dlp_pipeline/project_task.py
from __future__ import annotations
import os
import glob
import logging
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from dlp_pipeline.projector_interface import ProjectorWindow
from dlp_pipeline.utils import save_image

log = logging.getLogger(__name__)

class ProjectTask:
    """
    existing masks -> projector window(1080p) 생성만 담당
    - binary: raw/mask_input -> raw/window_1080p
    - gray:   raw/mask_gray  -> raw/window_1080p_gray
    """

    def __init__(self, cfg, ds_manager):
        self.cfg = cfg
        self.ds = ds_manager
        self.proj = ProjectorWindow(cfg)

    def run(self):
        which = str(getattr(self.cfg.task, "which", "both"))
        source = str(getattr(self.cfg.task, "source", "manifest"))
        overwrite = bool(getattr(self.cfg.task, "overwrite", False))

        if which in ("binary", "both"):
            self._run_binary(source=source, overwrite=overwrite)

        if which in ("gray", "both"):
            self._run_gray(source=source, overwrite=overwrite)

        log.info("ProjectTask complete.")

    # -----------------------
    # binary -> window_1080p
    # -----------------------
    def _run_binary(self, source: str, overwrite: bool):
        items = []

        if source == "manifest":
            df = self.ds.manifest
            if df is None or df.empty:
                log.warning("manifest.csv is empty; skip binary projection.")
                return
            for r in df.to_dict("records"):
                sid = r.get("sample_id")
                mp = r.get("mask_path")
                if sid and isinstance(mp, str) and mp:
                    items.append((sid, mp))
        else:
            pattern = os.path.join(self.ds.dirs["mask_input"], "*.png")
            for f in sorted(glob.glob(pattern)):
                stem = Path(f).stem
                sid = stem.replace("_mask", "") if stem.endswith("_mask") else stem
                items.append((sid, os.path.basename(f)))

        if not items:
            log.warning("No binary masks found to project.")
            return

        rows_out = []
        for sid, mask_fname in tqdm(items, desc="Project(binary)"):
            src = os.path.join(self.ds.dirs["mask_input"], mask_fname)
            bin_img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
            if bin_img is None:
                log.warning(f"Failed to read: {src}")
                continue

            win_name = f"{sid}_window.png"
            dst = os.path.join(self.ds.dirs["window"], win_name)
            if (not overwrite) and os.path.exists(dst):
                continue

            win = self.proj.insert_mask(bin_img)
            save_image(dst, win)

            rows_out.append({"sample_id": sid, "window_path": win_name})

        # manifest.csv에 window_path backfill (있으면 업데이트)
        if rows_out:
            self.ds.update_manifest(rows_out)
            log.info(f"Binary windows written: {len(rows_out)}")

    # -----------------------
    # gray -> window_1080p_gray
    # -----------------------
    def _run_gray(self, source: str, overwrite: bool):
        gray_manifest_path = os.path.join(self.ds.path, "manifest_gray.csv")
        if not os.path.exists(gray_manifest_path):
            log.warning("manifest_gray.csv not found; skip gray projection.")
            return

        df = pd.read_csv(gray_manifest_path)
        if df.empty:
            log.warning("manifest_gray.csv is empty; skip gray projection.")
            return

        rows_out = []
        for r in tqdm(df.to_dict("records"), desc="Project(gray)"):
            sid = r.get("sample_id")
            gp = r.get("mask_gray_path")
            if not sid or not isinstance(gp, str) or not gp:
                continue

            src = os.path.join(self.ds.dirs["mask_gray"], gp)
            gray_img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
            if gray_img is None:
                log.warning(f"Failed to read: {src}")
                continue

            win_name = f"{sid}_window_gray.png"
            dst = os.path.join(self.ds.dirs["window_gray"], win_name)
            if (not overwrite) and os.path.exists(dst):
                continue

            win = self.proj.insert_mask(gray_img)
            save_image(dst, win)

            rows_out.append({"sample_id": sid, "window_gray_path": win_name})

        # manifest_gray.csv 업데이트: sample_id 기준으로 window_gray_path 채우기
        if rows_out:
            df_new = pd.DataFrame(rows_out)
            # left join 스타일 업데이트
            df = df.drop(columns=[c for c in ["window_gray_path"] if c in df.columns], errors="ignore") \
                   .merge(df_new, on="sample_id", how="left")
            df.to_csv(gray_manifest_path, index=False)
            log.info(f"Gray windows written: {len(rows_out)}")

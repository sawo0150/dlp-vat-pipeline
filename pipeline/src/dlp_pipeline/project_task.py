# pipeline/src/dlp_pipeline/project_task.py
from __future__ import annotations
import os
import glob
import logging
import re
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
    
    LabVIEW 제약:
    - 한 폴더당 최대 10,000개만 허용
    - 파일명은 window####.png 형태로 저장
    + LabVIEW가 실행 후 light distribution 결과를 저장할 output 폴더도 미리 생성
      (입력 batch 구조와 동일하게 batch_XXXX 폴더까지 생성)
    """

    def __init__(self, cfg, ds_manager):
        self.cfg = cfg
        self.ds = ds_manager
        self.proj = ProjectorWindow(cfg)


        # ---- output sharding config (defaults) ----
        # 필요하면 rig.projector 쪽에 추후 yaml로 내려도 됨.
        self.max_per_folder = int(getattr(getattr(cfg, "rig", {}).get("projector", {}), "max_per_folder", 10000)) \
            if hasattr(getattr(cfg, "rig", None), "projector") else int(getattr(cfg, "max_per_folder", 10000))
        self.batch_prefix = str(getattr(getattr(cfg, "rig", {}).get("projector", {}), "batch_prefix", "batch_")) \
            if hasattr(getattr(cfg, "rig", None), "projector") else "batch_"
        self.file_prefix = str(getattr(getattr(cfg, "rig", {}).get("projector", {}), "file_prefix", "window")) \
            if hasattr(getattr(cfg, "rig", None), "projector") else "window"

        # ---- LabVIEW output dirs (light distribution) ----
        # window_1080p 와 같은 레벨(raw/)에 parallel로 만든다:
        # raw/light_distribution/batch_XXXX/
        # raw/light_distribution_gray/batch_XXXX/
        self.light_dirname_bin = str(getattr(getattr(cfg, "rig", {}).get("projector", {}), "light_dirname_bin", "light_distribution")) \
            if hasattr(getattr(cfg, "rig", None), "projector") else "light_distribution"
        self.light_dirname_gray = str(getattr(getattr(cfg, "rig", {}).get("projector", {}), "light_dirname_gray", "light_distribution_gray")) \
            if hasattr(getattr(cfg, "rig", None), "projector") else "light_distribution_gray"

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
    # helpers: shard path + next index
    # -----------------------
    def _find_next_global_index(self, out_root: str) -> int:
        """
        out_root 아래의 batch 폴더들을 스캔해서,
        (batch_idx * max_per_folder + local_idx) 기준으로 가장 큰 global index를 찾고
        다음 index를 반환한다.
        """
        if not os.path.exists(out_root):
            return 0

        batch_re = re.compile(rf"^{re.escape(self.batch_prefix)}(\d+)$")
        file_re = re.compile(rf"^{re.escape(self.file_prefix)}(\d+)\.png$")

        max_global = -1
        for entry in os.scandir(out_root):
            if not entry.is_dir():
                continue
            m = batch_re.match(entry.name)
            if not m:
                continue
            batch_idx = int(m.group(1))
            batch_dir = entry.path
            try:
                for f in os.scandir(batch_dir):
                    if not f.is_file():
                        continue
                    fm = file_re.match(f.name)
                    if not fm:
                        continue
                    local_idx = int(fm.group(1))
                    if local_idx < 0 or local_idx >= self.max_per_folder:
                        continue
                    g = batch_idx * self.max_per_folder + local_idx
                    if g > max_global:
                        max_global = g
            except FileNotFoundError:
                continue

        return max_global + 1

    def _resolve_output_path(self, out_root: str, global_index: int, light_root: str | None = None):
        """
        global_index -> (dst_abs_path, rel_path_for_manifest)
        저장 규칙:
          out_root/batch_XXXX/windowYYYY.png  (YYYY는 0~9999)
        """
        batch_idx = global_index // self.max_per_folder
        local_idx = global_index % self.max_per_folder

        batch_name = f"{self.batch_prefix}{batch_idx:04d}"
        fname = f"{self.file_prefix}{local_idx:04d}.png"

        batch_dir = os.path.join(out_root, batch_name)
        os.makedirs(batch_dir, exist_ok=True)

        # LabVIEW가 저장할 light distribution 폴더도 같이 준비
        if light_root:
            os.makedirs(light_root, exist_ok=True)
            os.makedirs(os.path.join(light_root, batch_name), exist_ok=True)

        dst_abs = os.path.join(batch_dir, fname)
        rel = os.path.join(batch_name, fname).replace("\\", "/")
        return dst_abs, rel

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

        out_root = self.ds.dirs["window"]
        # overwrite면 0부터 다시, 아니면 기존 뒤부터 이어서
        next_global = 0 if overwrite else self._find_next_global_index(out_root)

        # raw/ 아래에 light distribution 출력 루트 생성
        raw_root = os.path.dirname(out_root)  # raw/
        light_root = os.path.join(raw_root, self.light_dirname_bin)

        rows_out = []
        for sid, mask_fname in tqdm(items, desc="Project(binary)"):
            # overwrite=False면 manifest에 이미 window_path가 있으면 스킵 (번호도 소모하지 않음)
            if (not overwrite) and self.ds.manifest is not None and (not self.ds.manifest.empty):
                try:
                    prev = self.ds.manifest.loc[self.ds.manifest["sample_id"] == sid, "window_path"]
                    if len(prev) > 0 and isinstance(prev.iloc[0], str) and prev.iloc[0]:
                        continue
                except Exception:
                    pass

            src = os.path.join(self.ds.dirs["mask_input"], mask_fname)
            bin_img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
            if bin_img is None:
                log.warning(f"Failed to read: {src}")
                continue

            dst_abs, rel_path = self._resolve_output_path(out_root, next_global, light_root=light_root)
            # overwrite=False인데 이미 파일이 존재하면(외부 요인), 다음 인덱스로 재시도
            if (not overwrite) and os.path.exists(dst_abs):
                # 충돌 회피: 빈 자리 찾기 (최악 케이스 대비)
                while os.path.exists(dst_abs):
                    next_global += 1
                    dst_abs, rel_path = self._resolve_output_path(out_root, next_global, light_root=light_root)

            win = self.proj.insert_mask(bin_img)
            save_image(dst_abs, win)

            rows_out.append({"sample_id": sid, "window_path": rel_path})
            next_global += 1

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

        out_root = self.ds.dirs["window_gray"]
        next_global = 0 if overwrite else self._find_next_global_index(out_root)

        # raw/ 아래에 gray light distribution 출력 루트 생성
        raw_root = os.path.dirname(out_root)  # raw/
        light_root = os.path.join(raw_root, self.light_dirname_gray)

        rows_out = []
        for r in tqdm(df.to_dict("records"), desc="Project(gray)"):
            sid = r.get("sample_id")
            gp = r.get("mask_gray_path")
            if not sid or not isinstance(gp, str) or not gp:
                continue

            # overwrite=False면 이미 window_gray_path가 있으면 스킵 (번호도 소모하지 않음)
            if not overwrite:
                prev = r.get("window_gray_path")
                if isinstance(prev, str) and prev:
                    continue

            src = os.path.join(self.ds.dirs["mask_gray"], gp)
            gray_img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
            if gray_img is None:
                log.warning(f"Failed to read: {src}")
                continue

            dst_abs, rel_path = self._resolve_output_path(out_root, next_global, light_root=light_root)

            if (not overwrite) and os.path.exists(dst_abs):
                while os.path.exists(dst_abs):
                    next_global += 1
                    dst_abs, rel_path = self._resolve_output_path(out_root, next_global, light_root=light_root)
 
            win = self.proj.insert_mask(gray_img)
            save_image(dst_abs, win)

            rows_out.append({"sample_id": sid, "window_gray_path": rel_path})
            next_global += 1

        # manifest_gray.csv 업데이트: sample_id 기준으로 window_gray_path 채우기
        if rows_out:
            df_new = pd.DataFrame(rows_out)
            # left join 스타일 업데이트
            df = df.drop(columns=[c for c in ["window_gray_path"] if c in df.columns], errors="ignore") \
                   .merge(df_new, on="sample_id", how="left")
            df.to_csv(gray_manifest_path, index=False)
            log.info(f"Gray windows written: {len(rows_out)}")

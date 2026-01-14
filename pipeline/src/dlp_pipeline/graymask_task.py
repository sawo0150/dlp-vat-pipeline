# pipeline/src/dlp_pipeline/graymask_task.py
from __future__ import annotations
import os
import json
import logging
import glob
import re
import hashlib
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf

from dlp_pipeline.utils import save_image, should_dump_debug
from dlp_pipeline.graymask_synth import GrayMaskSynthesizer

log = logging.getLogger(__name__)

def _stable_u01(s: str) -> float:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
    v = int(h, 16)
    return v / float(0xFFFFFFFF)

def _sanitize_stem(s: str) -> str:
    # 파일명 안전 처리: |, 공백, = 등 -> __
    return re.sub(r"[^A-Za-z0-9._-]+", "__", s)

class GrayMaskTask:
    def __init__(self, cfg, ds_manager):
        self.cfg = cfg
        self.ds = ds_manager
        self.synth = GrayMaskSynthesizer(cfg, base_seed=int(getattr(cfg, "seed", 0) or 0))

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

        # 새 manifest(1 gray sample = 1 row)
        gray_manifest_path = os.path.join(self.ds.path, "manifest_gray.csv")
        if os.path.exists(gray_manifest_path):
            df_gray = pd.read_csv(gray_manifest_path)
        else:
            df_gray = pd.DataFrame()

        rows_out = []

        debug_saved = 0

        for idx, (sample_id, mask_fname) in enumerate(tqdm(items, desc="GrayMask")):
            src_mask_path = os.path.join(self.ds.dirs["mask_input"], mask_fname)
            bin_img = cv2.imread(src_mask_path, cv2.IMREAD_GRAYSCALE)
            if bin_img is None:
                log.warning(f"Failed to read mask: {src_mask_path}")
                continue

            base_id = str(sample_id)  # "id 형식 기반" (원하면 여기서 숫자만 추출 등 커스텀 가능)

            # ----------------------------
            # 그룹 분기 (순서 무관, ID-hash)
            # ----------------------------
            gs = getattr(self.cfg.task, "group_split", None)
            sob_ratio = float(getattr(gs, "sobolev_ratio", 0.0)) if gs is not None else 0.0
            salt = str(getattr(gs, "hash_salt", "v1")) if gs is not None else "v1"
            u = _stable_u01(f"{salt}|{base_id}")
            is_sobolev = (u < sob_ratio)
            group = "B" if is_sobolev else "A"

            prof_cfg = getattr(self.cfg.task, "profiles", None)

            def get_override(key: str):
                if prof_cfg is None or not hasattr(prof_cfg, key):
                    return None
                node = getattr(prof_cfg, key)
                if hasattr(node, "graymask_override"):
                    return OmegaConf.to_container(getattr(node, "graymask_override"), resolve=True)
                return None

            # ----------------------------
            # profile 생성(각 base 1장 -> 3장)
            # ----------------------------
            if group == "A":
                variants = [
                    ("A0", f"{base_id}|A0|smooth_phys"),
                    ("A1", f"{base_id}|A1|ring_aniso"),
                    ("A2", f"{base_id}|A2|stress_quant_partial"),
                ]

                for pkey, sid in variants:
                    out_stem = _sanitize_stem(sid)
                    gray_name = f"{out_stem}_mask_gray.png"
                    band_name = f"{out_stem}_band.png"
                    meta_name = f"{out_stem}_meta.json"

                    dst_gray_path = os.path.join(self.ds.dirs["mask_gray"], gray_name)
                    dst_band_path = os.path.join(self.ds.dirs["mask_band"], band_name)
                    dst_meta_path = os.path.join(self.ds.dirs["mask_gray_meta"], meta_name)

                    if (not overwrite) and (os.path.exists(dst_gray_path) and os.path.exists(dst_band_path) and os.path.exists(dst_meta_path)):
                        continue

                    override = get_override(pkey)
                    meta_extra = {"group": "A", "profile": pkey, "base_id": base_id, "base_mask_path": mask_fname}

                    gray_img, band_img, meta = self.synth.synthesize(
                        bin_img, sample_id=sid, cfg_override=override, meta_extra=meta_extra
                    )

                    save_image(dst_gray_path, gray_img)
                    save_image(dst_band_path, band_img)

                    with open(dst_meta_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)

                    if should_dump_debug(idx, self.cfg) and debug_saved < int(self.cfg.debug.max_images):
                        dbg_dir = self.ds.dirs["debug"]
                        save_image(os.path.join(dbg_dir, f"{out_stem}_dbg_bin.png"), bin_img)
                        save_image(os.path.join(dbg_dir, f"{out_stem}_dbg_gray.png"), gray_img)
                        save_image(os.path.join(dbg_dir, f"{out_stem}_dbg_band.png"), band_img)
                        debug_saved += 1

                    rows_out.append({
                        "base_id": base_id,
                        "base_sample_id": sample_id,
                        "group": "A",
                        "profile": pkey,
                        "sample_id": sid,
                        "mask_path": mask_fname,
                        "mask_gray_path": gray_name,
                        "band_path": band_name,
                        "meta_path": meta_name,
                    })

            else:
                # Sobolev: anchor/plus/minus
                sob = getattr(self.cfg.task, "sobolev", None)
                eps_choices = list(getattr(sob, "eps_choices", [8])) if sob is not None else [8]
                basis_choices = list(getattr(sob, "basis_choices", ["patch"])) if sob is not None else ["patch"]

                # deterministic pick (order-independent)
                u_eps = _stable_u01(f"{salt}|{base_id}|eps")
                u_bas = _stable_u01(f"{salt}|{base_id}|basis")
                eps = int(eps_choices[int(u_eps * len(eps_choices)) % len(eps_choices)])
                basis = str(basis_choices[int(u_bas * len(basis_choices)) % len(basis_choices)])

                sobolev_group_id = f"{base_id}|basis={basis}|eps={eps}"

                sid_anchor = f"{base_id}|B0|sobolev_anchor|basis={basis}|eps={eps}"
                sid_plus   = f"{base_id}|B1|sobolev_plus|basis={basis}|eps={eps}"
                sid_minus  = f"{base_id}|B2|sobolev_minus|basis={basis}|eps={eps}"

                # 1) anchor 생성
                override_anchor = get_override("B0")
                meta_extra_anchor = {
                    "group": "B", "profile": "B0", "base_id": base_id, "base_mask_path": mask_fname,
                    "sobolev_group_id": sobolev_group_id, "epsilon": eps, "basis_type": basis,
                }
                gray_anchor, band_anchor, meta_anchor = self.synth.synthesize(
                    bin_img, sample_id=sid_anchor, cfg_override=override_anchor, meta_extra=meta_extra_anchor
                )

                # 2) phi 생성(공유)
                phi, meta_phi = self.synth.make_sobolev_phi(band_anchor, sobolev_group_id=sobolev_group_id, basis_type=basis)
                # 3) plus/minus 생성
                gray_plus, gray_minus = self.synth.apply_sobolev_plus_minus(gray_anchor, band_anchor, phi, epsilon=eps)

                def save_one(sid: str, prof: str, gray_img_u8: "np.ndarray", extra_meta: dict):
                    out_stem = _sanitize_stem(sid)
                    gray_name = f"{out_stem}_mask_gray.png"
                    band_name = f"{out_stem}_band.png"
                    meta_name = f"{out_stem}_meta.json"

                    dst_gray_path = os.path.join(self.ds.dirs["mask_gray"], gray_name)
                    dst_band_path = os.path.join(self.ds.dirs["mask_band"], band_name)
                    dst_meta_path = os.path.join(self.ds.dirs["mask_gray_meta"], meta_name)

                    if (not overwrite) and (os.path.exists(dst_gray_path) and os.path.exists(dst_band_path) and os.path.exists(dst_meta_path)):
                        return

                    save_image(dst_gray_path, gray_img_u8)
                    save_image(dst_band_path, band_anchor)  # band는 anchor와 동일

                    meta = dict(extra_meta)
                    meta["phi"] = meta_phi
                    with open(dst_meta_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)

                    rows_out.append({
                        "base_id": base_id,
                        "base_sample_id": sample_id,
                        "group": "B",
                        "profile": prof,
                        "sample_id": sid,
                        "mask_path": mask_fname,
                        "mask_gray_path": gray_name,
                        "band_path": band_name,
                        "meta_path": meta_name,
                    })

                # anchor 저장
                save_one(sid_anchor, "B0", gray_anchor, meta_anchor)
                # plus/minus meta는 anchor meta를 베이스로 profile만 변경
                meta_plus = dict(meta_anchor);  meta_plus["profile"] = "B1"; meta_plus["sample_id"] = sid_plus
                meta_minus = dict(meta_anchor); meta_minus["profile"] = "B2"; meta_minus["sample_id"] = sid_minus
                save_one(sid_plus,  "B1", gray_plus,  meta_plus)
                save_one(sid_minus, "B2", gray_minus, meta_minus)

        # 새 manifest_gray.csv append 저장
        if rows_out:
            df_new = pd.DataFrame(rows_out)
            if df_gray is None or df_gray.empty:
                df_gray = df_new
            else:
                df_gray = pd.concat([df_gray, df_new], ignore_index=True)
                # sample_id 기준으로 최신 유지(덮어쓰기 시)
                df_gray = df_gray.drop_duplicates(subset=["sample_id"], keep="last")
            df_gray.to_csv(gray_manifest_path, index=False)

        log.info(f"GrayMaskTask complete. Wrote {len(rows_out)} gray samples -> {gray_manifest_path}")

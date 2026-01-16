# pipeline/src/dlp_pipeline/preprocessor.py

import cv2
import numpy as np
import logging
import os
import shutil
from tqdm import tqdm
import pandas as pd
from dlp_pipeline.utils import ensure_dir

log = logging.getLogger(__name__)

class Preprocessor:
    def __init__(self, cfg, ds_manager):
        self.cfg = cfg
        self.ds = ds_manager
        self.p_cfg = cfg.preprocess # preprocess config shortcut
        # 네이밍 설정 로드 (없으면 기본값 A/B 사용)
        self.name_cfg = self.p_cfg.get('naming', {'process_name': 'default', 'input_folder': 'A', 'target_folder': 'B'})

    def run(self):
        log.info("Starting Preprocessing Task...")
        
        # [NEW] mapping 기반 pair 전처리 모드
        if hasattr(self.p_cfg, "paired"):
            return self._run_paired_light()
         
        manifest = self.ds.manifest
        if manifest.empty:
            log.error("Manifest is empty!")
            return

        processed_data = []
        # 폴더명에 프로세스 이름 반영 (확장성)
        # 예: interim/processed_standard/digital_mask
        proc_dir_name = f"processed_{self.name_cfg.get('process_name', 'default')}"
        base_proc_dir = ensure_dir(os.path.join(self.ds.path, "interim", proc_dir_name))

        # ML 데이터셋 폴더 준비 (interim/processed)
        dir_input = ensure_dir(os.path.join(base_proc_dir, self.name_cfg['input_folder']))
        dir_target = ensure_dir(os.path.join(base_proc_dir, self.name_cfg['target_folder']))

        for idx, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Preprocessing"):
            if pd.isna(row['camera_path']) or pd.isna(row['mask_path']):
                continue

            sample_id = row['sample_id']
            
            # 1. 경로 로드
            mask_src = os.path.join(self.ds.dirs['mask_input'], row['mask_path'])
            cam_src = os.path.join(self.ds.dirs['camera_raw'], row['camera_path'])

            # 2. 이미지 로드
            mask_img = cv2.imread(mask_src, cv2.IMREAD_GRAYSCALE)
            cam_img = cv2.imread(cam_src, cv2.IMREAD_GRAYSCALE) # 혹은 UNCHANGED

            if mask_img is None or cam_img is None:
                log.warning(f"Failed to load image for {sample_id}")
                continue

            # 3. 처리 (Process)
            # 3-1. Camera Processing (S3 Logic)
            proc_cam = self._process_camera(cam_img)
            
            # 3-2. Mask Processing (S4 Logic + Dynamic Resize)
            # Manifest에 있는 width 정보를 활용
            curr_w = row.get('mask_width', 128) # 없으면 기본 128 가정
            proc_mask = self._process_mask(mask_img, curr_w)

            # 4. 저장 (Save)
            # 파일명은 간단하게 유지 (중간 산출물이므로)
            fname = f"{sample_id}.png"
            cv2.imwrite(os.path.join(dir_input, fname), proc_mask)
            cv2.imwrite(os.path.join(dir_target, fname), proc_cam)

            processed_data.append({
                "sample_id": sample_id,
                # [수정] 저장된 경로 기록 (상대 경로)
                "processed_input_path": os.path.join(proc_dir_name, self.name_cfg['input_folder'], fname),
                "processed_target_path": os.path.join(proc_dir_name, self.name_cfg['target_folder'], fname),
                "pattern_type": row.get('pattern_type', 'unknown') # 나중에 파일명에 쓰기 위해
            })

        # Manifest 업데이트
        if processed_data:
            df_proc = pd.DataFrame(processed_data)
            
            # [수정] 중복 컬럼 처리 (Merge 시 _x, _y 발생하는 문제 해결)
            # df_proc에 있는 컬럼이 manifest에도 있다면, df_proc(새로운 값)을 우선시하기 위한 로직
            self.ds.manifest = pd.merge(self.ds.manifest, df_proc, on='sample_id', how='left', suffixes=('', '_new'))
            
            # _new가 붙은 컬럼이 생겼다면 원본 컬럼을 업데이트하고 _new 삭제
            for col in df_proc.columns:
                if col == 'sample_id': continue
                if f'{col}_new' in self.ds.manifest.columns:
                    self.ds.manifest[col] = self.ds.manifest[f'{col}_new'].fillna(self.ds.manifest[col])
                    self.ds.manifest.drop(columns=[f'{col}_new'], inplace=True)            

            self.ds.manifest.to_csv(self.ds.manifest_path, index=False)
            
        # 5. ML Dataset Split 실행 (Preprocessing 직후 수행)
        self._build_ml_dataset()

    # ------------------------------------------------------------------
    # [NEW] mapping 기반: window(or mask) <-> light_distribution pairing
    # ------------------------------------------------------------------
    def _run_paired_light(self):
        paired = self.p_cfg.paired

        # gray/binary 판별: dataset에 window_1080p_gray가 있으면 gray를 기본으로
        use_gray = os.path.isdir(self.ds.dirs.get("window_gray", "")) and \
                   os.path.isdir(os.path.join(self.ds.path, "raw", "light_distribution_gray"))

        win_root = self.ds.dirs["window_gray"] if use_gray else self.ds.dirs["window"]
        light_root = os.path.join(self.ds.path, "raw", "light_distribution_gray" if use_gray else "light_distribution")

        input_source = str(getattr(paired, "input_source", "mask_gray")).lower()
        if input_source == "mask_input":
            in_root = self.ds.dirs["mask_input"]
        elif input_source == "mask_gray":
            in_root = self.ds.dirs["mask_gray"]
        else:
            in_root = win_root

        mapping_dir = str(getattr(paired, "mapping_dir", "pairing"))
        mapping_name = getattr(paired, "mapping_name", None)
        map_root = ensure_dir(os.path.join(self.ds.path, mapping_dir))

        # mapping 자동 선택: 가장 최근 csv
        if mapping_name is None or str(mapping_name).lower() == "null":
            cands = [f for f in os.listdir(map_root) if f.endswith(".csv")]
            if not cands:
                raise FileNotFoundError(f"No mapping csv in {map_root}. Run task=pair_gui first.")
            cands = sorted(cands, key=lambda x: os.path.getmtime(os.path.join(map_root, x)), reverse=True)
            mapping_name = cands[0]

        mapping_path = os.path.join(map_root, str(mapping_name))
        log.info(f"[paired] using mapping: {mapping_path}")

        df_map = pd.read_csv(mapping_path)
        if df_map.empty:
            log.warning("Mapping CSV is empty.")
            return

        # output dirs
        process_name = str(getattr(paired, "process_name", "paired"))
        input_folder = str(getattr(paired, "input_folder", "digital_mask"))
        target_folder = str(getattr(paired, "target_folder", "light_dist"))

        base_proc_dir = ensure_dir(os.path.join(self.ds.path, "interim", f"processed_{process_name}"))
        dir_input = ensure_dir(os.path.join(base_proc_dir, input_folder))
        dir_target = ensure_dir(os.path.join(base_proc_dir, target_folder))
        dir_debug = self.ds.dirs["debug"]

        # ECC params
        do_register = bool(getattr(paired, "do_register", True))
        warp_mode = str(getattr(paired, "warp_mode", "translation")).lower()
        ecc_iters = int(getattr(paired, "ecc_iters", 2000))
        ecc_eps = float(getattr(paired, "ecc_eps", 1e-6))
        gauss = int(getattr(paired, "gauss_filt", 5))

        out_size = tuple(getattr(paired, "out_size", [256, 256]))
        roi = getattr(paired, "crop_roi", None)
        save_debug = bool(getattr(paired, "save_debug", True))

        def warp_mode_cv(s: str) -> int:
            if s == "translation": return cv2.MOTION_TRANSLATION
            if s == "euclidean": return cv2.MOTION_EUCLIDEAN
            if s == "affine": return cv2.MOTION_AFFINE
            if s == "homography": return cv2.MOTION_HOMOGRAPHY
            return cv2.MOTION_TRANSLATION

        def apply_crop(img, roi_):
            if roi_ is None: return img
            x,y,w,h = [int(v) for v in roi_]
            return img[y:y+h, x:x+w]

        def apply_resize(img, size_):
            ow, oh = int(size_[0]), int(size_[1])
            return cv2.resize(img, (ow, oh), interpolation=cv2.INTER_AREA)

        def ecc_align(ref_u8, mov_u8):
            ref = ref_u8.astype(np.float32) / 255.0
            mov = mov_u8.astype(np.float32) / 255.0
            if gauss and gauss > 0:
                k = gauss if gauss % 2 == 1 else gauss + 1
                ref = cv2.GaussianBlur(ref, (k,k), 0)
                mov = cv2.GaussianBlur(mov, (k,k), 0)
            wm = warp_mode_cv(warp_mode)
            if wm == cv2.MOTION_HOMOGRAPHY:
                warp = np.eye(3,3, dtype=np.float32)
            else:
                warp = np.eye(2,3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ecc_iters, ecc_eps)
            try:
                cc, warp = cv2.findTransformECC(ref, mov, warp, wm, criteria, None, 1)
            except cv2.error:
                return mov_u8, -1.0
            if wm == cv2.MOTION_HOMOGRAPHY:
                aligned = cv2.warpPerspective(
                    mov_u8, warp, (ref_u8.shape[1], ref_u8.shape[0]),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0
                )
            else:
                aligned = cv2.warpAffine(
                    mov_u8, warp, (ref_u8.shape[1], ref_u8.shape[0]),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0
                )
            return aligned, float(cc)

        processed_rows = []

        # mapping CSV 형식: window_file, light_file
        for _, r in tqdm(df_map.iterrows(), total=len(df_map), desc="Paired preprocess"):
            wfile = str(r.get("window_file", "")).strip()
            lfile = str(r.get("light_file", "")).strip()
            if not wfile or not lfile:
                continue

            # window_file은 batch 내부 파일명이라고 가정 (GUI 저장 방식)
            # batch는 window_file에서 유추 불가하므로 mapping_name에 batch를 포함시키는 것을 권장
            # 여기서는 모든 batch를 스캔하지 않고, "mapping 파일명에 batch_XXXX가 포함"된다고 가정
            # fallback: window 폴더 전체에서 검색
            # ---- find window absolute path ----
            win_abs = None
            for b in os.scandir(win_root):
                if not b.is_dir() or not b.name.startswith("batch_"):
                    continue
                cand = os.path.join(b.path, wfile)
                if os.path.exists(cand):
                    win_abs = cand
                    batch_name = b.name
                    break
            if win_abs is None:
                log.warning(f"window not found: {wfile}")
                continue

            # light는 같은 batch 아래에 있다고 가정
            light_abs = os.path.join(light_root, batch_name, lfile)
            if not os.path.exists(light_abs):
                log.warning(f"light not found: {batch_name}/{lfile}")
                continue

            # input source 로드
            if input_source == "window":
                inp_abs = win_abs

            else:
                # window파일명을 sample_id로 바로 못 바꾸므로, manifest를 통해 window_path->sample_id 매핑이 이상적
                # 일단: window####.png -> sample_id를 manifest에서 검색 (window_path endswith window####.png)
                sid = None
                if os.path.exists(os.path.join(self.ds.path, "manifest_gray.csv")):
                    dfm = pd.read_csv(os.path.join(self.ds.path, "manifest_gray.csv"))
                    hit = dfm[dfm.get("window_gray_path","").astype(str).str.endswith(f"{batch_name}/{wfile}")]
                    if len(hit) > 0:
                        sid = str(hit.iloc[0]["sample_id"])
                        gp = str(hit.iloc[0].get("mask_gray_path",""))

                        mp = str(hit.iloc[0].get("mask_path",""))
                    else:
                        gp = mp = ""

                else:
                    dfm = self.ds.manifest

                    hit = dfm[dfm.get("window_path","").astype(str).str.endswith(f"{batch_name}/{wfile}")]

                    if len(hit) > 0:
                        sid = str(hit.iloc[0]["sample_id"])
                        mp = str(hit.iloc[0].get("mask_path",""))
                        gp = ""
                    else:
                        gp = mp = ""

                if sid is None:
                    # fallback: 파일명 기반 sid 생성
                    sid = os.path.splitext(wfile)[0]

                if input_source == "mask_gray":
                    if gp:
                        inp_abs = os.path.join(self.ds.dirs["mask_gray"], gp)
                    else:
                        inp_abs = None
                else:  # mask_input
                    if mp:
                        inp_abs = os.path.join(self.ds.dirs["mask_input"], mp)
                    else:
                        inp_abs = None

                if inp_abs is None or not os.path.exists(inp_abs):
                    log.warning(f"input not found for {wfile} (source={input_source})")
                    continue

            # 이미지 로드
            inp = cv2.imread(inp_abs, cv2.IMREAD_GRAYSCALE)
            win = cv2.imread(win_abs, cv2.IMREAD_GRAYSCALE)
            light = cv2.imread(light_abs, cv2.IMREAD_GRAYSCALE)
            if inp is None or win is None or light is None:
                continue

            # 정합은 "window 기준으로 light를 정합"
            cc = None
            aligned = light
            if do_register:
                aligned, cc = ecc_align(win, light)

            # crop/resize
            inp2 = apply_resize(apply_crop(inp, roi), out_size)
            tgt2 = apply_resize(apply_crop(aligned, roi), out_size)

            # 저장
            # sample id는 window파일명 기반으로 안전하게
            sid2 = os.path.splitext(wfile)[0]
            out_in = os.path.join(dir_input, f"{sid2}.png")
            out_tg = os.path.join(dir_target, f"{sid2}.png")
            cv2.imwrite(out_in, inp2)
            cv2.imwrite(out_tg, tgt2)

            if save_debug:
                try:
                    overlay = cv2.addWeighted(win, 0.5, aligned, 0.5, 0.0)
                    cv2.imwrite(os.path.join(dir_debug, f"{sid2}_overlay.png"), overlay)
                except Exception:
                    pass

            processed_rows.append({
                "sample_id": sid2,
                "paired_window_file": f"{batch_name}/{wfile}",
                "paired_light_file": f"{batch_name}/{lfile}",
                "ecc_cc": cc,
                "processed_input_path": os.path.join(f"processed_{process_name}", input_folder, f"{sid2}.png"),
                "processed_target_path": os.path.join(f"processed_{process_name}", target_folder, f"{sid2}.png"),
            })

        # manifest 업데이트 (기본 manifest.csv에 기록)
        if processed_rows:
            dfp = pd.DataFrame(processed_rows)
            self.ds.manifest = pd.merge(self.ds.manifest, dfp, on="sample_id", how="outer", suffixes=("", "_new"))
            for col in dfp.columns:
                if col == "sample_id": 
                    continue
                cnew = f"{col}_new"
                if cnew in self.ds.manifest.columns:
                    self.ds.manifest[col] = self.ds.manifest[cnew].fillna(self.ds.manifest.get(col))
                    self.ds.manifest.drop(columns=[cnew], inplace=True)
            self.ds.manifest.to_csv(self.ds.manifest_path, index=False)

        # 기존 split 로직 재사용
        self._build_ml_dataset()
        return

    def _process_camera(self, img):
        """S3 Logic Porting: Transpose -> Rotate/Scale -> Pad -> Crop"""
        c_cfg = self.p_cfg.camera
        
        # 1. Transpose (MATLAB: II.')
        if c_cfg.transpose:
            img = cv2.transpose(img)
            
        # [추가] Flip Logic (180도 회전 이슈 해결)
        if c_cfg.vflip:
            img = cv2.flip(img, 0) # Vertical Flip
        if c_cfg.hflip:
            img = cv2.flip(img, 1) # Horizontal Flip

        # 2. Affine (Rotate & Scale)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        # OpenCV getRotationMatrix2D는 (Center, Angle, Scale)
        # Angle: 반시계 방향이 양수. MATLAB -1.1은 시계방향 -> OpenCV 1.1? 확인 필요.
        # 일단 MATLAB 로직 그대로 적용 시도
        M = cv2.getRotationMatrix2D(center, c_cfg.rotation, c_cfg.scale)
        warped = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)

        # 3. Padding
        pad = c_cfg.pad_size
        padded = cv2.copyMakeBorder(warped, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

        # 4. Crop
        cx, cy, cw, ch = c_cfg.crop.x, c_cfg.crop.y, c_cfg.crop.w, c_cfg.crop.h
        # Boundary Check
        if cy+ch > padded.shape[0] or cx+cw > padded.shape[1]:
            log.warning("Crop region out of bounds! Resizing instead.")
            cropped = cv2.resize(padded, (cw, ch))
        else:
            cropped = padded[cy:cy+ch, cx:cx+cw]

        # 5. Final Resize (Optional, 안전장치)
        if cropped.shape[0] != c_cfg.target_size:
            cropped = cv2.resize(cropped, (c_cfg.target_size, c_cfg.target_size))
            
        return cropped

    def _process_mask(self, img, current_width):
        """S4 Logic: Upscale/Pad to target size"""
        m_cfg = self.p_cfg.mask
        target = m_cfg.target_size # 1280
        
        # Case 1: 이미 Target Size (Legacy Data)
        if current_width == target:
            return img
        
        # Case 2: 작은 마스크 (128 -> 1280)
        # Nearest Neighbor로 픽셀 깨짐 없이 확대
        scale = target / current_width
        
        # 정수배 확대인지 확인
        if target % current_width == 0:
            resized = cv2.resize(img, (target, target), interpolation=cv2.INTER_NEAREST)
        else:
            # 정수배가 아니면(예: 64 -> 1280) 일단 확대 후 Padding/Crop 해야함
            # 여기서는 일단 Resize로 처리
            resized = cv2.resize(img, (target, target), interpolation=cv2.INTER_NEAREST)
            
        return resized

    def _build_ml_dataset(self):
        """
        Split Train/Val/Test and organize folders with Descriptive Names
        """
        log.info("Building ML Dataset structure...")
        
        manifest = self.ds.manifest
        # 처리된 데이터만 필터링
        valid_df = manifest.dropna(subset=['processed_input_path'])
        
        # Shuffle
        if self.p_cfg.split.shuffle:
            valid_df = valid_df.sample(frac=1).reset_index(drop=True)
            
        n = len(valid_df)
        n_train = int(n * self.p_cfg.split.train_ratio)
        n_val = int(n * self.p_cfg.split.val_ratio)
        
        splits = {
            'train': valid_df.iloc[:n_train],
            'val': valid_df.iloc[n_train:n_train+n_val],
            'test': valid_df.iloc[n_train+n_val:]
        }
        
        # [수정] 최종 데이터셋 폴더명에도 프로세스 이름 반영
        final_folder_name = f"final_dataset_{self.name_cfg.get('process_name', 'standard')}"
        final_root = ensure_dir(os.path.join(self.ds.path, final_folder_name))        

        for split_name, df_split in splits.items():
            # 폴더 생성 (예: final_standard/train/digital_mask)
            path_in = ensure_dir(os.path.join(final_root, split_name, self.name_cfg['input_folder']))
            path_out = ensure_dir(os.path.join(final_root, split_name, self.name_cfg['target_folder']))         

            for i, row in df_split.iterrows():
                # [주의] processed_path는 interim 폴더 기준 상대경로임. ds.path와 결합해야 함
                src_in = os.path.join(self.ds.path, "interim", row['processed_input_path'])
                src_out = os.path.join(self.ds.path, "interim", row['processed_target_path'])

                # [추가] 사람이 읽기 편한 파일명 (Config 옵션 확인)
                if self.name_cfg.get('descriptive_files', False):
                    # 안전한 접근 (KeyError 방지)
                    ptype = str(row.get('pattern_type', 'unknown')).replace(" ", "_")
                    fname = f"{split_name}_{i:04d}_{ptype}.png"
                else:
                    fname = f"{row['sample_id']}.png"
                
                shutil.copy2(src_in, os.path.join(path_in, fname))
                shutil.copy2(src_out, os.path.join(path_out, fname))                

        log.info(f"ML Dataset built at: {final_root}")
        log.info(f"Counts - Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
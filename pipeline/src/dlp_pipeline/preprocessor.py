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
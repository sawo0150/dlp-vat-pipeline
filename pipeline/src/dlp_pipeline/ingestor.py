# dlp_pipeline/ingestor.py

import os
import shutil
import glob
import cv2
from tqdm import tqdm
import pandas as pd
import logging
from natsort import natsorted
from pathlib import Path
from dlp_pipeline.utils import ensure_dir

log = logging.getLogger(__name__)

class DataIngestor:
    def __init__(self, cfg, ds_manager):
        self.cfg = cfg
        self.ds = ds_manager
        
    def run(self):
        mode = self.cfg.ingest.mode
        log.info(f"Running Ingest Task in [{mode.upper()}] mode.")
        
        if mode == "external":
            self._ingest_external_legacy()
        elif mode == "pipeline":
            self._ingest_pipeline_labview()
        else:
            log.error(f"Unknown ingest mode: {mode}")

    def _ingest_external_legacy(self):
        """
        Mode A: 기존(사수님) 데이터셋 주입
        - Mask 폴더와 Camera 폴더를 각각 읽어서 파일명(stem) 기준으로 짝을 맞춤
        - 새로운 manifest.csv를 생성함
        """
        mask_dir = self.cfg.ingest.mask_dir
        cam_dir = self.cfg.ingest.camera_dir
        
        # 1. 파일 리스트 확보
        mask_files = natsorted(glob.glob(os.path.join(mask_dir, self.cfg.ingest.mask_ext)))
        cam_files = natsorted(glob.glob(os.path.join(cam_dir, self.cfg.ingest.camera_ext)))
        
        log.info(f"Found {len(mask_files)} masks and {len(cam_files)} camera images.")

        # 2. 매칭 로직 (파일명 기준)
        # { '9001': 'path/to/9001.png', ... }
        mask_map = {Path(f).stem: f for f in mask_files}
        cam_map = {Path(f).stem: f for f in cam_files}
        
        # 교집합 키(파일명) 찾기
        common_keys = natsorted(list(set(mask_map.keys()) & set(cam_map.keys())))
        
        if not common_keys:
            log.error("No matching filenames found between mask and camera folders!")
            return

        log.info(f"Matched {len(common_keys)} pairs based on filenames.")

        # 3. 데이터 복사 및 Manifest 생성
        manifest_data = []
        
        for idx, key in enumerate(tqdm(common_keys, desc="Ingesting External")):
            src_mask = mask_map[key]
            src_cam = cam_map[key]

            # --- [수정] 이미지 크기 메타데이터 확보 ---
            # 이미지를 잠깐 읽어서 크기만 잽니다.
            _m_img = cv2.imread(src_mask, cv2.IMREAD_GRAYSCALE)
            h, w = _m_img.shape if _m_img is not None else (0, 0)

            # ID 생성 (예: sample_0000)
            sample_id = f"sample_{idx:05d}"
            
            # 대상 파일명 표준화
            dst_mask_name = f"{sample_id}_mask.png"
            dst_cam_name = f"{sample_id}_cam.png"
            
            # 목적지 경로
            dst_mask_path = os.path.join(self.ds.dirs['mask_input'], dst_mask_name) # mask_128 -> mask_input으로 변경
            dst_cam_path = os.path.join(self.ds.dirs['camera_raw'], dst_cam_name)
            
            try:
                shutil.copy2(src_mask, dst_mask_path)
                shutil.copy2(src_cam, dst_cam_path)
                
                manifest_data.append({
                    "sample_id": sample_id,
                    "mask_path": dst_mask_name,
                    "window_path": None, # 외부 주입이라 Window 정보는 없음
                    "camera_path": dst_cam_name,
                    "original_filename": key, # 추적용
                    "pattern_type": "legacy_external",
                    "mask_width": w,
                    "mask_height": h
                })
            except Exception as e:
                log.error(f"Error copying {key}: {e}")

        # 4. 저장
        self.ds.update_manifest(manifest_data)
        log.info("External Legacy Injection Complete.")

    def _ingest_pipeline_labview(self):
        """
        Mode B: 파이프라인 연동 (LabVIEW 결과물 가져오기)
        - 기존 manifest.csv가 존재해야 함
        - 순서대로(sort) 매칭하여 가져옴
        """
        source_dir = self.cfg.ingest.source_camera_dir
        
        # 1. 소스 파일 확인
        src_files = natsorted(glob.glob(os.path.join(source_dir, self.cfg.ingest.file_ext)))
        if not src_files:
            log.error(f"No files found in {source_dir}")
            return
            
        # 2. Manifest 로드
        if self.ds.manifest.empty:
            log.error("Manifest is empty. Run 'generate' task first.")
            return
            
        manifest = self.ds.manifest
        
        # 3. 매칭 및 업데이트
        new_camera_paths = []
        
        # 기존에 camera_path가 이미 있으면 건너뛰거나 덮어쓰기 정책 결정 필요 (여기선 덮어쓰기)
        for idx, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Ingesting Pipeline"):
            if idx >= len(src_files):
                log.warning(f"Not enough source images for sample {row['sample_id']}")
                new_camera_paths.append(None)
                continue
                
            src_file = src_files[idx]
            sample_id = row['sample_id']
            
            # 확장자는 원본 따름 (또는 png로 변환)
            ext = Path(src_file).suffix
            dst_filename = f"{sample_id}_cam{ext}"
            dst_path = os.path.join(self.ds.dirs['camera_raw'], dst_filename)
            
            try:
                shutil.copy2(src_file, dst_path)
                new_camera_paths.append(dst_filename)
            except Exception as e:
                log.error(f"Failed copy {src_file}: {e}")
                new_camera_paths.append(None)
                
        # 컬럼 업데이트
        manifest['camera_path'] = new_camera_paths
        self.ds.manifest = manifest
        self.ds.manifest.to_csv(self.ds.manifest_path, index=False)
        log.info(f"Pipeline Ingest Complete. Updated {len(src_files)} images.")
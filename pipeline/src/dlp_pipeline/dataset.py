# dlp-vat-pipeline/pipeline/src/dlp_pipeline/dataset.py
import os
import pandas as pd
from datetime import datetime
from dlp_pipeline.utils import ensure_dir
import logging

log = logging.getLogger(__name__)

class DatasetManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.root = cfg.paths.dataset_root
        self.id = self._determine_id()
        self.path = os.path.join(self.root, self.id)

        # [추가] 로드 모드일 때 폴더 유무 확인
        if self.cfg.dataset.load_id and not os.path.exists(self.path):
            raise FileNotFoundError(f"Dataset ID '{self.id}' not found in {self.root}")

        self.path = ensure_dir(self.path)
        self.manifest_path = os.path.join(self.path, "manifest.csv")
        
        # [추가] Manifest 로드 (기존 파일이 있으면 읽어옴)
        if os.path.exists(self.manifest_path):
            self.manifest = pd.read_csv(self.manifest_path)
            log.info(f"Loaded existing manifest with {len(self.manifest)} records.")
        else:
            self.manifest = pd.DataFrame()
            if self.cfg.dataset.load_id:
                log.warning("Loaded dataset folder but manifest.csv is missing!")
        
        # 기본 폴더 구조 생성
        self.dirs = {
            "mask_input": ensure_dir(os.path.join(self.path, "raw", "mask_input")),
            "mask_gray": ensure_dir(os.path.join(self.path, "raw", "mask_gray")),
            "mask_band": ensure_dir(os.path.join(self.path, "raw", "mask_band")),
            "mask_gray_meta": ensure_dir(os.path.join(self.path, "raw", "mask_gray_meta")),
            "window": ensure_dir(os.path.join(self.path, "raw", "window_1080p")),
            "window_gray": ensure_dir(os.path.join(self.path, "raw", "window_1080p_gray")),
            "camera_raw": ensure_dir(os.path.join(self.path, "raw", "camera_raw")),
            "processed": ensure_dir(os.path.join(self.path, "interim", "processed")),
            "debug": ensure_dir(os.path.join(self.path, "interim", "debug")),
            "rig": ensure_dir(os.path.join(self.path, "rig")),
        }   
        
        log.info(f"Dataset initialized at: {self.path}")
    def _determine_id(self):
        # 1. 커맨드라인/Config에서 load_id가 지정되었으면 최우선 사용
        if self.cfg.dataset.load_id:
            log.info(f"Loading EXISTING Dataset ID: {self.cfg.dataset.load_id}")
            return str(self.cfg.dataset.load_id)

        # 2. 아니면 새로 생성
        if self.cfg.dataset.name:
            return self.cfg.dataset.name
        # timestamp strategy
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def update_manifest(self, new_data):
        """new_data: list of dicts"""
        df_new = pd.DataFrame(new_data)
        if os.path.exists(self.manifest_path):
            self.manifest = pd.read_csv(self.manifest_path)
            # sample_id 기준으로 병합 (여기서는 단순 append 후 중복제거로 구현)
            self.manifest = pd.concat([self.manifest, df_new]).drop_duplicates(subset=['sample_id'], keep='last')
        else:
            self.manifest = df_new
        
        self.manifest.to_csv(self.manifest_path, index=False)
        log.info(f"Manifest updated. Total records: {len(self.manifest)}")

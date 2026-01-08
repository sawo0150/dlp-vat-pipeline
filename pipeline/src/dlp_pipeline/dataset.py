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
        self.id = self._create_id()
        self.path = ensure_dir(os.path.join(self.root, self.id))
        self.manifest_path = os.path.join(self.path, "manifest.csv")
        self.manifest = pd.DataFrame()
        
        # 기본 폴더 구조 생성
        self.dirs = {
            "mask_128": ensure_dir(os.path.join(self.path, "raw", "mask_128")),
            "window": ensure_dir(os.path.join(self.path, "raw", "window_1080p")),
            "camera_raw": ensure_dir(os.path.join(self.path, "raw", "camera_raw")),
            "processed": ensure_dir(os.path.join(self.path, "interim", "processed")),
            "debug": ensure_dir(os.path.join(self.path, "interim", "debug")),
        }
        
        log.info(f"Dataset initialized at: {self.path}")

    def _create_id(self):
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

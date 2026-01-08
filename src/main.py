import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os
from tqdm import tqdm

from src.dataset import DatasetManager
from src.generator import MaskGenerator
from src.projector_interface import ProjectorWindow
from src.preprocessor import ImagePreprocessor
from src.utils import save_image

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    log.info(f"Task Started: {cfg.task.name}")
    log.info(f"Dataset Root: {cfg.paths.dataset_root}")
    
    # 1. Dataset Manager Init (폴더 생성)
    ds = DatasetManager(cfg)
    
    # Task 분기
    if cfg.task.name == "generate":
        run_generate(cfg, ds)
    elif cfg.task.name == "ingest":
        run_ingest(cfg, ds)
    elif cfg.task.name == "preprocess":
        run_preprocess(cfg, ds)
    else:
        log.error("Unknown task!")

def run_generate(cfg, ds):
    """마스크 생성 및 프로젝터 윈도우 생성"""
    gen = MaskGenerator(cfg)
    proj = ProjectorWindow(cfg)
    
    count = cfg.task.num_images
    log.info(f"Generating {count} masks...")
    
    samples = gen.generate_batch(count)
    manifest_data = []

    for idx, item in enumerate(tqdm(samples)):
        sample_id = f"sample_{idx:04d}"
        mask_name = f"{sample_id}_mask.png"
        win_name = f"{sample_id}_window.png"
        
        # Save Mask (128x128)
        save_image(os.path.join(ds.dirs['mask_128'], mask_name), item['image'])
        
        # Process Window (1080p)
        win_img = proj.insert_mask(item['image'])
        save_image(os.path.join(ds.dirs['window'], win_name), win_img)
        
        manifest_data.append({
            "sample_id": sample_id,
            "mask_path": mask_name,
            "window_path": win_name,
            "pattern_type": item['type']
        })
        
    ds.update_manifest(manifest_data)
    log.info("Generation Complete.")

def run_ingest(cfg, ds):
    log.info("Ingest Logic to be implemented (Copying camera files...)")

def run_preprocess(cfg, ds):
    log.info("Preprocess Logic (Running S3/S4 logic...)")
    prep = ImagePreprocessor(cfg)
    # 여기에 manifest 로드 후 Loop 돌면서 prep.process_camera_image() 호출 구현 예정

if __name__ == "__main__":
    main()

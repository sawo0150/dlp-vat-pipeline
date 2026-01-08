# dlp-vat-pipeline/pipeline/src/dlp_pipeline/main.py
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os
from tqdm import tqdm

from dlp_pipeline.dataset import DatasetManager
from dlp_pipeline.generator import MaskGenerator
from dlp_pipeline.graymask_task import GrayMaskTask
from dlp_pipeline.projector_interface import ProjectorWindow
from dlp_pipeline.preprocessor import Preprocessor
from dlp_pipeline.ingestor import DataIngestor
from dlp_pipeline.utils import save_image, seed_everything

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    seed_everything(int(cfg.seed), bool(cfg.deterministic))

    log.info(f"Task Started: {cfg.task.name}")
    log.info(f"Dataset Root: {cfg.paths.dataset_root}")
    
    # 1. Dataset Manager Init (폴더 생성)
    ds = DatasetManager(cfg)
    
    # Task 분기
    if cfg.task.name == "generate":
        run_generate(cfg, ds)
    elif cfg.task.name == "graymask":
        run_graymask(cfg, ds)
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

    # 설정에서 사이즈 가져오기
    gen_size = cfg.rig.mask.base_size # 보통 128

    samples = gen.generate_batch(count)
    manifest_data = []

    for idx, item in enumerate(tqdm(samples)):
        sample_id = f"sample_{idx:05d}"
        mask_name = f"{sample_id}_mask.png"
        win_name = f"{sample_id}_window.png"
        
        # Save Mask (128x128)
        save_image(os.path.join(ds.dirs['mask_input'], mask_name), item['image'])
        
        # Process Window (1080p)
        win_img = proj.insert_mask(item['image'])
        save_image(os.path.join(ds.dirs['window'], win_name), win_img)
        
        manifest_data.append({
            "sample_id": sample_id,
            "mask_path": mask_name,
            "window_path": win_name,
            "pattern_type": item['type'],
            "mask_width": gen_size,
            "mask_height": gen_size
        })
        
    ds.update_manifest(manifest_data)
    log.info("Generation Complete.")

def run_graymask(cfg, ds):
    """
    기존 dataset(load_id)에 존재하는 binary mask들을 읽어서
    grayscale mask + editable band(mask) 생성 후 raw/에 저장하고 manifest 갱신
    """
    log.info("Starting GrayMask Task...")
    task = GrayMaskTask(cfg, ds)
    task.run()

def run_ingest(cfg, ds):
    log.info("Starting Ingest Task...")
    ingestor = DataIngestor(cfg, ds)
    ingestor.run()

def run_preprocess(cfg, ds):
    proc = Preprocessor(cfg, ds)
    proc.run()
    
if __name__ == "__main__":
    main()

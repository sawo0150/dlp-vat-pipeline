# dlp-vat-pipeline/pipeline/src/dlp_pipeline/utils.py
import os
import cv2
import logging
import os, random
import numpy as np

log = logging.getLogger(__name__)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def save_image(path, img):
    # OpenCV는 폴더가 없으면 에러나므로 체크
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, img)

def should_dump_debug(index, cfg):
    """디버그 이미지를 저장할지 결정하는 헬퍼 함수"""
    if not cfg.debug.enable:
        return False
    if index >= cfg.debug.max_images:
        return False
    return (index % cfg.debug.sample_every == 0)

def seed_everything(seed: int, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # OpenCV 자체는 크게 상관 없지만,
    # 멀티스레드로 인해 미세하게 달라질 수 있는 경우가 있어서 잠그고 싶으면:
    try:
        import cv2
        if deterministic:
            cv2.setNumThreads(1)
            cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass
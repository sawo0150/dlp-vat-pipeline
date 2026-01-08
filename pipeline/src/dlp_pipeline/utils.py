# dlp-vat-pipeline/pipeline/src/dlp_pipeline/utils.py
import os
import cv2
import logging

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

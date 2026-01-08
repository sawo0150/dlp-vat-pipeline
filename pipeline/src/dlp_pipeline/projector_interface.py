# dlp-vat-pipeline/pipeline/src/dlp_pipeline/projector_interface.py
import numpy as np
import cv2

class ProjectorWindow:
    def __init__(self, cfg):
        self.p_width = cfg.rig.projector.width
        self.p_height = cfg.rig.projector.height
        self.ins_x = cfg.rig.projector.insert_x
        self.ins_y = cfg.rig.projector.insert_y
        self.win_size = cfg.rig.projector.window_size # 보통 128

    def insert_mask(self, mask_img):
        """S2_random_image2window.m 로직"""
        # 검은 배경 생성
        full_window = np.zeros((self.p_height, self.p_width), dtype=np.uint8)
        
        h, w = mask_img.shape
        # 위치 삽입 (Boundary check 생략 - MVP)
        # Python slicing: y:y+h, x:x+w
        full_window[self.ins_y:self.ins_y+h, self.ins_x:self.ins_x+w] = mask_img
        
        return full_window

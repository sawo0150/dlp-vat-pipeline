import cv2
import numpy as np

class ImagePreprocessor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rig = cfg.rig

    def process_camera_image(self, img):
        """S3_image_transformation.m 로직 포팅"""
        # 1. Transpose
        if self.rig.camera.transpose:
            img = cv2.transpose(img) # or np.transpose
            # img = cv2.flip(img, 1) # 필요시 축 반전 확인

        # 2. Affine Transform (Rotation + Scale)
        # MATLAB: Rotation -1.1 deg, Scale 8/9.98
        center = (img.shape[1]//2, img.shape[0]//2)
        angle = self.rig.camera.rotation_deg
        scale = self.rig.camera.scale_factor
        
        M = cv2.getRotationMatrix2D(center, angle, scale)
        warped = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        # 3. Padding
        pad_size = self.rig.camera.pad # 256
        padded = cv2.copyMakeBorder(warped, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=0)

        # 4. Crop
        # MATLAB imcrop rect is [xmin ymin width height] 1-based
        # Config: [x, y, w, h] 0-based
        cx, cy, cw, ch = self.rig.camera.crop_xywh
        cropped = padded[cy:cy+ch, cx:cx+cw]
        
        # 5. Resize to target if needed (Optional)
        if cropped.shape[0] != self.rig.camera.target_size:
            cropped = cv2.resize(cropped, (self.rig.camera.target_size, self.rig.camera.target_size))
            
        return cropped

    def process_mask_for_training(self, mask_img):
        """S4_image_padding.m: Mask -> 1280 Training Target"""
        # 1. Upscale (Nearest Neighbor to keep sharp edges)
        # 128 -> 1024 (x8)
        upscale_factor = self.rig.mask.upscale_factor
        upscaled = cv2.resize(mask_img, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_NEAREST)
        
        # 2. Pad to 1280
        # 1024 -> 1280 (Need 256 padding total, 128 each side)
        pad = self.rig.mask.pad
        final_mask = cv2.copyMakeBorder(upscaled, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        
        return final_mask

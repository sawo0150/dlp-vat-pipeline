#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyGame Manual Alignment Tool V2 (Multi-Sample Support)
------------------------------------------------------
1. Loads all valid pairs from pairs.csv.
2. Allows iterating through samples using '[' and ']'.
3. Maintains alignment parameters across samples to verify generalization.
4. Saves 'registration_params.json'.

Usage:
    python manual_alignment_pygame_v2.py --dataset /path/to/dataset --mode binary
"""

import argparse
import json
import logging
import os
import sys
import numpy as np
import pandas as pd
import cv2
import pygame

# Configure Logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("manual_align_v2")

# ------------------------------------------------------
# 1. Image Processing Logic
# ------------------------------------------------------

def process_mask_s4(mask_raw: np.ndarray, target_size=1280, upscale_factor=8) -> np.ndarray:
    if mask_raw.ndim == 3:
        mask_raw = cv2.cvtColor(mask_raw, cv2.COLOR_BGR2GRAY)
    
    h, w = mask_raw.shape[:2]
    upscaled = cv2.resize(mask_raw, (w * upscale_factor, h * upscale_factor), interpolation=cv2.INTER_NEAREST)
    
    uh, uw = upscaled.shape
    ph = (target_size - uh) // 2
    pw = (target_size - uw) // 2
    
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    # Safe slicing
    h_end = min(ph+uh, target_size)
    w_end = min(pw+uw, target_size)
    src_h = h_end - ph
    src_w = w_end - pw
    
    if src_h > 0 and src_w > 0:
        canvas[ph:h_end, pw:w_end] = upscaled[:src_h, :src_w]
    
    return canvas

def transform_ld_s3(ld_raw: np.ndarray, params: dict, target_size=1280) -> np.ndarray:
    if ld_raw.ndim == 3:
        ld_raw = cv2.cvtColor(ld_raw, cv2.COLOR_BGR2GRAY)
        
    h, w = ld_raw.shape[:2]
    center = (w // 2, h // 2)
    
    M = cv2.getRotationMatrix2D(center, params['angle'], params['scale'])
    M[0, 2] += params['tx']
    M[1, 2] += params['ty']
    
    warped = cv2.warpAffine(ld_raw, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    
    start_x = (w - target_size) // 2
    start_y = (h - target_size) // 2
    
    # Boundary handling for crop
    if start_x < 0 or start_y < 0:
        pad_w = max(0, -start_x)
        pad_h = max(0, -start_y)
        warped = cv2.copyMakeBorder(warped, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)
        start_x += pad_w
        start_y += pad_h

    cropped = warped[start_y:start_y+target_size, start_x:start_x+target_size]
    return cropped

# ------------------------------------------------------
# 2. PyGame GUI Class
# ------------------------------------------------------

class PyGameAlignerV2:
    def __init__(self, dataset_path, mode):
        self.dataset_path = dataset_path
        self.mode = mode
        
        # 1. Load Pairs List
        self.pairs_df = self._load_pairs()
        self.total_samples = len(self.pairs_df)
        self.current_idx = 0
        
        if self.total_samples == 0:
            raise ValueError("No valid pairs found in pairs.csv")
            
        print(f"[INFO] Found {self.total_samples} samples. Loading index 0...")

        # 2. Init PyGame
        pygame.init()
        self.target_size = 1280
        self.display_scale = 0.7
        self.win_w = int(self.target_size * self.display_scale)
        self.win_h = int(self.target_size * self.display_scale)
        self.ui_sidebar = 320
        
        self.screen = pygame.display.set_mode((self.win_w + self.ui_sidebar, self.win_h))
        pygame.display.set_caption("DLP Alignment Tool V2 (Multi-Sample)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 16)
        self.font_bold = pygame.font.SysFont("Arial", 20, bold=True)
        self.font_big = pygame.font.SysFont("Arial", 24, bold=True)

        # 3. Default Params (MATLAB S3 Defaults)
        self.params = {
            "angle": 1.15,
            "scale": 0.8031, 
            "tx": 56.0,
            "ty": 11.0,
            "opacity": 0.5,
            "threshold": 30  # Default Threshold
        }
        
        # 4. State Containers
        self.mask_raw = None
        self.ld_raw = None
        self.mask_processed = None
        self.surf_overlay = None
        self.sample_info = {}
        self.iou_score = 0.0
                
        # 6. View Control State
        self.zoom = 1.0
        self.pan_x = 0  # Offset from center
        self.pan_y = 0
        self.dragging = False
        self.last_mouse_pos = (0, 0)
        self.view_mode = 0  # 0: Overlay, 1: Diff, 2: Threshold Preview

        # 5. Load First Sample
        self.load_sample(self.current_idx)
        
        self.running = True
        self.needs_update = True

    def _load_pairs(self):
        csv_path = os.path.join(self.dataset_path, "pairing", "pairs.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"pairs.csv not found: {csv_path}")
        df = pd.read_csv(csv_path)
        # Filter only OK
        return df[(df["mode"] == self.mode) & (df["status"] == "OK")].reset_index(drop=True)

    def load_sample(self, idx):
        """Loads Mask and LD for the given index"""
        if idx < 0 or idx >= self.total_samples:
            return

        row = self.pairs_df.iloc[idx]
        mask_name = str(row["mask_name"])
        mask_stem = os.path.splitext(os.path.basename(mask_name))[0]
        
        # Resolve Paths
        pairing_root = os.path.join(self.dataset_path, "pairing")
        if self.mode == "binary":
            mask_path = os.path.join(pairing_root, "binary_mask_128", mask_name)
            ld_dir = os.path.join(pairing_root, "binary_rawLD_1600")
        else:
            mask_path = os.path.join(pairing_root, "gray_mask_128", mask_name)
            ld_dir = os.path.join(pairing_root, "gray_rawLD_1600")

        # Find LD
        ld_path = None
        if os.path.exists(ld_dir):
            for f in os.listdir(ld_dir):
                if f.startswith(mask_stem):
                    ld_path = os.path.join(ld_dir, f)
                    break
        if not ld_path:
             ld_path = os.path.join(ld_dir, str(row["src_ld_file"]))

        # Load Images
        try:
            if not os.path.exists(mask_path): raise FileNotFoundError(f"Mask missing: {mask_path}")
            if not os.path.exists(ld_path): raise FileNotFoundError(f"LD missing: {ld_path}")

            self.mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            self.ld_raw = cv2.imread(ld_path, cv2.IMREAD_GRAYSCALE)
            
            # *** Transpose LD (MATLAB Logic) ***
            self.ld_raw = cv2.transpose(self.ld_raw)

            # Pre-process Mask
            self.mask_processed = process_mask_s4(self.mask_raw, self.target_size)
            
            # Store Info
            self.sample_info = {
                "index": idx,
                "mask_name": mask_name,
                "ld_name": os.path.basename(ld_path)
            }
            self.needs_update = True
            
        except Exception as e:
            print(f"[ERROR] Failed to load sample {idx}: {e}")

    def cv2_to_pygame(self, img):
        if img.ndim == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                          
        # --- Zoom & Pan Logic ---
        h, w = img_rgb.shape[:2]
        
        # Calculate visible area size based on zoom
        visible_w = w / self.zoom
        visible_h = h / self.zoom
        
        # Calculate top-left corner based on pan
        # Center of image is (w/2, h/2). Pan is offset from center.
        cx = w / 2 - self.pan_x
        cy = h / 2 - self.pan_y
        
        x1 = int(max(0, cx - visible_w / 2))
        y1 = int(max(0, cy - visible_h / 2))
        x2 = int(min(w, cx + visible_w / 2))
        y2 = int(min(h, cy + visible_h / 2))
        
        # Crop and Resize to Window
        img_cropped = img_rgb[y1:y2, x1:x2]
        if img_cropped.size == 0: return pygame.Surface((self.win_w, self.win_h))
        
        img_disp = cv2.resize(img_cropped, (self.win_w, self.win_h), interpolation=cv2.INTER_NEAREST)
        return pygame.surfarray.make_surface(img_disp.swapaxes(0, 1))

    def update_transform(self):
        if self.ld_raw is None: return

        ld_warped = transform_ld_s3(self.ld_raw, self.params, self.target_size)

        # --- Thresholding & IoU Calc ---
        _, ld_bin = cv2.threshold(ld_warped, int(self.params['threshold']), 255, cv2.THRESH_BINARY)
        
        # Calculate IoU (Intersection over Union)
        # Mask > 127, LD > 127
        intersection = np.logical_and(self.mask_processed > 127, ld_bin > 127)
        union = np.logical_or(self.mask_processed > 127, ld_bin > 127)
        u_sum = np.sum(union)
        self.iou_score = np.sum(intersection) / u_sum if u_sum > 0 else 0.0
        # -------------------------------

        if self.view_mode == 0:
            # Mode 0: Color Overlay (Green + Magenta)
            mask_color = cv2.cvtColor(self.mask_processed, cv2.COLOR_GRAY2BGR)
            mask_color[:, :, 0] = 0 # B=0
            mask_color[:, :, 2] = 0 # R=0 -> Green
            
            ld_color = cv2.cvtColor(ld_warped, cv2.COLOR_GRAY2BGR)
            ld_color[:, :, 1] = 0   # G=0 -> Magenta
            
            alpha = self.params['opacity']
            result = cv2.addWeighted(mask_color, 1.0 - alpha, ld_color, alpha, 0)
            
        elif self.view_mode == 1:
            # Mode 1: Absolute Difference (High Contrast)
            # Perfect overlap -> Black, Error -> White
            result = cv2.absdiff(self.mask_processed, ld_warped)
            # Normalize to make differences visible
            result = cv2.multiply(result, 3.0) 
            
        else:
            # Mode 2: Threshold Preview (Binary Mask vs Binary LD)
            # Green: Mask (Target), Red: LD (Prediction), Yellow: Overlap
            result = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
            result[:, :, 1] = self.mask_processed  # Green Channel
            result[:, :, 2] = ld_bin               # Red Channel

        self.surf_overlay = self.cv2_to_pygame(result)
        self.needs_update = False

    def auto_tune_threshold(self):
        """Finds the threshold that maximizes IoU for current geometric params"""
        if self.ld_raw is None: return
        
        print("[INFO] Auto-tuning threshold...")
        ld_warped = transform_ld_s3(self.ld_raw, self.params, self.target_size)
        mask_bool = self.mask_processed > 127
        
        best_iou = -1.0
        best_t = self.params['threshold']
        
        # Sweep threshold from 10 to 250 with step 2
        for t in range(10, 251, 2):
            _, bin_img = cv2.threshold(ld_warped, t, 255, cv2.THRESH_BINARY)
            ld_bool = bin_img > 127
            
            inter = np.logical_and(mask_bool, ld_bool).sum()
            union = np.logical_or(mask_bool, ld_bool).sum()
            
            if union > 0:
                iou = inter / union
                if iou > best_iou:
                    best_iou = iou
                    best_t = t
                    
        self.params['threshold'] = best_t
        print(f"[INFO] Found Best Threshold: {best_t} (IoU: {best_iou:.4f})")
        self.needs_update = True

    def draw_ui(self):
        rect = pygame.Rect(self.win_w, 0, self.ui_sidebar, self.win_h)
        pygame.draw.rect(self.screen, (35, 35, 35), rect)
        
        x_off = self.win_w + 20
        y_off = 20
        
        # Title
        self.screen.blit(self.font_big.render("Alignment Tool V2", True, (255, 255, 255)), (x_off, y_off))
        y_off += 40
        
        # Sample Info
        idx_str = f"Sample: {self.current_idx + 1} / {self.total_samples}"
        self.screen.blit(self.font_bold.render(idx_str, True, (0, 255, 255)), (x_off, y_off))
        y_off += 25
        
        name_str = self.sample_info.get("mask_name", "N/A")
        # Truncate if too long
        if len(name_str) > 25: name_str = name_str[:22] + "..."
        self.screen.blit(self.font.render(name_str, True, (200, 200, 200)), (x_off, y_off))
        y_off += 30
        
        pygame.draw.line(self.screen, (100, 100, 100), (x_off, y_off), (self.win_w + self.ui_sidebar - 20, y_off), 1)
        y_off += 20

        # Instructions
        help_texts = [
            "[ [ / ] ] Prev / Next Sample",
            "[ V ] Toggle View (Color/Diff)",
            "[ T ] Auto-Tune Threshold",
            "---------------------------",
            "[Q/A] Rotate (+/- 0.1)",
            "[W/S] Scale  (+/- 0.001)",
            "[Arrows] Move (Translate)",
            "[Z/X] Threshold (+/- 1)",
            "[Mouse] Drag to Pan",
            "[Wheel] Zoom In/Out",
            "[Shift] Fast Move",
            "[O/P] Opacity",
            "---------------------------",
            "[Enter] Save JSON",
            "[Esc] Exit"
        ]
        
        for h in help_texts:
            t = self.font.render(h, True, (180, 180, 180))
            self.screen.blit(t, (x_off, y_off))
            y_off += 22
            
        y_off += 20
        pygame.draw.line(self.screen, (100, 100, 100), (x_off, y_off), (self.win_w + self.ui_sidebar - 20, y_off), 1)
        y_off += 20

        # Values

        if self.view_mode == 0: v_mode_str = "Color Overlay"
        elif self.view_mode == 1: v_mode_str = "Diff (Gray)"
        else: v_mode_str = "Threshold (Bin)"

        val_texts = [
            f"Angle: {self.params['angle']:.2f} deg",
            f"Scale: {self.params['scale']:.4f}",
            f"Trans X: {self.params['tx']:.1f} px",
            f"Trans Y: {self.params['ty']:.1f} px",
            f"Threshold: {self.params['threshold']}",
            f"IoU: {self.iou_score:.4f}",
            f"View: {v_mode_str}",
            f"Zoom: {self.zoom:.2f}x",
            f"Opacity: {self.params['opacity']:.1f}"
        ]
        
        for v in val_texts:
            self.screen.blit(self.font_bold.render(v, True, (0, 255, 0)), (x_off, y_off))
            y_off += 28

    def save_json(self):
        out_data = {
            "dataset_path": os.path.abspath(self.dataset_path),
            "mode": self.mode,
            "target_size": self.target_size,
            "params": {
                "rotation_degree": round(self.params['angle'], 3),
                "scale": round(self.params['scale'], 5),
                "translation_x": round(self.params['tx'], 2),
                "translation_y": round(self.params['ty'], 2),
                "threshold": int(self.params['threshold'])
            },
            "validation_sample": self.sample_info.get("mask_name", "unknown")
        }
        
        out_dir = os.path.join(self.dataset_path, "interim", "processed")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"registration_params_{self.mode}.json")
        
        with open(out_path, "w") as f:
            json.dump(out_data, f, indent=4)
        
        print(f"\n[INFO] Saved parameters to: {out_path}")
        
        # Flash
        pygame.draw.rect(self.screen, (255, 255, 255), (0, 0, self.win_w + self.ui_sidebar, self.win_h))
        pygame.display.flip()
        pygame.time.delay(100)

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False                 
                # --- Mouse Events (Zoom/Pan) ---
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1: # Left Click
                        if event.pos[0] < self.win_w: # Only in image area
                            self.dragging = True
                            self.last_mouse_pos = event.pos
                    elif event.button == 4: # Wheel Up
                        self.zoom = min(self.zoom * 1.1, 10.0)
                        self.needs_update = True
                    elif event.button == 5: # Wheel Down
                        self.zoom = max(self.zoom / 1.1, 1.0)
                        if self.zoom == 1.0: self.pan_x, self.pan_y = 0, 0 # Reset pan on full zoom out
                        self.needs_update = True
                        
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1: self.dragging = False
                    
                elif event.type == pygame.MOUSEMOTION:
                    if self.dragging:
                        dx = event.pos[0] - self.last_mouse_pos[0]
                        dy = event.pos[1] - self.last_mouse_pos[1]
                        # Adjust pan based on zoom level (inverted movement)
                        self.pan_x += dx / self.zoom
                        self.pan_y += dy / self.zoom
                        self.last_mouse_pos = event.pos
                        self.needs_update = True
                elif event.type == pygame.KEYDOWN:
                    mods = pygame.key.get_mods()
                    shift = mods & pygame.KMOD_SHIFT
                    
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_RETURN:
                        self.save_json()
                    
                    # Navigation
                    elif event.key == pygame.K_LEFTBRACKET: # [
                        if self.current_idx > 0:
                            self.current_idx -= 1
                            self.load_sample(self.current_idx)
                    elif event.key == pygame.K_RIGHTBRACKET: # ]
                        if self.current_idx < self.total_samples - 1:
                            self.current_idx += 1
                            self.load_sample(self.current_idx)
                    # View Mode Toggle
                    elif event.key == pygame.K_v:
                        self.view_mode = (self.view_mode + 1) % 3
                        self.needs_update = True
                    # Params
                    elif event.key == pygame.K_q: self.params['angle'] -= 0.1
                    elif event.key == pygame.K_a: self.params['angle'] += 0.1
                    elif event.key == pygame.K_w: self.params['scale'] += 0.001
                    elif event.key == pygame.K_s: self.params['scale'] -= 0.001
                    elif event.key == pygame.K_o: self.params['opacity'] = max(0.0, self.params['opacity'] - 0.1)
                    elif event.key == pygame.K_p: self.params['opacity'] = min(1.0, self.params['opacity'] + 0.1)
                    # Threshold
                    elif event.key == pygame.K_z: self.params['threshold'] = max(0, self.params['threshold'] - 1)
                    elif event.key == pygame.K_x: self.params['threshold'] = min(255, self.params['threshold'] + 1)
                    elif event.key == pygame.K_t: self.auto_tune_threshold()

                    step = 10.0 if shift else 1.0
                    if event.key == pygame.K_LEFT: self.params['tx'] -= step
                    elif event.key == pygame.K_RIGHT: self.params['tx'] += step
                    elif event.key == pygame.K_UP: self.params['ty'] -= step
                    elif event.key == pygame.K_DOWN: self.params['ty'] += step
                    
                    self.needs_update = True

            if self.needs_update:
                self.update_transform()
            
            self.screen.fill((0, 0, 0))
            if self.surf_overlay:
                self.screen.blit(self.surf_overlay, (0, 0))
            
            self.draw_ui()
            pygame.display.flip()
            self.clock.tick(30)
            
        pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to dataset root")
    parser.add_argument("--mode", required=True, choices=["binary", "gray"])
    args = parser.parse_args()

    try:
        app = PyGameAlignerV2(args.dataset, args.mode)
        app.run()
    except Exception as e:
        log.error(f"Error: {e}")
        sys.exit(1)
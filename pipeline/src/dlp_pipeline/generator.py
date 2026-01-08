# dlp-vat-pipeline/pipeline/src/dlp_pipeline/generator.py
import numpy as np
import cv2
import random

class MaskGenerator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.size = cfg.rig.mask.base_size

    def generate_batch(self, count):
        """설정된 비율에 따라 여러 패턴 생성"""
        samples = []
        ratios = self.cfg.generator.mix_ratios
        
        # 단순화를 위해 순차적으로 생성 (실제로는 셔플 가능)
        n_shapes = int(count * ratios.random_shapes)
        n_grid = int(count * ratios.grid)
        n_stripe = count - n_shapes - n_grid # 나머지는 stripe

        for i in range(count):
            if i < n_shapes:
                ptype = "shape"
                img = self._gen_random_shapes()
            elif i < n_shapes + n_grid:
                ptype = "grid"
                img = self._gen_grid()
            else:
                ptype = "stripe"
                img = self._gen_stripe()
                
            samples.append({"type": ptype, "image": img})
        return samples

    def _gen_random_shapes(self):
        # S1_random_image_generation.m 로직 포팅 (간소화)
        img = np.zeros((self.size, self.size), dtype=np.uint8)
        num_shapes = random.randint(3, 10)
        for _ in range(num_shapes):
            color = random.choice([0, 255]) # Binary
            shape_type = random.choice(['circle', 'rect'])
            if shape_type == 'circle':
                c = (random.randint(0, self.size), random.randint(0, self.size))
                r = random.randint(5, 30)
                cv2.circle(img, c, r, color, -1)
            else:
                pt1 = (random.randint(0, self.size), random.randint(0, self.size))
                pt2 = (pt1[0] + random.randint(10, 50), pt1[1] + random.randint(10, 50))
                cv2.rectangle(img, pt1, pt2, color, -1)
        return img

    def _gen_grid(self):
        img = np.zeros((self.size, self.size), dtype=np.uint8)
        spacing = random.choice(self.cfg.generator.grid.spacing)
        thick = random.choice(self.cfg.generator.grid.thickness)
        
        # Vertical lines
        for x in range(0, self.size, spacing):
            cv2.line(img, (x, 0), (x, self.size), 255, thick)
        # Horizontal lines
        for y in range(0, self.size, spacing):
            cv2.line(img, (0, y), (self.size, y), 255, thick)
        return img

    def _gen_stripe(self):
        img = np.zeros((self.size, self.size), dtype=np.uint8)
        period = random.choice(self.cfg.generator.stripe.period)
        orientation = random.choice(self.cfg.generator.stripe.orientation)
        
        if orientation == 'v':
            for x in range(0, self.size, period * 2):
                cv2.rectangle(img, (x, 0), (x + period, self.size), 255, -1)
        else:
            for y in range(0, self.size, period * 2):
                cv2.rectangle(img, (0, y), (self.size, y + period), 255, -1)
        return img

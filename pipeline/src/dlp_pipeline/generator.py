# dlp-vat-pipeline/pipeline/src/dlp_pipeline/generator.py
import numpy as np
import cv2, math
import os, glob
from PIL import Image  # 파일 상단에 import 확인
from tqdm import tqdm

class MaskGenerator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.size = cfg.rig.mask.base_size
        # seed는 main에서 이미 고정해도 되지만, generator 내부도 RNG로 통일하면 재현성이 더 안정적임
        seed = int(getattr(cfg, "seed", 0) or 0)
        self.rng = np.random.default_rng(seed)
        # ImageNet 데이터셋 파일 리스트 미리 로드
        self.imagenet_files = []
        imagenet_cfg = getattr(cfg.generator, "imagenet", None)

        if imagenet_cfg is not None:
            # 1. Root 경로 탐색 (cfg.dataset_root 또는 cfg.paths.dataset_root 시도)
            root_dir = getattr(cfg, "dataset_root", None)
            if root_dir is None and hasattr(cfg, "paths"):
                root_dir = getattr(cfg.paths, "dataset_root", None)
            
            # 2. 상대 경로/폴더명 가져오기
            # mix_imagenet.yaml에 정의된 'dir_name' 혹은 'data_dir' 사용
            rel_dir = getattr(imagenet_cfg, "dir_name", getattr(imagenet_cfg, "data_dir", "../processed_imagenet"))
            
            # 3. 경로 결합
            if root_dir is not None:
                # root가 있으면 결합 (예: /home/.../dlp_datasets/processed_imagenet)
                data_dir = os.path.join(root_dir, rel_dir)
            else:
                # root가 없으면 설정된 경로 그대로 사용 (상대 경로로 가정)
                data_dir = rel_dir

            # 4. 파일 스캔
            print(f"[Info] Scanning ImageNet files in: {os.path.abspath(data_dir)}")
            patterns = [os.path.join(data_dir, "*.png"), os.path.join(data_dir, "*.jpg")]
            for p in patterns:
                self.imagenet_files.extend(glob.glob(p))
            
            if len(self.imagenet_files) == 0:
                print(f"[Warning] No ImageNet files found in {data_dir}. 'imagenet' layer will fallback to shapes.")
            else:
                print(f"[Info] Loaded {len(self.imagenet_files)} ImageNet file paths.")
    
    def generate_batch(self, count):
        """
        (업그레이드)
        - 기존: sample마다 shape/grid/stripe 중 하나만 생성
        - 신규: sample마다 여러 레이어를 뽑아 합성(OR / MAX)하여 다양성 증가
        - 추가 파라미터가 YAML에 없어도 기존처럼 동작하도록 fallback 포함
        """
        samples = []

        # ---- 1) layer mixing 설정 (없으면 fallback) ----
        mix_cfg = getattr(self.cfg.generator, "mix", None)
        if mix_cfg is None:
            # 구버전 fallback: 기존 mix_ratios를 이용해 "한 장당 1개 패턴" 생성
            return self._generate_batch_legacy(count)

        nmin, nmax = self._get_list2(mix_cfg, "num_layers", [1, 3])
        nmin = int(max(1, nmin))
        nmax = int(max(nmin, nmax))

        # [수정] tqdm으로 감싸서 진행바 표시
        for _ in tqdm(range(count), desc="Generating Masks"):
            img, meta = self.generate_one()
            samples.append({"type": meta["type_summary"], "image": img, "meta": meta})
        return samples

    def _generate_batch_legacy(self, count):
        """기존 로직 유지(호환)."""
        samples = []
        ratios = self.cfg.generator.mix_ratios

        # 단순화를 위해 순차적으로 생성 (실제로는 셔플 가능)
        n_shapes = int(count * float(ratios.random_shapes))
        n_grid = int(count * float(ratios.grid))
        n_stripe = count - n_shapes - n_grid
        
        # [수정] tqdm으로 감싸서 진행바 표시
        for i in tqdm(range(count), desc="Generating Legacy Masks"):
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

    def generate_one(self):
        """
        레이어 합성 한 장 생성.
        - layer types는 mix.layer_probs 또는 (fallback) mix_ratios로부터 샘플링
        - 각 layer는 ROI, 회전, jitter, waviness 등 랜덤 파라미터 포함
        - Post-filter: 흰색 비율이 설정 범위를 벗어나면 재시도
        """
        # 1. 필터 설정 로드
        post_cfg = getattr(self.cfg.generator, "post", None)
        filter_cfg = getattr(post_cfg, "filter", None) if post_cfg else None
        
        should_filter = False
        if filter_cfg and getattr(filter_cfg, "enable", False):
            should_filter = True
            min_r = float(getattr(filter_cfg, "min_white_ratio", 0.0))
            max_r = float(getattr(filter_cfg, "max_white_ratio", 1.0))
            max_retries = int(getattr(filter_cfg, "max_retries", 10))
        else:
            max_retries = 1

        # 2. 재시도 루프
        final_img = None
        final_meta = {}
        
        # print("디버깅중!!", max_retries, post_cfg)
        for attempt in range(max_retries):
            # ---- 기존 생성 로직 시작 ----
            mix_cfg = getattr(self.cfg.generator, "mix", None)
            if mix_cfg is None:
                # 안전 fallback
                img = self._gen_random_shapes()
                final_img = img
                final_meta = {"layers": ["shape"], "type_summary": "shape"}
            else:
                nmin, nmax = self._get_list2(mix_cfg, "num_layers", [1, 3])
                n_layers = int(self.rng.integers(int(nmin), int(nmax) + 1))

                layer_types = self._sample_layer_types(n_layers)
                canvas = np.zeros((self.size, self.size), dtype=np.uint8)

                for lt in layer_types:
                    layer = self._render_layer(lt)
                    canvas = np.maximum(canvas, layer)  # OR 합성

                canvas = self._postprocess(canvas)
                
                final_img = canvas
                final_meta = {"layers": layer_types, "type_summary": "+".join(layer_types)}
            # ---- 기존 생성 로직 끝 ----

            # 3. 비율 검사 (필터링 꺼져있으면 바로 break)
            if not should_filter:
                break
                
            # 흰색(255) 비율 계산
            ratio = np.count_nonzero(final_img) / final_img.size
            final_meta['white_ratio'] = ratio # 메타데이터에 기록

            if min_r <= ratio <= max_r:
                # print("조건 만족!!", attempt, ratio)
                # 조건 만족 -> 루프 탈출
                break
            # print("조건 불만족!!", attempt)
            
            # 조건 불만족 -> attempt 반복 (마지막 시도였다면 그냥 그 결과 반환)
            # 디버깅용 로그가 필요하면 print 등을 추가 가능
            # if attempt == max_retries - 1:
            #     print(f"[Warning] Mask generation failed validation after {max_retries} tries. Ratio: {ratio:.3f}")

        return final_img, final_meta

    # ------------------------------------------------------------------
    # Layer sampling / rendering
    # ------------------------------------------------------------------
    def _sample_layer_types(self, n_layers: int):
        """
        YAML 우선순위:
        1) generator.mix.layer_probs가 있으면 그걸 사용
        2) 없으면 generator.mix_ratios를 fallback으로 사용(shape/grid/stripe)
        """
        mix_cfg = getattr(self.cfg.generator, "mix", None)
        probs = None
        if mix_cfg is not None and hasattr(mix_cfg, "layer_probs"):
            # OmegaConf dict-like
            probs = dict(mix_cfg.layer_probs)

        if not probs:
            # fallback to old ratios
            r = self.cfg.generator.mix_ratios
            probs = {
                "shape": float(getattr(r, "random_shapes", 0.4)),
                "grid": float(getattr(r, "grid", 0.3)),
                "stripe": float(getattr(r, "stripe", 0.3)),
                "imagenet": 0.0
            }

        keys = list(probs.keys())
        w = np.array([float(probs[k]) for k in keys], dtype=np.float64)
        w = np.clip(w, 0, None)
        if w.sum() <= 0:
            # 안전장치
            keys = ["shape", "grid", "stripe"]
            w = np.array([0.4, 0.3, 0.3], dtype=np.float64)
        w = w / w.sum()
        sampled = self.rng.choice(keys, size=n_layers, replace=True, p=w).tolist()

       # shuffle 옵션
        mix_cfg = getattr(self.cfg.generator, "mix", None)
        shuffle = bool(getattr(mix_cfg, "shuffle", True)) if mix_cfg is not None else True
        if shuffle:
            self.rng.shuffle(sampled)
        return sampled

    def _render_layer(self, layer_type: str):
        if layer_type in ("shape", "shapes"):
            return self._gen_random_shapes()
        if layer_type == "grid":
            return self._gen_grid_var()
        if layer_type == "stripe":
            return self._gen_stripe_var()
        # 확장 포인트: polygon/blob 등
        if layer_type == "imagenet":
            return self._gen_imagenet()
        return self._gen_random_shapes()

    # ------------------------------------------------------------------
    # Shapes (고도화)
    # ------------------------------------------------------------------
    def _gen_random_shapes(self):
        """
        다양한 도형 생성:
        - circle / rect / triangle / parallelogram / polygon / ellipse / arc
        - 도형을 더하기(255) / 빼기(0) 둘 다 허용 -> 네거티브 공간 생성
        - fill/outline 혼합, 두께 랜덤
        """
        img = np.zeros((self.size, self.size), dtype=np.uint8)

        s_cfg = getattr(self.cfg.generator, "shapes", None)
        nmin, nmax = self._get_list2(s_cfg, "num_shapes", [3, 12])
        num_shapes = int(self.rng.integers(int(nmin), int(nmax) + 1))

        # 기본 도형 타입들
        default_types = ["circle", "rect", "triangle", "parallelogram", "polygon", "ellipse", "arc"]
        types = self._get_list(s_cfg, "types", default_types)

        fill_prob = float(getattr(s_cfg, "fill_prob", 0.8)) if s_cfg is not None else 0.8
        # outline thickness range
        tmin, tmax = self._get_list2(s_cfg, "thickness", [1, 4])
        # size range (대략적인 반경/스케일)
        zmin, zmax = self._get_list2(s_cfg, "size", [4, max(8, self.size // 2)])
        zmin = int(max(2, zmin))
        zmax = int(max(zmin, zmax))

        cutout_prob = float(getattr(s_cfg, "cutout_prob", 0.15)) if s_cfg is not None else 0.15

        for _ in range(num_shapes):
            stype = str(self.rng.choice(types))
            filled = bool(self.rng.random() < fill_prob)
            thickness = -1 if filled else int(self.rng.integers(int(tmin), int(tmax) + 1))

            # add or subtract
            color = 255
            if self.rng.random() < cutout_prob:
                color = 0

            # 공통: 중심점
            cx = int(self.rng.integers(0, self.size))
            cy = int(self.rng.integers(0, self.size))
            scale = int(self.rng.integers(zmin, zmax + 1))

            if stype == "circle":
                r = scale
                cv2.circle(img, (cx, cy), r, color, thickness)

            elif stype == "rect":
                w = int(self.rng.integers(max(2, scale // 2), scale * 2 + 1))
                h = int(self.rng.integers(max(2, scale // 2), scale * 2 + 1))
                x1 = cx - w // 2
                y1 = cy - h // 2
                x2 = x1 + w
                y2 = y1 + h
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            elif stype == "triangle":
                pts = self._rand_polygon(cx, cy, n_vertices=3, radius=scale)
                cv2.fillPoly(img, [pts], color) if filled else cv2.polylines(img, [pts], True, color, int(thickness))

            elif stype == "parallelogram":
                pts = self._rand_parallelogram(cx, cy, scale)
                cv2.fillPoly(img, [pts], color) if filled else cv2.polylines(img, [pts], True, color, int(thickness))

            elif stype == "polygon":
                vmin, vmax = self._get_list2(s_cfg, "poly_vertices", [4, 9])
                n_vertices = int(self.rng.integers(int(vmin), int(vmax) + 1))
                pts = self._rand_polygon(cx, cy, n_vertices=n_vertices, radius=scale)
                cv2.fillPoly(img, [pts], color) if filled else cv2.polylines(img, [pts], True, color, int(thickness))

            elif stype == "ellipse":
                ax = int(self.rng.integers(max(2, scale // 2), scale * 2 + 1))
                ay = int(self.rng.integers(max(2, scale // 2), scale * 2 + 1))
                angle = float(self.rng.uniform(0, 180))
                cv2.ellipse(img, (cx, cy), (ax, ay), angle, 0, 360, color, thickness)

            elif stype == "arc":
                # arc는 "원호/부채꼴" 느낌을 위해 ellipse의 일부 구간만 그리기
                ax = int(self.rng.integers(max(2, scale // 2), scale * 2 + 1))
                ay = int(self.rng.integers(max(2, scale // 2), scale * 2 + 1))
                angle = float(self.rng.uniform(0, 180))
                start = float(self.rng.uniform(0, 360))
                span = float(self.rng.uniform(30, 270))  # 원호 길이
                end = start + span
                # filled arc(부채꼴) 구현: 폴리곤 근사
                if filled:
                    pts = self._arc_sector_points(cx, cy, ax, ay, angle, start, end, steps=24)
                    cv2.fillPoly(img, [pts], color)
                else:
                    cv2.ellipse(img, (cx, cy), (ax, ay), angle, start, end, color, int(thickness))

            else:
                # fallback
                r = scale
                cv2.circle(img, (cx, cy), r, color, thickness)

        # (선택) 전체를 한 번 회전시키면 도형 방향 다양성 증가
        rot_cfg = getattr(s_cfg, "rotate", None) if s_cfg is not None else None
        amin, amax = self._get_list2(rot_cfg, "angle", [-20, 20])
        if rot_cfg is None or bool(getattr(rot_cfg, "enable", True)):
            if self.rng.random() < float(getattr(rot_cfg, "prob", 0.35) if rot_cfg is not None else 0.35):
                ang = float(self.rng.uniform(float(amin), float(amax)))
                img = self._rotate(img, ang)
        # 1) 먼저 binary로 확정
        img = self._binarize(img)

        # 2) [신규] shape boundary speckle 적용 (경계 band에서만 a×b 패치 추가)
        img = self._maybe_add_boundary_speckle(img, s_cfg)

        return self._binarize(img)

    def _gen_grid(self):
        # legacy 유지 (호환)
        img = np.zeros((self.size, self.size), dtype=np.uint8)
        spacing = int(self.rng.choice(list(self.cfg.generator.grid.spacing)))
        thick = int(self.rng.choice(list(self.cfg.generator.grid.thickness)))
        for x in range(0, self.size, spacing):
            cv2.line(img, (x, 0), (x, self.size), 255, thick)
        for y in range(0, self.size, spacing):
            cv2.line(img, (0, y), (self.size, y), 255, thick)
        return img
    

    def _gen_stripe(self):
        # legacy 유지 (호환)
        img = np.zeros((self.size, self.size), dtype=np.uint8)
        period = int(self.rng.choice(list(self.cfg.generator.stripe.period)))
        orientation = str(self.rng.choice(list(self.cfg.generator.stripe.orientation)))
        if orientation == 'v':
            for x in range(0, self.size, period * 2):
                cv2.rectangle(img, (x, 0), (x + period, self.size), 255, -1)
        else:
            for y in range(0, self.size, period * 2):
                cv2.rectangle(img, (0, y), (self.size, y + period), 255, -1)
        return img

    # ------------------------------------------------------------------
    # Grid / Stripe (자유도 업그레이드 버전)
    # ------------------------------------------------------------------
    def _gen_grid_var(self):
        """
        그리드 자유도 증가:
        - spacing jitter (라인마다 간격이 달라짐)
        - thickness jitter (라인마다 두께 달라짐)
        - rotation
        - line dropout / segment dropout
        - ROI(불규칙 영역) 내부에만 그리드 적용
        - 약간의 warp(구불구불 느낌은 stripe 쪽이 더 강함)
        """
        g_cfg = getattr(self.cfg.generator, "grid", None)
        base_spacing = int(self.rng.choice(self._get_list(g_cfg, "spacing", [4, 8, 16])))
        base_thick = int(self.rng.choice(self._get_list(g_cfg, "thickness", [1, 2, 4])))

        # jitter 강도
        jitter = float(getattr(g_cfg, "spacing_jitter", 0.35)) if g_cfg is not None else 0.35
        thick_jitter = float(getattr(g_cfg, "thickness_jitter", 0.6)) if g_cfg is not None else 0.6
        dropout = float(getattr(g_cfg, "drop_prob", 0.05)) if g_cfg is not None else 0.05

        # ROI
        roi = self._maybe_make_roi(getattr(g_cfg, "roi", None), default_enable_prob=0.6)

        img = np.zeros((self.size, self.size), dtype=np.uint8)

        # vertical-like lines
        x = int(self.rng.integers(-base_spacing, base_spacing + 1))
        while x < self.size:
            if self.rng.random() >= dropout:
                thick = max(1, int(round(base_thick * (1 + self.rng.uniform(-thick_jitter, thick_jitter)))))
                # segment dropout: 랜덤 구간을 비워두기
                if self.rng.random() < 0.25:
                    self._draw_segmented_line(img, (x, 0), (x, self.size - 1), thick)
                else:
                    cv2.line(img, (x, 0), (x, self.size - 1), 255, thick)
            step = max(1, int(round(base_spacing * (1 + self.rng.uniform(-jitter, jitter)))))
            x += step

        # horizontal-like lines
        y = int(self.rng.integers(-base_spacing, base_spacing + 1))
        while y < self.size:
            if self.rng.random() >= dropout:
                thick = max(1, int(round(base_thick * (1 + self.rng.uniform(-thick_jitter, thick_jitter)))))
                if self.rng.random() < 0.25:
                    self._draw_segmented_line(img, (0, y), (self.size - 1, y), thick)
                else:
                    cv2.line(img, (0, y), (self.size - 1, y), 255, thick)
            step = max(1, int(round(base_spacing * (1 + self.rng.uniform(-jitter, jitter)))))
            y += step

        # rotation
        amin, amax = self._get_list2(g_cfg, "angle", [-30, 30])
        if self.rng.random() < float(getattr(g_cfg, "rotate_prob", 0.8) if g_cfg is not None else 0.8):
            ang = float(self.rng.uniform(float(amin), float(amax)))
            img = self._rotate(img, ang)

        # ROI 적용
        if roi is not None:
            img = cv2.bitwise_and(img, roi)

        return self._binarize(img)

    def _gen_stripe_var(self):
        """
       스트라이프 자유도 증가:
        - rotation
        - duty cycle(띠 두께 비율)
        - spacing jitter(띠 간격 변동)
        - ROI(불규칙 영역) 내부만 stripe
        - waviness(구불구불): sin warp를 이용해 곡선 스트라이프 생성
        - multi-ROI (여러 영역에 stripe)
        """
        s_cfg = getattr(self.cfg.generator, "stripe", None)
        period = int(self.rng.choice(self._get_list(s_cfg, "period", [4, 8, 16])))
        duty_min, duty_max = self._get_list2(s_cfg, "duty_cycle", [0.35, 0.65])
        duty = float(self.rng.uniform(float(duty_min), float(duty_max)))
        on_width = max(1, int(round(period * duty)))

        jitter = float(getattr(s_cfg, "period_jitter", 0.25)) if s_cfg is not None else 0.25
        # rotation angle
        amin, amax = self._get_list2(s_cfg, "angle", [-60, 60])
        ang = float(self.rng.uniform(float(amin), float(amax)))

        # ROI
        roi = self._maybe_make_roi(getattr(s_cfg, "roi", None), default_enable_prob=0.75)

        # waviness
        w_cfg = getattr(s_cfg, "waviness", None) if s_cfg is not None else None
        w_prob = float(getattr(w_cfg, "enable_prob", 0.55)) if w_cfg is not None else 0.55
        use_wavy = self.rng.random() < w_prob
        amp_min, amp_max = self._get_list2(w_cfg, "amp", [0, max(2, self.size // 10)])
        wl_min, wl_max = self._get_list2(w_cfg, "wavelength", [12, 48])

        # create stripe pattern in rotated coord
        yy, xx = np.mgrid[0:self.size, 0:self.size].astype(np.float32)
        # center
        cx = (self.size - 1) / 2.0
        cy = (self.size - 1) / 2.0
        x0 = xx - cx
        y0 = yy - cy

        # rotate coords: we want stripes roughly along y direction after rotation
        th = math.radians(ang)
        xr = x0 * math.cos(th) + y0 * math.sin(th)
        yr = -x0 * math.sin(th) + y0 * math.cos(th)

        # waviness: warp xr by sin(yr / wavelength)
        if use_wavy:
            amp = float(self.rng.uniform(float(amp_min), float(amp_max)))
            wl = float(self.rng.uniform(float(wl_min), float(wl_max)))
            phase = float(self.rng.uniform(0, 2 * math.pi))
            xr = xr + amp * np.sin((2 * math.pi / max(1e-3, wl)) * yr + phase)

        # spacing jitter: local varying period by slow sinusoid
        if self.rng.random() < 0.6:
            j_amp = float(self.rng.uniform(0.0, jitter))
            j_wl = float(self.rng.uniform(20, 80))
            xr = xr / (1.0 + j_amp * np.sin((2 * math.pi / j_wl) * yr))

        # stripe: modulo with (period*2) but allow varying on_width
        # shift random offset
        offset = float(self.rng.uniform(-period, period))
        xr_shift = xr + offset
        mod = np.mod(xr_shift, float(period * 2))
        stripe = (mod < float(on_width)).astype(np.uint8) * 255

        if roi is not None:
            stripe = cv2.bitwise_and(stripe, roi)

        return stripe

    # ------------------------------------------------------------------
    # Postprocess
    # ------------------------------------------------------------------
    def _postprocess(self, img: np.ndarray):
        """
        합성 후 후처리(선택):
        - invert 확률
        - morphology(open/close/erode/dilate) 랜덤
        """
        p_cfg = getattr(self.cfg.generator, "post", None)
        invert_prob = float(getattr(p_cfg, "invert_prob", 0.08)) if p_cfg is not None else 0.08
        if self.rng.random() < invert_prob:
            img = 255 - img

        m_cfg = getattr(p_cfg, "morph", None) if p_cfg is not None else None
        if m_cfg is not None:
            enable_prob = float(getattr(m_cfg, "enable_prob", 0.30))
            if self.rng.random() < enable_prob:
                ops = self._get_list(m_cfg, "op", ["open", "close", "erode", "dilate"])
                op = str(self.rng.choice(ops))
                kmin, kmax = self._get_list2(m_cfg, "ksize", [1, 5])
                itmin, itmax = self._get_list2(m_cfg, "iters", [1, 2])
                k = int(self.rng.integers(int(kmin), int(kmax) + 1))
                k = max(1, k)
                if k % 2 == 0:
                    k += 1
                iters = int(self.rng.integers(int(itmin), int(itmax) + 1))
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                if op == "open":
                    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iters)
                elif op == "close":
                    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iters)
                elif op == "erode":
                    img = cv2.erode(img, kernel, iterations=iters)
                elif op == "dilate":
                    img = cv2.dilate(img, kernel, iterations=iters)

        return self._binarize(img)

    # ------------------------------------------------------------------
    # ROI / helpers
    # ------------------------------------------------------------------
    def _maybe_make_roi(self, roi_cfg, default_enable_prob=0.7):
        """
        불규칙 ROI 마스크 생성(255 inside, 0 outside)
        - 여러 덩어리 (multi-ROI)
        - polygon/circle 혼합
        - smooth(blur+threshold / morphology)로 "구불구불"한 영역 만들기
        """
        if roi_cfg is None:
            if self.rng.random() > default_enable_prob:
                return None
            # 기본 ROI 파라미터
            num_regions = int(self.rng.integers(1, 3))
            smooth = True
        else:
            enable_prob = float(getattr(roi_cfg, "enable_prob", default_enable_prob))
            if self.rng.random() > enable_prob:
                return None
            rmin, rmax = self._get_list2(roi_cfg, "num_regions", [1, 3])
            num_regions = int(self.rng.integers(int(rmin), int(rmax) + 1))
            smooth = bool(getattr(roi_cfg, "smooth", True))

        roi = np.zeros((self.size, self.size), dtype=np.uint8)

        for _ in range(num_regions):
            if self.rng.random() < 0.55:
                # polygon blob
                cx = int(self.rng.integers(0, self.size))
                cy = int(self.rng.integers(0, self.size))
                rad = int(self.rng.integers(max(6, self.size // 8), max(10, self.size // 2)))
                n_vertices = int(self.rng.integers(4, 10))
                pts = self._rand_polygon(cx, cy, n_vertices=n_vertices, radius=rad)
                cv2.fillPoly(roi, [pts], 255)
            else:
                # circle blob
                cx = int(self.rng.integers(0, self.size))
                cy = int(self.rng.integers(0, self.size))
                rad = int(self.rng.integers(max(5, self.size // 10), max(8, self.size // 3)))
                cv2.circle(roi, (cx, cy), rad, 255, -1)

        # smooth to get irregular boundary
        if smooth:
            # blur + threshold
            k = int(self.rng.integers(3, 9))
            if k % 2 == 0:
                k += 1
            blurred = cv2.GaussianBlur(roi, (k, k), 0)
            thr = int(self.rng.integers(60, 160))
            _, roi = cv2.threshold(blurred, thr, 255, cv2.THRESH_BINARY)

            # morphology cleanup
            if self.rng.random() < 0.7:
                kk = int(self.rng.integers(3, 7))
                if kk % 2 == 0:
                    kk += 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))
                op = "close" if self.rng.random() < 0.5 else "open"
                roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE if op == "close" else cv2.MORPH_OPEN, kernel)

        return self._binarize(roi)

    def _draw_segmented_line(self, img, p1, p2, thick: int):
        """
        라인을 통째로 그리지 말고, 랜덤하게 몇 구간만 그려서 '끊긴 선' 만들기.
        """
       # p1->p2가 수평/수직인 경우만 간단 지원(우리 grid에서만 씀)
        x1, y1 = p1
        x2, y2 = p2
        if x1 == x2:
            # vertical
            y = 0
            while y < self.size:
                seg = int(self.rng.integers(3, max(4, self.size // 3)))
                gap = int(self.rng.integers(2, max(3, self.size // 6)))
                if self.rng.random() < 0.65:
                    cv2.line(img, (x1, y), (x1, min(self.size - 1, y + seg)), 255, thick)
                y += seg + gap
        elif y1 == y2:
            # horizontal
            x = 0
            while x < self.size:
                seg = int(self.rng.integers(3, max(4, self.size // 3)))
                gap = int(self.rng.integers(2, max(3, self.size // 6)))
                if self.rng.random() < 0.65:
                    cv2.line(img, (x, y1), (min(self.size - 1, x + seg), y1), 255, thick)
                x += seg + gap
        else:
            # fallback: normal line
            cv2.line(img, p1, p2, 255, thick)
    def _rotate(self, img: np.ndarray, angle_deg: float):
        h, w = img.shape[:2]
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        return out

    def _binarize(self, img: np.ndarray):
        # 안전하게 0/255로
        return (img > 127).astype(np.uint8) * 255

    # ------------------------------------------------------------------
    # [신규] Boundary Speckle (for shapes)
    # ------------------------------------------------------------------
    def _maybe_add_boundary_speckle(self, mask: np.ndarray, shapes_cfg) -> np.ndarray:
        """
        이미 생성된 binary mask(0/255)에 대해,
        경계 주변에서만 speckle patch를 찍되,
        패치 내부는 0/255가 랜덤하게 섞인 micro-binary가 되도록 한다.

        - anchor를 "edge pixel"에서 직접 뽑으면 패치가 경계에 더 밀착된다.
        - patch 적용 방식:
            overwrite: micro-binary로 덮어쓰기 (흰/검 섞임)
            xor: 기존 mask를 patch 내부에서 토글(0<->255)
            mix: overwrite/xor 혼합
        """
        if shapes_cfg is None:
            return mask

        sp_cfg = getattr(shapes_cfg, "speckle", None)
        if sp_cfg is None:
            return mask

        if not bool(getattr(sp_cfg, "enable", False)):
            return mask

        prob = float(getattr(sp_cfg, "prob", 0.0))
        if self.rng.random() >= prob:
            return mask

        # mask가 비어있거나(전부 0) 혹은 꽉 차있으면(전부 255) 의미가 약하니 빠르게 리턴
        nnz = int(np.count_nonzero(mask))
        if nnz == 0 or nnz == mask.size:
            return mask

        # ---------------------------
        # 1) anchor 픽셀(경계) 뽑기
        # ---------------------------
        anchor_mode = str(getattr(sp_cfg, "anchor", "edge")).lower()

        ys, xs = self._sample_boundary_anchors(mask, sp_cfg, mode=anchor_mode)
        if xs is None or len(xs) == 0:
            return mask

        # 패치 개수
        nmin, nmax = self._get_list2(sp_cfg, "num_patches", [10, 80])
        n_patches = int(self.rng.integers(int(nmin), int(nmax) + 1))
        n_patches = max(1, n_patches)

        # 샘플링 인덱스 (중복 허용해도 되지만, 일단 replace=True로 두면 간단/빠름)
        idxs = self.rng.choice(len(xs), size=n_patches, replace=True)

        # a,b 후보: [1,2,3]
        patch_sizes = self._get_list(sp_cfg, "patch_sizes", [1, 2, 3])
        swap_prob = float(getattr(sp_cfg, "swap_prob", 0.5))

        # patch 내부 micro-binary 확률 (패치마다 랜덤)
        fp_min, fp_max = self._get_list2(sp_cfg, "fill_prob", [0.4, 0.6])
        fp_min = float(fp_min); fp_max = float(fp_max)
        fp_min, fp_max = (min(fp_min, fp_max), max(fp_min, fp_max))

        # mode
        mode = str(getattr(sp_cfg, "mode", "mix")).lower()
        mix_prob = float(getattr(sp_cfg, "mix_prob", 0.5))
 
        out = mask.copy()

        for ii in idxs:
            y = int(ys[ii])
            x = int(xs[ii])

            a = int(self.rng.choice(patch_sizes))
            b = int(self.rng.choice(patch_sizes))
            a = max(1, a)
            b = max(1, b)

            # a×b vs b×a 랜덤 스왑
            if self.rng.random() < swap_prob:
                a, b = b, a

            # -----------------------------------------
            # 2) "경계에 붙게" 배치: anchor 픽셀이 패치에 포함되도록
            #    (anchor를 patch center로 삼기보단, patch가 anchor를 확실히 덮도록)
            # -----------------------------------------
            # anchor를 포함하도록 top-left 계산 (anchor가 패치 내부에 들어가기만 하면 됨)
            x0 = x - int(self.rng.integers(0, a))   # [0, a-1] 만큼 왼쪽으로
            y0 = y - int(self.rng.integers(0, b))   # [0, b-1] 만큼 위로
            x1 = x0 + a
            y1 = y0 + b

            # clip to image bounds
            x0 = max(0, min(self.size - 1, x0))
            y0 = max(0, min(self.size - 1, y0))
            x1 = max(0, min(self.size, x1))
            y1 = max(0, min(self.size, y1))

            if x1 <= x0 or y1 <= y0:
                continue

            # -----------------------------------------
            # 3) 패치 내부 micro-binary 생성
            # -----------------------------------------
            fill_p = float(self.rng.uniform(fp_min, fp_max))
            h = y1 - y0
            w = x1 - x0
            patch = (self.rng.random((h, w)) < fill_p).astype(np.uint8) * 255

            # -----------------------------------------
            # 4) 적용 방식 (overwrite / xor / mix)
            # -----------------------------------------
            use_mode = mode
            if mode == "mix":
                use_mode = "overwrite" if (self.rng.random() < mix_prob) else "xor"

            if use_mode == "overwrite":
                out[y0:y1, x0:x1] = patch
            elif use_mode == "xor":
                # patch의 255인 곳만 토글(0<->255), 0인 곳은 유지
                region = out[y0:y1, x0:x1]
                toggle = (patch > 0)
                region[toggle] = 255 - region[toggle]
                out[y0:y1, x0:x1] = region
            else:
                # fallback: overwrite
                out[y0:y1, x0:x1] = patch

        return out

    def _sample_boundary_anchors(self, mask: np.ndarray, sp_cfg, mode: str = "edge"):
        """
        speckle patch를 경계에 최대한 붙이기 위한 anchor 픽셀 샘플링.
        - edge: morphology gradient(=dilate-erode)로 edge 픽셀을 뽑음 (가장 밀착)
        - band: dilate XOR erode band에서 뽑음 (조금 더 넓게)
        """
        mode = (mode or "edge").lower()
        bw_min, bw_max = self._get_list2(sp_cfg, "band_width", [1, 2])
        bw = int(self.rng.integers(int(bw_min), int(bw_max) + 1))
        bw = max(1, bw)

        k = 2 * bw + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        dil = cv2.dilate(mask, kernel, iterations=1)
        ero = cv2.erode(mask, kernel, iterations=1)

        if mode == "band":
            sel = cv2.bitwise_xor(dil, ero)
        else:
            # edge: morphology gradient (dilate - erode). binary에선 사실상 edge 강하게 나옴
            sel = cv2.subtract(dil, ero)

        ys, xs = np.where(sel > 0)
        if len(xs) == 0:
            return None, None

        # anchor jitter (옵션)
        jitter = float(getattr(sp_cfg, "anchor_jitter", 0.0))
        if jitter > 0:
            j = int(round(jitter))
            xs = np.clip(xs + self.rng.integers(-j, j + 1, size=xs.shape[0]), 0, self.size - 1)
            ys = np.clip(ys + self.rng.integers(-j, j + 1, size=ys.shape[0]), 0, self.size - 1)

        return ys, xs

    def _rand_polygon(self, cx: int, cy: int, n_vertices: int, radius: int):
        """
        중심(cx,cy)에서 각도를 랜덤하게 뽑아 '볼록에 가까운' 폴리곤 생성.
        """
        angles = np.sort(self.rng.uniform(0, 2 * math.pi, size=n_vertices))
        # radius jitter
        r = radius * (0.6 + 0.8 * self.rng.random(size=n_vertices))
        xs = cx + (r * np.cos(angles)).astype(np.int32)
        ys = cy + (r * np.sin(angles)).astype(np.int32)
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        return pts

    def _rand_parallelogram(self, cx: int, cy: int, scale: int):
        """
        평행사변형: 중심 + 두 벡터(u, v)로 4점을 만듦.
        """
        # vector lengths
        a = float(self.rng.uniform(scale * 0.6, scale * 1.8))
        b = float(self.rng.uniform(scale * 0.6, scale * 1.8))
        # angles
        th1 = float(self.rng.uniform(0, 2 * math.pi))
        # v는 u와 완전 수직이 아니게(평행사변형 느낌)
        th2 = th1 + float(self.rng.uniform(math.pi / 6, 5 * math.pi / 6))
        u = np.array([math.cos(th1), math.sin(th1)]) * a
        v = np.array([math.cos(th2), math.sin(th2)]) * b

        p1 = np.array([cx, cy]) - u / 2 - v / 2
        p2 = p1 + u
        p3 = p2 + v
        p4 = p1 + v
        pts = np.stack([p1, p2, p3, p4], axis=0).astype(np.int32)
        return pts

    def _arc_sector_points(self, cx, cy, ax, ay, angle_deg, start_deg, end_deg, steps=24):
        """
        ellipse sector(부채꼴) 폴리곤 근사.
        """
        pts = []
        pts.append([cx, cy])
        th_rot = math.radians(angle_deg)

        start = math.radians(start_deg)
        end = math.radians(end_deg)
        if end < start:
            end += 2 * math.pi
        for t in np.linspace(start, end, steps):
            x = ax * math.cos(t)
            y = ay * math.sin(t)
            # rotate
            xr = x * math.cos(th_rot) - y * math.sin(th_rot)
            yr = x * math.sin(th_rot) + y * math.cos(th_rot)
            pts.append([int(round(cx + xr)), int(round(cy + yr))])
        return np.array(pts, dtype=np.int32)

    # ------------------------------------------------------------------
    # Config helpers (OmegaConf-safe)
    # ------------------------------------------------------------------
    def _get_list(self, node, key: str, default):
        if node is None:
            return list(default)
        if hasattr(node, key):
            v = getattr(node, key)
            # OmegaConf ListConfig -> list()
            try:
                return list(v)
            except Exception:
                return list(default)
        return list(default)

    def _get_list2(self, node, key: str, default2):
        arr = self._get_list(node, key, default2)
        if len(arr) < 2:
            return default2[0], default2[1]
        return arr[0], arr[1]
    
    # ------------------------------------------------------------------
    # [신규] ImageNet Generation Logic
    # ------------------------------------------------------------------
    def _gen_imagenet(self):
        """
        ImageNet 이미지를 로드하고 YAML 설정에 따라 이진화 수행.
        모든 파라미터(임계값, 커널 등)는 cfg.generator.imagenet.params에서 가져옴.
        """
        if not self.imagenet_files:
            return self._gen_random_shapes()

        # 1. 랜덤 이미지 로드
        fpath = self.rng.choice(self.imagenet_files)
        try:
            # [수정] cv2.imread 대신 PIL로 읽어서 numpy 변환 (경고 제거용)
            # PIL은 메타데이터 불일치 경고를 띄우지 않습니다.
            pil_img = Image.open(fpath).convert("L") 
            img = np.array(pil_img)
        except Exception as e:
            print(f"Error loading {fpath}: {e}")
            return self._gen_random_shapes()

        if img.shape[0] != self.size or img.shape[1] != self.size:
            img = cv2.resize(img, (self.size, self.size))

        # 2. Config 로드
        i_cfg = getattr(self.cfg.generator, "imagenet", None)
        
        # Method 선택
        default_methods = ["threshold", "otsu", "adaptive", "canny", "quantize"]
        methods = self._get_list(i_cfg, "methods", default_methods)
        method = str(self.rng.choice(methods))
        
        # Params 섹션 로드
        p_cfg = getattr(i_cfg, "params", None)

        mask = np.zeros_like(img)

        # ------------------- Binarization Methods -------------------
        if method == "threshold":
            t_cfg = getattr(p_cfg, "threshold", None)
            tmin, tmax = self._get_list2(t_cfg, "range", [50, 200])
            thresh = int(self.rng.integers(int(tmin), int(tmax) + 1))
            _, mask = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)

        elif method == "otsu":
            # Otsu는 파라미터가 따로 없으므로 Blur만 살짝 적용
            blur = cv2.GaussianBlur(img, (5, 5), 0)
            _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        elif method == "adaptive":
            a_cfg = getattr(p_cfg, "adaptive", None)
            # Block Size: 홀수여야 함
            candidates = self._get_list(a_cfg, "block_sizes", [11, 15, 21, 31])
            block_size = int(self.rng.choice(candidates))
            
            cmin, cmax = self._get_list2(a_cfg, "c_range", [2, 10])
            C = int(self.rng.integers(int(cmin), int(cmax) + 1))
            
            mask = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C
            )

        elif method == "canny":
            c_cfg = getattr(p_cfg, "canny", None)
            lmin, lmax = self._get_list2(c_cfg, "low_range", [30, 100])
            hmin, hmax = self._get_list2(c_cfg, "high_range", [150, 250])
            
            low = int(self.rng.integers(int(lmin), int(lmax) + 1))
            high = int(self.rng.integers(int(hmin), int(hmax) + 1))
            mask = cv2.Canny(img, low, high)
            
            # Dilation (얇은 선 보강)
            d_prob = float(getattr(c_cfg, "dilate_prob", 0.7)) if c_cfg else 0.7
            if self.rng.random() < d_prob:
                dk_candidates = self._get_list(c_cfg, "dilate_k", [1, 2])
                k = int(self.rng.choice(dk_candidates))
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k+1, 2*k+1))
                mask = cv2.dilate(mask, kernel)

        elif method == "quantize":
            q_cfg = getattr(p_cfg, "quantize", None)
            divs = self._get_list(q_cfg, "divs", [32, 64, 80])
            div = int(self.rng.choice(divs))
            # 짝수 밴드만 선택
            mask = np.where((img // div) % 2 == 0, 255, 0).astype(np.uint8)

        # ------------------- Global Augmentation -------------------
        # Invert
        inv_prob = float(getattr(i_cfg, "invert_prob", 0.5)) if i_cfg else 0.5
        if self.rng.random() < inv_prob:
            mask = 255 - mask

        # Rotate
        rot_prob = float(getattr(i_cfg, "rotate_prob", 0.5)) if i_cfg else 0.5
        if self.rng.random() < rot_prob:
            amin, amax = self._get_list2(i_cfg, "rotate_angle", [-180, 180])
            ang = float(self.rng.uniform(float(amin), float(amax)))
            mask = self._rotate(mask, ang)
        
        # ------------------- [신규] Force Sparse Logic -------------------
        # 설정된 비율보다 흰색이 많으면 강제로 반전시킴
        force_sparse = bool(getattr(i_cfg, "force_sparse", False))
        
        if force_sparse:
            # 255값(흰색)의 비율 계산 (0.0 ~ 1.0)
            white_ratio = np.count_nonzero(mask) / mask.size
            
            # 임계값 (설정 없으면 0.5)
            sparse_thr = float(getattr(i_cfg, "sparse_threshold", 0.5))
            
            # 흰색이 임계값보다 많으면 -> 반전
            if white_ratio > sparse_thr:
                # print("반전!!")
                mask = 255 - mask

        return self._binarize(mask)
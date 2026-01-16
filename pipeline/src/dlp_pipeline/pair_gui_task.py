from __future__ import annotations
import os
import re
import csv
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)


def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _parse_int_stem(fname: str) -> Optional[int]:
    # window0123.png -> 123,  -1.bmp -> -1,  0001.bmp -> 1
    stem = os.path.splitext(os.path.basename(fname))[0]
    m = re.search(r"-?\d+", stem)
    return int(m.group(0)) if m else None


def _sorted_files_by_int(folder: str, exts: Tuple[str, ...]) -> List[str]:
    files = []
    if not os.path.isdir(folder):
        return files
    for f in os.listdir(folder):
        p = os.path.join(folder, f)
        if not os.path.isfile(p):
            continue
        if os.path.splitext(f)[1].lower() in exts:
            files.append(f)
    # 정렬 키: stem 숫자, 없으면 이름
    def key_fn(x):
        v = _parse_int_stem(x)
        return (0, v) if v is not None else (1, x)
    return sorted(files, key=key_fn)


class PairGUITask:
    """
    pygame GUI로 window <-> light 매칭을 수동으로 만들고 CSV로 저장.
    (GUI는 별도 모듈로 분리하지 않고 여기서 최소 동작 구현)
    """

    def __init__(self, cfg, ds_manager):
        self.cfg = cfg
        self.ds = ds_manager

    def run(self):
        mode = str(getattr(self.cfg.task, "mode", "gray")).lower()
        batch = getattr(self.cfg.task, "batch", None)
        mapping_name = str(getattr(self.cfg.task, "mapping_name", "mapping.csv"))

        # 폴더 선택
        if mode == "gray":
            win_root = self.ds.dirs["window_gray"]
            light_root = os.path.join(self.ds.path, "raw", "light_distribution_gray")
            win_exts = (".png",)
            light_exts = (".bmp", ".png")
        else:
            win_root = self.ds.dirs["window"]
            light_root = os.path.join(self.ds.path, "raw", "light_distribution")
            win_exts = (".png",)
            light_exts = (".bmp", ".png")

        # batch 자동 선택
        if batch is None:
            batches = [d.name for d in os.scandir(win_root) if d.is_dir() and d.name.startswith("batch_")]
            batch = sorted(batches)[0] if batches else "batch_0000"

        win_batch = os.path.join(win_root, batch)
        light_batch = os.path.join(light_root, batch)

        win_files = _sorted_files_by_int(win_batch, win_exts)
        light_files = _sorted_files_by_int(light_batch, light_exts)

        if not win_files:
            raise FileNotFoundError(f"No window files in: {win_batch}")
        if not light_files:
            raise FileNotFoundError(f"No light files in: {light_batch}")

        # mapping 저장 위치
        pair_dir = _ensure_dir(os.path.join(self.ds.path, "pairing"))
        mapping_path = os.path.join(pair_dir, mapping_name)

        # 기존 mapping 로드 (이어하기)
        mapping: Dict[str, str] = {}
        if os.path.exists(mapping_path):
            with open(mapping_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    w = r.get("window_file")
                    l = r.get("light_file")
                    if w and l:
                        mapping[w] = l
            log.info(f"Loaded existing mapping: {len(mapping)} pairs from {mapping_path}")

        # GUI 실행
        self._run_pygame(
            mode=mode,
            batch=batch,
            win_batch=win_batch,
            light_batch=light_batch,
            win_files=win_files,
            light_files=light_files,
            mapping=mapping,
            mapping_path=mapping_path,
        )

    def _run_pygame(
        self,
        mode: str,
        batch: str,
        win_batch: str,
        light_batch: str,
        win_files: List[str],
        light_files: List[str],
        mapping: Dict[str, str],
        mapping_path: str,
    ):
        import pygame

        pygame.init()
        W, H = 1500, 800
        screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption(f"Pair GUI ({mode}) - {batch}")

        font = pygame.font.SysFont("consolas", 18)
        clock = pygame.time.Clock()

        # 시작 인덱스/범위
        w_start = int(getattr(self.cfg.task, "window_start", 0))
        w_end = int(getattr(self.cfg.task, "window_end", len(win_files) - 1))
        w_end = min(w_end, len(win_files) - 1)

        # win_files는 정렬된 “파일 리스트 index”이므로
        # window_start/end는 “리스트 인덱스”로 간주(가장 단순)
        wi = max(0, min(w_start, len(win_files) - 1))
        li = 0
        offset = int(getattr(self.cfg.task, "init_offset", 0))

        # light_start를 “파일 리스트 index”로 매핑
        light_start = int(getattr(self.cfg.task, "light_start", 0))
        li = max(0, min(light_start, len(light_files) - 1))

        # preview ECC
        prev_cfg = getattr(self.cfg.task, "preview_registration", {})
        preview_reg = bool(getattr(prev_cfg, "enable", True)) if hasattr(prev_cfg, "enable") else bool(prev_cfg.get("enable", True))
        warp_mode = (prev_cfg.warp_mode if hasattr(prev_cfg, "warp_mode") else prev_cfg.get("warp_mode", "translation")).lower()
        ecc_iters = int(prev_cfg.ecc_iters) if hasattr(prev_cfg, "ecc_iters") else int(prev_cfg.get("ecc_iters", 1500))
        ecc_eps = float(prev_cfg.ecc_eps) if hasattr(prev_cfg, "ecc_eps") else float(prev_cfg.get("ecc_eps", 1e-6))
        gauss = int(prev_cfg.gauss_filt) if hasattr(prev_cfg, "gauss_filt") else int(prev_cfg.get("gauss_filt", 5))

        def warp_mode_cv(s: str) -> int:
            if s == "translation":
                return cv2.MOTION_TRANSLATION
            if s == "euclidean":
                return cv2.MOTION_EUCLIDEAN
            if s == "affine":
                return cv2.MOTION_AFFINE
            if s == "homography":
                return cv2.MOTION_HOMOGRAPHY
            return cv2.MOTION_TRANSLATION

        def load_u8(path: str) -> Optional[np.ndarray]:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            return img

        def to_surface(gray_u8: np.ndarray, max_wh: Tuple[int, int]) -> pygame.Surface:
            h, w = gray_u8.shape
            max_w, max_h = max_wh
            scale = min(max_w / w, max_h / h)
            nw, nh = int(w * scale), int(h * scale)
            resized = cv2.resize(gray_u8, (nw, nh), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            rgb = np.transpose(rgb, (1, 0, 2))  # pygame expects (w,h)
            return pygame.surfarray.make_surface(rgb)

        def ecc_align(ref_u8: np.ndarray, mov_u8: np.ndarray) -> Tuple[np.ndarray, float]:
            ref = ref_u8.astype(np.float32) / 255.0
            mov = mov_u8.astype(np.float32) / 255.0
            if gauss and gauss > 0:
                k = gauss if gauss % 2 == 1 else gauss + 1
                ref = cv2.GaussianBlur(ref, (k, k), 0)
                mov = cv2.GaussianBlur(mov, (k, k), 0)

            wm = warp_mode_cv(warp_mode)
            if wm == cv2.MOTION_HOMOGRAPHY:
                warp = np.eye(3, 3, dtype=np.float32)
            else:
                warp = np.eye(2, 3, dtype=np.float32)

            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ecc_iters, ecc_eps)
            try:
                cc, warp = cv2.findTransformECC(ref, mov, warp, wm, criteria, None, 1)
            except cv2.error:
                return mov_u8, -1.0

            if wm == cv2.MOTION_HOMOGRAPHY:
                aligned = cv2.warpPerspective(
                    mov_u8, warp, (ref_u8.shape[1], ref_u8.shape[0]),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0
                )
            else:
                aligned = cv2.warpAffine(
                    mov_u8, warp, (ref_u8.shape[1], ref_u8.shape[0]),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0
                )
            return aligned, float(cc)

        def save_mapping():
            rows = []
            for w, l in mapping.items():
                rows.append({"window_file": w, "light_file": l})
            rows = sorted(rows, key=lambda r: (_parse_int_stem(r["window_file"]) is None, _parse_int_stem(r["window_file"]) or 0))
            with open(mapping_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["window_file", "light_file"])
                writer.writeheader()
                writer.writerows(rows)
            log.info(f"Saved mapping: {len(rows)} pairs -> {mapping_path}")

        show_help = True
        show_overlay = True  # 정합/오버레이 미리보기

        running = True
        while running:
            clock.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN:
                    k = event.key

                    # quit
                    if k == pygame.K_q or k == pygame.K_ESCAPE:
                        running = False

                    # window index
                    if k == pygame.K_RIGHT:
                        wi = min(wi + 1, w_end)
                    if k == pygame.K_LEFT:
                        wi = max(wi - 1, 0)

                    # light index
                    if k == pygame.K_d:
                        li = min(li + 1, len(light_files) - 1)
                    if k == pygame.K_a:
                        li = max(li - 1, 0)

                    # offset adjust (auto suggestion)
                    if k == pygame.K_p:
                        offset += 1
                    if k == pygame.K_o:
                        offset -= 1

                    # toggle overlay / help
                    if k == pygame.K_h:
                        show_help = not show_help
                    if k == pygame.K_v:
                        show_overlay = not show_overlay

                    # accept match: Enter
                    if k == pygame.K_RETURN:
                        wfile = win_files[wi]
                        lfile = light_files[li]
                        mapping[wfile] = lfile

                    # clear match: Backspace
                    if k == pygame.K_BACKSPACE:
                        wfile = win_files[wi]
                        if wfile in mapping:
                            del mapping[wfile]

                    # save mapping: S
                    if k == pygame.K_s:
                        save_mapping()

                    # auto-fill with current offset: R
                    if k == pygame.K_r:
                        # light index를 window index + offset으로 채움 (범위 내에서만)
                        for wj in range(0, w_end + 1):
                            lj = wj + offset
                            if 0 <= lj < len(light_files):
                                mapping[win_files[wj]] = light_files[lj]

                    # jump to mapped (if exists): M
                    if k == pygame.K_m:
                        wfile = win_files[wi]
                        if wfile in mapping:
                            target = mapping[wfile]
                            try:
                                li = light_files.index(target)
                            except ValueError:
                                pass

            # auto-suggest light index based on offset (단, 사용자가 a/d로 움직이면 유지)
            # 여기서는 "i" 키로만 강제 적용
            keys = pygame.key.get_pressed()
            if keys[pygame.K_i]:
                li2 = wi + offset
                if 0 <= li2 < len(light_files):
                    li = li2

            # load current
            wfile = win_files[wi]
            wpath = os.path.join(win_batch, wfile)

            lfile = light_files[li]
            lpath = os.path.join(light_batch, lfile)

            win_u8 = load_u8(wpath)
            light_u8 = load_u8(lpath)

            # render
            screen.fill((20, 20, 20))

            if win_u8 is None or light_u8 is None:
                txt = font.render("Failed to load images.", True, (255, 80, 80))
                screen.blit(txt, (30, 30))
                pygame.display.flip()
                continue

            # preview align + overlay
            aligned = light_u8
            cc = None
            if preview_reg and show_overlay:
                aligned, cc = ecc_align(win_u8, light_u8)

            # surfaces
            left = to_surface(win_u8, (700, 650))
            right_src = aligned if (preview_reg and show_overlay) else light_u8
            right = to_surface(right_src, (700, 650))

            screen.blit(left, (30, 100))
            screen.blit(right, (770, 100))

            # status text
            mapped = mapping.get(wfile, "")
            t1 = f"[W] {wi}/{w_end}  {wfile}"
            t2 = f"[L] {li}/{len(light_files)-1}  {lfile}"
            t3 = f"offset={offset}  (hold I: suggest light=window+offset)"
            t4 = f"mapped: {mapped}" if mapped else "mapped: (none)"
            t5 = f"ECC cc={cc:.4f}" if (cc is not None) else ""

            y = 10
            for t in [t1, t2, t3, t4, t5]:
                if not t:
                    continue
                screen.blit(font.render(t, True, (230, 230, 230)), (30, y))
                y += 22

            if show_help:
                help_lines = [
                    "Keys: LEFT/RIGHT=window  A/D=light  O/P=offset  (hold I) apply suggested light",
                    "Enter=accept match  Backspace=clear  S=save CSV  R=auto-fill by offset  M=jump to mapped",
                    "V=toggle overlay(aligned preview)  H=toggle help  Q/Esc=quit",
                ]
                yy = 730
                for hl in help_lines:
                    screen.blit(font.render(hl, True, (180, 180, 180)), (30, yy))
                    yy += 20

            pygame.display.flip()

        # 종료 시 저장 한 번 권장 (자동 저장은 취향인데 여기선 안전하게 저장)
        save_mapping()
        pygame.quit()

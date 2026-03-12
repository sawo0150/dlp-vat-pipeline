# pipeline/src/dlp_pipeline/graymask_synth.py
from __future__ import annotations
import json
import math
import random
import hashlib
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional, List

import numpy as np
import cv2
from omegaconf import OmegaConf


def _stable_int_from_str(s: str) -> int:
    """sample_id 같은 문자열로부터 안정적인 정수 seed 생성."""
    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
    return int(h, 16)


def _to_list(x, default):
    if x is None:
        return list(default)
    try:
        return list(x)
    except Exception:
        return list(default)


def _get2(node, key: str, default2):
    if node is None:
        return default2[0], default2[1]
    if hasattr(node, key):
        v = getattr(node, key)
        v = _to_list(v, default2)
        if len(v) >= 2:
            return v[0], v[1]
    return default2[0], default2[1]


def _get(node, key: str, default):
    if node is None:
        return default
    if hasattr(node, key):
        return getattr(node, key)
    return default


class GrayMaskSynthesizer:
    """
    Binary mask(0/255) -> Gray mask(0..255) + editable band(0/255) + meta(json)
    - band 모양 다양화:
      (1) variable width band (저주파 noise로 두께가 위치마다 변동)
      (2) partial band (blob/sector/stripe gating으로 일부 구간만 선택)
    - grayscale 모드 혼합:
      aa_ramp / multi_ring / edge_patches / anisotropic / gamma_quant
    """

    def __init__(self, cfg, base_seed: int = 0):
        self.cfg = cfg
        self.base_seed = int(base_seed)

    def synthesize(
        self,
        bin_mask_u8: np.ndarray,
        sample_id: str = "sample",
        cfg_override: Optional[Dict[str, Any]] = None,
        meta_extra: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        cfg_override:
          - OmegaConf.merge(self.cfg, cfg_override)로 이번 호출에만 override 적용
        meta_extra:
          - meta에 추가로 기록할 필드들(group/profile/epsilon 등)
        """
        old_cfg = self.cfg
        if cfg_override:
            # dict or OmegaConf OK
            self.cfg = OmegaConf.merge(self.cfg, cfg_override)
        try:
            out = self._synthesize_core(bin_mask_u8, sample_id=sample_id, meta_extra=meta_extra)
        finally:
            self.cfg = old_cfg
        return out

    def _synthesize_core(
        self,
        bin_mask_u8: np.ndarray,
        sample_id: str,
        meta_extra: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:

        # per-sample deterministic RNG (order independent)
        seed = self.base_seed + _stable_int_from_str(sample_id)
        rng = np.random.default_rng(seed)

        m01 = (bin_mask_u8 > 127).astype(np.uint8)
        h, w = m01.shape

        # SDF: outside - inside
        dist_in = cv2.distanceTransform(m01, cv2.DIST_L2, 3)
        dist_out = cv2.distanceTransform(1 - m01, cv2.DIST_L2, 3)
        sdf = dist_out - dist_in  # >0 outside, <0 inside

        # ---- band 만들기 (가변 두께 + partial) ----
        band_cfg = getattr(self.cfg.graymask, "band", None)

        w_in_min, w_in_max = _get2(band_cfg, "w_in", [1, 4])
        w_out_min, w_out_max = _get2(band_cfg, "w_out", [1, 4])
        w_in0 = int(rng.integers(int(w_in_min), int(w_in_max) + 1))
        w_out0 = int(rng.integers(int(w_out_min), int(w_out_max) + 1))

        # variable width maps
        vw_cfg = getattr(band_cfg, "variable_width", None) if band_cfg is not None else None
        vw_prob = float(_get(vw_cfg, "enable_prob", 0.85))
        use_vw = rng.random() < vw_prob

        if use_vw:
            alpha_min, alpha_max = _get2(vw_cfg, "alpha", [0.2, 0.9])
            sigma_min, sigma_max = _get2(vw_cfg, "sigma", [2.0, 6.0])
            alpha = float(rng.uniform(float(alpha_min), float(alpha_max)))
            sigma = float(rng.uniform(float(sigma_min), float(sigma_max)))

            noise = rng.normal(0, 1, size=(h, w)).astype(np.float32)
            noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=sigma, sigmaY=sigma)
            noise = noise / (np.std(noise) + 1e-6)
            # normalize to [-1,1]
            noise = np.clip(noise, -2.5, 2.5) / 2.5

            # thickness maps (clip to valid positive)
            w_in_map = np.clip(w_in0 * (1.0 + alpha * noise), 0.5, 10.0).astype(np.float32)
            w_out_map = np.clip(w_out0 * (1.0 + alpha * noise), 0.5, 10.0).astype(np.float32)
            meta_vw = {"enabled": True, "w_in0": w_in0, "w_out0": w_out0, "alpha": alpha, "sigma": sigma}
        else:
            w_in_map = np.full((h, w), float(w_in0), dtype=np.float32)
            w_out_map = np.full((h, w), float(w_out0), dtype=np.float32)
            meta_vw = {"enabled": False, "w_in0": w_in0, "w_out0": w_out0}

        band = (sdf >= -w_in_map) & (sdf <= w_out_map)

        # partial band gating
        part_cfg = getattr(band_cfg, "partial", None) if band_cfg is not None else None
        part_prob = float(_get(part_cfg, "enable_prob", 0.70))
        use_partial = rng.random() < part_prob
        meta_partial = {"enabled": False}

        if use_partial:
            gate = self._make_partial_gate(rng, part_cfg, h, w)
            band = band & (gate > 0)
            meta_partial = {"enabled": True, "mode": gate.shape and gate.dtype.name}

        band_u8 = (band.astype(np.uint8) * 255)

        # band 구성 자체를 일부 흔들거나(추후) / band 기반 노이즈를 추가하기 위한 훅 포인트

        # ---- grayscale 생성 ----
        gray = (m01 * 255).astype(np.float32)

        modes_cfg = getattr(self.cfg.graymask, "modes", None)
        nmin, nmax = _get2(modes_cfg, "num_modes", [1, 3])
        n_modes = int(rng.integers(int(nmin), int(nmax) + 1))

        probs = {}
        probs_cfg = getattr(modes_cfg, "probs", None) if modes_cfg is not None else None
        if probs_cfg is not None:
            probs = dict(probs_cfg)
        else:
            probs = {"aa_ramp": 0.35, "multi_ring": 0.20, "edge_patches": 0.20, "anisotropic": 0.15, "gamma_quant": 0.10}

        mode_names = list(probs.keys())
        weights = np.array([float(probs[m]) for m in mode_names], dtype=np.float64)
        weights = np.clip(weights, 0, None)
        weights = weights / max(1e-9, weights.sum())

        chosen = rng.choice(mode_names, size=min(n_modes, len(mode_names)), replace=False, p=weights).tolist()

        meta_modes: List[Dict[str, Any]] = []

        for mode in chosen:
            if mode == "aa_ramp":
                gray, mm = self._aa_ramp(rng, gray, sdf, band)
            elif mode == "multi_ring":
                gray, mm = self._multi_ring(rng, gray, m01, sdf, band)
            elif mode == "edge_patches":
                gray, mm = self._edge_patches(rng, gray, band)
            elif mode == "anisotropic":
                gray, mm = self._anisotropic(rng, gray, sdf, band)
            elif mode == "gamma_quant":
                gray, mm = self._gamma_quant(rng, gray)
            else:
                mm = {"skipped": True}
            meta_modes.append({mode: mm})

        # ------------------------------------------------------------
        # [NEW] band 내부 픽셀 일부를 0~255로 과감하게 덮어쓰기(sparse override noise)
        # - 모드들이 만든 "구조적인 그레이" 위에 강한 분포 확장을 추가
        # - 기본은 final stage(모드 적용 후)
        # ------------------------------------------------------------
        gray, meta_sparse = self._maybe_sparse_override_noise(rng, gray, band)
        if meta_sparse is not None:
            meta_modes.append({"sparse_override_noise": meta_sparse})


        # ---- band 밖을 binary로 clamp (권장) ----
        clamp_outside = bool(_get(self.cfg.graymask, "clamp_outside_band", True))
        if clamp_outside:
            outside = ~band
            gray[(outside) & (m01 == 1)] = 255.0
            gray[(outside) & (m01 == 0)] = 0.0

        gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)

        meta = {
            "sample_id": sample_id,
            "seed": seed,
            "variable_width": meta_vw,
            "partial_band": meta_partial,
            "chosen_modes": meta_modes,
        }
        if meta_extra:
            meta.update(meta_extra)

        return gray_u8, band_u8, meta

    # ------------------------------------------------------------
    # Sobolev helpers
    # ------------------------------------------------------------
    def make_sobolev_phi(
        self,
        band_u8: np.ndarray,
        sobolev_group_id: str,
        basis_type: str = "patch",
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        band_u8: 0/255
        반환:
          phi: float32, 대략 [-1,1], band 밖은 0
        """
        h, w = band_u8.shape
        band = (band_u8 > 127)

        seed_phi = self.base_seed + _stable_int_from_str(f"{sobolev_group_id}|phi|{basis_type}")
        rng = np.random.default_rng(seed_phi)

        if basis_type == "patch":
            noise = rng.normal(0, 1, size=(h, w)).astype(np.float32)
            sigma = float(rng.uniform(2.0, 6.0))
            noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=sigma, sigmaY=sigma)
            noise = noise / (np.std(noise) + 1e-6)
            phi = np.clip(noise, -2.5, 2.5) / 2.5  # ~[-1,1]
        else:
            # fallback: patch
            noise = rng.normal(0, 1, size=(h, w)).astype(np.float32)
            noise = noise / (np.std(noise) + 1e-6)
            phi = np.clip(noise, -2.5, 2.5) / 2.5

        phi = phi * band.astype(np.float32)
        meta = {
            "seed_phi": int(seed_phi),
            "basis_type": basis_type,
            "phi_min": float(phi.min()) if phi.size else 0.0,
            "phi_max": float(phi.max()) if phi.size else 0.0,
        }
        return phi, meta

    def apply_sobolev_plus_minus(
        self,
        gray_anchor_u8: np.ndarray,
        band_u8: np.ndarray,
        phi: np.ndarray,
        epsilon: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        gray_plus  = gray_anchor + eps * phi
        gray_minus = gray_anchor - eps * phi
        - band 밖은 phi=0이라 동일(anchor 유지)
        """
        g = gray_anchor_u8.astype(np.float32)
        eps = float(epsilon)
        plus = np.clip(g + eps * phi, 0, 255).astype(np.uint8)
        minus = np.clip(g - eps * phi, 0, 255).astype(np.uint8)
        return plus, minus

    def _maybe_sparse_override_noise(self, rng, gray: np.ndarray, band: np.ndarray):
        """
        band 픽셀 중 일부를 강하게 랜덤 값으로 덮어써서 분포 확장.
        반환: (gray_out, meta or None)
        """
        band_cfg = getattr(self.cfg.graymask, "band", None)
        sn_cfg = getattr(band_cfg, "sparse_override_noise", None) if band_cfg is not None else None
        if sn_cfg is None:
            return gray, None

        enable_prob = float(_get(sn_cfg, "enable_prob", 0.0))
        if rng.random() >= enable_prob:
            return gray, None

        rmin, rmax = _get2(sn_cfg, "ratio", [0.05, 0.30])
        ratio = float(rng.uniform(float(rmin), float(rmax)))

        # band 픽셀 인덱스
        idxs = np.flatnonzero(band.reshape(-1))
        n_band = int(idxs.size)
        if n_band <= 0:
            return gray, {"enabled": True, "skipped": "empty_band"}

        k = int(round(n_band * ratio))
        k = max(1, min(k, n_band))

        chosen = rng.choice(idxs, size=k, replace=False)

        value_mode = str(_get(sn_cfg, "value_mode", "uniform"))
        if value_mode == "two_sided":
            low = int(_get(sn_cfg, "two_sided_low", 40))
            high = int(_get(sn_cfg, "two_sided_high", 215))
            # 50:50로 low-side vs high-side
            side = rng.random(k) < 0.5
            vals = np.empty(k, dtype=np.float32)
            vals[side] = rng.integers(0, max(1, low + 1), size=int(side.sum()))
            vals[~side] = rng.integers(min(255, high), 256, size=int((~side).sum()))
        else:
            # uniform default
            vals = rng.integers(0, 256, size=k).astype(np.float32)

        out = gray.copy().reshape(-1)
        out[chosen] = vals
        out = out.reshape(gray.shape)

        # optional blur to avoid ultra-sparkle noise
        blur_cfg = getattr(sn_cfg, "blur_after", None)
        if blur_cfg is not None and bool(_get(blur_cfg, "enable", False)):
            smin, smax = _get2(blur_cfg, "sigma", [0.6, 1.6])
            sigma = float(rng.uniform(float(smin), float(smax)))
            out = cv2.GaussianBlur(out.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
            # blur 후 band 밖으로 새는 것 방지: band 밖은 원래 gray 유지
            out[~band] = gray[~band]
        else:
            sigma = None

        meta = {
            "enabled": True,
            "ratio": ratio,
            "k": k,
            "n_band": n_band,
            "value_mode": value_mode,
            "blur_sigma": sigma,
        }
        return out, meta

    # ----------------------- band partial gate -----------------------
    def _make_partial_gate(self, rng, part_cfg, h: int, w: int) -> np.ndarray:
        # modes weights
        modes = {"blob_gate": 0.50, "sector_gate": 0.25, "stripe_gate": 0.25}
        if part_cfg is not None and hasattr(part_cfg, "modes"):
            modes = dict(part_cfg.modes)

        keys = list(modes.keys())
        ww = np.array([float(modes[k]) for k in keys], dtype=np.float64)
        ww = np.clip(ww, 0, None)
        ww = ww / max(1e-9, ww.sum())
        mode = str(rng.choice(keys, p=ww))

        if mode == "blob_gate":
            return self._gate_blob(rng, getattr(part_cfg, "blob", None), h, w)
        if mode == "sector_gate":
            return self._gate_sector(rng, h, w)
        if mode == "stripe_gate":
            return self._gate_stripe(rng, h, w)

        return np.full((h, w), 255, dtype=np.uint8)

    def _gate_blob(self, rng, blob_cfg, h: int, w: int) -> np.ndarray:
        out = np.zeros((h, w), dtype=np.uint8)
        nb_min, nb_max = _get2(blob_cfg, "num_blobs", [1, 4])
        rad_min, rad_max = _get2(blob_cfg, "radius", [10, 40])
        n = int(rng.integers(int(nb_min), int(nb_max) + 1))
        for _ in range(n):
            cx = int(rng.integers(0, w))
            cy = int(rng.integers(0, h))
            r = int(rng.integers(int(rad_min), int(rad_max) + 1))
            cv2.circle(out, (cx, cy), r, 255, -1)

        smooth = bool(_get(blob_cfg, "smooth", True))
        if smooth:
            k = int(rng.integers(3, 9))
            if k % 2 == 0:
                k += 1
            blur = cv2.GaussianBlur(out, (k, k), 0)
            thr = int(rng.integers(60, 170))
            _, out = cv2.threshold(blur, thr, 255, cv2.THRESH_BINARY)
        return out

    def _gate_sector(self, rng, h: int, w: int) -> np.ndarray:
        # 중심 기준 각도 범위만 남김 (원형 sector gate)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx = (w - 1) / 2.0
        cy = (h - 1) / 2.0
        ang = np.arctan2(yy - cy, xx - cx)  # [-pi,pi]

        center = float(rng.uniform(-math.pi, math.pi))
        width = float(rng.uniform(math.pi / 6, math.pi * 0.9))
        # wrap-safe: compute smallest angle difference
        diff = np.angle(np.exp(1j * (ang - center)))
        keep = (np.abs(diff) < width / 2.0)
        return (keep.astype(np.uint8) * 255)

    def _gate_stripe(self, rng, h: int, w: int) -> np.ndarray:
        # 큰 띠(선형 band)로 일부만 남김. 회전 포함
        out = np.zeros((h, w), dtype=np.uint8)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx = (w - 1) / 2.0
        cy = (h - 1) / 2.0
        x0 = xx - cx
        y0 = yy - cy
        angle = float(rng.uniform(-math.pi/2, math.pi/2))
        th = angle
        xr = x0 * math.cos(th) + y0 * math.sin(th)
        # keep |xr - offset| < thickness
        offset = float(rng.uniform(-w * 0.15, w * 0.15))
        thick = float(rng.uniform(w * 0.10, w * 0.35))
        keep = (np.abs(xr - offset) < thick)
        out[keep] = 255
        return out

    # ----------------------- modes -----------------------
    def _aa_ramp(self, rng, gray, sdf, band):
        aa_cfg = getattr(self.cfg.graymask, "aa_ramp", None)
        bmin, bmax = _get2(aa_cfg, "bias", [-0.6, 0.6])
        smin, smax = _get2(aa_cfg, "slope", [0.35, 1.4])
        bias = float(rng.uniform(float(bmin), float(bmax)))
        slope = float(rng.uniform(float(smin), float(smax)))

        # logistic ramp: inside(neg sdf) -> 255
        x = (-(sdf - bias)) / max(1e-6, slope)
        # x값이 너무 작거나 커지는 것을 방지 (-500 ~ 500 사이로 제한)
        x_safe = np.clip(x, -500.0, 500.0) 
        ramp = 1.0 / (1.0 + np.exp(-x_safe))
        target = 255.0 * ramp

        out = gray.copy()
        out[band] = target[band]
        return out, {"bias": bias, "slope": slope}

    def _multi_ring(self, rng, gray, m01, sdf, band):
        mr_cfg = getattr(self.cfg.graymask, "multi_ring", None)
        kin_min, kin_max = _get2(mr_cfg, "k_in", [1, 4])
        kout_min, kout_max = _get2(mr_cfg, "k_out", [1, 4])
        kin = int(rng.integers(int(kin_min), int(kin_max) + 1))
        kout = int(rng.integers(int(kout_min), int(kout_max) + 1))

        ker_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*kin+1, 2*kin+1))
        ker_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*kout+1, 2*kout+1))
        er = cv2.erode(m01, ker_in, iterations=1)
        di = cv2.dilate(m01, ker_out, iterations=1)

        ring_in = (m01 == 1) & (er == 0)
        ring_out = (di == 1) & (m01 == 0)

        gin_min, gin_max = _get2(mr_cfg, "g_in", [180, 255])
        gout_min, gout_max = _get2(mr_cfg, "g_out", [0, 90])
        g_in = float(rng.uniform(float(gin_min), float(gin_max)))
        g_out = float(rng.uniform(float(gout_min), float(gout_max)))

        out = gray.copy()
        out[ring_in] = g_in
        out[ring_out] = g_out

        # 옵션: ring에서 거리 기반 그라데이션(조금 더 물리적으로)
        grad_prob = float(_get(mr_cfg, "gradient_prob", 0.4))
        if rng.random() < grad_prob:
            # inside ring: sdf in [-kin, 0]
            # outside ring: sdf in [0, kout]
            # linear fade
            inside_mask = ring_in & band
            outside_mask = ring_out & band
            if np.any(inside_mask):
                t = np.clip((-sdf[inside_mask]) / max(1e-6, float(kin)), 0, 1)
                out[inside_mask] = (g_in * (1 - 0.35 * t))  # 안쪽으로 갈수록 조금 덜
            if np.any(outside_mask):
                t = np.clip((sdf[outside_mask]) / max(1e-6, float(kout)), 0, 1)
                out[outside_mask] = (g_out * (1 - 0.35 * t))  # 바깥으로 갈수록 더 어둡게

        return out, {"k_in": kin, "k_out": kout, "g_in": g_in, "g_out": g_out, "grad": (grad_prob,)}

    def _edge_patches(self, rng, gray, band):
        ep_cfg = getattr(self.cfg.graymask, "edge_patches", None)
        smin, smax = _get2(ep_cfg, "sigma", [2.0, 7.0])
        amin, amax = _get2(ep_cfg, "amp", [10, 80])
        sigma = float(rng.uniform(float(smin), float(smax)))
        amp = float(rng.uniform(float(amin), float(amax)))

        noise = rng.normal(0, 1, size=gray.shape).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=sigma, sigmaY=sigma)
        noise = noise / (np.std(noise) + 1e-6)
        out = gray.copy()
        out[band] = np.clip(out[band] + amp * noise[band], 0, 255)

        # 고주파 noise 약간 섞기
        hf_prob = float(_get(ep_cfg, "hf_prob", 0.35))
        if rng.random() < hf_prob:
            hfmin, hfmax = _get2(ep_cfg, "hf_amp", [0, 18])
            hf_amp = float(rng.uniform(float(hfmin), float(hfmax)))
            hf = rng.normal(0, 1, size=gray.shape).astype(np.float32)
            out[band] = np.clip(out[band] + hf_amp * hf[band], 0, 255)
        return out, {"sigma": sigma, "amp": amp, "hf": (hf_prob,)}

    def _anisotropic(self, rng, gray, sdf, band):
        an_cfg = getattr(self.cfg.graymask, "anisotropic", None)
        amin, amax = _get2(an_cfg, "a", [0.05, 0.30])
        harm_min, harm_max = _get2(an_cfg, "harmonic", [1, 2])
        a = float(rng.uniform(float(amin), float(amax)))
        harmonic = int(rng.integers(int(harm_min), int(harm_max) + 1))
        phi = float(rng.uniform(-math.pi, math.pi))

        gy, gx = np.gradient(sdf.astype(np.float32))
        theta = np.arctan2(gy, gx)
        gain = 1.0 + a * np.cos(float(harmonic) * (theta - phi))

        out = gray.copy()
        out[band] = np.clip(out[band] * gain[band], 0, 255)
        return out, {"a": a, "harmonic": harmonic, "phi": phi}

    def _gamma_quant(self, rng, gray):
        gq_cfg = getattr(self.cfg.graymask, "gamma_quant", None)
        gmin, gmax = _get2(gq_cfg, "gamma", [1.6, 2.6])
        gamma = float(rng.uniform(float(gmin), float(gmax)))

        out = 255.0 * np.power(np.clip(gray / 255.0, 0, 1), gamma)

        qprob = float(_get(gq_cfg, "quant_prob", 0.65))
        if rng.random() < qprob:
            levels = _to_list(_get(gq_cfg, "levels", [16, 32, 64, 256]), [16, 32, 64, 256])
            lv = int(rng.choice(levels))
            if lv < 256:
                step = 255.0 / (lv - 1)
                out = np.round(out / step) * step

        return out, {"gamma": gamma, "quant_prob": qprob}

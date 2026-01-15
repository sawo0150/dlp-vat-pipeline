# ml/scripts/build_manifest_from_folders.py

"""
사용 예시
python ml/scripts/build_manifest_from_folders.py \
  --root "/home/wosasa/Desktop/25-1_UROP/전달한 사항들/소분" \
  --out_dir "ml/manifests" \
  --shuffle --seed 42
"""

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(d: Path) -> List[Path]:
    out = []
    for p in d.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out.append(p)
    return sorted(out)


def parse_base_id(stem: str) -> str:
    """
    val3 같은 파일명에서 base id를 뽑는다.
      예) "sample_00012_thr060" -> "sample_00012"
          "abc_threshold_80"    -> "abc"
    규칙:
      - _thr\d+ / _threshold\d+ / -thr\d+ 같은 접미사 제거
    """
    s = stem
    s = re.sub(r"([_\-])(thr|threshold)\d+$", "", s, flags=re.IGNORECASE)
    return s


def relpath(p: Path, base: Path) -> str:
    return str(p.relative_to(base)).replace("\\", "/")


def build_index_by_stem(paths: List[Path]) -> Dict[str, Path]:
    idx = {}
    for p in paths:
        idx[p.stem] = p
    return idx


def write_jsonl(records: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def maybe_shuffle(records: List[dict], seed: int, shuffle: bool) -> List[dict]:
    if not shuffle:
        return records
    rng = random.Random(seed)
    rng.shuffle(records)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="dataset root containing val/val3/val4 folders")
    ap.add_argument("--out_dir", type=str, required=True, help="where to write jsonl manifests")
    ap.add_argument("--val", type=str, default="val")
    ap.add_argument("--val3", type=str, default="val3")
    ap.add_argument("--val4", type=str, default="val4")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--dry_run", action="store_true")

    args = ap.parse_args()
    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve()

    d_val = root / args.val
    d_val3 = root / args.val3
    d_val4 = root / args.val4

    if not d_val.exists():
        raise FileNotFoundError(f"Missing folder: {d_val}")
    if not d_val4.exists():
        raise FileNotFoundError(f"Missing folder: {d_val4}")
    if not d_val3.exists():
        print(f"[WARN] val3 folder not found: {d_val3} (sub1 / thr-based manifests will be empty)")

    masks = list_images(d_val)
    lds = list_images(d_val4)
    thrs = list_images(d_val3) if d_val3.exists() else []

    idx_mask = build_index_by_stem(masks)
    idx_ld = build_index_by_stem(lds)

    # ---------- (1) binary_ld2mask ----------
    rec_binary_ld2mask = []
    for ld in lds:
        sid = ld.stem
        m = idx_mask.get(sid)
        if m is None:
            continue
        rec_binary_ld2mask.append({
            "id": sid,
            "input_path": relpath(ld, root),
            "target_path": relpath(m, root),
            "meta": {"pair": "binary_ld2mask"}
        })

    # ---------- (2) binary_thr2mask ----------
    rec_binary_thr2mask = []
    for t in thrs:
        sid = parse_base_id(t.stem)
        m = idx_mask.get(sid)
        if m is None:
            continue
        rec_binary_thr2mask.append({
            "id": sid,
            "input_path": relpath(t, root),
            "target_path": relpath(m, root),
            "meta": {"pair": "binary_thr2mask", "thr_stem": t.stem}
        })

    # ---------- (3) gray_sub1_thr2ld ----------
    rec_gray_sub1 = []
    for t in thrs:
        sid = parse_base_id(t.stem)
        ld = idx_ld.get(sid)
        if ld is None:
            continue
        rec_gray_sub1.append({
            "id": sid,
            "input_path": relpath(t, root),
            "target_path": relpath(ld, root),
            "meta": {"pair": "gray_sub1_thr2ld", "thr_stem": t.stem}
        })

    # ---------- (4) gray_sub2_ld2mask ----------
    rec_gray_sub2 = []
    for ld in lds:
        sid = ld.stem
        m = idx_mask.get(sid)
        if m is None:
            continue
        rec_gray_sub2.append({
            "id": sid,
            "input_path": relpath(ld, root),
            "target_path": relpath(m, root),
            "meta": {"pair": "gray_sub2_ld2mask"}
        })

    # shuffle/limit
    def postprocess(recs: List[dict], name: str) -> List[dict]:
        recs = maybe_shuffle(recs, seed=args.seed, shuffle=args.shuffle)
        if args.limit and args.limit > 0:
            recs = recs[:args.limit]
        print(f"{name}: {len(recs)} pairs")
        return recs

    rec_binary_ld2mask = postprocess(rec_binary_ld2mask, "binary_ld2mask")
    rec_binary_thr2mask = postprocess(rec_binary_thr2mask, "binary_thr2mask")
    rec_gray_sub1 = postprocess(rec_gray_sub1, "gray_sub1_thr2ld")
    rec_gray_sub2 = postprocess(rec_gray_sub2, "gray_sub2_ld2mask")

    if args.dry_run:
        print("[DRY RUN] not writing files.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(rec_binary_ld2mask, out_dir / "binary_ld2mask.jsonl")
    write_jsonl(rec_binary_thr2mask, out_dir / "binary_thr2mask.jsonl")
    write_jsonl(rec_gray_sub1, out_dir / "gray_sub1_thr2ld.jsonl")
    write_jsonl(rec_gray_sub2, out_dir / "gray_sub2_ld2mask.jsonl")

    print("Done.")
    print(f"Manifests written under: {out_dir}")


if __name__ == "__main__":
    main()

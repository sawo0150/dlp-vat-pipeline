# dlp_pipeline/pair_task.py
import os
import re
import json
import shutil
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd

log = logging.getLogger(__name__)


# -----------------------------
# Helper utilities
# -----------------------------
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_files_sorted(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    files = [f for f in os.listdir(folder) if not f.startswith(".")]
    files.sort(key=natural_key)
    return files


def parse_index(filename: str, regex: str) -> Optional[int]:
    """
    filename에서 index 뽑기.
    regex는 반드시 숫자 캡쳐 그룹을 포함해야 함. (ex: r'(\\d+)' 또는 r'frame_(\\d+)' )
    여러 그룹이면 마지막 그룹을 사용.
    """
    m = re.search(regex, filename)
    if not m:
        return None
    # 마지막 group
    for g in reversed(m.groups()):
        if g is not None and str(g).isdigit():
            return int(g)
    # 혹시 전체가 숫자면
    if m.group(0).isdigit():
        return int(m.group(0))
    return None


def safe_link_or_copy(src: str, dst: str, mode: str = "copy") -> str:
    """
    mode: copy | hardlink | symlink
    실패하면 copy로 fallback.
    """
    ensure_dir(os.path.dirname(dst))
    if os.path.abspath(src) == os.path.abspath(dst):
        return "same"

    # clean existing
    if os.path.exists(dst):
        os.remove(dst)

    try:
        if mode == "hardlink":
            os.link(src, dst)
            return "hardlink"
        if mode == "symlink":
            os.symlink(src, dst)
            return "symlink"
        shutil.copy2(src, dst)
        return "copy"
    except Exception as e:
        log.warning(f"[fallback copy] link/copy failed: {e} | src={src} -> dst={dst}")
        shutil.copy2(src, dst)
        return "copy"


def load_json_if_exists(path: str) -> Optional[dict]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"Failed to read json: {path} ({e})")
        return None


# -----------------------------
# PairTask
# -----------------------------
@dataclass
class PairStats:
    tried: int = 0
    ok: int = 0
    skipped: int = 0
    missing_window: int = 0
    missing_ld: int = 0
    missing_maskmap: int = 0
    missing_maskfile: int = 0
    duplicate_mask: int = 0


class PairTask:
    """
    목적:
      - 사람이 준 batch별 index rule을 yaml(cfg.pairing)로 받고
      - CSV/JSON으로 window_index -> mask_filename 확정
      - pairing/ 아래에 아래 6 폴더로 재구성하여 저장:
        binary_mask_128 / binary_rawLD_1600 / gray_mask_128 / gray_rawLD_1600 / binary_meta / gray_meta
      - pairs.csv + pairing_report.json 생성

    실행 권장:
      task=pair dataset.source=raw dataset.load_id=B1_Shape_Cutout pairing=lab_pair
    """

    def __init__(self, cfg, ds):
        self.cfg = cfg
        self.ds = ds

        self.root = ds.path  # raw dataset의 Bx 폴더
        self.pair_root = ds.dirs["pairing_root"]

        # outputs (DatasetManager가 만들어둠)
        self.out = {
            "binary_mask": ds.dirs["pair_binary_mask_128"],
            "binary_ld": ds.dirs["pair_binary_rawld_1600"],
            "gray_mask": ds.dirs["pair_gray_mask_128"],
            "gray_ld": ds.dirs["pair_gray_rawld_1600"],
            "binary_meta": ds.dirs["pair_binary_meta"],
            "gray_meta": ds.dirs["pair_gray_meta"],
        }

        self.pairs_csv_path = os.path.join(self.pair_root, "pairs.csv")
        self.report_path = os.path.join(self.pair_root, "pairing_report.json")

    # -----------------------------
    # CSV mapping
    # -----------------------------
    def _load_mask_map(self, mode: str, filter_str: str = None) -> Tuple[Dict[int, str], Dict[str, Any]]:
        """
        window_index(int) -> mask_filename(str) 매핑 생성.
        cfg.pairing.modes[mode].csv_* 설정을 최대한 유연하게 지원.

        return:
          mapping, meta_info
        """
        p = self.cfg.pairing
        mcfg = p.modes.get(mode)

        mapping: Dict[int, str] = {}
        meta = {"mode": mode, "csv_path": None, "strategy": None, "rows": 0}

        if mcfg is None:
            return mapping, meta

        csv_path = getattr(mcfg, "csv_path", None)
        if csv_path:
            if not os.path.isabs(csv_path):
                csv_path = os.path.join(self.root, csv_path)
            meta["csv_path"] = csv_path

        if not csv_path or not os.path.exists(csv_path):
            meta["strategy"] = "none"
            return mapping, meta

        df = pd.read_csv(csv_path)
        
        # [추가] Batch 필터링 로직
        window_file_col = getattr(mcfg, "window_file_col", None)
        if filter_str and window_file_col and window_file_col in df.columns:
            # 컬럼 값에 filter_str(예: "batch_0000")이 포함된 행만 필터링
            original_len = len(df)
            df = df[df[window_file_col].astype(str).str.contains(filter_str)]
            log.info(f"[DEBUG] Filtered CSV by '{filter_str}': {original_len} -> {len(df)} rows")

        meta["rows"] = len(df)
        
        # [DEBUG] 실제 로드된 CSV 컬럼 확인
        log.info(f"[DEBUG] CSV Path: {csv_path}")
        log.info(f"[DEBUG] CSV Columns: {list(df.columns)}")

        # column hints
        window_index_col = getattr(mcfg, "window_index_col", "window_index")
        mask_name_col = getattr(mcfg, "mask_name_col", "mask_name")
        window_file_col = getattr(mcfg, "window_file_col", None)

        # [DEBUG] 설정된 타겟 컬럼 확인
        log.info(f"[DEBUG] Config - WindowFileCol: '{window_file_col}', MaskNameCol: '{mask_name_col}'")

        # 1) 가장 선호: window_index_col + mask_name_col
        if window_index_col in df.columns and mask_name_col in df.columns:
            for _, r in df.iterrows():
                try:
                    wi = int(r[window_index_col])
                except Exception:
                    continue
                mn = str(r[mask_name_col])
                if mn and mn != "nan":
                    mapping[wi] = os.path.basename(mn)
            meta["strategy"] = f"cols:{window_index_col}->{mask_name_col}"
            return mapping, meta

        # 2) window filename col이 있으면 filename에서 index 파싱
        if window_file_col and window_file_col in df.columns and mask_name_col in df.columns:
            idx_regex = getattr(mcfg, "index_regex", r"(\d+)")
            log.info(f"[DEBUG] Strategy 2 Active. Regex: {idx_regex}")

            for _, r in df.iterrows():
                wf = str(r[window_file_col])
                wi = parse_index(os.path.basename(wf), idx_regex)
                                
                # [DEBUG] 첫 5개만 파싱 결과 출력해보기
                if len(mapping) < 5:
                     log.info(f"[DEBUG] Parse Sample: File='{wf}' -> Index={wi}")

                if wi is None:
                    continue
                mn = str(r[mask_name_col])
                if mn and mn != "nan":
                    mapping[wi] = os.path.basename(mn)
            meta["strategy"] = f"cols:{window_file_col}(parse)->{mask_name_col}"
            return mapping, meta

        # 3) fallback: sample_id + mask_path 같은 흔한 조합
        # (여기서는 window_index가 없으면 순서 기반밖에 불가능. 순서 fallback은 batch rule에서 수행)
        meta["strategy"] = "unsupported_cols"
        return mapping, meta

    # -----------------------------
    # File index maps for a folder
    # -----------------------------
    def _build_index_to_file(self, folder: str, idx_regex: str) -> Dict[int, str]:
        out: Dict[int, str] = {}
        for f in list_files_sorted(folder):
            idx = parse_index(f, idx_regex)
            if idx is None:
                continue
            # 중복 index는 먼저 나온 걸 유지(또는 overwrite 정책 가능)
            if idx not in out:
                out[idx] = f
        return out

    # -----------------------------
    # Main run
    # -----------------------------
    def run(self):
        p = self.cfg.pairing
        policy = getattr(p, "policy", {})
        copy_mode = getattr(policy, "copy_mode", "copy")
        rename_strategy = getattr(policy, "rename_strategy", "stem")  # stem | full
        dry_run = bool(getattr(policy, "dry_run", False))

        # mode별 stats / used mask tracking
        used_masks = {"binary": set(), "gray": set()}
        stats = {"binary": PairStats(), "gray": PairStats()}
        pair_rows: List[dict] = []

        # mode loop
        for mode in ["binary", "gray"]:
            mcfg = p.modes.get(mode) if hasattr(p, "modes") else None
            if mcfg is None or not bool(getattr(mcfg, "enable", True)):
                log.info(f"[PairTask] mode={mode} disabled. skip.")
                continue

            # roots
            window_root = os.path.join(self.root, str(getattr(mcfg, "window_root", "")).strip("/"))
            ld_root = os.path.join(self.root, str(getattr(mcfg, "ld_root", "")).strip("/"))
            mask_root = os.path.join(self.root, str(getattr(mcfg, "mask_root", "")).strip("/"))
            gray_meta_root = os.path.join(self.root, str(getattr(mcfg, "gray_meta_root", "raw/mask_gray_meta")).strip("/"))

            idx_regex = str(getattr(mcfg, "index_regex", r"(\d+)"))
            ld_ext_override = getattr(mcfg, "ld_ext_override", None)  # 필요하면 강제 확장자

            batches = list(getattr(mcfg, "batches", []))
            if len(batches) == 0:
                log.warning(f"[PairTask] mode={mode} has no batches in config. skip.")
                continue

            for br in batches:
                # batch id / folders
                batch_id = str(getattr(br, "id", "batch_0000"))
                window_batch = str(getattr(br, "window_batch", batch_id))
                ld_batch = str(getattr(br, "ld_batch", batch_id))

                wdir = os.path.join(window_root, window_batch)
                ldir = os.path.join(ld_root, ld_batch)
                # [추가] 배치 루프 안에서 해당 배치용 맵을 따로 만듭니다.
                # filter_str 인자에 batch_id를 넘겨줍니다.
                mask_map, map_meta = self._load_mask_map(mode, filter_str=batch_id)
                log.info(f"[PairTask] mode={mode} batch={batch_id} mask_map size={len(mask_map)}")

                # build index maps
                w_map = self._build_index_to_file(wdir, idx_regex)
                l_map = self._build_index_to_file(ldir, idx_regex)

                # range config
                w_start = int(getattr(getattr(br, "window_index", {}), "start", getattr(br, "window_start", 0)))
                w_end = int(getattr(getattr(br, "window_index", {}), "end", getattr(br, "window_end", -1)))
                w_step = int(getattr(getattr(br, "window_index", {}), "step", getattr(br, "window_step", 1)))

                # mapping rule (offset)
                offset = int(getattr(getattr(br, "mapping", {}), "offset", getattr(br, "offset", 0)))

                # policy per batch
                on_missing_window = str(getattr(policy, "on_missing_window", "skip"))
                on_missing_ld = str(getattr(policy, "on_missing_ld", "skip"))
                on_missing_maskmap = str(getattr(policy, "on_missing_maskmap", "skip"))
                on_missing_maskfile = str(getattr(policy, "on_missing_maskfile", "skip"))
                on_duplicate_mask = str(getattr(policy, "on_duplicate_mask", "error"))

                if w_end < w_start:
                    log.warning(f"[PairTask] bad window range: {w_start}..{w_end} for {mode}/{batch_id}. skip batch.")
                    continue

                log.info(f"[PairTask] mode={mode} batch={batch_id} wdir={wdir} ldir={ldir} range={w_start}..{w_end} step={w_step} offset={offset}")

                for wi in range(w_start, w_end + 1, w_step):
                    stats[mode].tried += 1
                    li = wi + offset

                    wfile = w_map.get(wi)
                    lfile = l_map.get(li)

                    # resolve mask_name from csv mapping
                    mask_name = mask_map.get(wi)

                    status = "OK"
                    reason = ""

                    if wfile is None:
                        stats[mode].missing_window += 1
                        status = "MISSING_WINDOW"
                        reason = f"window index {wi} not found"
                        if on_missing_window == "error":
                            raise FileNotFoundError(f"{mode}/{batch_id}: {reason}")
                        # window는 검증용일 수 있으니 skip 처리
                        stats[mode].skipped += 1
                        pair_rows.append(self._row(mode, batch_id, wi, li, mask_name, wfile, lfile, status, reason))
                        continue

                    if lfile is None:
                        stats[mode].missing_ld += 1
                        status = "MISSING_LD"
                        reason = f"ld index {li} not found"
                        if on_missing_ld == "error":
                            raise FileNotFoundError(f"{mode}/{batch_id}: {reason}")
                        stats[mode].skipped += 1
                        pair_rows.append(self._row(mode, batch_id, wi, li, mask_name, wfile, lfile, status, reason))
                        continue

                    if not mask_name:
                        stats[mode].missing_maskmap += 1
                        status = "MISSING_MASKMAP"
                        # [DEBUG] 왜 매핑 실패했는지 첫 번째 케이스만 출력
                        if stats[mode].missing_maskmap == 1:
                            log.info(f"[DEBUG] Map Lookup Failed! WindowIndex={wi}. Map Size={len(mask_map)}. Keys Sample={list(mask_map.keys())[:5]}")
                        reason = f"mask mapping for window index {wi} not found"
                        if on_missing_maskmap == "error":
                            raise KeyError(f"{mode}/{batch_id}: {reason}")
                        stats[mode].skipped += 1
                        pair_rows.append(self._row(mode, batch_id, wi, li, mask_name, wfile, lfile, status, reason))
                        continue

                    # duplicate check
                    if mask_name in used_masks[mode]:
                        # [수정] 정책에 따라 분기 처리
                        if on_duplicate_mask == "error":
                            stats[mode].duplicate_mask += 1
                            reason = f"mask {mask_name} already used"
                            raise ValueError(f"{mode}/{batch_id}: {reason}")
                        
                        elif on_duplicate_mask == "skip":
                            stats[mode].duplicate_mask += 1
                            stats[mode].skipped += 1
                            status = "DUPLICATE_MASK"
                            reason = f"mask {mask_name} already used (skipped)"
                            pair_rows.append(self._row(mode, batch_id, wi, li, mask_name, wfile, lfile, status, reason))
                            continue
                        
                        elif on_duplicate_mask == "allow":
                            # 중복이지만 허용함 -> 진행 (stats만 기록하고 continue 안 함)
                            # 단, used_masks에 이미 있으므로 추가 동작은 필요 없음
                            pass

                    # source paths
                    src_mask = os.path.join(mask_root, mask_name)
                    src_ld = os.path.join(ldir, lfile)

                    if not os.path.exists(src_mask):
                        stats[mode].missing_maskfile += 1
                        status = "MISSING_MASKFILE"
                        reason = f"mask file not found: {src_mask}"
                        if on_missing_maskfile == "error":
                            raise FileNotFoundError(f"{mode}/{batch_id}: {reason}")
                        stats[mode].skipped += 1
                        pair_rows.append(self._row(mode, batch_id, wi, li, mask_name, wfile, lfile, status, reason))
                        continue

                    # destination filenames
                    mask_stem, mask_ext = os.path.splitext(mask_name)
                    ld_src_ext = os.path.splitext(lfile)[1]

                    if ld_ext_override:
                        ld_dst_ext = str(ld_ext_override)
                    else:
                        ld_dst_ext = ld_src_ext

                    if rename_strategy == "full":
                        # rawLD를 mask 파일명과 "완전히 동일"하게 (확장자도 mask_ext)
                        ld_dst_name = mask_name
                    else:
                        # stem만 동일, rawLD 확장자는 유지
                        ld_dst_name = mask_stem + ld_dst_ext

                    # outputs
                    if mode == "binary":
                        dst_mask = os.path.join(self.out["binary_mask"], mask_name)
                        dst_ld = os.path.join(self.out["binary_ld"], ld_dst_name)
                        dst_meta = os.path.join(self.out["binary_meta"], f"{mask_stem}.json")
                    else:
                        dst_mask = os.path.join(self.out["gray_mask"], mask_name)
                        dst_ld = os.path.join(self.out["gray_ld"], ld_dst_name)
                        dst_meta = os.path.join(self.out["gray_meta"], f"{mask_stem}.json")

                    if not dry_run:
                        # write files
                        safe_link_or_copy(src_mask, dst_mask, copy_mode)
                        safe_link_or_copy(src_ld, dst_ld, copy_mode)

                        # meta: csv row + mapping info + optional gray meta json
                        meta_obj = {
                            "dataset_id": os.path.basename(self.root),
                            "mode": mode,
                            "batch_id": batch_id,
                            "window_index": wi,
                            "ld_index": li,
                            "src": {
                                "window_dir": wdir,
                                "window_file": wfile,
                                "ld_dir": ldir,
                                "ld_file": lfile,
                                "mask_root": mask_root,
                                "mask_file": mask_name,
                            },
                            "dst": {
                                "mask_file": os.path.basename(dst_mask),
                                "rawld_file": os.path.basename(dst_ld),
                            },
                        }

                        # try attach gray meta json (if exists)
                        if mode == "gray":
                            cand = os.path.join(gray_meta_root, f"{mask_stem}.json")
                            meta_obj["gray_meta_src"] = cand
                            meta_obj["gray_meta"] = load_json_if_exists(cand)

                        # write meta json
                        ensure_dir(os.path.dirname(dst_meta))
                        with open(dst_meta, "w", encoding="utf-8") as f:
                            json.dump(meta_obj, f, ensure_ascii=False, indent=2)

                    used_masks[mode].add(mask_name)
                    stats[mode].ok += 1

                    pair_rows.append(self._row(
                        mode, batch_id, wi, li, mask_name, wfile, lfile,
                        "OK", ""
                    ))

        # write pairs.csv + report
        self._write_pairs_and_report(pair_rows, stats, dry_run=dry_run)

        log.info("[PairTask] Done.")
        log.info(f"[PairTask] binary: {stats['binary']}")
        log.info(f"[PairTask] gray: {stats['gray']}")

    def _row(self, mode, batch_id, wi, li, mask_name, wfile, lfile, status, reason):
        return {
            "dataset_id": os.path.basename(self.root),
            "mode": mode,
            "batch_id": batch_id,
            "window_index": wi,
            "ld_index": li,
            "mask_name": mask_name if mask_name else "",
            "src_window_file": wfile if wfile else "",
            "src_ld_file": lfile if lfile else "",
            "status": status,
            "reason": reason,
        }

    def _write_pairs_and_report(self, pair_rows: List[dict], stats: Dict[str, PairStats], dry_run: bool):
        df = pd.DataFrame(pair_rows)
        # deterministic sort
        if not df.empty:
            df = df.sort_values(by=["mode", "batch_id", "window_index"], kind="stable").reset_index(drop=True)

        if not dry_run:
            df.to_csv(self.pairs_csv_path, index=False)

        report = {
            "dataset_id": os.path.basename(self.root),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "dry_run": bool(dry_run),
            "outputs": {
                "pairing_root": self.pair_root,
                "pairs_csv": self.pairs_csv_path,
                "report_json": self.report_path,
            },
            "stats": {
                "binary": stats["binary"].__dict__,
                "gray": stats["gray"].__dict__,
            },
        }

        if not dry_run:
            with open(self.report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

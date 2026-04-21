#!/usr/bin/env python
"""
Publish trained WLASL-2000 artifacts into sign_language_web backend.

What it does:
- Validates expected checkpoints exist and contain required keys.
- Validates class dimension (expected 2000 by default).
- Copies checkpoints to backend/app/checkpoints/wlasl2000.
- Optionally applies optimized ensemble weights into backend config_wlasl2000.py.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import torch


DEFAULT_MODELS = ["hybrid", "tcn", "bilstm", "transformer"]
REQUIRED_KEYS = ["model_state_dict", "model_kwargs"]


def resolve_paths() -> Tuple[Path, Path, Path, Path, Path]:
    script_dir = Path(__file__).resolve().parent
    source_ckpt_dir = script_dir / "checkpoints_wlasl2000"
    source_runs_dir = script_dir / "runs_wlasl2000"
    web_root = script_dir.parent / "sign_language_web" / "sign_language_web"
    target_ckpt_dir = web_root / "backend" / "app" / "checkpoints" / "wlasl2000"
    target_cfg = web_root / "backend" / "app" / "config_wlasl2000.py"
    target_class_list = web_root / "backend" / "app" / "class_lists" / "wlasl2000_class_list.txt"
    return source_ckpt_dir, source_runs_dir, target_ckpt_dir, target_cfg, target_class_list


def validate_class_list(path: Path, expected_classes: int) -> Tuple[bool, str]:
    if not path.exists():
        return False, f"class list not found: {path}"

    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != expected_classes:
        return False, f"class list size mismatch: got {len(lines)}, expected {expected_classes} ({path})"

    return True, f"ok class list {path.name} classes={len(lines)}"


def read_ckpt_num_classes(state_dict: Dict[str, torch.Tensor], fallback: int | None) -> int | None:
    if fallback:
        return int(fallback)
    for key, val in state_dict.items():
        if (
            ("classifier" in key or "fc" in key)
            and key.endswith(".weight")
            and getattr(val, "ndim", 0) == 2
        ):
            return int(val.shape[0])
    return None


def validate_checkpoint(path: Path, expected_classes: int) -> Tuple[bool, str]:
    if not path.exists():
        return False, f"missing checkpoint: {path}"

    ckpt = torch.load(path, map_location="cpu")
    for k in REQUIRED_KEYS:
        if k not in ckpt:
            return False, f"checkpoint missing key '{k}': {path.name}"

    state_dict = ckpt["model_state_dict"]
    if not isinstance(state_dict, dict) or not state_dict:
        return False, f"invalid model_state_dict in {path.name}"

    num_classes = read_ckpt_num_classes(state_dict, ckpt.get("num_classes"))
    if num_classes != expected_classes:
        return False, (
            f"num_classes mismatch in {path.name}: got {num_classes}, "
            f"expected {expected_classes}"
        )

    return True, f"ok {path.name} num_classes={num_classes}"


def apply_weights_if_available(source_runs_dir: Path, target_cfg: Path) -> Tuple[bool, str]:
    opt_json = source_runs_dir / "ensemble_wlasl_optimized.json"
    if not opt_json.exists():
        return False, "optimized weight file not found; skipped"

    with opt_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    weights = data.get("optimized_weights_list")
    if not isinstance(weights, list) or not weights:
        return False, "optimized_weights_list missing or empty; skipped"

    content = target_cfg.read_text(encoding="utf-8")
    new_line = f"ENSEMBLE_WEIGHTS = [{', '.join(f'{float(w):.6f}' for w in weights)}]"
    pattern = re.compile(r"^ENSEMBLE_WEIGHTS\s*=\s*\[[^\]]*\].*$", re.MULTILINE)
    if not pattern.search(content):
        return False, "could not find ENSEMBLE_WEIGHTS assignment in target config"

    updated = pattern.sub(new_line, content)
    if updated != content:
        target_cfg.write_text(updated, encoding="utf-8")
        return True, f"applied optimized weights to {target_cfg}"

    return True, "optimized weights already applied"


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish WLASL2000 checkpoints to sign_language_web")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--expected-classes", type=int, default=2000)
    parser.add_argument("--apply-weights-if-available", action="store_true")
    parser.add_argument(
        "--class-list-src",
        type=Path,
        default=None,
        help="Optional source class list path to sync into backend app/class_lists",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_ckpt_dir, source_runs_dir, target_ckpt_dir, target_cfg, target_class_list = resolve_paths()

    print("[Publish] Source checkpoints:", source_ckpt_dir)
    print("[Publish] Target checkpoints:", target_ckpt_dir)
    print("[Publish] Target config:", target_cfg)
    print("[Publish] Target class list:", target_class_list)

    if not source_ckpt_dir.exists():
        print(f"[ERROR] Source checkpoint dir does not exist: {source_ckpt_dir}")
        return 2

    target_ckpt_dir.mkdir(parents=True, exist_ok=True)

    all_ok = True
    for model in args.models:
        src = source_ckpt_dir / f"{model}_best.pt"
        ok, msg = validate_checkpoint(src, args.expected_classes)
        print("[Validate]", msg)
        if not ok:
            all_ok = False
            continue

        dst = target_ckpt_dir / src.name
        if args.dry_run:
            print(f"[DryRun] Would copy {src} -> {dst}")
        else:
            shutil.copy2(src, dst)
            print(f"[Copy] {src.name} -> {dst}")

    if args.class_list_src is not None:
        if not args.class_list_src.exists():
            print(f"[ERROR] class-list source does not exist: {args.class_list_src}")
            return 2
        ok, msg = validate_class_list(args.class_list_src, args.expected_classes)
        print("[ClassList-Validate]", msg)
        if not ok:
            return 1

        if args.dry_run:
            print(f"[DryRun] Would copy {args.class_list_src} -> {target_class_list}")
        else:
            target_class_list.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.class_list_src, target_class_list)
            print(f"[ClassList-Copy] {args.class_list_src} -> {target_class_list}")

    ok, msg = validate_class_list(target_class_list, args.expected_classes)
    print("[ClassList-Target]", msg)
    if not ok:
        all_ok = False

    if args.apply_weights_if_available:
        if args.dry_run:
            print("[DryRun] Would attempt optimized weight apply")
        else:
            changed, msg = apply_weights_if_available(source_runs_dir, target_cfg)
            tag = "[Weights]" if changed else "[Weights-Skip]"
            print(tag, msg)

    if not all_ok:
        print("[Result] Publish incomplete due to validation failures")
        return 1

    print("[Result] Publish completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

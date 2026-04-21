#!/usr/bin/env python
"""
orchestrate_wlasl_robust.py
Robust end-to-end pipeline:
  tune -> train -> optimize ensemble -> evaluate ensemble

Features:
- Variant-aware (wlasl100, wlasl2000)
- Per-model retry with exponential backoff
- Resume state saved to RUNS_DIR
- Partial success handling (requires at least N successful models)
- Optional auto-apply optimized ensemble weights into config

Usage:
  python orchestrate_wlasl_robust.py --variant wlasl2000 --tune-trials 50 --trial-epochs 20 --train-epochs 300
  python orchestrate_wlasl_robust.py --variant wlasl2000 --resume --skip-tune
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent


def run_cmd(cmd: List[str], log_file: Path) -> Tuple[int, str]:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    text = f"\n[CMD] {' '.join(cmd)}\n"
    print(text, flush=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(text)

    p = subprocess.Popen(
        cmd,
        cwd=str(SCRIPT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: List[str] = []
    assert p.stdout is not None
    for line in p.stdout:
        print(line, end="", flush=True)
        lines.append(line)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(line)
    p.wait()
    out = "".join(lines)
    return p.returncode, out


def sleep_backoff(base_seconds: int, attempt: int) -> None:
    wait_s = base_seconds * (2 ** max(attempt - 1, 0))
    print(f"[Retry] Waiting {wait_s}s before retry...", flush=True)
    time.sleep(wait_s)


def load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def parse_opt_weights(opt_json: Path) -> List[float] | None:
    if not opt_json.exists():
        return None
    data = load_json(opt_json, {})
    weights = data.get("optimized_weights_list")
    if isinstance(weights, list) and weights:
        return [float(w) for w in weights]
    return None


def apply_weights_to_config(config_file: Path, weights: List[float]) -> bool:
    if not config_file.exists():
        return False

    content = config_file.read_text(encoding="utf-8")
    new_line = f"ENSEMBLE_WEIGHTS = [{', '.join(f'{w:.4f}' for w in weights)}]"
    pattern = re.compile(r"^ENSEMBLE_WEIGHTS\s*=\s*\[[^\]]*\].*$", re.MULTILINE)

    if not pattern.search(content):
        return False

    updated = pattern.sub(new_line, content)
    if updated == content:
        return True

    config_file.write_text(updated, encoding="utf-8")
    return True


@dataclass
class PipelineConfig:
    variant: str
    models: List[str]
    tune_trials: int
    trial_epochs: int
    train_epochs: int
    max_retries: int
    retry_wait_seconds: int
    min_success_models: int
    skip_tune: bool
    skip_train: bool
    skip_optimize: bool
    skip_eval: bool
    resume: bool
    auto_apply_weights: bool
    dry_run: bool


def make_paths(variant: str):
    import config_selector

    cfg = config_selector.activate(variant)
    runs_dir = Path(cfg.RUNS_DIR)
    ckpt_dir = Path(cfg.CHECKPOINT_DIR)
    state = runs_dir / f"robust_state_{variant}.json"
    summary = runs_dir / f"robust_summary_{variant}.json"
    log_file = runs_dir / f"robust_pipeline_{variant}.log"
    return cfg, runs_dir, ckpt_dir, state, summary, log_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Robust WLASL orchestrator")
    parser.add_argument("--variant", choices=["wlasl100", "wlasl2000"], default="wlasl2000")
    parser.add_argument("--models", nargs="+", default=["hybrid", "tcn", "bilstm", "transformer"])
    parser.add_argument("--tune-trials", type=int, default=50)
    parser.add_argument("--trial-epochs", type=int, default=20)
    parser.add_argument("--train-epochs", type=int, default=300)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-wait-seconds", type=int, default=15)
    parser.add_argument("--min-success-models", type=int, default=2)
    parser.add_argument("--skip-tune", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-optimize", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--auto-apply-weights", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Use 1-epoch dry run in training phase")
    args = parser.parse_args()

    cfg = PipelineConfig(
        variant=args.variant,
        models=args.models,
        tune_trials=args.tune_trials,
        trial_epochs=args.trial_epochs,
        train_epochs=args.train_epochs,
        max_retries=args.max_retries,
        retry_wait_seconds=args.retry_wait_seconds,
        min_success_models=args.min_success_models,
        skip_tune=args.skip_tune,
        skip_train=args.skip_train,
        skip_optimize=args.skip_optimize,
        skip_eval=args.skip_eval,
        resume=args.resume,
        auto_apply_weights=args.auto_apply_weights,
        dry_run=args.dry_run,
    )

    _, runs_dir, ckpt_dir, state_path, summary_path, log_file = make_paths(cfg.variant)

    state = {
        "variant": cfg.variant,
        "models": cfg.models,
        "tuned": {},
        "trained": {},
        "optimized": False,
        "evaluated": False,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if cfg.resume and state_path.exists():
        state = load_json(state_path, state)

    print("=" * 72)
    print("Robust WLASL Orchestrator")
    print("=" * 72)
    print(f"variant={cfg.variant} models={cfg.models}")
    print(f"tune_trials={cfg.tune_trials} trial_epochs={cfg.trial_epochs} train_epochs={cfg.train_epochs}")
    print(f"retries={cfg.max_retries} min_success_models={cfg.min_success_models}")

    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

    py = sys.executable

    # Phase 1: Tuning
    if not cfg.skip_tune:
        for model in cfg.models:
            if state.get("tuned", {}).get(model) == "ok":
                continue
            ok = False
            for attempt in range(1, cfg.max_retries + 2):
                cmd = [
                    py,
                    str(SCRIPT_DIR / "tune.py"),
                    "--variant",
                    cfg.variant,
                    "--model",
                    model,
                    "--n-trials",
                    str(cfg.tune_trials),
                    "--trial-epochs",
                    str(cfg.trial_epochs),
                ]
                if cfg.resume:
                    cmd.append("--resume-study")

                code, _ = run_cmd(cmd, log_file)
                if code == 0:
                    ok = True
                    break
                if attempt <= cfg.max_retries:
                    sleep_backoff(cfg.retry_wait_seconds, attempt)

            state.setdefault("tuned", {})[model] = "ok" if ok else "failed"
            save_json(state_path, state)

    # Phase 2: Training
    if not cfg.skip_train:
        for model in cfg.models:
            if state.get("trained", {}).get(model) == "ok":
                continue
            ok = False
            for attempt in range(1, cfg.max_retries + 2):
                cmd = [
                    py,
                    str(SCRIPT_DIR / "train_wlasl.py"),
                    "--variant",
                    cfg.variant,
                    "--model",
                    model,
                    "--epochs",
                    str(cfg.train_epochs),
                ]
                if cfg.dry_run:
                    cmd.append("--dry-run")
                code, _ = run_cmd(cmd, log_file)
                ckpt_ok = (ckpt_dir / f"{model}_best.pt").exists()
                if code == 0 and ckpt_ok:
                    ok = True
                    break
                if attempt <= cfg.max_retries:
                    sleep_backoff(cfg.retry_wait_seconds, attempt)

            state.setdefault("trained", {})[model] = "ok" if ok else "failed"
            save_json(state_path, state)

    successful_models = [m for m, st in state.get("trained", {}).items() if st == "ok"]

    # Phase 3: Optimize ensemble
    if not cfg.skip_optimize and len(successful_models) >= cfg.min_success_models:
        cmd = [py, str(SCRIPT_DIR / "optimize_ensemble_wlasl.py"), "--variant", cfg.variant]
        code, _ = run_cmd(cmd, log_file)
        state["optimized"] = code == 0

        if state["optimized"] and cfg.auto_apply_weights:
            opt_json = runs_dir / "ensemble_wlasl_optimized.json"
            weights = parse_opt_weights(opt_json)
            if weights:
                config_name = "config_wlasl2000.py" if cfg.variant == "wlasl2000" else "config_wlasl100.py"
                state["weights_applied"] = apply_weights_to_config(SCRIPT_DIR / config_name, weights)
            else:
                state["weights_applied"] = False

        save_json(state_path, state)
    elif not cfg.skip_optimize:
        print(
            f"[WARN] Skip optimize: only {len(successful_models)} successful models; "
            f"requires >= {cfg.min_success_models}"
        )

    # Phase 4: Evaluate ensemble
    if not cfg.skip_eval:
        cmd = [py, str(SCRIPT_DIR / "eval_ensemble_wlasl.py"), "--variant", cfg.variant]
        code, _ = run_cmd(cmd, log_file)
        state["evaluated"] = code == 0
        save_json(state_path, state)

    state["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    state["successful_models"] = successful_models
    save_json(state_path, state)

    summary = {
        "pipeline": "robust_wlasl",
        "variant": cfg.variant,
        "models": cfg.models,
        "successful_models": successful_models,
        "tune_trials": cfg.tune_trials,
        "trial_epochs": cfg.trial_epochs,
        "train_epochs": cfg.train_epochs,
        "skip_tune": cfg.skip_tune,
        "skip_train": cfg.skip_train,
        "skip_optimize": cfg.skip_optimize,
        "skip_eval": cfg.skip_eval,
        "dry_run": cfg.dry_run,
        "optimized": state.get("optimized", False),
        "evaluated": state.get("evaluated", False),
        "weights_applied": state.get("weights_applied", None),
        "state_file": str(state_path),
        "log_file": str(log_file),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(summary_path, summary)

    print("\n" + "=" * 72)
    print("Robust orchestrator finished")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

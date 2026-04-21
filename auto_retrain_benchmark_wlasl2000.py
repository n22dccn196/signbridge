#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs_wlasl2000"
SUMMARY_OUT = RUNS_DIR / "auto_retrain_benchmark_summary.json"


def run_cmd(cmd: list[str]) -> int:
    print("[CMD]", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, cwd=str(ROOT))
    proc.wait()
    return proc.returncode


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def extract_model_metrics(data: dict[str, Any]) -> dict[str, Any]:
    if not data:
        return {}

    return {
        "best_val_acc": data.get("best_val") if data.get("best_val") is not None else data.get("best_val_acc"),
        "test_acc": data.get("test_top1") if data.get("test_top1") is not None else data.get("test_acc"),
        "test_top5": data.get("test_top5"),
        "best_epoch": data.get("best_epoch"),
        "total_time_s": data.get("total_time_s"),
        "num_params": data.get("num_params"),
        "best_params": data.get("best_params"),
        "checkpoint": data.get("best_ckpt") or data.get("checkpoint"),
    }


def collect_summary() -> dict[str, Any]:
    model_names = ["hybrid", "tcn", "bilstm", "transformer"]
    model_results = {}

    for name in model_names:
        path = RUNS_DIR / f"{name}_wlasl2000_result.json"
        model_results[name] = extract_model_metrics(read_json(path))

    ensemble_result = read_json(RUNS_DIR / "wlasl2000_ensemble_result.json")
    optimized = read_json(RUNS_DIR / "ensemble_wlasl_optimized.json")
    robust_summary = read_json(RUNS_DIR / "robust_summary_wlasl2000.json")

    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": model_results,
        "ensemble": {
            "test_acc": (
                ensemble_result.get("ensemble_test_top1")
                if ensemble_result.get("ensemble_test_top1") is not None
                else (ensemble_result.get("test_top1") if ensemble_result.get("test_top1") is not None else ensemble_result.get("test_acc"))
            ),
            "test_top5": (
                ensemble_result.get("ensemble_test_top5")
                if ensemble_result.get("ensemble_test_top5") is not None
                else ensemble_result.get("test_top5")
            ),
            "weights": ensemble_result.get("weights"),
        },
        "optimized_weights": optimized.get("optimized_weights_list"),
        "optimized_acc": optimized.get("optimized_accuracy") if optimized.get("optimized_accuracy") is not None else optimized.get("best_accuracy"),
        "pipeline": robust_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full WLASL2000 retrain + benchmark summary collection")
    parser.add_argument("--mode", choices=["full", "quick", "collect-only"], default="full")
    parser.add_argument("--tune-trials", type=int, default=50)
    parser.add_argument("--trial-epochs", type=int, default=20)
    parser.add_argument("--train-epochs", type=int, default=300)
    args = parser.parse_args()

    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

    if args.mode in {"full", "quick"}:
        retrain_cmd = [
            sys.executable,
            str(ROOT / "orchestrate_wlasl_robust.py"),
            "--variant",
            "wlasl2000",
            "--auto-apply-weights",
        ]
        if args.mode == "full":
            retrain_cmd += [
                "--tune-trials",
                str(args.tune_trials),
                "--trial-epochs",
                str(args.trial_epochs),
                "--train-epochs",
                str(args.train_epochs),
            ]
        else:
            retrain_cmd += ["--skip-tune", "--train-epochs", "20"]

        code = run_cmd(retrain_cmd)
        if code != 0:
            print(f"[ERROR] orchestrate_wlasl_robust failed with code={code}")
            return code

        publish_cmd = [
            sys.executable,
            str(ROOT / "publish_wlasl2000_to_sign_language_web.py"),
            "--apply-weights-if-available",
        ]
        code = run_cmd(publish_cmd)
        if code != 0:
            print(f"[ERROR] publish_wlasl2000_to_sign_language_web failed with code={code}")
            return code

    summary = collect_summary()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] Summary written: {SUMMARY_OUT}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

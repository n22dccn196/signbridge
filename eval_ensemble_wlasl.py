#!/usr/bin/env python
"""
eval_ensemble_wlasl.py — Evaluate WLASL ensemble on test set.

Usage:
    python eval_ensemble_wlasl.py --variant wlasl100
    python eval_ensemble_wlasl.py --variant wlasl2000
    python eval_ensemble_wlasl.py --variant wlasl2000 --weights 0.30 0.30 0.20 0.20
"""
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import config_selector


def _detect_variant(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--variant", choices=["wlasl100", "wlasl2000"], default="wlasl100")
    args, _ = parser.parse_known_args(argv)
    return args.variant


ACTIVE_VARIANT = _detect_variant(sys.argv[1:])
RESULT_TAG = "wlasl2000" if ACTIVE_VARIANT == "wlasl2000" else "wlasl"
config_selector.activate(ACTIVE_VARIANT)

from config import (
    CLASSES, NUM_CLASSES, CHECKPOINT_DIR, RUNS_DIR,
    ENSEMBLE_MODELS, ENSEMBLE_WEIGHTS, BATCH_SIZE,
)
from dataset import get_loaders
from ensemble import Ensemble


def evaluate_ensemble(weights=None, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    w = weights or ENSEMBLE_WEIGHTS

    print(f"\n{'='*60}")
    print(f"  WLASL Ensemble Evaluation ({ACTIVE_VARIANT})")
    print(f"  Models: {ENSEMBLE_MODELS}")
    print(f"  Weights: {w}")
    print(f"{'='*60}\n")

    ens = Ensemble(
        model_names=ENSEMBLE_MODELS,
        weights=w,
        mode="soft",
        device=device,
    )

    _, _, tel = get_loaders(batch_size=BATCH_SIZE)
    print(f"  Test samples: {len(tel.dataset)}")

    correct, total = 0, 0
    top5_correct = 0
    top10_correct = 0
    conf_sum = 0.0
    ent_sum = 0.0
    conf_correct_sum = 0.0
    conf_wrong_sum = 0.0
    ent_correct_sum = 0.0
    ent_wrong_sum = 0.0
    n_correct = 0
    n_wrong = 0
    class_correct = np.zeros(NUM_CLASSES)
    class_total = np.zeros(NUM_CLASSES)
    class_conf_sum = np.zeros(NUM_CLASSES, dtype=np.float64)
    class_ent_sum = np.zeros(NUM_CLASSES, dtype=np.float64)

    for x, y in tel:
        probs = ens.predict_probs(x)
        if probs.ndim == 1:
            probs = probs[np.newaxis, :]

        preds = probs.argmax(axis=1)
        for i in range(len(y)):
            label = y[i].item()
            p = probs[i]
            conf = float(np.max(p))
            ent = float(-np.sum(p * np.log(p + 1e-12)))

            class_total[label] += 1
            class_conf_sum[label] += conf
            class_ent_sum[label] += ent
            conf_sum += conf
            ent_sum += ent

            if preds[i] == label:
                correct += 1
                class_correct[label] += 1
                conf_correct_sum += conf
                ent_correct_sum += ent
                n_correct += 1
            else:
                conf_wrong_sum += conf
                ent_wrong_sum += ent
                n_wrong += 1
            # Top-5
            top5_idx = np.argsort(probs[i])[-5:]
            if label in top5_idx:
                top5_correct += 1
            # Top-10
            top10_idx = np.argsort(probs[i])[-10:]
            if label in top10_idx:
                top10_correct += 1
            total += 1

    acc_t1 = 100 * correct / max(total, 1)
    acc_t5 = 100 * top5_correct / max(total, 1)
    acc_t10 = 100 * top10_correct / max(total, 1)
    mean_conf = conf_sum / max(total, 1)
    mean_ent = ent_sum / max(total, 1)
    mean_conf_correct = conf_correct_sum / max(n_correct, 1)
    mean_conf_wrong = conf_wrong_sum / max(n_wrong, 1)
    mean_ent_correct = ent_correct_sum / max(n_correct, 1)
    mean_ent_wrong = ent_wrong_sum / max(n_wrong, 1)
    print(f"\n  Ensemble Test Top-1: {acc_t1:.2f}%")
    print(f"  Ensemble Test Top-5: {acc_t5:.2f}%")
    print(f"  Ensemble Test Top-10: {acc_t10:.2f}%")
    print(f"  Mean confidence: {mean_conf:.4f}")
    print(f"  Mean entropy: {mean_ent:.4f}")
    print(f"  Confidence (correct/wrong): {mean_conf_correct:.4f} / {mean_conf_wrong:.4f}")
    print(f"  Entropy (correct/wrong): {mean_ent_correct:.4f} / {mean_ent_wrong:.4f}")

    # Per-class
    print(f"\n  Per-class accuracy:")
    results_per_class = []
    for i, cls in enumerate(CLASSES):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            results_per_class.append((acc, cls, int(class_total[i])))
            print(f"    {cls:>15}: {acc:5.1f}% ({int(class_correct[i])}/{int(class_total[i])})")

    results_per_class.sort()
    if results_per_class:
        worst_5 = results_per_class[:min(5, len(results_per_class))]
        avg_worst = np.mean([w[0] for w in worst_5])
        print(f"\n  Worst 5 avg: {avg_worst:.1f}%")

    result = {
        "ensemble_test_top1": acc_t1,
        "ensemble_test_top5": acc_t5,
        "ensemble_test_top10": acc_t10,
        "confidence": {
            "overall_mean": mean_conf,
            "correct_mean": mean_conf_correct,
            "wrong_mean": mean_conf_wrong,
        },
        "entropy": {
            "overall_mean": mean_ent,
            "correct_mean": mean_ent_correct,
            "wrong_mean": mean_ent_wrong,
        },
        "weights": w,
        "models": ENSEMBLE_MODELS,
        "num_classes": NUM_CLASSES,
        "test_samples": total,
        "per_class": {cls: {
                            "acc": float(100 * class_correct[i] / max(class_total[i], 1)),
                            "n": int(class_total[i]),
                            "mean_confidence": float(class_conf_sum[i] / max(class_total[i], 1)),
                            "mean_entropy": float(class_ent_sum[i] / max(class_total[i], 1)),
                            }
                      for i, cls in enumerate(CLASSES) if class_total[i] > 0},
    }
    rpath = os.path.join(RUNS_DIR, f"{RESULT_TAG}_ensemble_result.json")
    os.makedirs(RUNS_DIR, exist_ok=True)
    json.dump(result, open(rpath, "w"), indent=2)
    print(f"\n  Saved: {rpath}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["wlasl100", "wlasl2000"], default=ACTIVE_VARIANT)
    parser.add_argument("--weights", type=float, nargs="+", default=None)
    args = parser.parse_args()
    evaluate_ensemble(weights=args.weights)


if __name__ == "__main__":
    main()

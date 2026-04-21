"""Optimize ensemble weights for WLASL models via SLSQP, then report."""
import os, sys, json, time, argparse
import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import config_selector


def _detect_variant(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--variant", choices=["wlasl100", "wlasl2000"], default="wlasl100")
    args, _ = parser.parse_known_args(argv)
    return args.variant


ACTIVE_VARIANT = _detect_variant(sys.argv[1:])
config_selector.activate(ACTIVE_VARIANT)

from config import (
    NUM_CLASSES, IDX2CLASS, CLASSES,
    CHECKPOINT_DIR, RUNS_DIR, SEED,
    ENSEMBLE_MODELS, NPY_DIR, BATCH_SIZE,
)
from dataset import get_loaders
from models import get_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Optimize] Active variant: {ACTIVE_VARIANT}", flush=True)

# ── CUDA warm-up ──────────────────────────────────────────────
if DEVICE.type == "cuda":
    print("CUDA warm-up...", flush=True)
    _m = nn.Linear(150, NUM_CLASSES).to(DEVICE)
    _x = torch.randn(16, 150).to(DEVICE)
    _y = torch.randint(0, NUM_CLASSES, (16,)).to(DEVICE)
    for _ in range(10):
        _o = _m(_x); _l = nn.functional.cross_entropy(_o, _y); _l.backward()
    del _m, _x, _y, _o, _l
    torch.cuda.empty_cache(); torch.cuda.synchronize()
    print("CUDA warm-up OK", flush=True)

# ── Load data ─────────────────────────────────────────────────
_, vl, tel = get_loaders(NPY_DIR, BATCH_SIZE, seed=SEED)
print(f"Val samples: {len(vl.dataset)}, Test samples: {len(tel.dataset)}", flush=True)


def get_logits_single(mname, loader):
    """Load one model, collect logits, free GPU memory."""
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{mname}_best.pt")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    mkw = ckpt.get("model_kwargs", {})
    nc = ckpt.get("num_classes", NUM_CLASSES)
    model = get_model(mname, num_classes=nc, **mkw).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  {mname}: epoch={ckpt['epoch']}, val={ckpt['best_val_acc']:.2f}%, "
          f"nc={nc}, params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    logits_list, labels_list = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            logits_list.append(model(x).cpu())
            labels_list.append(y)

    del model
    torch.cuda.empty_cache(); torch.cuda.synchronize()
    return torch.cat(logits_list, 0), torch.cat(labels_list, 0)


# ── Collect logits ────────────────────────────────────────────
print("\nCollecting val logits...", flush=True)
val_logits, val_labels = {}, None
for m in ENSEMBLE_MODELS:
    logits, labels = get_logits_single(m, vl)
    val_logits[m] = logits
    val_labels = labels

print("\nCollecting test logits...", flush=True)
test_logits, test_labels = {}, None
for m in ENSEMBLE_MODELS:
    logits, labels = get_logits_single(m, tel)
    test_logits[m] = logits
    test_labels = labels

# ── Convert to probabilities ─────────────────────────────────
val_probs  = {m: torch.softmax(val_logits[m],  dim=1).numpy() for m in ENSEMBLE_MODELS}
test_probs = {m: torch.softmax(test_logits[m], dim=1).numpy() for m in ENSEMBLE_MODELS}
val_labels_np  = val_labels.numpy()
test_labels_np = test_labels.numpy()

# Stack for vectorized operations: (n_models, n_samples, n_classes)
val_probs_stack  = np.stack([val_probs[m]  for m in ENSEMBLE_MODELS], axis=0)
test_probs_stack = np.stack([test_probs[m] for m in ENSEMBLE_MODELS], axis=0)

# ── Individual model results ─────────────────────────────────
print("\n" + "=" * 60)
print("INDIVIDUAL MODEL RESULTS")
print("=" * 60)
individual = {}
for m in ENSEMBLE_MODELS:
    v_pred = val_logits[m].argmax(1)
    v_t1 = (v_pred == val_labels).float().mean().item() * 100
    t_pred = test_logits[m].argmax(1)
    t_t1 = (t_pred == test_labels).float().mean().item() * 100
    _, t5 = test_logits[m].topk(5, dim=1)
    t_t5 = t5.eq(test_labels.view(-1, 1)).any(1).float().mean().item() * 100
    individual[m] = {"val_t1": v_t1, "test_t1": t_t1, "test_t5": t_t5}
    print(f"  {m:>12}: val_t1={v_t1:.2f}%  test_t1={t_t1:.2f}%  test_t5={t_t5:.2f}%")


# ── SLSQP Optimization (cross-entropy surrogate) ─────────────
# Accuracy is piecewise constant -> zero gradient -> SLSQP stalls.
# Use negative log-likelihood (cross-entropy) as a smooth, differentiable
# proxy that correlates with accuracy. Then verify with grid search.

def neg_log_likelihood(w):
    """Cross-entropy loss of weighted ensemble on val set (smooth surrogate)."""
    w = np.array(w)
    combined = np.einsum('i,ijk->jk', w, val_probs_stack)
    combined = np.clip(combined, 1e-12, None)  # numerical safety
    log_probs = np.log(combined)
    return -log_probs[np.arange(len(val_labels_np)), val_labels_np].mean()

def val_accuracy(w):
    """Actual val accuracy for a weight vector."""
    w = np.array(w)
    combined = np.einsum('i,ijk->jk', w, val_probs_stack)
    return (combined.argmax(axis=1) == val_labels_np).mean() * 100

print("\n[1/2] SLSQP optimization (cross-entropy surrogate)...", flush=True)
n = len(ENSEMBLE_MODELS)
x0 = np.ones(n) / n
constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
bounds = [(0.0, 1.0)] * n

result = minimize(
    neg_log_likelihood, x0,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options={'maxiter': 500, 'ftol': 1e-12},
)
slsqp_w = result.x.tolist()
slsqp_val = val_accuracy(slsqp_w)
print(f"  SLSQP converged: {result.success} ({result.nit} iterations)")
print(f"  SLSQP weights: {[f'{w:.4f}' for w in slsqp_w]}")
print(f"  SLSQP val acc: {slsqp_val:.2f}%")

# ── Grid search verification ─────────────────────────────────
print("\n[2/2] Grid search verification (step=0.05)...", flush=True)
best_grid_acc, best_grid_w = 0, None
step = 0.05
weight_range = np.arange(0.0, 1.01, step)
count = 0
for w0 in weight_range:
    for w1 in weight_range:
        for w2 in weight_range:
            w3 = 1.0 - w0 - w1 - w2
            if w3 < -0.01 or w3 > 1.01:
                continue
            w3 = max(0, w3)
            acc = val_accuracy([w0, w1, w2, w3])
            count += 1
            if acc > best_grid_acc:
                best_grid_acc = acc
                best_grid_w = [w0, w1, w2, w3]
print(f"  Grid searched {count} combos")
print(f"  Grid best weights: {[f'{w:.2f}' for w in best_grid_w]}")
print(f"  Grid best val acc: {best_grid_acc:.2f}%")

# ── Pick the winner ──────────────────────────────────────────
if best_grid_acc >= slsqp_val:
    opt_w = best_grid_w
    opt_val_acc = best_grid_acc
    opt_source = "grid"
else:
    opt_w = slsqp_w
    opt_val_acc = slsqp_val
    opt_source = "slsqp"

opt_w_rounded = [round(float(w), 4) for w in opt_w]
print(f"\n  Winner: {opt_source} -> weights={opt_w_rounded}, val_acc={opt_val_acc:.2f}%")


# ── Evaluate on test set ─────────────────────────────────────
def eval_weights(w, probs_stack, labels_np, label=""):
    w = np.array(w)
    combined = np.einsum('i,ijk->jk', w, probs_stack)
    preds = combined.argmax(axis=1)
    t1 = (preds == labels_np).mean() * 100
    top5 = np.argsort(combined, axis=1)[:, -5:]
    t5 = np.any(top5 == labels_np[:, None], axis=1).mean() * 100

    per_class = {}
    for c in range(NUM_CLASSES):
        mask = labels_np == c
        if mask.sum() == 0:
            continue
        cls_acc = (preds[mask] == c).mean() * 100
        per_class[IDX2CLASS[c]] = {"acc": round(cls_acc, 1), "n": int(mask.sum())}

    if label:
        print(f"  {label}: test_t1={t1:.2f}%  test_t5={t5:.2f}%")
    return t1, t5, preds, per_class


print(f"\n{'=' * 60}")
print("ENSEMBLE COMPARISON ON TEST SET")
print("=" * 60)

opt_t1, opt_t5, opt_preds, opt_per_class = eval_weights(
    opt_w, test_probs_stack, test_labels_np, "Optimized")
eq_t1, eq_t5, _, _ = eval_weights(
    [0.25] * 4, test_probs_stack, test_labels_np, "Equal [0.25x4]")

# TCN-only baseline
tcn_only = [0.0, 1.0, 0.0, 0.0]
tcn_t1, tcn_t5, _, _ = eval_weights(
    tcn_only, test_probs_stack, test_labels_np, "TCN-only")

# ── Per-class worst-10 analysis ──────────────────────────────
sorted_classes = sorted(opt_per_class.items(), key=lambda x: x[1]["acc"])
worst_10 = sorted_classes[:10]
print(f"\n  Worst 10 classes (optimized ensemble):")
for cls, info in worst_10:
    print(f"    {cls:>15}: {info['acc']:5.1f}% (n={info['n']})")
avg_worst = np.mean([info["acc"] for _, info in worst_10])
print(f"  Avg worst 10: {avg_worst:.1f}%")

# ── Save results ─────────────────────────────────────────────
os.makedirs(RUNS_DIR, exist_ok=True)
result_data = {
    "optimized_weights": dict(zip(ENSEMBLE_MODELS, opt_w_rounded)),
    "optimized_weights_list": opt_w_rounded,
    "optimization_source": opt_source,
    "val_top1": round(opt_val_acc, 2),
    "test_top1": round(opt_t1, 2),
    "test_top5": round(opt_t5, 2),
    "equal_test_top1": round(eq_t1, 2),
    "equal_test_top5": round(eq_t5, 2),
    "tcn_only_test_top1": round(tcn_t1, 2),
    "slsqp_weights": [round(w, 4) for w in slsqp_w],
    "slsqp_val_acc": round(slsqp_val, 2),
    "grid_weights": [round(w, 4) for w in best_grid_w] if best_grid_w else None,
    "grid_val_acc": round(best_grid_acc, 2),
    "individual": individual,
    "per_class_acc": opt_per_class,
    "worst_10": {c: info for c, info in worst_10},
    "avg_worst_10": round(avg_worst, 1),
    "slsqp_iterations": result.nit,
    "slsqp_converged": bool(result.success),
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
}
out_path = os.path.join(RUNS_DIR, "ensemble_wlasl_optimized.json")
with open(out_path, "w") as f:
    json.dump(result_data, f, indent=2)
print(f"\nSaved: {out_path}", flush=True)

# ── Print config update suggestion ───────────────────────────
print(f"\n{'=' * 60}")
print("FINAL SUMMARY")
print("=" * 60)
print(f"  Optimal weights: {dict(zip(ENSEMBLE_MODELS, opt_w_rounded))}")
print(f"  Val  Top-1: {opt_val_acc:.2f}%")
print(f"  Test Top-1: {opt_t1:.2f}%  (equal={eq_t1:.2f}%, tcn_only={tcn_t1:.2f}%)")
print(f"  Test Top-5: {opt_t5:.2f}%")
target_config = "config_wlasl2000.py" if ACTIVE_VARIANT == "wlasl2000" else "config_wlasl100.py"
print(f"\n>>> Config update for {target_config}:")
print(f"ENSEMBLE_WEIGHTS = [{', '.join(f'{w:.4f}' for w in opt_w_rounded)}]")
print(f"\nDone: {time.strftime('%Y-%m-%d %H:%M:%S')}")

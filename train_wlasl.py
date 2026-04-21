#!/usr/bin/env python
"""
train_wlasl.py — Train models on WLASL skeletal data variants.

Uses config_selector + active `config` module.
Supports WLASL-100 and WLASL-2000 via `--variant`.

Usage:
    python train_wlasl.py --variant wlasl100                     # train all 4 models
    python train_wlasl.py --variant wlasl2000 --model hybrid     # train one model
    python train_wlasl.py --variant wlasl2000 --epochs 300       # custom epoch count
    python train_wlasl.py --variant wlasl2000 --dry-run          # 1-epoch validation only
"""
import os, sys, json, time, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ─── Activate WLASL config ───────────────────────────────────
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
    NPY_DIR, CLASSES, NUM_CLASSES, BATCH_SIZE, LR, WEIGHT_DECAY,
    PATIENCE, GRAD_CLIP, SEED, CHECKPOINT_DIR, RUNS_DIR, NUM_EPOCHS,
    WARMUP_EPOCHS,
)
from dataset import get_loaders, build_splits, BSLDataset
from models import get_model

MAX_BS = 64
MODELS_TO_TRAIN = ["hybrid", "tcn", "bilstm", "transformer"]


# ─── FocalLoss ────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal Loss with optional per-class alpha weighting."""
    def __init__(self, alpha=None, gamma=2, num_classes=None):
        super().__init__()
        self.gamma = gamma
        if alpha is None:
            self.register_buffer("alpha", None)
        elif isinstance(alpha, (int, float)):
            self.register_buffer("alpha", torch.tensor(float(alpha)))
        else:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if self.alpha.dim() == 0:
                focal = self.alpha * focal
            else:
                at = self.alpha.gather(0, targets)
                focal = at * focal

        return focal.mean()


def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def cuda_warmup(device, num_classes):
    """CUDA warm-up: prevents CUBLAS crashes on RTX 3060 Ti."""
    print("  CUDA warm-up...", flush=True)
    m = nn.Linear(150, num_classes).to(device)
    x = torch.randn(32, 150).to(device)
    y = torch.randint(0, num_classes, (32,)).to(device)
    for _ in range(20):
        o = m(x)
        loss = F.cross_entropy(o, y)
        loss.backward()
    del m, x, y, o, loss
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("  CUDA warm-up OK", flush=True)


def train_model(model_name, epochs, device, dry_run=False):
    print(f"\n{'='*60}", flush=True)
    print(f"  TRAINING [{ACTIVE_VARIANT.upper()}]: {model_name.upper()}", flush=True)
    print(f"{'='*60}", flush=True)

    set_seed()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"  Classes: {NUM_CLASSES}, Data: {NPY_DIR}", flush=True)

    # Load tuned hyperparams if available
    pp = os.path.join(RUNS_DIR, f"{model_name}_best_params.json")
    bp = json.load(open(pp)) if os.path.exists(pp) else {}
    lr = bp.get("lr", LR)
    wd = bp.get("weight_decay", WEIGHT_DECAY)
    bs = min(bp.get("batch_size", BATCH_SIZE), MAX_BS)
    ls = bp.get("label_smoothing", 0.0)
    gc = bp.get("grad_clip", GRAD_CLIP)
    mkw = {k: v for k, v in bp.items()
           if k not in {"lr", "weight_decay", "batch_size",
                        "label_smoothing", "grad_clip", "model", "val_top1"}}

    print(f"  lr={lr:.6f} bs={bs} ls={ls:.3f} gc={gc:.2f}", flush=True)

    tl, vl, tel = get_loaders(NPY_DIR, bs, seed=SEED, use_adaptive_aug=True)
    print(f"  Data: train={len(tl.dataset)} val={len(vl.dataset)} test={len(tel.dataset)}",
          flush=True)

    tl.dataset.print_class_distribution("train")

    if len(tl.dataset) == 0:
        print(f"  ERROR: No training data! Check NPY_DIR={NPY_DIR}", flush=True)
        return None

    model = get_model(model_name, num_classes=NUM_CLASSES, **mkw).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {model_name} ({n_params:,} params)", flush=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    # FocalLoss(gamma=2) — uniform alpha.
    # WeightedRandomSampler handles class balance.
    criterion = FocalLoss(alpha=1.0, gamma=2.0).to(device)
    print(f"  Loss: FocalLoss(alpha=1.0, gamma=2) + WeightedRandomSampler", flush=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    if dry_run:
        epochs = 1
        print("  *** DRY RUN: 1 epoch only ***", flush=True)

    best_val = 0.0
    no_improve = 0
    warmup_ep = WARMUP_EPOCHS
    t_start = time.time()

    for ep in range(1, epochs + 1):
        if ep <= warmup_ep:
            for pg in optimizer.param_groups:
                pg['lr'] = lr * ep / warmup_ep

        # Train
        model.train()
        tloss, tcorr, tn = 0, 0, 0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gc)
            optimizer.step()
            tloss += loss.item() * y.size(0)
            tcorr += (out.argmax(1) == y).sum().item()
            tn += y.size(0)

        # Validate
        model.eval()
        vcorr, vn, v5corr = 0, 0, 0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(device), y.to(device)
                out = model(x)
                vcorr += (out.argmax(1) == y).sum().item()
                _, t5 = out.topk(min(5, out.size(1)), dim=1)
                v5corr += t5.eq(y.view(-1, 1)).any(1).sum().item()
                vn += y.size(0)

        if ep > warmup_ep:
            scheduler.step()

        vacc = 100 * vcorr / max(vn, 1)
        v5acc = 100 * v5corr / max(vn, 1)

        star = ""
        if vacc > best_val:
            best_val = vacc
            no_improve = 0
            star = " *BEST*"
            torch.save({
                "epoch": ep,
                "model_name": model_name,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val,
                "num_classes": NUM_CLASSES,
                "model_kwargs": mkw,
                "best_params": bp,
            }, os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pt"))
        else:
            no_improve += 1

        cur_lr = optimizer.param_groups[0]['lr']
        print(f"  Ep {ep:>3}/{epochs}  loss={tloss/max(tn,1):.4f}  "
              f"t1={100*tcorr/max(tn,1):.1f}%  |  "
              f"v_t1={vacc:.1f}%  v_t5={v5acc:.1f}%  lr={cur_lr:.6f}{star}",
              flush=True)

        if not dry_run and no_improve >= PATIENCE:
            print(f"\n  >>> Early stop at ep {ep} (patience={PATIENCE}, best={best_val:.2f}%)",
                  flush=True)
            break

    # Test
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pt")
    if not os.path.exists(ckpt_path):
        print("  No checkpoint saved (validation never improved)", flush=True)
        return {"model": model_name, "status": "NO_CHECKPOINT"}

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    te1, te5, ten = 0, 0, 0
    class_correct = np.zeros(NUM_CLASSES)
    class_total = np.zeros(NUM_CLASSES)

    with torch.no_grad():
        for x, y in tel:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(1)
            te1 += (preds == y).sum().item()
            _, t5 = out.topk(min(5, out.size(1)), dim=1)
            te5 += t5.eq(y.view(-1, 1)).any(1).sum().item()
            ten += y.size(0)

            for i in range(y.size(0)):
                class_total[y[i].item()] += 1
                if preds[i].item() == y[i].item():
                    class_correct[y[i].item()] += 1

    total_time = time.time() - t_start
    test_t1 = 100 * te1 / max(ten, 1)
    test_t5 = 100 * te5 / max(ten, 1)

    print(f"\n  RESULT: {model_name} val={best_val:.2f}% test_t1={test_t1:.2f}% "
          f"test_t5={test_t5:.2f}%", flush=True)
    print(f"  Best epoch: {ckpt['epoch']}, Time: {total_time:.0f}s", flush=True)

    # Per-class results
    print(f"\n  Per-class accuracy:", flush=True)
    worst_classes = []
    for i, cls in enumerate(CLASSES):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            worst_classes.append((acc, cls, int(class_total[i])))
            print(f"    {cls:>15}: {acc:5.1f}% ({int(class_correct[i])}/{int(class_total[i])})",
                  flush=True)

    worst_classes.sort()
    if worst_classes:
        worst_5 = worst_classes[:min(5, len(worst_classes))]
        avg_worst = np.mean([w[0] for w in worst_5])
        print(f"\n  Worst 5 avg: {avg_worst:.1f}%", flush=True)

    result = {
        "model": model_name,
        "best_val": best_val,
        "best_epoch": ckpt["epoch"],
        "test_top1": test_t1,
        "test_top5": test_t5,
        "total_time_s": total_time,
        "num_classes": NUM_CLASSES,
        "num_params": n_params,
        "dataset": ACTIVE_VARIANT,
        "dry_run": dry_run,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    rpath = os.path.join(RUNS_DIR, f"{model_name}_{RESULT_TAG}_result.json")
    json.dump(result, open(rpath, "w"), indent=2)
    print(f"  Saved: {rpath}", flush=True)

    # Cleanup
    del model, optimizer, criterion, scheduler
    torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Train single model")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--variant", choices=["wlasl100", "wlasl2000"], default=ACTIVE_VARIANT,
                        help="Dataset variant (activated before import)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run 1 epoch only for validation")
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)

    models = [args.model] if args.model else MODELS_TO_TRAIN

    print(f"\n{'='*60}", flush=True)
    print(f"  WLASL Training Pipeline ({ACTIVE_VARIANT})", flush=True)
    print(f"  Models: {models}", flush=True)
    print(f"  Classes: {NUM_CLASSES}", flush=True)
    print(f"  Epochs: {args.epochs} (patience={PATIENCE})", flush=True)
    print(f"  Data: {NPY_DIR}", flush=True)
    print(f"  Dry-run: {args.dry_run}", flush=True)
    print(f"  Start: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"{'='*60}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        cuda_warmup(device, NUM_CLASSES)

    results = {}
    for m in models:
        try:
            results[m] = train_model(m, args.epochs, device, dry_run=args.dry_run)
        except Exception as e:
            print(f"\n  >>> {m}: FAILED — {e}", flush=True)
            import traceback
            traceback.print_exc()
            results[m] = {"model": m, "status": "FAILED", "error": str(e)}
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}", flush=True)
    print(f"  FINAL SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    for m, r in results.items():
        if r and "test_top1" in r:
            print(f"  {m:>12}: val={r['best_val']:.2f}%  test={r['test_top1']:.2f}%  "
                  f"ep={r['best_epoch']}  params={r.get('num_params', '?'):,}", flush=True)
        else:
            status = r.get("status", "UNKNOWN") if r else "NO RESULT"
            print(f"  {m:>12}: {status}", flush=True)

    all_path = os.path.join(RUNS_DIR, f"{RESULT_TAG}_all_results.json")
    json.dump(results, open(all_path, "w"), indent=2)
    print(f"\n  Results saved: {all_path}", flush=True)
    print(f"  Done: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()

# =============================================================
# tune.py — Automatic hyperparameter optimisation with Optuna
# =============================================================
"""
Usage:
    # Tune Option 5 (hybrid) with 30 trials, 15 epochs each:
    python tune.py

    # Tune specific model:
    python tune.py --model bilstm --n-trials 20 --trial-epochs 20

    # Tune all 5 models sequentially:
    python tune.py --all-models --n-trials 20 --trial-epochs 15

    # Resume existing study:
    python tune.py --model hybrid --resume-study

Output files (in runs/):
    {model}_best_params.json   — best hyperparameters found
    {model}_optuna.csv         — all trial results
    {model}_optuna_study.db    — SQLite study (for resuming)
"""

import os
import sys
import json
import time
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).parent))
import config_selector


def _detect_variant(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--variant", choices=["wlasl100", "wlasl2000"], default="wlasl100")
    args, _ = parser.parse_known_args(argv)
    return args.variant


ACTIVE_VARIANT = _detect_variant(sys.argv[1:])
config_selector.activate(ACTIVE_VARIANT)

from config import (
    NPY_DIR, CHECKPOINT_DIR, RUNS_DIR,
    NUM_CLASSES, CLASSES,
    BATCH_SIZE, NUM_EPOCHS, LR, WEIGHT_DECAY,
    WARMUP_EPOCHS, GRAD_CLIP, SEED,
)
from dataset import get_loaders
from models import get_model

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None
    MedianPruner = None
    TPESampler = None


# ─── Config ───────────────────────────────────────────────────

TRIAL_EPOCHS_DEFAULT = 15      # epochs per trial (keep short for speed)
N_TRIALS_DEFAULT     = 30      # total Optuna trials per model
TOPK                 = min(5, NUM_CLASSES)

ALL_MODELS = ["bilstm", "transformer", "stgcn", "tcn", "hybrid"]


# ─── Shared search space ──────────────────────────────────────

def suggest_shared(trial):
    """Hyperparams shared by all models."""
    return {
        "lr":              trial.suggest_float("lr",            1e-5, 5e-3, log=True),
        "weight_decay":    trial.suggest_float("weight_decay",  1e-6, 1e-2, log=True),
        # RTX 3060 Ti safety: cap batch-size at 64.
        "batch_size":      trial.suggest_categorical("batch_size", [32, 64]),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.2),
        "grad_clip":       trial.suggest_float("grad_clip",     0.5, 3.0),
    }


# ─── Per-model search spaces ──────────────────────────────────

def suggest_model_params(trial, model_name):
    """Model-specific architecture hyperparams."""
    if model_name == "bilstm":
        return {
            "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
            "num_layers":  trial.suggest_int("num_layers", 1, 3),
            "dropout":     trial.suggest_float("dropout", 0.1, 0.5),
        }
    elif model_name == "transformer":
        return {
            "d_model":    trial.suggest_categorical("d_model",    [64, 128, 256]),
            "nhead":      trial.suggest_categorical("nhead",      [2, 4, 8]),
            "num_layers": trial.suggest_int("num_layers", 2, 6),
            "dropout":    trial.suggest_float("dropout", 0.1, 0.4),
        }
    elif model_name == "stgcn":
        return {
            "dropout": trial.suggest_float("dropout", 0.1, 0.4),
        }
    elif model_name == "tcn":
        return {
            "channels": trial.suggest_categorical("channels", [64, 128, 256]),
            "dropout":  trial.suggest_float("dropout", 0.1, 0.4),
        }
    elif model_name == "hybrid":
        return {
            "cnn_channels": trial.suggest_categorical("cnn_channels", [64, 128, 256]),
            "lstm_hidden":  trial.suggest_categorical("lstm_hidden",  [64, 128, 256]),
            "lstm_layers":  trial.suggest_int("lstm_layers", 1, 3),
            "attn_heads":   trial.suggest_categorical("attn_heads",   [2, 4, 8]),
            "drop_cnn":     trial.suggest_float("drop_cnn",  0.0, 0.3),
            "drop_lstm":    trial.suggest_float("drop_lstm", 0.1, 0.5),
            "drop_cls":     trial.suggest_float("drop_cls",  0.2, 0.6),
            "fc_hidden":    trial.suggest_categorical("fc_hidden",    [64, 128, 256]),
        }
    return {}


# ─── Single trial ─────────────────────────────────────────────

def run_trial(trial, model_name, loaders, device, trial_epochs):
    """Train for trial_epochs, return best val top-1."""
    shared = suggest_shared(trial)
    model_kw = suggest_model_params(trial, model_name)

    train_loader, val_loader, _ = loaders

    # Build model
    model = get_model(model_name, num_classes=NUM_CLASSES, **model_kw).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=shared["label_smoothing"])
    optimizer = optim.AdamW(
        model.parameters(),
        lr=shared["lr"],
        weight_decay=shared["weight_decay"],
        betas=(0.9, 0.98),
    )
    total_steps = len(train_loader) * trial_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=shared["lr"],
        total_steps=total_steps,
        pct_start=min(0.3, WARMUP_EPOCHS / trial_epochs),
        anneal_strategy="cos",
        div_factor=10,
        final_div_factor=100,
    )

    best_val = 0.0

    for epoch in range(trial_epochs):
        # Train
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            if shared["grad_clip"] > 0:
                nn.utils.clip_grad_norm_(model.parameters(), shared["grad_clip"])
            optimizer.step()
            scheduler.step()

        # Validate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += pred.eq(y).sum().item()
                total   += y.size(0)
        val_top1 = 100.0 * correct / total if total else 0.0
        best_val = max(best_val, val_top1)

        # Optuna pruning — report intermediate value
        trial.report(val_top1, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val


# ─── Tune one model ───────────────────────────────────────────

def tune_model(model_name, n_trials, trial_epochs, resume_study, device):
    if optuna is None:
        raise ImportError("optuna not found. Install with: pip install optuna")
    os.makedirs(RUNS_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  TUNING: {model_name.upper()}  ({n_trials} trials x {trial_epochs} epochs)")
    print(f"{'='*60}")

    # Build loaders once (shared across trials for speed)
    train_loader, val_loader, test_loader = get_loaders(npy_dir=NPY_DIR, batch_size=64)
    loaders = (train_loader, val_loader, test_loader)

    db_path    = os.path.join(RUNS_DIR, f"{model_name}_optuna_study.db")
    study_name = f"bsl_{model_name}"
    storage    = f"sqlite:///{db_path}"
    load_if_exists = resume_study

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=TPESampler(seed=SEED),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        load_if_exists=True,      # always resume if DB exists
    )

    def objective(trial):
        return run_trial(trial, model_name, loaders, device, trial_epochs)

    t0 = time.time()
    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[_progress_callback],
        show_progress_bar=False,
    )
    elapsed = time.time() - t0

    best = study.best_trial
    print(f"\n[Best] Trial #{best.number}  val_top1={best.value:.2f}%  ({elapsed:.0f}s total)")
    print(f"[Best] Params:")
    for k, v in best.params.items():
        print(f"  {k:20s} = {v}")

    # Save best params
    best_params = {"model": model_name, "val_top1": best.value, **best.params}
    params_path = os.path.join(RUNS_DIR, f"{model_name}_best_params.json")
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"[Saved] Best params -> {params_path}")

    # Save all trial results as CSV
    csv_path = os.path.join(RUNS_DIR, f"{model_name}_optuna.csv")
    df = study.trials_dataframe()
    df.to_csv(csv_path, index=False)
    print(f"[Saved] All trials  -> {csv_path}")

    return best.value, best.params


def _progress_callback(study, trial):
    n = len(study.trials)
    best = study.best_value if study.best_value is not None else 0.0
    status = "PRUNED" if trial.state == optuna.trial.TrialState.PRUNED else f"{trial.value:.2f}%"
    print(f"  Trial {trial.number+1:>3}/{n}  val={status}  best={best:.2f}%", flush=True)


# ─── Entry point ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="BSL Hyperparameter Tuning (Optuna)")
    p.add_argument("--variant",      choices=["wlasl100", "wlasl2000"], default=ACTIVE_VARIANT,
                   help="Dataset variant (activated before imports)")
    p.add_argument("--model",        default="hybrid",
                   choices=ALL_MODELS)
    p.add_argument("--all-models",   action="store_true",
                   help="Tune all 5 models sequentially")
    p.add_argument("--n-trials",     type=int, default=N_TRIALS_DEFAULT,
                   help=f"Optuna trials per model (default {N_TRIALS_DEFAULT})")
    p.add_argument("--trial-epochs", type=int, default=TRIAL_EPOCHS_DEFAULT,
                   help=f"Training epochs per trial (default {TRIAL_EPOCHS_DEFAULT})")
    p.add_argument("--resume-study", action="store_true",
                   help="Resume existing Optuna study from SQLite DB")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[Device] {gpu_name}")
    print(f"[Variant] {ACTIVE_VARIANT}")
    print(f"[Classes] {NUM_CLASSES}")

    models_to_tune = ALL_MODELS if args.all_models else [args.model]
    summary = {}

    for model_name in models_to_tune:
        best_val, best_params = tune_model(
            model_name, args.n_trials, args.trial_epochs,
            args.resume_study, device,
        )
        summary[model_name] = {"best_val_top1": best_val, "params": best_params}

    if len(models_to_tune) > 1:
        print(f"\n{'='*55}")
        print(f"  TUNING SUMMARY — All Models")
        print(f"{'='*55}")
        print(f"  {'Model':<16}  {'Best Val Top-1':>14}")
        print(f"  {'-'*35}")
        for name, r in summary.items():
            marker = " <-- FOCUS" if name == "hybrid" else ""
            print(f"  {name:<16}  {r['best_val_top1']:>13.2f}%{marker}")

        with open(os.path.join(RUNS_DIR, "tune_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

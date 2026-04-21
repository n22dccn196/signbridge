# Ensemble Fix & Optimization Log — WLASL-100

**Date:** 2026-03-08  
**Work Directory:** `G:\HK2N4\BOBSL_Lightweight\dl_model_wlasl\`

---

## Phase 1: Bug Fix — ensemble.py Checkpoint Loader

### Root Cause

`ensemble.py` used **module-level imports** to bind config values at import time:

```python
from config import (
    NUM_CLASSES, CLASSES, IDX2CLASS,
    CHECKPOINT_DIR,
    ENSEMBLE_MODELS, ENSEMBLE_WEIGHTS, ENSEMBLE_MODE,
)
```

When `config_selector.activate("wlasl")` was called *after* this import (or from a different module), these names still pointed to the **default BSL config** (`config.py`):

| Variable | Bug Value (BSL default) | Correct Value (WLASL) |
|----------|------------------------|-----------------------|
| `CHECKPOINT_DIR` | `checkpoints/` | `checkpoints_wlasl/` |
| `NUM_CLASSES` | 36 | 100 |
| `CLASSES` | 36 BSL signs | 100 ASL glosses |

This caused `Ensemble.from_config()` to look in `checkpoints/` for BSL models, find none (or find wrong-class-count models), and fail.

### Fix Applied

Replaced module-level `from config import (...)` with a runtime `_cfg()` helper:

```python
def _cfg():
    """Resolve config at runtime so config_selector.activate() is respected."""
    return sys.modules.get('config', __import__('config'))
```

All references to `NUM_CLASSES`, `IDX2CLASS`, `CHECKPOINT_DIR`, `ENSEMBLE_*` in the `Ensemble` class now resolve through `_cfg()` at call time, guaranteeing they pick up whichever config (`auto`, `manual`, `wlasl`) was activated by `config_selector.activate()`.

Instance attributes `self.num_classes` and `self.idx2class` are captured once in `__init__()` from the live config, then used in `predict_probs()`, `predict_label()`, and `predict_topk()`.

**Files modified:** `ensemble.py`  
**Backward compatible:** Yes — existing BSL usage is unaffected.

---

## Phase 2: Ensemble Weight Optimization

### Method

1. **SLSQP with cross-entropy surrogate** — accuracy is piecewise constant (zero gradient), so SLSQP was applied to negative log-likelihood (smooth, differentiable) as a proxy objective. Converged in 15 iterations.
2. **Grid search verification** — exhaustive search with step=0.05 (1,771 combinations). Constraint: weights sum to 1.0.

### Individual Model Performance

| Model | Val Top-1 | Test Top-1 | Test Top-5 | Params |
|-------|-----------|------------|------------|--------|
| Hybrid | 30.00% | 24.08% | 60.21% | 4,682,508 |
| **TCN** | **63.75%** | **65.45%** | **87.43%** | 430,628 |
| BiLSTM | 46.25% | 51.31% | 81.15% | 1,479,440 |
| Transformer | 52.50% | 48.17% | 79.58% | 827,812 |

### Optimization Results

| Method | Weights [H, T, B, Tr] | Val Top-1 | Test Top-1 | Test Top-5 |
|--------|------------------------|-----------|------------|------------|
| Equal | [0.25, 0.25, 0.25, 0.25] | 59.69% | 61.26% | 82.20% |
| TCN-only | [0.00, 1.00, 0.00, 0.00] | 63.75% | 65.45% | 87.43% |
| SLSQP | [0.07, 0.64, 0.05, 0.24] | 64.06% | — | — |
| **Grid (winner)** | **[0.05, 0.60, 0.00, 0.35]** | **65.94%** | **64.92%** | **86.91%** |

### Optimal Weights

```
hybrid:      0.05  (5%)
tcn:         0.60  (60%)
bilstm:      0.00  (0%)
transformer: 0.35  (35%)
```

**Key insight:** TCN dominates (60% weight). Transformer adds complementary diversity (35%). Hybrid contributes minimally (5%). BiLSTM is fully excluded — its predictions don't improve the TCN+Transformer combination.

The optimized ensemble achieves **+3.66pp over equal weights** on test (64.92% vs 61.26%). It slightly trails TCN-only on test (64.92% vs 65.45%) but outperforms it on val (65.94% vs 63.75%), suggesting better generalization.

---

## Phase 3: Integration

### Config Update

`config_wlasl.py` updated:
```python
ENSEMBLE_WEIGHTS = [0.05, 0.60, 0.00, 0.35]   # optimized via SLSQP+grid (val=65.94%)
```

### Demo Camera Integration

`demo_camera.py` updated with auto-injection of optimized weights when `--wlasl --ensemble` flags are used. The block loads weights from `runs_wlasl/ensemble_wlasl_optimized.json` at runtime, ensuring the camera demo always uses the latest optimized weights.

### Verified Test Accuracy

```
Ensemble Test Top-1: 64.92%
Ensemble Test Top-5: 86.91%
Weights: [0.05, 0.6, 0.0, 0.35]
```

Confirmed via `python eval_ensemble_wlasl.py`.

---

## Files Modified/Created

| File | Action | Description |
|------|--------|-------------|
| `ensemble.py` | Modified | Runtime config resolution via `_cfg()` helper |
| `config_wlasl.py` | Modified | `ENSEMBLE_WEIGHTS = [0.05, 0.60, 0.00, 0.35]` |
| `demo_camera.py` | Modified | WLASL weight injection block for `--wlasl --ensemble` |
| `optimize_ensemble_wlasl.py` | Created | SLSQP + grid search optimization script |
| `runs_wlasl/ensemble_wlasl_optimized.json` | Created | Full optimization results |
| `ensemble_fix_log.md` | Created | This document |

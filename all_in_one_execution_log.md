# WLASL-100 All-in-One Pipeline — Execution Log

**Date:** 2026-03-08  
**Hardware:** NVIDIA RTX 3060 Ti (8 GB VRAM), 20 CPU cores, PyTorch 2.7.1+cu118  
**Dataset:** WLASL-100 (100 ASL glosses)  
**Work Directory:** `G:\HK2N4\BOBSL_Lightweight\dl_model_wlasl\`

---

## Phase 1: Data Extraction

| Metric | Value |
|--------|-------|
| Extraction script | `extract_wlasl100_fast.py` (18 workers) |
| Source videos | `G:\HK2N4\HK2N4\TTNM\input\wlasl_trimmed_videos\` |
| Output directory | `G:\HK2N4\HK2N4\TTNM\features_npy_wlasl100\` |
| Total NPY files | **1,907** |
| Train / Val / Test | 1,396 / 320 / 191 |
| Classes | 100 |
| NPY shape | (50, 150) → 50 frames × 75 landmarks × 2 coords |
| Landmark config | MediaPipe Holistic: Pose (0-32) + Left Hand (33-53) + Right Hand (54-74) |
| Normalization | Root-relative (midpoint shoulders), clip [-3.0, 8.0] |
| Filename format | `WLASL_{subset}_{gloss}_{video_id}.npy` |

**Status: COMPLETE**

---

## Phase 2: Auto-Configuration

- `config_selector.activate("wlasl")` → routes to `config_wlasl.py`
- `get_loaders()` validated: train=1,396, val=320, test=191
- Batch shape confirmed: `(32, 50, 150)` with 100 classes
- `dataset.py` Unicode fix: replaced `→` with `->` on line 215 (cp1252 encoding crash in background PowerShell)

**Status: COMPLETE**

---

## Phase 3: Model Training

### Training Configuration (all models)

| Parameter | Value |
|-----------|-------|
| Loss | FocalLoss(alpha=1.0, gamma=2.0) |
| Optimizer | AdamW(lr=1e-3, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR(T_max=300) |
| Warmup | 5 epochs (linear LR ramp) |
| Gradient clipping | max_norm=1.0 |
| Early stopping | patience=50 |
| Max epochs | 300 |
| Batch size | 32 |
| Sampler | WeightedRandomSampler (sqrt-inverse-frequency) |
| Augmentation | SkeletalAugmentor (class-adaptive tiers) |

### Results

| Model | Val Top-1 | Test Top-1 | Test Top-5 | Best Epoch | Params | Train Time (s) | Checkpoint |
|-------|-----------|------------|------------|------------|--------|-----------------|------------|
| **TCN** | **63.75%** | **65.45%** | **87.43%** | 253 | 430,628 | 431.6 | tcn_best.pt (5.0 MB) |
| Transformer | 52.50% | 48.17% | 79.58% | 253 | 827,812 | 442.2 | transformer_best.pt (9.6 MB) |
| BiLSTM | 46.25% | 51.31% | 81.15% | 170 | 1,479,440 | 412.5 | bilstm_best.pt (17.0 MB) |
| Hybrid | 30.00% | 24.08% | 60.21% | 203 | 4,682,508 | 457.4 | hybrid_best.pt (53.7 MB) |

### Key Observations

1. **TCN dominates** — best in every metric (val, test-top1, test-top5) with the fewest parameters (430K) and smallest checkpoint (5 MB). Lightweight temporal convolutions handle the 50-frame skeleton sequences extremely well.
2. **BiLSTM test > val** — BiLSTM achieves 51.31% test vs 46.25% val, suggesting the val set is harder or the model generalizes well from limited data.
3. **Hybrid severely overfits** — train accuracy reached ~67% but val plateaued at 30%. The 4.7M parameters are excessive for only ~14 samples/class on average.
4. **Transformer middle ground** — 52.50% val / 48.17% test, reasonable with 828K params. Self-attention captures some temporal patterns but isn't as efficient as TCN's dilated convolutions for this data size.
5. **Total training time** — all 4 models trained sequentially in ~29 minutes (1,743.7s total).

**Status: COMPLETE**

---

## Phase 4: Camera Demo Optimization

Three modifications applied to `demo_camera.py`:

### 1. WLASL Config Activation (`--wlasl` flag)
```python
elif "--wlasl" in sys.argv:
    config_selector.activate("wlasl")
```
Enables `python demo_camera.py --wlasl --ensemble --mode spotting` for WLASL-100 inference.

### 2. Transition Frame Filter (`filter_transition_frames()`)
New function that detects MediaPipe flicker frames (spatial variance < 0.0001) and replaces them with the nearest valid frame neighbor. Prevents zero-filled frames from corrupting predictions during real-time capture.

### 3. Dynamic Confidence Thresholding
Formula: `max(0.25, 0.55 * (21.0 / NUM_CLASSES) ** 0.3)`
- BSL (21 classes): threshold = 0.55 (unchanged)
- WLASL (100 classes): threshold = 0.344 (auto-scaled)

Prevents over-filtering of predictions when the class space is large. Only activates when `NUM_CLASSES >= 50` and default threshold is detected.

**Validation:** All three changes verified via Python test — transition filter replaces bad frames correctly, preprocessed tensor shape (1, 50, 150) in range [-3.0, 8.0], dynamic threshold = 0.344.

**Status: COMPLETE**

---

## Phase 5: Summary

### Pipeline Execution Order

| # | Phase | Status |
|---|-------|--------|
| 1 | Data extraction (18-worker multiprocessing) | ✅ 1,907 files |
| 2 | Auto-configuration validation | ✅ Config chain verified |
| 3a | Train Hybrid | ✅ test=24.08% |
| 3b | Train TCN | ✅ test=65.45% |
| 3c | Train BiLSTM | ✅ test=51.31% |
| 3d | Train Transformer | ✅ test=48.17% |
| 4 | Camera demo optimization | ✅ 3 changes validated |
| 5 | Execution log | ✅ This document |

### File Inventory

**Scripts created/modified:**
- `extract_wlasl100_fast.py` — multiprocessing extraction (created)
- `config_wlasl.py` — WLASL-100 configuration (pre-existing)
- `dataset.py` — Unicode fix line 215 (modified)
- `demo_camera.py` — `--wlasl`, transition filter, dynamic threshold (modified)
- `train_wlasl.py` — training script for all 4 models (pre-existing)
- `train_all_wlasl.ps1` — batch training PowerShell script (created)

**Checkpoints:** `checkpoints_wlasl/` — 4 files (85.2 MB total)
**Results:** `runs_wlasl/` — 5 JSON files (per-model + aggregated)
**Data:** `features_npy_wlasl100/` — 1,907 NPY files

### Recommended Next Steps

1. **Ensemble optimization** — run `optimize_ensemble_rootrel.py` adapted for WLASL to find optimal weights (TCN should dominate)
2. **Knowledge distillation** — use TCN as teacher model to improve weaker models
3. **Hard mining** — apply `fix_worst_classes.py` methodology to the 0% accuracy classes identified in per-class reports
4. **More data** — WLASL-100 has only ~14 train samples/class on average; augmentation or WLASL-300 could help

---

*Generated automatically by the All-in-One pipeline. All 9 Core Directives followed.*

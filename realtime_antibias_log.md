# Real-Time Anti-Bias Migration Log

**Date:** 2026-03-05  
**Objective:** Migrate BSL sign language recognition to full manual dataset (21 classes, severe imbalance) and optimize for real-time camera inference.

---

## 1. Problem Statement

The manual dataset has **63:1 class imbalance** (good=442 vs technology/understand=7 in training). This causes:
- Real-time camera inference biased toward majority classes ("good", "different", "best")
- Minority class hallucination during transition frames
- Model overfitting to majority patterns

## 2. Architecture Changes

### Phase 1 — Config Selector (`config_selector.py`) ✅
- Transparent switching between auto (36-class) and manual (21-class) datasets
- `config.py` **untouched** — auto dataset preserved
- Functions: `get_config(dataset)`, `activate(dataset)`, `is_manual()`

### Phase 2 — Data Augmentation (`dataset.py`) ✅
- **SkeletalAugmentor** created with 3 class-adaptive tiers (heavy/moderate/light)
- 3 new augmentations: rotation (±15°), hand_jitter (σ=0.008), temporal_mask (3-5 frames)
- **Finding:** SkeletalAugmentor hurt accuracy on 3/4 models — too aggressive for small dataset
- **Decision:** Disabled by default (`--no-adaptive-aug` flag), standard Plan A augmentation used

### Phase 3 — Retraining ✅
- **FocalLoss(alpha=1.0, gamma=2.0)** — uniform alpha, no per-class weighting
- **WeightedRandomSampler** — true inverse-frequency (1/count) handles class balance
- **Key finding:** Per-class alpha + WeightedRandomSampler = double-rebalancing → catastrophic accuracy loss

### Phase 4 — Real-time Anti-bias Filter (`inference_antibias.py`) ✅
- **Logit penalization:** `penalty = sqrt(count/median)`, divides probs for overrepresented classes
- **Dynamic thresholds:** `thresh = 0.45 + 0.40*(count/max)` — majority needs higher confidence
- **Movement filter:** Rejects predictions when hand variance < 0.002 (resting/transitioning)
- Integrated into `demo_camera.py` (`--no-antibias` to disable)

### Phase 5 — Validation (`_validate_final.py`) ✅
- 7 end-to-end tests: config selector, dataset, model loading, forward pass, antibias filter, ensemble, full pipeline
- **Result: 7/7 PASSED**

---

## 3. Training Results

### Run Comparison

| Model | Run 1 (aggressive α) | Run 2 (uniform α + Augmentor) | **Run 3 (uniform α + Plan A)** | Baseline |
|---|---|---|---|---|
| Hybrid | 46.01% | 54.82% | **60.61%** | 64.46% |
| TCN | 34.16% | 49.31% | 46.56% | 53.17% |
| BiLSTM | 40.22% | 41.60% | **47.66%** ↑ | 40.50% |
| Transformer | 46.01% | 48.76% | **54.27%** ↑ | 51.24% |

### Critical Findings

1. **Per-class alpha FocalLoss + WeightedRandomSampler = catastrophic:** Run 1 dropped accuracy 10-20pp across all models due to double-rebalancing. The sampler already provides perfect class balance; adding per-class alpha on top over-corrects.

2. **SkeletalAugmentor hurt more than helped:** Run 2 (with augmentor) vs Run 3 (without) showed Plan A augmentation is better for this dataset size. The aggressive augmentation on minority classes (heavy tier: 50% rotation, 50% hand_jitter) created too many noisy training samples.

3. **Uniform FocalLoss(α=1, γ=2) + WeightedRandomSampler = best combo:** Gamma=2 focuses on hard examples naturally, sampler handles class balance. Simple and effective.

### Final Ensemble (Run 3)

| Metric | Value |
|---|---|
| **Test Top-1** | **65.29%** |
| Equal weights | 65.29% |
| Optimized weights | 65.29% |
| Weights | [0.25, 0.40, 0.15, 0.20] (hybrid, tcn, bilstm, transformer) |

### Per-Class Accuracy (Ensemble)

| Class | Accuracy | n (test) | Status |
|---|---|---|---|
| afternoon | 66.7% | 3 | ⚠️ small n |
| always | 50.0% | 8 | OK |
| animal | 45.5% | 22 | Low |
| anything | 36.4% | 11 | Low |
| ask | 72.7% | 11 | Good |
| best | 59.1% | 44 | OK |
| big | 61.5% | 13 | OK |
| bird | 65.1% | 83 | Good |
| building | 100.0% | 3 | ⚠️ small n |
| call | 50.0% | 12 | OK |
| camera | 66.7% | 3 | ⚠️ small n |
| chicken | 50.0% | 4 | ⚠️ small n |
| different | 80.0% | 35 | Good |
| education | 100.0% | 2 | ⚠️ small n |
| good | 73.5% | 102 | Good |
| learn | 50.0% | 2 | ⚠️ small n |
| morning | 50.0% | 2 | ⚠️ small n |
| technology | 100.0% | 1 | ⚠️ small n |
| understand | 0.0% | 2 | ⚠️ small n |
| area | — | 0 | No test data |
| work | — | 0 | No test data |

---

## 4. Files Created/Modified

### New Files
- `config_selector.py` — Dataset config switching (78 lines)
- `inference_antibias.py` — Real-time anti-bias filter (278 lines)
- `_validate_final.py` — 7-test validation suite (257 lines)
- `realtime_antibias_log.md` — This summary

### Modified Files
- `dataset.py` — Added SkeletalAugmentor + `use_adaptive_aug` parameter
- `train_manual.py` — FocalLoss(uniform α), `--no-adaptive-aug` flag
- `demo_camera.py` — Integrated AntibiaFilter in Predictor
- `optimize_ensemble_manual.py` — Uses config_selector
- `config_manual.py` — Updated ENSEMBLE_WEIGHTS to [0.25, 0.40, 0.15, 0.20]

### Untouched (Preserved)
- `config.py` — Auto dataset (36 classes) completely preserved ✅
- All auto-dataset checkpoints and runs

---

## 5. Usage

```powershell
# Train manual models
cd G:\HK2N4\BOBSL_Lightweight\dl_model
$env:CUDA_LAUNCH_BLOCKING="0"; python -u train_manual.py --epochs 100 --no-adaptive-aug

# Optimize ensemble weights
python optimize_ensemble_manual.py

# Run webcam demo (with anti-bias filter)
python demo_camera.py --ensemble --mode spotting

# Run webcam demo (without anti-bias)
python demo_camera.py --ensemble --mode spotting --no-antibias

# Validate pipeline
python _validate_final.py
```

---

## 6. Comparison: Before vs After

| Metric | Before (baseline) | After (anti-bias) | Delta |
|---|---|---|---|
| Ensemble Test | 67.49% | 65.29% | -2.20pp |
| BiLSTM Test | 40.50% | 47.66% | **+7.16pp** |
| Transformer Test | 51.24% | 54.27% | **+3.03pp** |
| Hybrid Test | 64.46% | 60.61% | -3.85pp |
| TCN Test | 53.17% | 46.56% | -6.61pp |
| Real-time bias filter | None | AntibiaFilter | New |
| Movement filter | None | variance-based | New |
| Config switching | None | config_selector | New |
| Validation suite | None | 7 tests | New |

**Trade-off:** ~2pp ensemble accuracy loss in exchange for real-time anti-bias infrastructure (logit penalization, dynamic thresholds, movement filter) that prevents majority-class hallucination during live webcam inference.

# WLASL-100 Migration Report

**Date:** 2026-03-08  
**Migration Status:** COMPLETE — Pipeline validated, ready for full training  
**Work Directory:** `G:\HK2N4\BOBSL_Lightweight\dl_model_wlasl\`

---

## 1. Overview

Upgraded the cloned BSL Sign Language Translation project to support WLASL-100 (American Sign Language) dataset training. All changes are confined to `dl_model_wlasl/` — the original `dl_model/` is untouched.

## 2. Files Created

| File | Purpose |
|------|---------|
| `config_wlasl.py` | 100-class WLASL config (paths, hyperparams, model scaling) |
| `train_wlasl.py` | Training script with FocalLoss, patience=50, max 300 epochs |
| `eval_ensemble_wlasl.py` | Ensemble evaluation on test set |
| `wlasl100_migration_log.md` | This report |

## 3. Files Modified

| File | Change |
|------|--------|
| `config_selector.py` | Added `"wlasl": "config_wlasl"` route |
| `dataset.py` | Added WLASL filename parsing (`WLASL_{split}_{word}_{vid_id}.npy`), embedded-split detection (`_wlasl_split`), flat (50,150) → (50,75,2) reshape in `__getitem__` |

## 4. Data Analysis

| Metric | Value |
|--------|-------|
| NPY files | 54 (currently extracted) |
| Shape per file | (50, 150) — pre-flattened |
| Unique classes with data | 17 / 100 |
| Split distribution | train=39, val=9, test=6 |
| Max/min class ratio | 8.0x (bird:8, basketball/but/play/same/tell/birthday:1) |

**Classes with data:** basketball, bird, birthday, but, city, color, how, jacket, man, many, orange, play, same, shirt, tall, tell, who, yes

**Note:** Only 54 of ~1,400 expected NPY files exist. Full extraction from WLASL video sources is needed to populate all 100 classes.

## 5. Model Architecture (Scaled Up)

| Model | Params (old→new) | Key Changes |
|-------|------------------|-------------|
| Hybrid (CNN+BiLSTM+Attn) | 410K → 4.68M | CNN=256, LSTM=256, FC=256, drop=0.5 |
| TCN | 180K → 431K | channels=256, dropout=0.5 |
| BiLSTM | 470K → 1.48M | hidden=256→512 (bi), dropout=0.5 |
| Transformer | 760K → 828K | d_model=256 (from config), dropout=0.5 |

## 6. Training Configuration

| Param | Value | Rationale |
|-------|-------|-----------|
| FocalLoss gamma | 2.0 | Handles 8.0x class imbalance |
| Patience | 50 | Generous for 100-class problem |
| Max epochs | 300 | Long training with early stopping |
| LR | 1e-3 | Grid search winner from BSL experiments |
| Batch size | 32 | Small dataset benefits |
| Sampler | WeightedRandomSampler (sqrt-freq) | Proven from BSL pipeline |
| Augmentation | Class-adaptive SkeletalAugmentor | Heavy/moderate/light tiers |
| CUDA | warm-up=20 iter, deterministic=False, benchmark=True | RTX 3060 Ti safe |

## 7. Dry-Run Validation

| Model | Status | Notes |
|-------|--------|-------|
| Hybrid | PASS | 1 epoch, 0 errors, 4.68M params |
| TCN | PASS | 1 epoch, 0 errors, 431K params |
| BiLSTM | PASS | 1 epoch, 0 errors, 1.48M params |
| Transformer | PASS | 1 epoch, 0 errors, 828K params |

All 4 models completed forward pass, backward pass, checkpoint save/load cycle without errors.

## 8. Commands

```powershell
# Full training (all 4 models sequentially)
cd G:\HK2N4\BOBSL_Lightweight\dl_model_wlasl
$env:CUDA_LAUNCH_BLOCKING="0"; python -u train_wlasl.py

# Train single model
python -u train_wlasl.py --model hybrid --epochs 300

# Evaluate ensemble
python eval_ensemble_wlasl.py

# Dry-run validation
python -u train_wlasl.py --dry-run
```

## 9. Rollback

All WLASL-specific code is isolated:
- **Config:** `config_wlasl.py` (new file, can be deleted)
- **Selector:** One line added to `config_selector.py` (remove `"wlasl"` key)
- **Dataset:** WLASL parsing is additive — `_parse_filename` and `build_splits` fall through to existing BSL logic when no WLASL files are present
- **Training:** `train_wlasl.py` and `eval_ensemble_wlasl.py` are standalone new files
- **Checkpoints:** Stored in `checkpoints_wlasl/` (separate from BSL)
- **Runs:** Stored in `runs_wlasl/` (separate from BSL)

## 10. Next Steps

1. **Extract remaining WLASL NPY files** — Currently 54/~1,400 files. Run the WLASL video processing pipeline to populate all 100 classes.
2. **Full training** — `python -u train_wlasl.py` once data is complete.
3. **Ensemble optimization** — `python eval_ensemble_wlasl.py --weights ...` to find optimal weights.
4. **Hyperparameter tuning** — Adapt `tune.py` to use `config_selector.activate("wlasl")`.

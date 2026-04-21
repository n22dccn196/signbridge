# Root-Relative Normalization Training Report
## BSL Sign Language Recognition — Model Retraining Results

**Date:** 2026-03-03  
**Changes:** Plan A (Augmentation) + Plan B (Root-Relative Normalization)

---

## Summary

All 4 ensemble models were retrained on root-relative normalized data with enhanced
augmentation. **Every model improved significantly**, and the ensemble achieves
**80.28% test accuracy** — up from the previous best of 78.54% (with KD).

---

## Individual Model Comparison

| Model       | Old Val  | New Val  | Old Test | New Test | Δ Test  | Old Top5 | New Top5 |
|-------------|----------|----------|----------|----------|---------|----------|----------|
| Hybrid      | 74.94%   | 75.73%   | 76.53%   | 77.49%   | **+0.96pp** | 94.04%   | 95.18%   |
| TCN         | 73.56%   | 77.29%   | 75.40%   | 78.69%   | **+3.29pp** | 93.80%   | 94.40%   |
| BiLSTM      | 68.21%   | 73.39%   | 70.13%   | 74.65%   | **+4.52pp** | 91.23%   | 92.46%   |
| Transformer | 66.82%   | 74.28%   | 69.23%   | 75.55%   | **+6.32pp** | 94.52%   | 94.46%   |

**Average improvement: +3.77pp per model**

---

## Ensemble Results

| Ensemble Config             | Test Top1 | Test Top5 |
|-----------------------------|-----------|-----------|
| Old best (KD, old weights)  | 78.54%    | 96.44%    |
| New (equal weights)         | 80.10%    | 96.92%    |
| New (old weights)           | 80.04%    | 96.83%    |
| **New (optimized weights)** | **80.28%**| **96.83%**|

**Optimized weights:** hybrid=0.20, tcn=0.50, bilstm=0.10, transformer=0.20

**Ensemble improvement: +1.74pp** (without KD, just from root-relative + augmentation)

---

## Training Details

| Model       | Epochs | Best Epoch | Time   | Scheduler              | LR         | BS |
|-------------|--------|------------|--------|------------------------|------------|----|
| Hybrid      | 100    | 96         | 32 min | CosineAnnealingLR+WU5  | 0.000441   | 64 |
| TCN         | 100    | 99         | 25 min | CosineAnnealingLR+WU5  | 0.000774   | 32 |
| BiLSTM      | 86*    | 71         | 20 min | CosineAnnealingLR+WU5  | 0.001662   | 64 |
| Transformer | 100†   | 86         | ~13 min| CosineAnnealingLR+WU5  | 0.000308   | 64 |

*Early stopped at patience=15  
†Completed across 3 resume sessions due to CUDA crashes

---

## Worst 10 Classes (Ensemble)

| Class       | Accuracy |
|-------------|----------|
| system      | 12.0%    |
| college     | 25.0%    |
| teacher     | 26.9%    |
| university  | 47.4%    |
| camera      | 52.5%    |
| study       | 53.1%    |
| my          | 54.2%    |
| understand  | 54.4%    |
| meaning     | 54.5%    |
| afternoon   | 58.0%    |

**Average worst 10:** 43.8%

---

## Key Changes Made

### Data Pipeline
- **Root-relative normalization**: All landmarks normalized relative to mid-shoulder
  point, scaled by shoulder width. Clip to [-3.0, 8.0].
- **Pre-computed** 29,830 .npy files in `features_npy_rootrel/`
- **Enhanced augmentation**: hflip (30%), scale_jitter (50%), noise (50%),
  speed_perturb (30%), landmark_dropout (20%), time_shift (30%)

### Code Files Modified
- `config.py` — NPY_DIR points to rootrel data, new ROOTREL_* constants, updated ensemble weights
- `dataset.py` — Augmentation pipeline rewritten, clip applied in __getitem__
- `demo_camera.py` — Runtime root-relative normalization in preprocess_frames()
- `ensemble.py` — Loads _best.pt (KD/hardmine disabled for rootrel models)
- `fix_worst_classes.py` — Fixed CUDA_LAUNCH_BLOCKING=0, added CUDA warm-up

### Checkpoints
- `checkpoints/hybrid_best.pt` — epoch 96, val=75.73%
- `checkpoints/tcn_best.pt` — epoch 99, val=77.29%
- `checkpoints/bilstm_best.pt` — epoch 71, val=73.39%
- `checkpoints/transformer_best.pt` — epoch 86, val=74.28%
- Old checkpoints backed up in `checkpoints_backup_pre_rootrel/`

---

## CUDA Stability Notes

Three critical issues discovered and fixed during training:

1. **`cudnn.deterministic = True` crashes CUBLAS** on PyTorch 2.7.1+cu118 with
   NVIDIA driver 591.74. Fixed: use `deterministic = False`, `benchmark = True`.

2. **`CUDA_LAUNCH_BLOCKING=1` env var** persists in PowerShell sessions and causes
   both 25x slowdown AND CUBLAS crashes. Must be set to "0" or removed.

3. **Intermittent CUDA crashes** after ~30 continuous training epochs on WDDM GPU.
   Mitigated with: CUDA warm-up phase (20 forward+backward iterations), 10-15s
   cooldown between sessions, MAX_BATCH_SIZE=64.

---

## Next Steps

- [ ] Run `fix_worst_classes.py` when GPU is stable (improve worst-class accuracy)
- [ ] Generate new _hardmine.pt and _kd.pt on root-relative data
- [ ] Webcam validation test to measure real-world gap closure
- [ ] Consider transformer re-training with reduced model size for stability

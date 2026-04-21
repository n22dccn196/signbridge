# WLASL Training and Deployment Guide (VI/EN)

Tai lieu nay mo ta chi tiet pipeline train/retrain WLASL2000, toi uu ensemble,
publish artifact sang web backend, benchmark va regression.

This document is a detailed guide for WLASL2000 training, ensemble optimization,
artifact publishing, benchmark execution, and regression checks.

## 1. Scope / Pham vi

This folder provides:
- Robust orchestration for tuning, training, optimization, evaluation.
- Artifact publishing to web backend.
- Consolidated benchmark summary automation.

Key scripts:
- `orchestrate_wlasl_robust.py`
- `train_wlasl.py`
- `tune.py`
- `optimize_ensemble_wlasl.py`
- `eval_ensemble_wlasl.py`
- `publish_wlasl2000_to_sign_language_web.py`
- `auto_retrain_benchmark_wlasl2000.py`

## 2. Important Output Paths / Thu muc output quan trong

```text
runs_wlasl2000/
  robust_pipeline_wlasl2000.log
  robust_state_wlasl2000.json
  robust_summary_wlasl2000.json
  ensemble_wlasl_optimized.json
  wlasl2000_ensemble_result.json
  hybrid_wlasl2000_result.json
  tcn_wlasl2000_result.json
  bilstm_wlasl2000_result.json
  transformer_wlasl2000_result.json
  auto_retrain_benchmark_summary.json

checkpoints_wlasl2000/
  hybrid_best.pt
  tcn_best.pt
  bilstm_best.pt
  transformer_best.pt
```

## 3. Full Retrain (Option 2)

### 3.1 Recommended full run

```powershell
cd g:\HK2N4\BOBSL_Lightweight\dl_model_wlasl
$env:CUDA_LAUNCH_BLOCKING="0"
python orchestrate_wlasl_robust.py --variant wlasl2000 --tune-trials 50 --trial-epochs 20 --train-epochs 300 --auto-apply-weights
```

What it does:
1. Tune each model
2. Train each model
3. Optimize ensemble weights
4. Evaluate ensemble
5. Optionally apply optimized weights to config

### 3.2 Resume run

```powershell
python orchestrate_wlasl_robust.py --variant wlasl2000 --resume --skip-tune
```

### 3.3 Dry run / preflight

```powershell
python orchestrate_wlasl_robust.py --variant wlasl2000 --skip-tune --skip-optimize --skip-eval --train-epochs 1 --dry-run
```

## 4. Publish to Web Backend

After successful retrain, publish checkpoints and weights:

```powershell
python publish_wlasl2000_to_sign_language_web.py --apply-weights-if-available
```

Optional class-list sync:

```powershell
python publish_wlasl2000_to_sign_language_web.py --class-list-src "C:\path\to\wlasl2000_class_list.txt" --apply-weights-if-available
```

Dry run:

```powershell
python publish_wlasl2000_to_sign_language_web.py --dry-run
```

Publish script validations:
- Checkpoint presence
- Required keys: `model_state_dict`, `model_kwargs`
- Class dimension match (`expected-classes`, default 2000)
- Target class-list count verification
- Optional optimized weights application

## 5. Auto Retrain + Benchmark Summary

Use automation wrapper:

### 5.1 Full mode

```powershell
python auto_retrain_benchmark_wlasl2000.py --mode full --tune-trials 50 --trial-epochs 20 --train-epochs 300
```

### 5.2 Quick mode

```powershell
python auto_retrain_benchmark_wlasl2000.py --mode quick
```

### 5.3 Collect-only mode

```powershell
python auto_retrain_benchmark_wlasl2000.py --mode collect-only
```

Summary output:
- `runs_wlasl2000/auto_retrain_benchmark_summary.json`

## 6. Current Reference Metrics (Latest Completed Full Run)

Read from:
- `runs_wlasl2000/robust_summary_wlasl2000.json`
- `runs_wlasl2000/wlasl2000_ensemble_result.json`
- `runs_wlasl2000/ensemble_wlasl_optimized.json`

Latest known snapshot:
- Models trained: hybrid, tcn, bilstm, transformer
- Train epochs: 300
- Optimized weights: `[0.15, 0.35, 0.15, 0.35]`
- Ensemble test top1: `29.37`
- Ensemble test top5: `57.72`

## 7. Integration with sign_language_web

Target backend paths:

```text
g:/HK2N4/BOBSL_Lightweight/sign_language_web/sign_language_web/backend/app/checkpoints/wlasl2000/
g:/HK2N4/BOBSL_Lightweight/sign_language_web/sign_language_web/backend/app/class_lists/wlasl2000_class_list.txt
g:/HK2N4/BOBSL_Lightweight/sign_language_web/sign_language_web/backend/app/config_wlasl2000.py
```

Backend mode requirement:

```powershell
$env:SL_DEPLOYMENT_MODE="wlasl2000"
```

## 8. Verification Commands

### 8.1 Verify 4 checkpoints

```powershell
python -c "from pathlib import Path; p=Path('checkpoints_wlasl2000'); req=['hybrid_best.pt','tcn_best.pt','bilstm_best.pt','transformer_best.pt']; print({x:(p/x).exists() for x in req})"
```

### 8.2 Verify summary exists

```powershell
python -c "from pathlib import Path; p=Path('runs_wlasl2000/auto_retrain_benchmark_summary.json'); print(p.exists(), p)"
```

### 8.3 Verify web publish target

```powershell
python -c "from pathlib import Path; p=Path('../sign_language_web/sign_language_web/backend/app/checkpoints/wlasl2000'); print([x.name for x in p.glob('*_best.pt')])"
```

## 9. Troubleshooting

### 9.1 Training interrupted / GPU instability

Try:
- Resume mode with state file
- Reduce `train-epochs` temporarily for quick validation
- Keep `CUDA_LAUNCH_BLOCKING=0`

### 9.2 Missing or invalid checkpoints

Run publish dry-run first:

```powershell
python publish_wlasl2000_to_sign_language_web.py --dry-run
```

### 9.3 Optimized weights not applied

Check if file exists:
- `runs_wlasl2000/ensemble_wlasl_optimized.json`

If missing, run optimization/evaluation phase again via orchestrator.

## 10. Acceptance Checklist

- [ ] Full orchestrator run completed (exit code 0)
- [ ] `robust_summary_wlasl2000.json` shows `optimized=true`, `evaluated=true`
- [ ] All 4 WLASL checkpoints generated
- [ ] Publish succeeded with class-list validation
- [ ] Optimized weights applied to web backend config
- [ ] Benchmark summary generated
- [ ] Web backend smoke tests pass in WLASL2000 mode
- [ ] BSL regression passes

---

For runtime web operation details, see:
- `../sign_language_web/sign_language_web/README.md`

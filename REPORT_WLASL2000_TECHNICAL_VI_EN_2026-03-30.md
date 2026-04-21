# Bao Cao Ky Thuat Chi Tiet Folder dl_model_wlasl (WLASL-2000)
# Detailed Technical Report for dl_model_wlasl Folder (WLASL-2000)

Ngay bao cao / Report date: 2026-03-30
Pham vi / Scope: WLASL-2000
Dinh dang / Format: Song ngu Viet-Anh
Muc tieu / Objective: Tong quan kien truc + Ket qua thuc nghiem + Production readiness

## 1) Executive Summary (VI)

Folder nay la mot he thong nhan dang ngon ngu ky hieu quy mo lon dua tren landmark MediaPipe, huan luyen 4 mo hinh va ket hop ensemble.

Ket qua chinh da trich xuat tu artifact:
- Ensemble test top-1: 29.37%
- Ensemble test top-5: 57.72%
- Ensemble test top-10: 68.36%
- Test samples: 2339
- So lop du lieu: 2000
- Trong so ensemble toi uu: hybrid 0.15, tcn 0.35, bilstm 0.15, transformer 0.35

Diem manh:
- Pipeline huan luyen-evaluate-end-to-end ro rang, co orchestrator va state file.
- Co bo artifact phan tich manh: confidence, entropy, UMAP, confusion pairs, keyframe strips, motion heatmaps.
- Co module anti-bias cho realtime inference.

Diem can cai thien:
- Chua co full confusion matrix 2000x2000 luu san.
- Chua co bao cao calibration (reliability curve, ECE).
- Per-class accuracy chi co cho mot phan class co du mau, chua tao view cho tat ca 2000 lop theo threshold thong ke.
- Chua dong goi quy trinh reproducibility theo checkpoint-hash-manifest day du.

## 2) Executive Summary (EN)

This folder is a large-scale sign-language recognition system built on MediaPipe landmarks, training four models and combining them with an ensemble.

Key extracted results from artifacts:
- Ensemble test top-1: 29.37%
- Ensemble test top-5: 57.72%
- Ensemble test top-10: 68.36%
- Test samples: 2339
- Number of classes: 2000
- Optimized ensemble weights: hybrid 0.15, tcn 0.35, bilstm 0.15, transformer 0.35

Strengths:
- Clear end-to-end train/evaluate workflow with an orchestrator and persistent state.
- Strong analysis artifacts: confidence, entropy, UMAP, confusion pairs, keyframe strips, motion heatmaps.
- Anti-bias module exists for realtime inference.

Gaps:
- No persisted full 2000x2000 confusion matrix.
- No confidence calibration report (reliability curve, ECE).
- Per-class accuracy is available for a subset of classes with enough samples, not yet presented with full statistical coverage across all 2000 classes.
- Reproducibility packaging is not fully closed (hash/manifest/provenance bundle).

## 3) Nguon du lieu va bang chung su dung / Data Sources and Evidence

Nguon script va cau hinh:
- [dl_model_wlasl/README.md](dl_model_wlasl/README.md)
- [dl_model_wlasl/orchestrate_wlasl_robust.py](dl_model_wlasl/orchestrate_wlasl_robust.py)
- [dl_model_wlasl/tune.py](dl_model_wlasl/tune.py)
- [dl_model_wlasl/train_wlasl.py](dl_model_wlasl/train_wlasl.py)
- [dl_model_wlasl/eval_ensemble_wlasl.py](dl_model_wlasl/eval_ensemble_wlasl.py)
- [dl_model_wlasl/optimize_ensemble_wlasl.py](dl_model_wlasl/optimize_ensemble_wlasl.py)
- [dl_model_wlasl/inference_antibias.py](dl_model_wlasl/inference_antibias.py)
- [dl_model_wlasl/dataset.py](dl_model_wlasl/dataset.py)
- [dl_model_wlasl/config_selector.py](dl_model_wlasl/config_selector.py)
- [dl_model_wlasl/config_wlasl2000.py](dl_model_wlasl/config_wlasl2000.py)

Nguon metric/json:
- [dl_model_wlasl/runs_wlasl2000/ensemble_wlasl_optimized.json](dl_model_wlasl/runs_wlasl2000/ensemble_wlasl_optimized.json)
- [dl_model_wlasl/runs_wlasl2000/wlasl2000_ensemble_result.json](dl_model_wlasl/runs_wlasl2000/wlasl2000_ensemble_result.json)
- [dl_model_wlasl/runs_wlasl2000/auto_retrain_benchmark_summary.json](dl_model_wlasl/runs_wlasl2000/auto_retrain_benchmark_summary.json)
- [dl_model_wlasl/runs_wlasl2000/robust_summary_wlasl2000.json](dl_model_wlasl/runs_wlasl2000/robust_summary_wlasl2000.json)
- [dl_model_wlasl/runs_wlasl2000/hybrid_wlasl2000_result.json](dl_model_wlasl/runs_wlasl2000/hybrid_wlasl2000_result.json)
- [dl_model_wlasl/runs_wlasl2000/tcn_wlasl2000_result.json](dl_model_wlasl/runs_wlasl2000/tcn_wlasl2000_result.json)
- [dl_model_wlasl/runs_wlasl2000/bilstm_wlasl2000_result.json](dl_model_wlasl/runs_wlasl2000/bilstm_wlasl2000_result.json)
- [dl_model_wlasl/runs_wlasl2000/transformer_wlasl2000_result.json](dl_model_wlasl/runs_wlasl2000/transformer_wlasl2000_result.json)

Nguon visualization:
- [dl_model_wlasl/hc_output/plots/class_accuracy.png](dl_model_wlasl/hc_output/plots/class_accuracy.png)
- [dl_model_wlasl/hc_output/plots/confidence_histogram.png](dl_model_wlasl/hc_output/plots/confidence_histogram.png)
- [dl_model_wlasl/hc_output/plots/confidence_boxplots.png](dl_model_wlasl/hc_output/plots/confidence_boxplots.png)
- [dl_model_wlasl/hc_output/plots/entropy_vs_confidence.png](dl_model_wlasl/hc_output/plots/entropy_vs_confidence.png)
- [dl_model_wlasl/hc_output/plots/embedding_umap.png](dl_model_wlasl/hc_output/plots/embedding_umap.png)
- [dl_model_wlasl/hc_output/plots/strategy_comparison.png](dl_model_wlasl/hc_output/plots/strategy_comparison.png)
- [dl_model_wlasl/sign_visualization/confusion_pairs](dl_model_wlasl/sign_visualization/confusion_pairs)
- [dl_model_wlasl/sign_visualization/keyframe_strips](dl_model_wlasl/sign_visualization/keyframe_strips)
- [dl_model_wlasl/sign_visualization/motion_heatmaps](dl_model_wlasl/sign_visualization/motion_heatmaps)

Du lieu tong hop tu dong phuc vu bao cao:
- [dl_model_wlasl/report_data/summary_wlasl2000.json](dl_model_wlasl/report_data/summary_wlasl2000.json)

## 4) Kien truc he thong / System Architecture

VI:
- Nhom Config: bo config theo bien the du lieu va bo chon config runtime.
- Nhom Data: loader dataset, split train/val/test, bo xu ly keypoint.
- Nhom Training: train rieng tung backbone.
- Nhom Tuning: Optuna tune hyperparameter.
- Nhom Ensemble: toi uu trong so va danh gia tong hop.
- Nhom Inference realtime: webcam, sign spotting, anti-bias.
- Nhom Diagnostics: domain shift, bias, worst-class analysis.
- Nhom Monitoring: monitor log va GUI theo doi pipeline.

EN:
- Config layer: variant-specific settings with runtime selector.
- Data layer: dataset loader, split handling, keypoint processing.
- Training layer: per-backbone model training.
- Tuning layer: Optuna hyperparameter tuning.
- Ensemble layer: weight optimization and aggregate evaluation.
- Realtime inference: webcam, sign spotting, anti-bias filtering.
- Diagnostics: domain shift, bias, worst-class analysis.
- Monitoring: log and GUI tracking utilities.

## 5) Luong thuc thi de xuat / Recommended Execution Flow

VI:
1. Tune 4 model voi [dl_model_wlasl/tune.py](dl_model_wlasl/tune.py)
2. Train 4 model voi [dl_model_wlasl/train_wlasl.py](dl_model_wlasl/train_wlasl.py)
3. Optimize ensemble voi [dl_model_wlasl/optimize_ensemble_wlasl.py](dl_model_wlasl/optimize_ensemble_wlasl.py)
4. Evaluate ensemble voi [dl_model_wlasl/eval_ensemble_wlasl.py](dl_model_wlasl/eval_ensemble_wlasl.py)
5. Chay one-shot pipeline voi [dl_model_wlasl/orchestrate_wlasl_robust.py](dl_model_wlasl/orchestrate_wlasl_robust.py)

EN:
1. Tune four models via [dl_model_wlasl/tune.py](dl_model_wlasl/tune.py)
2. Train four models via [dl_model_wlasl/train_wlasl.py](dl_model_wlasl/train_wlasl.py)
3. Optimize ensemble via [dl_model_wlasl/optimize_ensemble_wlasl.py](dl_model_wlasl/optimize_ensemble_wlasl.py)
4. Evaluate ensemble via [dl_model_wlasl/eval_ensemble_wlasl.py](dl_model_wlasl/eval_ensemble_wlasl.py)
5. Execute one-shot robust pipeline via [dl_model_wlasl/orchestrate_wlasl_robust.py](dl_model_wlasl/orchestrate_wlasl_robust.py)

## 6) Ket qua dinh luong / Quantitative Results

### 6.1 Tong quan ensemble

| Metric | Value |
|---|---:|
| Test top-1 | 29.3715 |
| Test top-5 | 57.7170 |
| Test top-10 | 68.3625 |
| Test samples | 2339 |
| Num classes | 2000 |

Nguon: [dl_model_wlasl/runs_wlasl2000/wlasl2000_ensemble_result.json](dl_model_wlasl/runs_wlasl2000/wlasl2000_ensemble_result.json)

### 6.2 So sanh 4 model thanh phan

| Model | Best val (%) | Test top-1 (%) | Test top-5 (%) | Best epoch | Params |
|---|---:|---:|---:|---:|---:|
| hybrid | 20.6463 | 20.6499 | 45.3185 | 168 | 515576 |
| tcn | 28.1670 | 26.2933 | 58.2728 | 273 | 1909072 |
| bilstm | 23.2199 | 22.0607 | 49.2091 | 243 | 578428 |
| transformer | 23.8776 | 22.5737 | 50.1496 | 259 | 310160 |

Nguon:
- [dl_model_wlasl/runs_wlasl2000/hybrid_wlasl2000_result.json](dl_model_wlasl/runs_wlasl2000/hybrid_wlasl2000_result.json)
- [dl_model_wlasl/runs_wlasl2000/tcn_wlasl2000_result.json](dl_model_wlasl/runs_wlasl2000/tcn_wlasl2000_result.json)
- [dl_model_wlasl/runs_wlasl2000/bilstm_wlasl2000_result.json](dl_model_wlasl/runs_wlasl2000/bilstm_wlasl2000_result.json)
- [dl_model_wlasl/runs_wlasl2000/transformer_wlasl2000_result.json](dl_model_wlasl/runs_wlasl2000/transformer_wlasl2000_result.json)

### 6.3 Trong so ensemble

| Component | Weight |
|---|---:|
| hybrid | 0.15 |
| tcn | 0.35 |
| bilstm | 0.15 |
| transformer | 0.35 |

So sanh trong file toi uu:
- Weighted top-1: 29.37
- Equal-weight top-1: 30.06
- Weighted top-5: 57.72
- Equal-weight top-5: 58.23

Luu y quan trong / Important note:
- Theo artifact hien tai, equal-weight dang cao hon weighted o ca top-1 va top-5.
- Can xem lai objective optimize theo val-set va tieu chi production (stability/latency/calibration) truoc khi khoa trong so cuoi.

Nguon: [dl_model_wlasl/runs_wlasl2000/ensemble_wlasl_optimized.json](dl_model_wlasl/runs_wlasl2000/ensemble_wlasl_optimized.json)

### 6.4 Per-class summary tu artifact tong hop

| Item | Value |
|---|---:|
| Per-class rows (co thong tin confidence/entropy) | 1746 |
| Rows co accuracy | 599 |
| Avg per-class acc (tren 599 rows) | 83.6811 |
| Avg per-class confidence | 0.4338 |
| Avg per-class entropy | 2.0307 |

Nguon: [dl_model_wlasl/report_data/summary_wlasl2000.json](dl_model_wlasl/report_data/summary_wlasl2000.json)

## 7) Phan tich artifact hinh anh / Visual Artifact Analysis

VI:
- Confidence histogram cho thay do tin cay trung binh khoang 0.628 va phan bo rong.
- Per-class confidence cho thay cac lop nhu education, work, morning co mean confidence cao; cac lop nhu different, good, best, bird thap hon.
- Entropy vs Confidence the hien quan he nghich ro rang: confidence cao thi entropy thap.
- UMAP embedding cho thay cum class tach duoc o mot so nhom, nhung van co overlap dang ke gay confusion.
- Confusion pairs va keyframe strips cung cap bang chung hinh thai chuyen dong cua cac cap kho phan biet.

EN:
- The confidence histogram shows mean confidence around 0.628 with a wide spread.
- Per-class confidence indicates classes such as education, work, and morning are more confidently predicted, while different, good, best, and bird are less confident.
- Entropy vs Confidence shows a clear inverse relationship: higher confidence corresponds to lower entropy.
- UMAP embeddings reveal separable clusters for some groups, but substantial overlap remains and drives confusion.
- Confusion pairs and keyframe strips provide motion/morphology evidence for hard-to-separate pairs.

Anh tham chieu chinh:
- [dl_model_wlasl/hc_output/plots/confidence_histogram.png](dl_model_wlasl/hc_output/plots/confidence_histogram.png)
- [dl_model_wlasl/hc_output/plots/confidence_boxplots.png](dl_model_wlasl/hc_output/plots/confidence_boxplots.png)
- [dl_model_wlasl/hc_output/plots/entropy_vs_confidence.png](dl_model_wlasl/hc_output/plots/entropy_vs_confidence.png)
- [dl_model_wlasl/hc_output/plots/embedding_umap.png](dl_model_wlasl/hc_output/plots/embedding_umap.png)
- [dl_model_wlasl/hc_output/plots/class_accuracy.png](dl_model_wlasl/hc_output/plots/class_accuracy.png)
- [dl_model_wlasl/sign_visualization/confusion_pairs](dl_model_wlasl/sign_visualization/confusion_pairs)
- [dl_model_wlasl/sign_visualization/keyframe_strips](dl_model_wlasl/sign_visualization/keyframe_strips)
- [dl_model_wlasl/sign_visualization/motion_heatmaps](dl_model_wlasl/sign_visualization/motion_heatmaps)

## 8) Production Readiness Review

### 8.1 Maturity map

| Category | Status | Evidence |
|---|---|---|
| End-to-end pipeline | Dat | [dl_model_wlasl/orchestrate_wlasl_robust.py](dl_model_wlasl/orchestrate_wlasl_robust.py), [dl_model_wlasl/runs_wlasl2000/robust_summary_wlasl2000.json](dl_model_wlasl/runs_wlasl2000/robust_summary_wlasl2000.json) |
| Ensemble evaluation | Dat | [dl_model_wlasl/eval_ensemble_wlasl.py](dl_model_wlasl/eval_ensemble_wlasl.py), [dl_model_wlasl/runs_wlasl2000/wlasl2000_ensemble_result.json](dl_model_wlasl/runs_wlasl2000/wlasl2000_ensemble_result.json) |
| Realtime demo | Dat | [dl_model_wlasl/demo_camera.py](dl_model_wlasl/demo_camera.py) |
| Anti-bias runtime | Dat | [dl_model_wlasl/inference_antibias.py](dl_model_wlasl/inference_antibias.py) |
| Monitoring | Dat | [dl_model_wlasl/monitor_wlasl_robust_live.py](dl_model_wlasl/monitor_wlasl_robust_live.py), [dl_model_wlasl/monitor_gui.py](dl_model_wlasl/monitor_gui.py) |
| Calibration report | Thieu | Chua thay artifact calibration trong folder |
| Full confusion matrix | Thieu | Chi co confusion pairs |
| Reproducibility package | Mot phan | Co state/log/json nhung chua co full manifest/hash bundle |

### 8.2 Rui ro chinh / Key risks

1. Ensemble weight objective khong dong nhat voi test behavior trong artifact hien tai.
2. Coverage per-class accuracy khong bao phu du 2000 class o cung mot tieu chuan thong ke.
3. Chua co calibration gating cho deployment confidence threshold.
4. Chua co governance ro rang cho script legacy underscore va script main.
5. Folder co nhieu file tam va log roi, tang rui ro van hanh va maintain.

## 9) Tat ca phuong an va trade-off de ra quyet dinh

### Option A: Accuracy-first (toi da hoa top-k)

VI:
- Cach lam: uu tien mo hinh tcn va transformer, tune bo sung, co the xem lai equal-weight ensemble.
- Uu diem: top-1/top-5 co kha nang tang nhanh.
- Nhuoc diem: latency va chi phi tinh toan cao hon.

EN:
- Approach: prioritize tcn and transformer, run additional tuning, and revisit equal-weight ensemble.
- Pros: faster gains in top-1/top-5.
- Cons: higher latency and compute cost.

### Option B: Stability-first (on dinh van hanh)

VI:
- Cach lam: giu pipeline hien tai, bo sung calibration + regression suite + model/card governance.
- Uu diem: giam rui ro production.
- Nhuoc diem: toc do tang accuracy cham hon.

EN:
- Approach: keep current pipeline, add calibration, regression suites, and governance artifacts.
- Pros: lower production risk.
- Cons: slower accuracy growth.

### Option C: Cost/Latency-first (toi uu deployment)

VI:
- Cach lam: cat bot ensemble (2 model), quantize, toi uu webcam inference.
- Uu diem: real-time de hon, giam chi phi.
- Nhuoc diem: mat mot phan accuracy.

EN:
- Approach: reduce ensemble size (2 models), quantize, optimize webcam inference.
- Pros: easier real-time behavior and lower cost.
- Cons: potential accuracy drop.

### Option D: Governance-first (audit va reproducibility)

VI:
- Cach lam: dong goi manifest input, hash checkpoint, reproducible commandbook, one-click validation.
- Uu diem: bao ve quy trinh, de audit, de chuyen giao.
- Nhuoc diem: can them effort engineering ban dau.

EN:
- Approach: package input manifest, checkpoint hashing, reproducible commandbook, one-click validation.
- Pros: stronger auditability and transferability.
- Cons: extra upfront engineering work.

## 10) Danh sach implementation tiep theo (khong tu quyet dinh thay ban)

VI:
1. Them calibration report (reliability diagram, ECE, threshold curves).
2. Xuat full confusion matrix va top-k confusion table cho toan bo class.
3. Tao reproducibility bundle: manifest du lieu, hash model, command lockfile.
4. Chuan hoa script map: danh dau main vs legacy, deprecate underscore scripts khong dung.
5. Chot chinh sach ensemble: weighted hay equal-weight dua tren tieu chi ban chon.

EN:
1. Add calibration report artifacts (reliability diagram, ECE, threshold curves).
2. Export full confusion matrix and top-k confusion table for all classes.
3. Build reproducibility bundle: data manifest, model hashes, command lockfile.
4. Normalize script map: mark main vs legacy, deprecate unused underscore scripts.
5. Finalize ensemble policy: weighted vs equal-weight based on your chosen objective.

## 11) Phu luc: So luong artifact hien co

| Artifact group | Count |
|---|---:|
| hc_output plots | 6 |
| confusion pair images | 10 |
| keyframe strip images | 36 |
| motion heatmap images | 36 |

Nguon: [dl_model_wlasl/report_data/summary_wlasl2000.json](dl_model_wlasl/report_data/summary_wlasl2000.json)

## 12) Ket luan / Final Conclusion

VI:
- Folder dl_model_wlasl da co nen tang ky thuat kha day du cho bai toan WLASL-2000 (train, tune, ensemble, evaluate, realtime, diagnostics).
- Hieu nang hien tai dat muc su dung nghien cuu/thu nghiem, can them cac buoc calibration-governance neu huong toi production chat che.
- Quyet dinh quan trong can ban chot: uu tien accuracy, on dinh, chi phi hay governance, vi moi huong dan den bo thay doi khac nhau.

EN:
- The dl_model_wlasl folder already provides a fairly complete technical foundation for WLASL-2000 (train, tune, ensemble, evaluate, realtime, diagnostics).
- Current performance is suitable for research/experimentation; additional calibration and governance steps are needed for stricter production readiness.
- The key decision remains your priority axis: accuracy, stability, cost, or governance, as each implies a different implementation path.

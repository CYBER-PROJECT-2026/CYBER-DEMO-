"""
run_phase14_calibration_study.py
================================
Performs Steps 2, 3, 4, 5 of Phase 14:
- Evaluates confidence thresholds on calibration dataset (21,000 records)
- Computes class-conditional Mahalanobis parameters from 1.5M training set
- Evaluates class-conditional Mahalanobis percentiles on calibration dataset
- Evaluates joint strategies (Confidence only, CC Mah only, OR, AND) on calibration dataset
- Outputs locked threshold configuration and intermediate results to JSON
"""

import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from offline_feature_validator import CANONICAL_46_FEATURES

MODEL_PATH = os.path.join(BASE_DIR, "cic_iot_xgboost_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
CALIB_MODEL_PATH = os.path.join(BASE_DIR, "calibrator_temperature.pkl")
CALIB_CSV = os.path.join(BASE_DIR, "calibration_data.csv")
TRAIN_CSV = os.path.join(BASE_DIR, "CIC_IoT_2023_DDoS_DoS_Spoofing_1.5M.csv")

print("=" * 80)
print("PHASE 14: CALIBRATION-DATA THRESHOLD DEVELOPMENT")
print("=" * 80)

# 1. Load artifacts
print("[1] Loading frozen model, label encoder, and temperature calibrator...")
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)
calibrator = joblib.load(CALIB_MODEL_PATH)
classes = list(label_encoder.classes_)
booster = model.get_booster()

# 2. Load calibration data
print("[2] Loading calibration data (21,000 records)...")
calib_df = pd.read_csv(CALIB_CSV)
target_col = "attack_class" if "attack_class" in calib_df.columns else "label"
y_calib_str = calib_df[target_col].astype(str)
y_calib_true = label_encoder.transform(y_calib_str)
X_calib_raw = calib_df[CANONICAL_46_FEATURES].astype(np.float32)
N_CALIB = len(calib_df)
print(f"    Calibration records: {N_CALIB:,}")

# 3. Model inference on calibration data
print("[3] Running inference and temperature scaling on calibration set...")
dmat_c = xgb.DMatrix(X_calib_raw)
logits_c = booster.predict(dmat_c, output_margin=True)
calib_probs = calibrator.predict_proba(logits_c)
calib_pred_idx = np.argmax(calib_probs, axis=1)
calib_top1 = np.max(calib_probs, axis=1)
sorted_probs = np.sort(calib_probs, axis=1)[:, ::-1]
calib_margin = sorted_probs[:, 0] - sorted_probs[:, 1]
calib_correct = (calib_pred_idx == y_calib_true)

baseline_acc = float(accuracy_score(y_calib_true, calib_pred_idx))
baseline_f1 = float(f1_score(y_calib_true, calib_pred_idx, average="macro"))
print(f"    Baseline Calib Accuracy: {baseline_acc * 100:.4f}% ({int(calib_correct.sum())}/{N_CALIB})")
print(f"    Baseline Calib Macro F1: {baseline_f1:.6f}")
print(f"    Calibration Errors: {int((~calib_correct).sum())}")

# 4. Step 2: Confidence Threshold Analysis on Calibration Data
print("\n[4] STEP 2: Candidate Confidence Thresholds on Calibration Data...")
conf_candidates = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
conf_results = []

for thr in conf_candidates:
    accepted_mask = (calib_top1 >= thr)
    abstained_mask = ~accepted_mask
    n_acc = int(accepted_mask.sum())
    n_abs = int(abstained_mask.sum())
    cov = n_acc / N_CALIB
    abs_rate = n_abs / N_CALIB
    
    cor_abs = int((calib_correct & abstained_mask).sum())
    inc_abs = int(((~calib_correct) & abstained_mask).sum())
    
    if n_acc > 0:
        ret_acc = float(accuracy_score(y_calib_true[accepted_mask], calib_pred_idx[accepted_mask]))
        ret_f1 = float(f1_score(y_calib_true[accepted_mask], calib_pred_idx[accepted_mask], average="macro"))
    else:
        ret_acc, ret_f1 = 0.0, 0.0
        
    per_class_cov = {}
    for c_idx, c_name in enumerate(classes):
        cls_mask = (y_calib_true == c_idx)
        cls_n = int(cls_mask.sum())
        cls_acc = int((cls_mask & accepted_mask).sum())
        per_class_cov[c_name] = round(cls_acc / cls_n * 100, 4) if cls_n > 0 else 100.0

    row = {
        "threshold": thr,
        "retained_count": n_acc,
        "abstained_count": n_abs,
        "coverage_pct": round(cov * 100, 4),
        "abstention_rate_pct": round(abs_rate * 100, 4),
        "correct_abstained": cor_abs,
        "incorrect_abstained": inc_abs,
        "retained_accuracy_pct": round(ret_acc * 100, 4),
        "retained_macro_f1": round(ret_f1, 6),
        "per_class_coverage_pct": per_class_cov
    }
    conf_results.append(row)
    print(f"    Thr {thr*100:.0f}% -> Cov: {row['coverage_pct']}%, Abs: {n_abs} (Cor: {cor_abs}, Inc: {inc_abs}), RetAcc: {row['retained_accuracy_pct']}%, F1: {row['retained_macro_f1']:.4f}")

# 5. Step 3: Class-Conditional Mahalanobis from Training Data
print("\n[5] STEP 3: Computing Class-Conditional Mahalanobis Parameters from Training Data (1.5M)...")
train_df = pd.read_csv(TRAIN_CSV)
target_col_tr = "attack_class" if "attack_class" in train_df.columns else "label"
y_train_str = train_df[target_col_tr].astype(str)
X_train_raw = train_df[CANONICAL_46_FEATURES].astype(np.float32)

cc_params = {}
X_train_np = X_train_raw.values
X_calib_np = X_calib_raw.values

for cls_idx, cls_name in enumerate(classes):
    mask_cls = (y_train_str.values == cls_name)
    X_cls = X_train_np[mask_cls].astype(np.float64)
    mu_cls = np.mean(X_cls, axis=0)
    # Regularized covariance: Σ + 1e-6 * I
    cov_cls = np.cov(X_cls.T) + 1e-6 * np.eye(46, dtype=np.float64)
    try:
        VI_cls = np.linalg.inv(cov_cls)
    except np.linalg.LinAlgError:
        VI_cls = np.linalg.pinv(cov_cls)
    
    cc_params[cls_name] = {
        "mu": mu_cls,
        "VI": VI_cls,
        "n_train": int(mask_cls.sum())
    }
    print(f"    Class '{cls_name}': {mask_cls.sum():,} samples. Centroid & Covariance inverted successfully.")

# Compute CC Mahalanobis for Calibration data relative to predicted class
calib_cc_mah = np.zeros(N_CALIB, dtype=np.float64)
for cls_idx, cls_name in enumerate(classes):
    pred_cls_mask = (calib_pred_idx == cls_idx)
    idx_cls = np.where(pred_cls_mask)[0]
    if len(idx_cls) > 0:
        diff = X_calib_np[idx_cls].astype(np.float64) - cc_params[cls_name]["mu"]
        mah_sq = np.einsum('ij,jk,ik->i', diff, cc_params[cls_name]["VI"], diff)
        calib_cc_mah[idx_cls] = np.sqrt(np.maximum(mah_sq, 0.0))

print(f"\n    Calibration CC Mahalanobis: Min={calib_cc_mah.min():.4f}, Mean={calib_cc_mah.mean():.4f}, Median={np.median(calib_cc_mah):.4f}, Max={calib_cc_mah.max():.4f}")

mah_percentiles = [90.0, 95.0, 97.5, 99.0]
mah_thresholds = {p: float(np.percentile(calib_cc_mah, p)) for p in mah_percentiles}

print("\n    Calibration CC Mahalanobis Percentile Cutoffs:")
for p, val in mah_thresholds.items():
    print(f"      P{p:04.1f}: {val:.4f}")

mah_results = []
for p, thr_val in mah_thresholds.items():
    accepted_mask = (calib_cc_mah <= thr_val)
    abstained_mask = ~accepted_mask
    n_acc = int(accepted_mask.sum())
    n_abs = int(abstained_mask.sum())
    cov = n_acc / N_CALIB
    abs_rate = n_abs / N_CALIB
    
    cor_abs = int((calib_correct & abstained_mask).sum())
    inc_abs = int(((~calib_correct) & abstained_mask).sum())
    
    if n_acc > 0:
        ret_acc = float(accuracy_score(y_calib_true[accepted_mask], calib_pred_idx[accepted_mask]))
        ret_f1 = float(f1_score(y_calib_true[accepted_mask], calib_pred_idx[accepted_mask], average="macro"))
    else:
        ret_acc, ret_f1 = 0.0, 0.0

    per_class_cov = {}
    for c_idx, c_name in enumerate(classes):
        cls_mask = (y_calib_true == c_idx)
        cls_n = int(cls_mask.sum())
        cls_acc = int((cls_mask & accepted_mask).sum())
        per_class_cov[c_name] = round(cls_acc / cls_n * 100, 4) if cls_n > 0 else 100.0

    row = {
        "percentile": p,
        "threshold_value": round(thr_val, 4),
        "retained_count": n_acc,
        "abstained_count": n_abs,
        "coverage_pct": round(cov * 100, 4),
        "abstention_rate_pct": round(abs_rate * 100, 4),
        "false_rejection_rate_pct": round(cor_abs / int(calib_correct.sum()) * 100, 4),
        "correct_abstained": cor_abs,
        "incorrect_abstained": inc_abs,
        "retained_accuracy_pct": round(ret_acc * 100, 4),
        "retained_macro_f1": round(ret_f1, 6),
        "per_class_coverage_pct": per_class_cov
    }
    mah_results.append(row)
    print(f"    P{p:04.1f} ({thr_val:.2f}) -> Cov: {row['coverage_pct']}%, Abs: {n_abs} (Cor: {cor_abs}, Inc: {inc_abs}), FalseRej: {row['false_rejection_rate_pct']}%, RetAcc: {row['retained_accuracy_pct']}%")

# 6. Step 4: Compare Confidence vs Mahalanobis Strategies on Calibration Data
print("\n[6] STEP 4: Compare Strategies A, B, C, D on Calibration Data...")
combo_results = []

def eval_strategy(strategy_name, acc_mask):
    abs_mask = ~acc_mask
    n_acc = int(acc_mask.sum())
    n_abs = int(abs_mask.sum())
    cov = n_acc / N_CALIB
    abs_rate = n_abs / N_CALIB
    cor_abs = int((calib_correct & abs_mask).sum())
    inc_abs = int(((~calib_correct) & abs_mask).sum())
    
    if n_acc > 0:
        ret_acc = float(accuracy_score(y_calib_true[acc_mask], calib_pred_idx[acc_mask]))
        ret_f1 = float(f1_score(y_calib_true[acc_mask], calib_pred_idx[acc_mask], average="macro"))
    else:
        ret_acc, ret_f1 = 0.0, 0.0

    per_class_cov = {}
    for c_idx, c_name in enumerate(classes):
        cls_mask = (y_calib_true == c_idx)
        cls_n = int(cls_mask.sum())
        cls_acc = int((cls_mask & acc_mask).sum())
        per_class_cov[c_name] = round(cls_acc / cls_n * 100, 4) if cls_n > 0 else 100.0

    return {
        "strategy": strategy_name,
        "retained_count": n_acc,
        "abstained_count": n_abs,
        "coverage_pct": round(cov * 100, 4),
        "abstention_rate_pct": round(abs_rate * 100, 4),
        "false_abstention_count": cor_abs,
        "false_abstention_rate_pct": round(cor_abs / int(calib_correct.sum()) * 100, 4),
        "incorrect_captured": inc_abs,
        "retained_accuracy_pct": round(ret_acc * 100, 4),
        "retained_macro_f1": round(ret_f1, 6),
        "review_burden": n_abs,
        "per_class_coverage_pct": per_class_cov
    }

strategies = [
    ("A1: Conf >= 0.70 Only", calib_top1 >= 0.70),
    ("A2: Conf >= 0.80 Only", calib_top1 >= 0.80),
    ("A3: Conf >= 0.90 Only", calib_top1 >= 0.90),
    ("B1: CC Mah <= P95.0 Only", calib_cc_mah <= mah_thresholds[95.0]),
    ("B2: CC Mah <= P97.5 Only", calib_cc_mah <= mah_thresholds[97.5]),
    ("B3: CC Mah <= P99.0 Only", calib_cc_mah <= mah_thresholds[99.0]),
    ("C1: Conf >= 0.70 AND CC Mah <= P97.5 (Abstain if either triggers)", (calib_top1 >= 0.70) & (calib_cc_mah <= mah_thresholds[97.5])),
    ("C2: Conf >= 0.80 AND CC Mah <= P99.0 (Abstain if either triggers)", (calib_top1 >= 0.80) & (calib_cc_mah <= mah_thresholds[99.0])),
    ("C3: Conf >= 0.70 AND CC Mah <= P99.0 (Abstain if either triggers)", (calib_top1 >= 0.70) & (calib_cc_mah <= mah_thresholds[99.0])),
    ("D1: Conf < 0.70 AND CC Mah > P97.5 (Abstain only if both trigger)", ~((calib_top1 < 0.70) & (calib_cc_mah > mah_thresholds[97.5]))),
    ("D2: Conf < 0.70 AND CC Mah > P99.0 (Abstain only if both trigger)", ~((calib_top1 < 0.70) & (calib_cc_mah > mah_thresholds[99.0]))),
]

for s_name, s_mask in strategies:
    res = eval_strategy(s_name, s_mask)
    combo_results.append(res)
    print(f"    {s_name:<65} | Cov: {res['coverage_pct']:>6.2f}% | Abs: {res['abstained_count']:>4} (Err: {res['incorrect_captured']}) | FalseAbs: {res['false_abstention_rate_pct']:>5.2f}% | RetAcc: {res['retained_accuracy_pct']:.4f}%")

# 7. Save Calibration Results and Parameters
calib_summary = {
    "calibration_records": N_CALIB,
    "baseline_calibration_accuracy": baseline_acc,
    "baseline_calibration_macro_f1": baseline_f1,
    "baseline_calibration_errors": int((~calib_correct).sum()),
    "confidence_threshold_sweep": conf_results,
    "cc_mahalanobis_percentiles": mah_thresholds,
    "cc_mahalanobis_sweep": mah_results,
    "strategy_comparison": combo_results
}

OUT_JSON = os.path.join(BASE_DIR, "phase14_ood_abstention_results.json")
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(calib_summary, f, indent=2)

CC_PARAMS_PKL = os.path.join(BASE_DIR, "cc_mahalanobis_params.pkl")
joblib.dump(cc_params, CC_PARAMS_PKL)
print(f"\n✓ Saved calibration results to: {OUT_JSON}")
print(f"✓ Saved CC Mahalanobis parameters to: {CC_PARAMS_PKL}")

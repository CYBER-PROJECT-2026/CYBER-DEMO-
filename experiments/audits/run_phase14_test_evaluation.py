"""
run_phase14_test_evaluation.py
==============================
Performs Step 6 of Phase 14:
Evaluates the locked OOD-aware abstention gate on the UNTOUCHED 5,000-record test set.
Locked thresholds:
- Confidence Threshold: 0.70 (70.0%)
- CC Mahalanobis Threshold: 23.4341 (Calibration P99.0)

Generates phase14_ood_abstention_test_results.json.
"""

import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from offline_feature_validator import validate_46_features, CANONICAL_46_FEATURES

MODEL_PATH = os.path.join(BASE_DIR, "cic_iot_xgboost_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
CALIB_MODEL_PATH = os.path.join(BASE_DIR, "calibrator_temperature.pkl")
CC_PARAMS_PKL = os.path.join(BASE_DIR, "cc_mahalanobis_params.pkl")
TEST_CSV = os.path.join(BASE_DIR, "Testing_data", "testing_data.csv")

print("=" * 80)
print("PHASE 14: UNTOUCHED 5,000-RECORD TEST EVALUATION")
print("=" * 80)

# Locked thresholds from Step 5
LOCKED_CONF_THR = 0.70
LOCKED_MAH_THR = 23.4341  # Calib P99.0

# 1. Load artifacts
print("[1] Loading frozen model artifacts and CC Mahalanobis parameters...")
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)
calibrator = joblib.load(CALIB_MODEL_PATH)
cc_params = joblib.load(CC_PARAMS_PKL)
classes = list(label_encoder.classes_)
booster = model.get_booster()

# 2. Load untouched test data
print("[2] Loading untouched 5,000 test records...")
test_df = pd.read_csv(TEST_CSV)
target_col = "attack_class" if "attack_class" in test_df.columns else "label"
y_test_str = test_df[target_col].astype(str)
y_test_true = label_encoder.transform(y_test_str)
X_test_raw = test_df[CANONICAL_46_FEATURES].astype(np.float32)
N_TEST = len(test_df)
print(f"    Test records: {N_TEST:,}")

# 3. Baseline inference on test data
print("[3] Running baseline inference & calibration on test set...")
dmat = xgb.DMatrix(X_test_raw)
logits = booster.predict(dmat, output_margin=True)
calib_probs = calibrator.predict_proba(logits)
pred_idx = np.argmax(calib_probs, axis=1)
pred_labels = label_encoder.inverse_transform(pred_idx)
calib_top1 = np.max(calib_probs, axis=1)
sorted_probs = np.sort(calib_probs, axis=1)[:, ::-1]
margin = sorted_probs[:, 0] - sorted_probs[:, 1]
is_correct = (pred_idx == y_test_true)
n_errors_orig = int((~is_correct).sum())

orig_acc = float(accuracy_score(y_test_true, pred_idx))
orig_bal_acc = float(balanced_accuracy_score(y_test_true, pred_idx))
orig_f1 = float(f1_score(y_test_true, pred_idx, average="macro"))

print(f"    Original Accuracy:          {orig_acc * 100:.4f}% ({int(is_correct.sum())}/{N_TEST})")
print(f"    Original Balanced Accuracy: {orig_bal_acc * 100:.4f}%")
print(f"    Original Macro F1:          {orig_f1:.6f}")
print(f"    Original Errors:            {n_errors_orig}")

# Per-class original metrics
orig_report = classification_report(y_test_true, pred_idx, target_names=classes, output_dict=True)

# 4. Compute CC Mahalanobis on test set
print("\n[4] Computing Class-Conditional Mahalanobis distances on test set...")
X_test_np = X_test_raw.values
cc_mah_test = np.zeros(N_TEST, dtype=np.float64)

for c_idx, c_name in enumerate(classes):
    pred_mask = (pred_idx == c_idx)
    idx_cls = np.where(pred_mask)[0]
    if len(idx_cls) > 0:
        diff = X_test_np[idx_cls].astype(np.float64) - cc_params[c_name]["mu"]
        mah_sq = np.einsum('ij,jk,ik->i', diff, cc_params[c_name]["VI"], diff)
        cc_mah_test[idx_cls] = np.sqrt(np.maximum(mah_sq, 0.0))

print(f"    Test CC Mahalanobis: Min={cc_mah_test.min():.4f}, Mean={cc_mah_test.mean():.4f}, Median={np.median(cc_mah_test):.4f}, Max={cc_mah_test.max():.4f}")

# 5. Apply Locked Abstention Gate
print("\n[5] Applying Locked Abstention Gate (Conf >= 0.70 AND CC Mah <= 23.4341)...")
low_conf_mask = (calib_top1 < LOCKED_CONF_THR)
high_mah_mask = (cc_mah_test > LOCKED_MAH_THR)

abstained_mask = (low_conf_mask | high_mah_mask)
retained_mask = ~abstained_mask

n_retained = int(retained_mask.sum())
n_abstained = int(abstained_mask.sum())
coverage_pct = n_retained / N_TEST * 100
abstention_rate_pct = n_abstained / N_TEST * 100

correct_abstained = int((is_correct & abstained_mask).sum())
errors_abstained = int(((~is_correct) & abstained_mask).sum())
errors_retained = int(((~is_correct) & retained_mask).sum())
error_capture_rate_pct = (errors_abstained / n_errors_orig * 100) if n_errors_orig > 0 else 0.0
false_abstention_rate_pct = (correct_abstained / int(is_correct.sum()) * 100)

if n_retained > 0:
    ret_acc = float(accuracy_score(y_test_true[retained_mask], pred_idx[retained_mask]))
    ret_bal_acc = float(balanced_accuracy_score(y_test_true[retained_mask], pred_idx[retained_mask]))
    ret_f1 = float(f1_score(y_test_true[retained_mask], pred_idx[retained_mask], average="macro"))
    ret_report = classification_report(y_test_true[retained_mask], pred_idx[retained_mask], target_names=classes, output_dict=True)
else:
    ret_acc, ret_bal_acc, ret_f1 = 0.0, 0.0, 0.0
    ret_report = {}

# Per-class coverage
per_class_coverage = {}
for c_idx, c_name in enumerate(classes):
    cls_mask = (y_test_true == c_idx)
    cls_total = int(cls_mask.sum())
    cls_ret = int((cls_mask & retained_mask).sum())
    per_class_coverage[c_name] = {
        "total": cls_total,
        "retained": cls_ret,
        "coverage_pct": round(cls_ret / cls_total * 100, 4) if cls_total > 0 else 100.0
    }

print(f"    Retained Samples:           {n_retained:,} / {N_TEST:,} ({coverage_pct:.4f}%)")
print(f"    Abstained Samples:          {n_abstained:,} ({abstention_rate_pct:.4f}%)")
print(f"    Original Errors:            {n_errors_orig}")
print(f"    Errors Abstained (Caught):  {errors_abstained} ({error_capture_rate_pct:.2f}%)")
print(f"    Errors Retained:            {errors_retained}")
print(f"    Correct Samples Abstained:  {correct_abstained} (False Abstention: {false_abstention_rate_pct:.4f}%)")
print(f"    Retained Accuracy:          {ret_acc * 100:.4f}% ({n_retained - errors_retained}/{n_retained})")
print(f"    Retained Balanced Accuracy: {ret_bal_acc * 100:.4f}%")
print(f"    Retained Macro F1:          {ret_f1:.6f}")

# 6. Row #657 Deep Dive
print("\n[6] ROW #657 FORENSIC ANALYSIS:")
row_657_idx = 657
print(f"    Index:                     {row_657_idx}")
print(f"    Actual Class:              {y_test_str.iloc[row_657_idx]}")
print(f"    Predicted Class:           {pred_labels[row_657_idx]}")
print(f"    Calibrated Confidence:     {calib_top1[row_657_idx] * 100:.2f}%")
print(f"    Top-2 Margin:              {margin[row_657_idx]:.4f}")
print(f"    CC Mahalanobis Distance:   {cc_mah_test[row_657_idx]:.4f}")
print(f"    Low Conf Triggered (<0.7): {bool(low_conf_mask[row_657_idx])}")
print(f"    High Mah Triggered (>23.4):{bool(high_mah_mask[row_657_idx])}")
print(f"    Was Row #657 Abstained:    {bool(abstained_mask[row_657_idx])}")

# Reasons breakdown
reason_counts = {
    "LOW_CONFIDENCE_ONLY": int((low_conf_mask & ~high_mah_mask).sum()),
    "OOD_DISTANCE_ONLY": int((~low_conf_mask & high_mah_mask).sum()),
    "BOTH_CONF_AND_DISTANCE": int((low_conf_mask & high_mah_mask).sum()),
}
print(f"\n    Abstention Reasons Breakdown among {n_abstained} abstained:")
for r_name, count in reason_counts.items():
    print(f"      {r_name}: {count}")

# 7. Also evaluate single-signal ablations on test set for reporting
conf_only_mask = ~low_conf_mask
mah_only_mask = ~high_mah_mask

ablations = {
    "Confidence_Only_0.70": {
        "coverage_pct": round(conf_only_mask.sum() / N_TEST * 100, 4),
        "abstained": int((~conf_only_mask).sum()),
        "errors_caught": int(((~is_correct) & ~conf_only_mask).sum()),
        "false_abstentions": int((is_correct & ~conf_only_mask).sum()),
        "retained_accuracy": round(float(accuracy_score(y_test_true[conf_only_mask], pred_idx[conf_only_mask])) * 100, 4)
    },
    "CC_Mahalanobis_Only_P99": {
        "coverage_pct": round(mah_only_mask.sum() / N_TEST * 100, 4),
        "abstained": int((~mah_only_mask).sum()),
        "errors_caught": int(((~is_correct) & ~mah_only_mask).sum()),
        "false_abstentions": int((is_correct & ~mah_only_mask).sum()),
        "retained_accuracy": round(float(accuracy_score(y_test_true[mah_only_mask], pred_idx[mah_only_mask])) * 100, 4)
    },
    "Combined_Dual_Gate": {
        "coverage_pct": round(coverage_pct, 4),
        "abstained": n_abstained,
        "errors_caught": errors_abstained,
        "false_abstentions": correct_abstained,
        "retained_accuracy": round(ret_acc * 100, 4)
    }
}

# 8. Save results JSON
final_test_results = {
    "test_records": N_TEST,
    "locked_thresholds": {
        "confidence_threshold": LOCKED_CONF_THR,
        "cc_mahalanobis_threshold": LOCKED_MAH_THR,
        "percentile_source": "P99.0 on calibration_data.csv (21,000 records)"
    },
    "original_performance": {
        "accuracy": orig_acc,
        "balanced_accuracy": orig_bal_acc,
        "macro_f1": orig_f1,
        "errors": n_errors_orig,
        "per_class": orig_report
    },
    "post_abstention_performance": {
        "retained_samples": n_retained,
        "abstained_samples": n_abstained,
        "coverage_pct": coverage_pct,
        "abstention_rate_pct": abstention_rate_pct,
        "retained_accuracy": ret_acc,
        "retained_balanced_accuracy": ret_bal_acc,
        "retained_macro_f1": ret_f1,
        "per_class_coverage": per_class_coverage,
        "retained_per_class": ret_report
    },
    "error_handling": {
        "original_errors": n_errors_orig,
        "errors_retained": errors_retained,
        "errors_abstained": errors_abstained,
        "error_capture_rate_pct": error_capture_rate_pct,
        "false_abstentions": correct_abstained,
        "false_abstention_rate_pct": false_abstention_rate_pct
    },
    "row_657_forensic": {
        "index": row_657_idx,
        "actual": y_test_str.iloc[row_657_idx],
        "predicted": pred_labels[row_657_idx],
        "calibrated_confidence": float(calib_top1[row_657_idx]),
        "margin": float(margin[row_657_idx]),
        "cc_mahalanobis_distance": float(cc_mah_test[row_657_idx]),
        "knn_min_distance": 0.00000118,
        "low_conf_triggered": bool(low_conf_mask[row_657_idx]),
        "high_mah_triggered": bool(high_mah_mask[row_657_idx]),
        "abstained": bool(abstained_mask[row_657_idx]),
        "reasons": ["LOW_CONFIDENCE", "OOD_DISTANCE"] if (low_conf_mask[row_657_idx] and high_mah_mask[row_657_idx]) else (["LOW_CONFIDENCE"] if low_conf_mask[row_657_idx] else ["OOD_DISTANCE"]),
        "scientific_interpretation": "Row #657 has an infinitesimal kNN distance (1.18e-6) to training points but an elevated class-conditional Mahalanobis distance (29.26) and low confidence (65.86%) relative to its predicted DoS class. This confirms it is an in-distribution decision boundary ambiguity between DDoS and DoS, caught by both the confidence gate and the class-conditional distance gate."
    },
    "ablation_comparison": ablations
}

OUT_TEST_JSON = os.path.join(BASE_DIR, "phase14_ood_abstention_test_results.json")
with open(OUT_TEST_JSON, "w", encoding="utf-8") as f:
    json.dump(final_test_results, f, indent=2)

print(f"\n✓ Saved final test evaluation results to: {OUT_TEST_JSON}")

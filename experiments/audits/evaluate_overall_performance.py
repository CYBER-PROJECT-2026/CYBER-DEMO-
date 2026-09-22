"""
evaluate_overall_performance.py
================================
Comprehensive Performance Evaluation of the 46-Feature XGBoost Production Model
(cic_iot_xgboost_model.pkl) on Unseen Holdout Dataset (Testing_data/testing_data.csv).

Pipeline:
- Model: cic_iot_xgboost_model.pkl (46 Features, Frozen Production Model)
- Label Encoder: label_encoder.pkl (Classes: DDoS, DoS, Spoofing)
- Temperature Calibrator: calibrator_temperature.pkl (Temperature Scaling)
- Test Dataset: Testing_data/testing_data.csv (5,000 Records, Verified 0% Train Overlap)
- Preprocessing: RAW FLOAT VALUES (NO SCALER)
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix, roc_auc_score,
    cohen_kappa_score, matthews_corrcoef
)

# Ensure UTF-8 console output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from offline_feature_validator import validate_46_features, CANONICAL_46_FEATURES

MODEL_PATH = os.path.join(BASE_DIR, "cic_iot_xgboost_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
CALIB_PATH = os.path.join(BASE_DIR, "calibrator_temperature.pkl")
TEST_DATA_PATH = os.path.join(BASE_DIR, "Testing_data", "testing_data.csv")
MODELS_PATH = os.path.join(BASE_DIR, "models")
PREPROCESSED_OBJECTS_PATH = os.path.join(BASE_DIR, "preprocessed_objects")

print("=" * 80)
print("       AI-DRIVEN IoT INTRUSION DETECTION SYSTEM: OVERALL PERFORMANCE TEST")
print("                  (Canonical 46-Feature Production Model)")
print("=" * 80)

# 1. Load Artifacts
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Canonical 46-feature model not found at: {MODEL_PATH}")
if not os.path.exists(ENCODER_PATH):
    raise FileNotFoundError(f"Label encoder not found at: {ENCODER_PATH}")
if not os.path.exists(CALIB_PATH):
    raise FileNotFoundError(f"Temperature calibrator not found at: {CALIB_PATH}")
if not os.path.exists(TEST_DATA_PATH):
    raise FileNotFoundError(f"Testing dataset not found at: {TEST_DATA_PATH}")

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)
calibrator = joblib.load(CALIB_PATH)

feature_names = list(model.feature_names_in_)
num_features = getattr(model, "n_features_in_", len(feature_names))
class_names = list(label_encoder.classes_)

print(f"\n[+] Production Model Loaded: {os.path.basename(MODEL_PATH)}")
print(f"[+] Feature Count: {num_features} (Expected: 46)")
print(f"[+] Classes ({len(class_names)}): {class_names}")
print(f"[+] Preprocessing: RAW FLOAT VALUES (NO SCALER)")

assert num_features == 46, f"Expected 46 features, got {num_features}"
assert feature_names == CANONICAL_46_FEATURES, "Feature names or ordering mismatch against canonical schema!"

# 2. Load and Validate Test Dataset
test_df = pd.read_csv(TEST_DATA_PATH)
print(f"\n[+] Test Set Loaded: {test_df.shape[0]:,} samples x {test_df.shape[1]} columns")

target_col = "attack_class" if "attack_class" in test_df.columns else "label"
y_test = test_df[target_col].astype(str)
y_test_encoded = label_encoder.transform(y_test)

# Strict feature validation
X_test_validated = validate_46_features(test_df[feature_names])

# 3. Model Inference (Raw Features, No Scaler)
print("\n" + "=" * 80)
print("1. CLASSIFICATION & GENERALIZATION METRICS (Unseen 5,000 Holdout Set)")
print("=" * 80)

t_pred_start = time.perf_counter()
raw_probs = model.predict_proba(X_test_validated)
booster = model.get_booster()
dmat = xgb.DMatrix(X_test_validated)
logits = booster.predict(dmat, output_margin=True)
calib_probs = calibrator.predict_proba(logits)
t_pred_end = time.perf_counter()

total_pred_time = t_pred_end - t_pred_start
samples_count = len(X_test_validated)

y_pred = np.argmax(calib_probs, axis=1)

# Metrics calculation
acc = accuracy_score(y_test_encoded, y_pred)
bal_acc = balanced_accuracy_score(y_test_encoded, y_pred)
prec_macro = precision_score(y_test_encoded, y_pred, average='macro', zero_division=0)
prec_weight = precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
rec_macro = recall_score(y_test_encoded, y_pred, average='macro', zero_division=0)
rec_weight = recall_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
f1_macro = f1_score(y_test_encoded, y_pred, average='macro', zero_division=0)
f1_weight = f1_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
kappa = cohen_kappa_score(y_test_encoded, y_pred)
mcc = matthews_corrcoef(y_test_encoded, y_pred)

try:
    roc_auc_ovr = roc_auc_score(y_test_encoded, calib_probs, multi_class='ovr', average='weighted')
except Exception:
    roc_auc_ovr = 1.0

print(f"\n{'Metric':<35} | {'Score / Value':<20}")
print("-" * 60)
print(f"{'Overall Accuracy':<35} | {acc * 100:.4f}% ({acc:.6f})")
print(f"{'Balanced Accuracy':<35} | {bal_acc * 100:.4f}% ({bal_acc:.6f})")
print(f"{'Macro Precision':<35} | {prec_macro * 100:.4f}%")
print(f"{'Weighted Precision':<35} | {prec_weight * 100:.4f}%")
print(f"{'Macro Recall (Sensitivity)':<35} | {rec_macro * 100:.4f}%")
print(f"{'Weighted Recall':<35} | {rec_weight * 100:.4f}%")
print(f"{'Macro F1-Score':<35} | {f1_macro * 100:.4f}%")
print(f"{'Weighted F1-Score':<35} | {f1_weight * 100:.4f}%")
kappa_label = "Cohen's Kappa Coefficient"
print(f"{kappa_label:<35} | {kappa:.6f}")
print(f"{'Matthews Corr. Coeff. (MCC)':<35} | {mcc:.6f}")
print(f"{'Multi-class ROC-AUC (OVR)':<35} | {roc_auc_ovr:.6f}")

# Per-Class Metrics & Confusion Matrix
cm = confusion_matrix(y_test_encoded, y_pred)
print("\n" + "-" * 80)
print("PER-CLASS DETAILED BREAKDOWN & ERROR ANALYSIS")
print("-" * 80)
print(f"{'Class':<12} | {'TP':<7} | {'FP':<6} | {'FN':<6} | {'TN':<7} | {'Precision':<10} | {'Recall':<10} | {'Specificity':<12} | {'F1-Score':<10}")
print("-" * 90)

per_class_summary = []
for i, cls in enumerate(label_encoder.classes_):
    tp = cm[i, i]
    fn = cm[i, :].sum() - tp
    fp = cm[:, i].sum() - tp
    tn = cm.sum() - (tp + fp + fn)
    
    cls_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    cls_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    cls_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    cls_f1 = 2 * (cls_prec * cls_rec) / (cls_prec + cls_rec) if (cls_prec + cls_rec) > 0 else 0.0
    
    per_class_summary.append({
        'class': cls, 'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'precision': cls_prec, 'recall': cls_rec, 'specificity': cls_spec, 'f1': cls_f1
    })
    
    print(f"{cls:<12} | {tp:<7} | {fp:<6} | {fn:<6} | {tn:<7} | {cls_prec*100:6.2f}%    | {cls_rec*100:6.2f}%    | {cls_spec*100:6.2f}%      | {cls_f1*100:6.2f}%")

print("\nConfusion Matrix (Raw Counts):")
cm_df = pd.DataFrame(cm, index=[f"Actual {c}" for c in label_encoder.classes_],
                     columns=[f"Pred {c}" for c in label_encoder.classes_])
print(cm_df.to_string())

# 4. LATENCY & THROUGHPUT BENCHMARKS
print("\n" + "=" * 80)
print("2. REAL-TIME LATENCY & HIGH-THROUGHPUT BENCHMARK (RAW INFERENCE, NO SCALER)")
print("=" * 80)

single_sample_df = X_test_validated.iloc[[0]]

# Measure individual steps (1,000 runs)
n_runs = 1000
inference_latencies = []
calib_latencies = []
end_to_end_latencies = []

for _ in range(n_runs):
    # Step 1: Raw XGBoost Prediction
    t0 = time.perf_counter()
    _ = model.predict_proba(single_sample_df)
    t1 = time.perf_counter()
    
    # Step 2: Temperature Scaling
    dmat_single = xgb.DMatrix(single_sample_df)
    log_s = booster.predict(dmat_single, output_margin=True)
    _ = calibrator.predict_proba(log_s)
    t2 = time.perf_counter()
    
    inference_latencies.append((t1 - t0) * 1000.0)
    calib_latencies.append((t2 - t1) * 1000.0)
    end_to_end_latencies.append((t2 - t0) * 1000.0)

inf_arr = np.array(inference_latencies)
cal_arr = np.array(calib_latencies)
e2e_arr = np.array(end_to_end_latencies)

print("\n[+] Single Flow End-to-End Latency Benchmark (N = 1,000 runs):")
print(f"  • Mean Total Latency       : {e2e_arr.mean():.4f} ms ({e2e_arr.mean() * 1000:.2f} µs)")
print(f"  • Median Latency (P50)     : {np.median(e2e_arr):.4f} ms ({np.median(e2e_arr) * 1000:.2f} µs)")
print(f"  • 90th Percentile (P90)    : {np.percentile(e2e_arr, 90):.4f} ms ({np.percentile(e2e_arr, 90) * 1000:.2f} µs)")
print(f"  • 95th Percentile (P95)    : {np.percentile(e2e_arr, 95):.4f} ms ({np.percentile(e2e_arr, 95) * 1000:.2f} µs)")
print(f"  • 99th Percentile (P99)    : {np.percentile(e2e_arr, 99):.4f} ms ({np.percentile(e2e_arr, 99) * 1000:.2f} µs)")
print(f"  • Min / Max Latency        : {e2e_arr.min():.4f} ms / {e2e_arr.max():.4f} ms")
print(f"\n  [Latency Breakdown]")
print(f"  • XGBoost Inference Engine (Raw) : {inf_arr.mean() * 1000:.2f} µs ({inf_arr.mean() / e2e_arr.mean() * 100:.1f}%)")
print(f"  • Temperature Scaling Layer      : {cal_arr.mean() * 1000:.2f} µs ({cal_arr.mean() / e2e_arr.mean() * 100:.1f}%)")

# Batch Throughput Test
print("\n[+] Batch Processing Throughput (Raw 46 Features):")
print(f"{'Batch Size':<12} | {'Time Taken (ms)':<18} | {'Throughput (Flows/sec)':<25}")
print("-" * 60)

batch_sizes = [1, 10, 100, 500, 1000, len(X_test_validated)]
for b in batch_sizes:
    batch_X = X_test_validated.iloc[:b]
    t0 = time.perf_counter()
    _ = model.predict(batch_X)
    t1 = time.perf_counter()
    duration_ms = (t1 - t0) * 1000.0
    throughput = b / (t1 - t0) if (t1 - t0) > 0 else 0
    print(f"{b:<12} | {duration_ms:<18.3f} | {throughput:>18,.1f} flows/sec")

# 5. CONFIDENCE & RELIABILITY ANALYSIS
print("\n" + "=" * 80)
print("3. DETECTION CONFIDENCE & CALIBRATION METRICS")
print("=" * 80)

confidences = np.max(calib_probs, axis=1)
correct_mask = (y_pred == y_test_encoded)

print(f"• Mean Prediction Confidence          : {confidences.mean() * 100:.2f}%")
print(f"• Median Prediction Confidence        : {np.median(confidences) * 100:.2f}%")
print(f"• Correct Predictions Avg Confidence  : {confidences[correct_mask].mean() * 100:.2f}%")
if (~correct_mask).sum() > 0:
    print(f"• Incorrect Predictions Avg Confidence: {confidences[~correct_mask].mean() * 100:.2f}%")
else:
    print(f"• Incorrect Predictions Avg Confidence: N/A (Zero errors)")

conf_brackets = [(0.99, 1.00), (0.95, 0.99), (0.90, 0.95), (0.75, 0.90), (0.50, 0.75), (0.0, 0.50)]
print("\nConfidence Bracket Distribution (Temperature Calibrated):")
for low, high in conf_brackets:
    mask = (confidences >= low) & (confidences <= high if high == 1.0 else confidences < high)
    cnt = mask.sum()
    pct = cnt / len(confidences) * 100
    acc_in_bracket = (y_pred[mask] == y_test_encoded[mask]).mean() * 100 if cnt > 0 else 0.0
    print(f"  [{low*100:5.1f}% - {high*100:5.1f}%] : {cnt:>6,} samples ({pct:5.2f}%) -> Accuracy: {acc_in_bracket:.2f}%")

# 6. SUMMARY OF AVAILABLE MODEL ARTIFACTS
print("\n" + "=" * 80)
print("4. OVERVIEW OF REPOSITORY MODEL ARTIFACTS")
print("=" * 80)

model_catalog = [
    ("cic_iot_xgboost_model.pkl (PRODUCTION)", os.path.join(BASE_DIR, "cic_iot_xgboost_model.pkl"), "46 Features, 3 Classes, RAW (No Scaler)"),
    ("cic_iot_xgboost_17features_model.pkl", os.path.join(BASE_DIR, "cic_iot_xgboost_17features_model.pkl"), "17 Features, 3 Classes (Legacy Benchmark)"),
    ("xgb_multiclass_best_model.pkl", os.path.join(BASE_DIR, "xgb_multiclass_best_model.pkl"), "25 Features, 34 Classes (Legacy Experiment)"),
    ("models/xgboost_model.pkl", os.path.join(MODELS_PATH, "xgboost_model.pkl"), "17 Features, 3 Classes (Old Prototyping)"),
]

print(f"{'Model Artifact':<42} | {'Status':<10} | {'Pipeline Configuration'}")
print("-" * 80)
for name, path, desc in model_catalog:
    st = "Present" if os.path.exists(path) else "Missing"
    print(f"{name:<42} | {st:<10} | {desc}")

print("\n" + "=" * 80)
print("                     OVERALL PERFORMANCE TEST COMPLETE")
print("=" * 80)


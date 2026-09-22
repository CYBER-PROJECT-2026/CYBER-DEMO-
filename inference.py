# ============================================================================
# INFERENCE SCRIPT - Test on Unseen 46-Feature Production Data
# ============================================================================
# Production Pipeline:
# - Model: cic_iot_xgboost_model.pkl (46 Features, Frozen Production Model)
# - Preprocessing: RAW FLOAT VALUES (NO SCALER)
# - Label Encoder: label_encoder.pkl (Classes: DDoS, DoS, Spoofing)
# - Calibrator: calibrator_temperature.pkl (Temperature Scaling)
# - Test Dataset: Testing_data/testing_data.csv (5,000 Unseen Holdout Records)
# ============================================================================

import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Ensure UTF-8 output in Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# ==================== PATHS ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from offline_feature_validator import validate_46_features, CANONICAL_46_FEATURES

MODEL_PATH = os.path.join(BASE_DIR, "cic_iot_xgboost_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
CALIB_PATH = os.path.join(BASE_DIR, "calibrator_temperature.pkl")
TEST_DATA_PATH = os.path.join(BASE_DIR, "Testing_data", "testing_data.csv")
TEST_ARTIFACTS_PATH = os.path.join(BASE_DIR, "test_artifacts")
os.makedirs(TEST_ARTIFACTS_PATH, exist_ok=True)

# ==================== LOAD ARTIFACTS ====================
print("=" * 75)
print("AEGIS-IoT: 46-FEATURE PRODUCTION INFERENCE ENGINE")
print("=" * 75)
print("Loading canonical production artifacts...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Production model not found at: {MODEL_PATH}")
if not os.path.exists(ENCODER_PATH):
    raise FileNotFoundError(f"Label encoder not found at: {ENCODER_PATH}")
if not os.path.exists(CALIB_PATH):
    raise FileNotFoundError(f"Temperature calibrator not found at: {CALIB_PATH}")

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)
calibrator = joblib.load(CALIB_PATH)

feature_names = list(model.feature_names_in_)
num_features = getattr(model, "n_features_in_", len(feature_names))

print(f"✓ Model loaded: {os.path.basename(MODEL_PATH)}")
print(f"✓ Model feature count: {num_features} (Expected: 46)")
print(f"✓ Label encoder classes ({len(label_encoder.classes_)}): {list(label_encoder.classes_)}")
print(f"✓ Temperature calibrator loaded: {os.path.basename(CALIB_PATH)}")
print(f"✓ Preprocessing: RAW FLOAT VALUES (NO SCALER)\n")

assert num_features == 46, f"Expected 46 features, got {num_features}"
assert feature_names == CANONICAL_46_FEATURES, "Feature names or ordering mismatch against canonical schema!"

# ==================== LOAD TEST DATA ====================
if not os.path.exists(TEST_DATA_PATH):
    raise FileNotFoundError(f"Holdout test data not found at: {TEST_DATA_PATH}")

test_df = pd.read_csv(TEST_DATA_PATH)
print(f"✓ Test data loaded from {TEST_DATA_PATH}: {test_df.shape[0]:,} records x {test_df.shape[1]} columns")

# Determine target column
target_col = "attack_class" if "attack_class" in test_df.columns else ("label" if "label" in test_df.columns else "AttackCategory")
y_test_true = test_df[target_col].astype(str)

print(f"✓ Ground truth target column: '{target_col}'")
print(f"✓ Class distribution:\n{y_test_true.value_counts()}\n")

# ==================== STRICT FEATURE VALIDATION (NO SCALER) ====================
print("Validating feature matrix against canonical 46-feature schema...")
X_test_validated = validate_46_features(test_df[feature_names])
print(f"✓ Feature validation passed: {X_test_validated.shape} (dtype: {X_test_validated.dtypes.iloc[0]})\n")

# ==================== MAKE PREDICTIONS & TEMPERATURE CALIBRATION ====================
print("Running production inference...")
t0 = time.perf_counter()

# Raw probabilities and margin logits from XGBoost
raw_probs = model.predict_proba(X_test_validated)
booster = model.get_booster()
dmat = xgb.DMatrix(X_test_validated)
logits = booster.predict(dmat, output_margin=True)

# Temperature Scaling
calib_probs = calibrator.predict_proba(logits)
pred_indices = np.argmax(calib_probs, axis=1)
y_pred_labels = label_encoder.inverse_transform(pred_indices)
calib_confidence = np.max(calib_probs, axis=1)

t1 = time.perf_counter()
total_time = t1 - t0
avg_lat_us = (total_time / len(X_test_validated)) * 1_000_000

print(f"✓ Inference completed in {total_time * 1000:.2f} ms ({avg_lat_us:.2f} µs/record)")
print(f"✓ Throughput: {len(X_test_validated) / total_time:,.1f} records/second\n")

# ==================== EVALUATE RESULTS ====================
y_test_encoded = label_encoder.transform(y_test_true)
accuracy = accuracy_score(y_test_encoded, pred_indices)

print("=" * 75)
print("PRODUCTION INFERENCE RESULTS")
print("=" * 75)
print(f"Overall Accuracy: {accuracy * 100:.4f}% ({accuracy:.6f})")
print(f"Mean Calibrated Confidence: {calib_confidence.mean() * 100:.2f}%\n")

print("Sample Predictions (First 15 records):")
print(f"{'#':<4} | {'Actual':<12} | {'Predicted':<12} | {'Confidence':<12} | {'Match':<6}")
print("-" * 55)
for i in range(min(15, len(X_test_validated))):
    match_str = "✓ YES" if y_test_true.iloc[i] == y_pred_labels[i] else "✗ NO"
    print(f"{i+1:<4} | {y_test_true.iloc[i]:<12} | {y_pred_labels[i]:<12} | {calib_confidence[i]:.4f} ({calib_confidence[i]*100:.1f}%) | {match_str}")

print("\nDetailed Classification Report:")
print(classification_report(y_test_encoded, pred_indices, target_names=label_encoder.classes_, digits=4))

print("Confusion Matrix:")
cm = confusion_matrix(y_test_encoded, pred_indices)
cm_df = pd.DataFrame(cm, index=[f"Actual {c}" for c in label_encoder.classes_],
                     columns=[f"Pred {c}" for c in label_encoder.classes_])
print(cm_df.to_string())

# ==================== SAVE INFERENCE RESULTS ====================
results_df = pd.DataFrame({
    'actual': y_test_true.values,
    'predicted': y_pred_labels,
    'calibrated_confidence': calib_confidence,
    'raw_confidence': np.max(raw_probs, axis=1),
    'correct': y_test_true.values == y_pred_labels
})

results_path = os.path.join(TEST_ARTIFACTS_PATH, "inference_results.csv")
results_df.to_csv(results_path, index=False)
print(f"\n✓ Saved detailed inference results to: {results_path}")
print("=" * 75)
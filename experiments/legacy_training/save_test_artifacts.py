# ============================================================================
# SAVE TEST ARTIFACTS FOR INFERENCE ON UNSEEN DATA
# ============================================================================
# This script creates test evaluation artifacts that were MISSING from the 
# original training code

import os
import sys
import pandas as pd
import numpy as np

# Ensure UTF-8 output in Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
import pickle
import joblib
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import xgboost as xgb
from imblearn.combine import SMOTETomek

# ==================== PATHS ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIRECTORY = BASE_DIR
PREPROCESSED_OBJECTS_PATH = os.path.join(BASE_DIR, "preprocessed_objects")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models")
TEST_ARTIFACTS_PATH = os.path.join(BASE_DIR, "test_artifacts")

# Create directories
os.makedirs(PREPROCESSED_OBJECTS_PATH, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(TEST_ARTIFACTS_PATH, exist_ok=True)

# ==================== STEP 1: LOAD & SPLIT DATASET ====================
print("=" * 70)
print("STEP 1: Loading Dataset & Creating Train/Test Split")
print("=" * 70)

dataset_file = os.path.join(DATASET_DIRECTORY, "Data160.csv")
if not os.path.exists(dataset_file):
    dataset_file = os.path.join(DATASET_DIRECTORY, "balanced_data.csv")

if not os.path.exists(dataset_file):
    raise FileNotFoundError(f"Dataset file not found at {dataset_file}")

dataset = pd.read_csv(dataset_file)
print(f"✓ Dataset loaded from {os.path.basename(dataset_file)}: {dataset.shape}")

# Map attack labels if needed
if 'AttackCategory' not in dataset.columns and 'label' in dataset.columns:
    attack_mapping = {
        'DDoS-RSTFINFlood': 'DDoS', 'DoS-TCP_Flood': 'DDoS', 'DDoS-ICMP_Flood': 'DDoS',
        'DoS-UDP_Flood': 'DoS', 'DoS-SYN_Flood': 'DoS',
        'MITM-ArpSpoofing': 'Spoofing', 'DNS_Spoofing': 'Spoofing',
        'BruteForce': 'BruteForce'
    }
    dataset['AttackCategory'] = dataset['label'].map(attack_mapping)
    dataset = dataset[dataset['AttackCategory'].notna()]
elif 'AttackCategory' in dataset.columns:
    dataset = dataset[dataset['AttackCategory'].notna()]

print(f"✓ After attack mapping: {dataset.shape}")

# Define features
selected_features = [
    'flow_duration', 'Protocol Type', 'Duration', 'Rate', 'Srate', 'Drate',
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number',
    'ack_flag_number', 'syn_count', 'fin_count', 'TCP', 'UDP', 'Tot sum', 'Weight'
]
selected_features = [f for f in selected_features if f in dataset.columns]

X = dataset[selected_features]
y = dataset['AttackCategory']

print(f"✓ Features shape: {X.shape}")
print(f"✓ Label distribution:\n{y.value_counts()}\n")

# ==================== STEP 2: STRATIFIED SPLIT (80% TRAIN / 20% TEST) ====================
print("=" * 70)
print("STEP 2: Stratified Train/Test Split (80/20)")
print("=" * 70)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in sss.split(X, y):
    X_train_raw = X.iloc[train_idx].copy()
    X_test_raw = X.iloc[test_idx].copy()
    y_train_raw = y.iloc[train_idx].copy()
    y_test_raw = y.iloc[test_idx].copy()

print(f"✓ Training set: {X_train_raw.shape[0]} samples ({100*len(X_train_raw)/len(X):.1f}%)")
print(f"✓ Test set: {X_test_raw.shape[0]} samples ({100*len(X_test_raw)/len(X):.1f}%)")
print(f"✓ Train label distribution:\n{y_train_raw.value_counts()}")
print(f"✓ Test label distribution:\n{y_test_raw.value_counts()}\n")

# ==================== STEP 3: ENCODE LABELS ====================
print("=" * 70)
print("STEP 3: Label Encoding")
print("=" * 70)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_raw)
y_test_encoded = label_encoder.transform(y_test_raw)

print(f"✓ Classes: {label_encoder.classes_}")
print(f"✓ Number of classes: {len(label_encoder.classes_)}\n")

# ==================== STEP 4: SCALE FEATURES (SEPARATELY) ====================
print("=" * 70)
print("STEP 4: Feature Scaling - FIT ON TRAINING ONLY")
print("=" * 70)

# ✅ IMPORTANT: Fit scaler on training data ONLY
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)  # Transform test WITHOUT refitting

print(f"✓ Scaler fitted on training data")
print(f"✓ Scaler mean: {scaler.center_}")
print(f"✓ Training data scaled: {X_train_scaled.shape}")
print(f"✓ Test data scaled: {X_test_scaled.shape}\n")

# ==================== STEP 5: BALANCE TRAINING DATA (SMOTE-TOMEK) ====================
print("=" * 70)
print("STEP 5: Balance Training Data with SMOTE-Tomek")
print("=" * 70)

smote_tomek = SMOTETomek(random_state=42, n_jobs=-1)
X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train_scaled, y_train_encoded)

print(f"✓ Original training size: {X_train_scaled.shape[0]}")
print(f"✓ Balanced training size: {X_train_balanced.shape[0]}")
print(f"✓ Balanced label distribution:\n{pd.Series(y_train_balanced).value_counts()}\n")

# ==================== STEP 6: TRAIN XGBOOST MODEL ====================
print("=" * 70)
print("STEP 6: Train XGBoost Model on Balanced Training Data")
print("=" * 70)

model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(label_encoder.classes_),
    max_depth=5,
    learning_rate=0.1,
    n_estimators=200,
    random_state=42,
    eval_metric='mlogloss',
    n_jobs=-1
)

model.fit(X_train_balanced, y_train_balanced)
print("✓ XGBoost model trained\n")

# ==================== STEP 7: EVALUATE ON TEST SET ====================
print("=" * 70)
print("STEP 7: Evaluate Model on Test Set (UNSEEN DATA)")
print("=" * 70)

y_test_pred = model.predict(X_test_scaled)
y_test_proba = model.predict_proba(X_test_scaled)

accuracy = accuracy_score(y_test_encoded, y_test_pred)
precision = precision_score(y_test_encoded, y_test_pred, average='weighted', zero_division=0)
recall = recall_score(y_test_encoded, y_test_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test_encoded, y_test_pred, average='weighted', zero_division=0)

print(f"✓ Accuracy:  {accuracy:.4f}")
print(f"✓ Precision: {precision:.4f}")
print(f"✓ Recall:    {recall:.4f}")
print(f"✓ F1-Score:  {f1:.4f}\n")

print("Classification Report:")
print(classification_report(y_test_encoded, y_test_pred, 
                          target_names=label_encoder.classes_,
                          zero_division=0))

# ==================== STEP 8: SAVE TRAINING ARTIFACTS ====================
print("=" * 70)
print("STEP 8: Save Training Artifacts")
print("=" * 70)

# Save model
model_path = os.path.join(MODEL_SAVE_PATH, "xgboost_model.pkl")
joblib.dump(model, model_path)
print(f"✓ Model saved: {model_path}")

preprocessed_model_path = os.path.join(PREPROCESSED_OBJECTS_PATH, "xgboost_model.pkl")
joblib.dump(model, preprocessed_model_path)
print(f"✓ Model saved: {preprocessed_model_path}")

# Save scaler
scaler_path = os.path.join(PREPROCESSED_OBJECTS_PATH, "robust_scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"✓ Scaler saved: {scaler_path}")

# Save label encoder
encoder_path = os.path.join(PREPROCESSED_OBJECTS_PATH, "label_encoder.pkl")
joblib.dump(label_encoder, encoder_path)
print(f"✓ Label encoder saved: {encoder_path}\n")

# ==================== STEP 9: SAVE TEST SET ARTIFACTS ====================
print("=" * 70)
print("STEP 9: Save Test Set Artifacts for Inference Testing")
print("=" * 70)

# Save raw test data
test_data_path = os.path.join(TEST_ARTIFACTS_PATH, "test_data.csv")
X_test_df = X_test_raw.copy()
X_test_df['label'] = y_test_raw.values
X_test_df.to_csv(test_data_path, index=False)
print(f"✓ Test data saved: {test_data_path}")

# Save scaled test data
test_scaled_path = os.path.join(TEST_ARTIFACTS_PATH, "test_data_scaled.csv")
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=selected_features)
X_test_scaled_df['label'] = y_test_raw.values
X_test_scaled_df.to_csv(test_scaled_path, index=False)
print(f"✓ Scaled test data saved: {test_scaled_path}")

# Save test predictions
test_predictions_path = os.path.join(TEST_ARTIFACTS_PATH, "test_predictions.csv")
predictions_df = pd.DataFrame({
    'actual_encoded': y_test_encoded,
    'actual_label': y_test_raw.values,
    'predicted_encoded': y_test_pred,
    'predicted_label': label_encoder.inverse_transform(y_test_pred),
    'confidence': np.max(y_test_proba, axis=1),
    'correct': y_test_pred == y_test_encoded
})
predictions_df.to_csv(test_predictions_path, index=False)
print(f"✓ Test predictions saved: {test_predictions_path}")

# Save test metrics
test_metrics_path = os.path.join(TEST_ARTIFACTS_PATH, "test_evaluation_metrics.txt")
with open(test_metrics_path, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("TEST SET EVALUATION METRICS\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Test Set Size: {len(y_test_encoded)} samples\n")
    f.write(f"Test Accuracy:  {accuracy:.4f}\n")
    f.write(f"Test Precision: {precision:.4f}\n")
    f.write(f"Test Recall:    {recall:.4f}\n")
    f.write(f"Test F1-Score:  {f1:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test_encoded, y_test_pred, 
                                 target_names=label_encoder.classes_,
                                 zero_division=0))
    f.write("\n\nConfusion Matrix:\n")
    cm = confusion_matrix(y_test_encoded, y_test_pred)
    f.write(str(cm))

print(f"✓ Test metrics saved: {test_metrics_path}")

# Save confusion matrix as numpy
confusion_matrix_path = os.path.join(TEST_ARTIFACTS_PATH, "confusion_matrix.npy")
np.save(confusion_matrix_path, confusion_matrix(y_test_encoded, y_test_pred))
print(f"✓ Confusion matrix saved: {confusion_matrix_path}\n")

# ==================== STEP 10: SAVE FEATURE & SPLIT INFO ====================
print("=" * 70)
print("STEP 10: Save Feature Schema & Split Information")
print("=" * 70)

# Save feature names
features_info = {
    'feature_names': selected_features,
    'num_features': len(selected_features),
    'label_classes': list(label_encoder.classes_),
    'num_classes': len(label_encoder.classes_),
    'random_state': 42,
    'test_size': 0.2,
    'train_size': 0.8,
    'scaler_type': 'RobustScaler',
    'balancing_method': 'SMOTE-Tomek'
}

features_info_path = os.path.join(TEST_ARTIFACTS_PATH, "features_info.pkl")
joblib.dump(features_info, features_info_path)
print(f"✓ Features info saved: {features_info_path}")

# Save as JSON too for readability
import json
features_info_json_path = os.path.join(TEST_ARTIFACTS_PATH, "features_info.json")
with open(features_info_json_path, 'w') as f:
    json.dump({k: (v if not isinstance(v, np.ndarray) else v.tolist()) 
               for k, v in features_info.items()}, f, indent=2)
print(f"✓ Features info saved (JSON): {features_info_json_path}\n")

# ==================== SUMMARY ====================
print("=" * 70)
print("✅ ALL ARTIFACTS SAVED SUCCESSFULLY")
print("=" * 70)
print("\n📁 SAVED ARTIFACTS STRUCTURE:")
print(f"""
Training Artifacts:
  - {model_path}
  - {scaler_path}
  - {encoder_path}

Test Artifacts (FOR INFERENCE TESTING):
  - {test_data_path}
  - {test_scaled_path}
  - {test_predictions_path}
  - {test_metrics_path}
  - {confusion_matrix_path}
  - {features_info_path}
  - {features_info_json_path}

📊 KEY STATISTICS:
  - Training samples: {len(X_train_balanced)} (after balancing)
  - Test samples: {len(y_test_encoded)} (unseen)
  - Features: {len(selected_features)}
  - Classes: {len(label_encoder.classes_)}
  - Test Accuracy: {accuracy:.4f}
""")

print("=" * 70)
print("✅ NEXT STEP: Use inference.py to test on new/unseen data")
print("=" * 70)
"""
Training Pipeline for 17-Feature XGBoost IoT Intrusion Detection Model
Dataset: CIC_IoT_2023_DDoS_DoS_Spoofing_1.5M.csv (1,500,000 records)
Target Classes: DDoS, DoS, Spoofing
Feature Count: 17 exact project features
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Ensure UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "CIC_IoT_2023_DDoS_DoS_Spoofing_1.5M.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "cic_iot_xgboost_17features_model.pkl")
ENCODER_SAVE_PATH = os.path.join(BASE_DIR, "label_encoder_17features.pkl")
METRICS_SAVE_PATH = os.path.join(BASE_DIR, "train_17features_metrics.json")

# EXACT 17 Features in Canonical Order from project
FEATURE_COLUMNS = [
    "flow_duration",
    "Protocol Type",
    "Duration",
    "Rate",
    "Srate",
    "Drate",
    "fin_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "psh_flag_number",
    "ack_flag_number",
    "syn_count",
    "fin_count",
    "TCP",
    "UDP",
    "Tot sum",
    "Weight"
]

TARGET_CLASSES = ["DDoS", "DoS", "Spoofing"]
RANDOM_SEED = 42

def main():
    print("=" * 80)
    print("  TRAINING NEW 17-FEATURE XGBOOST MODEL (1.5M DATASET: DDoS / DoS / Spoofing)")
    print("=" * 80)
    
    # 1. Verify Dataset
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Training dataset not found at: {DATASET_PATH}")
        
    print(f"\n[Step 1/5] Loading training dataset: {DATASET_PATH}")
    t0_load = time.perf_counter()
    df = pd.read_csv(DATASET_PATH)
    t1_load = time.perf_counter()
    print(f"✓ Loaded {len(df):,} records in {t1_load - t0_load:.2f}s")
    
    # Verify Columns
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")
        
    if "attack_class" not in df.columns:
        raise ValueError("Target column 'attack_class' not found in dataset!")
        
    # Verify Target Classes
    actual_classes = sorted(df['attack_class'].unique().tolist())
    print(f"✓ Target classes in dataset: {actual_classes}")
    assert set(actual_classes) == set(TARGET_CLASSES), f"Unexpected classes: {actual_classes}"
    
    print("\nClass Distribution in Training Dataset (N = 1,500,000):")
    class_counts = df['attack_class'].value_counts()
    for cls_name, count in class_counts.items():
        print(f"  • {cls_name:<10}: {count:>10,} ({count / len(df) * 100:6.2f}%)")
        
    # 2. Extract Features and Target
    print("\n[Step 2/5] Preparing feature matrix X (17 features) and target y...")
    X_train = df[FEATURE_COLUMNS].copy()
    y_raw = df['attack_class'].copy()
    
    # Free memory of full df if needed
    del df
    
    print(f"✓ Feature matrix shape: {X_train.shape}")
    print(f"✓ Feature names ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")
    
    # 3. Encode Target Labels
    print("\n[Step 3/5] Fitting LabelEncoder...")
    encoder = LabelEncoder()
    encoder.fit(TARGET_CLASSES)
    y_train = encoder.transform(y_raw)
    
    print("✓ Label encoding mapping:")
    for idx, cls in enumerate(encoder.classes_):
        print(f"  {idx} -> {cls}")
        
    # 4. Train XGBoost Classifier
    print("\n[Step 4/5] Training XGBoost Classifier on 1,500,000 samples...")
    print(f"  - Features       : {len(FEATURE_COLUMNS)}")
    print(f"  - Classes        : {len(encoder.classes_)} ({list(encoder.classes_)})")
    print(f"  - Algorithm      : XGBoost (hist)")
    print(f"  - n_estimators   : 300")
    print(f"  - max_depth      : 8")
    print(f"  - learning_rate  : 0.1")
    print(f"  - random_state   : {RANDOM_SEED}")
    
    model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    
    t0_train = time.perf_counter()
    model.fit(X_train, y_train)
    t1_train = time.perf_counter()
    train_time = t1_train - t0_train
    print(f"✓ Model training completed in {train_time:.2f} seconds ({train_time / 60:.2f} minutes)!")
    
    # 5. Evaluate on Training Data
    print("\n[Step 5/5] Evaluating model on Training Data (1,500,000 samples)...")
    t0_pred = time.perf_counter()
    y_pred = model.predict(X_train)
    t1_pred = time.perf_counter()
    
    train_acc = accuracy_score(y_train, y_pred)
    train_bal_acc = balanced_accuracy_score(y_train, y_pred)
    train_prec_macro = precision_score(y_train, y_pred, average='macro', zero_division=0)
    train_rec_macro = recall_score(y_train, y_pred, average='macro', zero_division=0)
    train_f1_macro = f1_score(y_train, y_pred, average='macro', zero_division=0)
    cm_train = confusion_matrix(y_train, y_pred)
    
    print("\n" + "=" * 80)
    print("TRAINING PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"• Total Training Records : {len(y_train):,}")
    print(f"• Training Time          : {train_time:.2f} seconds")
    print(f"• Training Accuracy      : {train_acc * 100:.4f}% ({train_acc:.6f})")
    print(f"• Balanced Accuracy      : {train_bal_acc * 100:.4f}% ({train_bal_acc:.6f})")
    print(f"• Macro Precision        : {train_prec_macro * 100:.4f}%")
    print(f"• Macro Recall           : {train_rec_macro * 100:.4f}%")
    print(f"• Macro F1-Score         : {train_f1_macro * 100:.4f}%")
    
    print("\nPer-Class Training Classification Report:")
    print(classification_report(y_train, y_pred, target_names=encoder.classes_, digits=4))
    
    print("Training Confusion Matrix:")
    cm_df = pd.DataFrame(cm_train, index=[f"Actual {c}" for c in encoder.classes_],
                         columns=[f"Pred {c}" for c in encoder.classes_])
    print(cm_df.to_string())
    
    # 6. Save Model and Encoder Artifacts
    print(f"\n[Saving Artifacts]")
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"✓ Saved 17-feature model to : {MODEL_SAVE_PATH}")
    
    joblib.dump(encoder, ENCODER_SAVE_PATH)
    print(f"✓ Saved label encoder to    : {ENCODER_SAVE_PATH}")
    
    train_metrics = {
        "model_file": os.path.basename(MODEL_SAVE_PATH),
        "encoder_file": os.path.basename(ENCODER_SAVE_PATH),
        "num_features": len(FEATURE_COLUMNS),
        "feature_names": FEATURE_COLUMNS,
        "classes": list(encoder.classes_),
        "num_train_records": len(y_train),
        "training_time_sec": float(train_time),
        "train_accuracy": float(train_acc),
        "train_balanced_accuracy": float(train_bal_acc),
        "train_macro_precision": float(train_prec_macro),
        "train_macro_recall": float(train_rec_macro),
        "train_macro_f1": float(train_f1_macro),
        "train_confusion_matrix": cm_train.tolist()
    }
    
    with open(METRICS_SAVE_PATH, "w") as f:
        json.dump(train_metrics, f, indent=2)
    print(f"✓ Saved training metrics to : {METRICS_SAVE_PATH}")
    
    print("\nTraining stage complete. Ready for unseen test evaluation.")

if __name__ == "__main__":
    main()

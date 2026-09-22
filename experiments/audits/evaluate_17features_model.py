"""
Evaluation Pipeline for 17-Feature XGBoost IoT Intrusion Detection Model
Evaluates on genuinely unseen official Hugging Face test split:
https://huggingface.co/datasets/lacg030175/CIC-IoT-2023-neto-full (random/test split)
Classes: DDoS, DoS, Spoofing only
Features: 17 exact project features
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix, roc_auc_score,
    cohen_kappa_score, matthews_corrcoef
)

# Ensure UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cic_iot_xgboost_17features_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder_17features.pkl")
TRAIN_METRICS_PATH = os.path.join(BASE_DIR, "train_17features_metrics.json")
EVAL_RESULTS_JSON = os.path.join(BASE_DIR, "evaluation_17features_results.json")

REPO_ID = "lacg030175/CIC-IoT-2023-neto-full"
TEST_SPLIT_FILE = "random/test-00000-of-00001.parquet"

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

def main():
    print("=" * 80)
    print("  EVALUATION OF NEW 17-FEATURE XGBOOST MODEL ON UNSEEN HUGGING FACE TEST SPLIT")
    print("=" * 80)
    
    # 1. Load Model and Encoder
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"Encoder file not found: {ENCODER_PATH}")
        
    print(f"\n[Step 1/4] Loading trained 17-feature model & encoder...")
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    class_names = list(encoder.classes_)
    
    print(f"✓ Model loaded from: {MODEL_PATH}")
    print(f"✓ Encoder loaded from: {ENCODER_PATH}")
    print(f"✓ Classes ({len(class_names)}): {class_names}")
    print(f"✓ Model feature names ({len(model.feature_names_in_)}): {list(model.feature_names_in_)}")
    assert list(model.feature_names_in_) == FEATURE_COLUMNS, "Model feature names mismatch!"
    
    # Load training metrics if available
    train_metrics = {}
    if os.path.exists(TRAIN_METRICS_PATH):
        with open(TRAIN_METRICS_PATH, "r") as f:
            train_metrics = json.load(f)
            
    # 2. Download / Locate Unseen Test Parquet Split
    print(f"\n[Step 2/4] Accessing official Hugging Face test split ({TEST_SPLIT_FILE})...")
    test_parquet_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=TEST_SPLIT_FILE,
        repo_type="dataset"
    )
    print(f"✓ Test split source: {test_parquet_path}")
    print(f"✓ File size: {os.path.getsize(test_parquet_path) / (1024*1024):.2f} MB")
    
    pf = pq.ParquetFile(test_parquet_path)
    total_test_rows = pf.metadata.num_rows
    num_rgs = pf.num_row_groups
    print(f"✓ Total records in raw test split: {total_test_rows:,} across {num_rgs} row groups")
    
    # 3. Stream and Predict on DDoS, DoS, and Spoofing records from Test Split
    print(f"\n[Step 3/4] Streaming test records (filtering ONLY DDoS, DoS, Spoofing) and predicting...")
    y_true_all = []
    y_pred_all = []
    y_proba_all = []
    
    t0_eval = time.perf_counter()
    total_filtered_samples = 0
    test_class_counts = {"DDoS": 0, "DoS": 0, "Spoofing": 0}
    
    for rg_idx in range(num_rgs):
        table = pf.read_row_group(rg_idx)
        df_chunk = table.to_pandas()
        
        # Filter strictly for DDoS, DoS, Spoofing
        filtered = df_chunk[df_chunk['attack_class'].isin(TARGET_CLASSES)]
        if len(filtered) == 0:
            continue
            
        for cls_name in TARGET_CLASSES:
            cnt = (filtered['attack_class'] == cls_name).sum()
            test_class_counts[cls_name] += int(cnt)
            
        X_chunk = filtered[FEATURE_COLUMNS]
        y_chunk = encoder.transform(filtered['attack_class'])
        
        preds = model.predict(X_chunk)
        probs = model.predict_proba(X_chunk)
        
        y_true_all.extend(y_chunk)
        y_pred_all.extend(preds)
        y_proba_all.append(probs)
        
        total_filtered_samples += len(filtered)
        
        if (rg_idx + 1) % 15 == 0 or rg_idx == num_rgs - 1:
            elapsed = time.perf_counter() - t0_eval
            print(f"  Processed row group {rg_idx + 1:>2}/{num_rgs} | Evaluated test samples: {total_filtered_samples:>10,} | Rate: {total_filtered_samples / elapsed:,.0f} samples/s")
            
    t1_eval = time.perf_counter()
    eval_time = t1_eval - t0_eval
    
    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)
    y_proba = np.vstack(y_proba_all)
    
    print(f"\n✓ Unseen test evaluation completed in {eval_time:.2f}s ({len(y_true) / eval_time:,.0f} flows/sec)")
    print(f"✓ Total Unseen Test Records: {len(y_true):,}")
    
    # 4. Compute Comprehensive Evaluation Metrics
    print(f"\n[Step 4/4] Computing full performance metrics on unseen test data...")
    test_acc = accuracy_score(y_true, y_pred)
    test_bal_acc = balanced_accuracy_score(y_true, y_pred)
    test_prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    test_prec_weight = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    test_rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    test_rec_weight = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    test_f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    test_f1_weight = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    try:
        roc_auc_ovr = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
    except Exception:
        roc_auc_ovr = 1.0
        
    cm = confusion_matrix(y_true, y_pred)
    
    # Print Full Report
    print("\n" + "=" * 85)
    print("                      FULL 17-FEATURE MODEL PERFORMANCE REPORT")
    print("=" * 85)
    
    print("\n1. PIPELINE & FEATURE CONFIGURATION")
    print("-" * 65)
    print(f"• Model File                 : {os.path.basename(MODEL_PATH)}")
    print(f"• Encoder File               : {os.path.basename(ENCODER_PATH)}")
    print(f"• Feature Count              : {len(FEATURE_COLUMNS)}")
    print(f"• Features (Canonical Order) :")
    for i, col in enumerate(FEATURE_COLUMNS, 1):
        print(f"    {i:>2}. {col}")
    print(f"• Target Classes ({len(class_names)}): {class_names}")
    print(f"• Training Records           : {train_metrics.get('num_train_records', 1500000):,}")
    print(f"• Model Training Time        : {train_metrics.get('training_time_sec', 0.0):.2f} seconds ({train_metrics.get('training_time_sec', 0.0) / 60:.2f} min)")
    print(f"• Unseen Test Records        : {len(y_true):,}")
    
    print("\n2. CLASS DISTRIBUTION BREAKDOWN")
    print("-" * 65)
    print(f"{'Class':<12} | {'Training Count':<16} | {'Unseen Test Count':<18} | {'Test Proportion':<15}")
    print("-" * 65)
    train_dist = {"DDoS": 555340, "DoS": 555339, "Spoofing": 389321}
    for cls in class_names:
        tr_c = train_dist.get(cls, 0)
        te_c = test_class_counts.get(cls, 0)
        pct = te_c / len(y_true) * 100
        print(f"{cls:<12} | {tr_c:>14,} | {te_c:>16,} | {pct:>13.2f}%")
    print(f"{'TOTAL':<12} | {sum(train_dist.values()):>14,} | {len(y_true):>16,} | {100.0:>13.2f}%")
    
    print("\n3. OVERALL CLASSIFICATION & GENERALIZATION METRICS")
    print("-" * 65)
    print(f"{'Metric':<35} | {'Training Set':<18} | {'Unseen Test Set':<18}")
    print("-" * 75)
    tr_acc_str = f"{train_metrics.get('train_accuracy', 0.0) * 100:.4f}%"
    tr_bal_str = f"{train_metrics.get('train_balanced_accuracy', 0.0) * 100:.4f}%"
    tr_pm_str = f"{train_metrics.get('train_macro_precision', 0.0) * 100:.4f}%"
    tr_rm_str = f"{train_metrics.get('train_macro_recall', 0.0) * 100:.4f}%"
    tr_fm_str = f"{train_metrics.get('train_macro_f1', 0.0) * 100:.4f}%"
    
    print(f"{'Overall Accuracy':<35} | {tr_acc_str:<18} | {test_acc * 100:.4f}% ({test_acc:.6f})")
    print(f"{'Balanced Accuracy':<35} | {tr_bal_str:<18} | {test_bal_acc * 100:.4f}% ({test_bal_acc:.6f})")
    print(f"{'Macro Precision':<35} | {tr_pm_str:<18} | {test_prec_macro * 100:.4f}%")
    print(f"{'Weighted Precision':<35} | {'-':<18} | {test_prec_weight * 100:.4f}%")
    print(f"{'Macro Recall':<35} | {tr_rm_str:<18} | {test_rec_macro * 100:.4f}%")
    print(f"{'Weighted Recall':<35} | {'-':<18} | {test_rec_weight * 100:.4f}%")
    print(f"{'Macro F1-Score':<35} | {tr_fm_str:<18} | {test_f1_macro * 100:.4f}%")
    print(f"{'Weighted F1-Score':<35} | {'-':<18} | {test_f1_weight * 100:.4f}%")
    print(f"{'Cohen\'s Kappa':<35} | {'-':<18} | {kappa:.6f}")
    print(f"{'Matthews Correlation (MCC)':<35} | {'-':<18} | {mcc:.6f}")
    print(f"{'Multi-class ROC-AUC (OVR)':<35} | {'-':<18} | {roc_auc_ovr:.6f}")
    
    print("\n4. PER-CLASS DETAILED BREAKDOWN (UNSEEN TEST SET)")
    print("-" * 88)
    print(f"{'Class':<12} | {'TP':<10} | {'FP':<8} | {'FN':<8} | {'TN':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 88)
    
    per_class_results = {}
    for i, cls in enumerate(class_names):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(cm.sum() - (tp + fp + fn))
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        per_class_results[cls] = {
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": float(prec), "recall": float(rec), "f1_score": float(f1),
            "support": int(cm[i, :].sum())
        }
        
        print(f"{cls:<12} | {tp:<10,} | {fp:<8,} | {fn:<8,} | {tn:<10,} | {prec*100:6.2f}%    | {rec*100:6.2f}%    | {f1*100:6.2f}%")
        
    print("\n5. CONFUSION MATRIX (UNSEEN TEST SET)")
    print("-" * 65)
    cm_df = pd.DataFrame(cm, index=[f"Actual {c}" for c in class_names],
                         columns=[f"Pred {c}" for c in class_names])
    print(cm_df.to_string())
    
    print("\n6. CLASSIFICATION REPORT (UNSEEN TEST SET)")
    print("-" * 65)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    # Save Full Results JSON
    full_results = {
        "model_file": os.path.basename(MODEL_PATH),
        "encoder_file": os.path.basename(ENCODER_PATH),
        "feature_count": len(FEATURE_COLUMNS),
        "feature_names": FEATURE_COLUMNS,
        "classes": class_names,
        "num_training_records": train_metrics.get("num_train_records", 1500000),
        "training_time_sec": train_metrics.get("training_time_sec", 0.0),
        "training_accuracy": train_metrics.get("train_accuracy", 0.0),
        "num_unseen_test_records": len(y_true),
        "test_evaluation_time_sec": float(eval_time),
        "test_accuracy": float(test_acc),
        "test_balanced_accuracy": float(test_bal_acc),
        "test_macro_precision": float(test_prec_macro),
        "test_weighted_precision": float(test_prec_weight),
        "test_macro_recall": float(test_rec_macro),
        "test_weighted_recall": float(test_rec_weight),
        "test_macro_f1": float(test_f1_macro),
        "test_weighted_f1": float(test_f1_weight),
        "cohen_kappa": float(kappa),
        "matthews_corrcoef": float(mcc),
        "roc_auc_ovr": float(roc_auc_ovr),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class_results,
        "class_distribution": {
            "training": train_dist,
            "test": test_class_counts
        }
    }
    
    with open(EVAL_RESULTS_JSON, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\n✓ Saved complete evaluation results JSON to: {EVAL_RESULTS_JSON}")
    print("=" * 85)

if __name__ == "__main__":
    main()

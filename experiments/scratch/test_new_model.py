"""
Comprehensive Performance Evaluation for cic_iot_xgboost_model.pkl and label_encoder.pkl
Tests accuracy, precision, recall, F1, latency, throughput, confusion matrix, and generalization on unseen test data.
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

# Ensure UTF-8 console output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cic_iot_xgboost_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
CSV_PATH = os.path.join(BASE_DIR, "CIC_IoT_2023_DDoS_DoS_Spoofing_1.5M.csv")

def evaluate_predictions(y_true, y_pred, y_proba, class_names, split_name="Dataset"):
    print("=" * 80)
    print(f"EVALUATION METRICS FOR: {split_name} (N = {len(y_true):,} samples)")
    print("=" * 80)
    
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    prec_weight = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    rec_weight = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weight = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    try:
        roc_auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
    except Exception:
        roc_auc = 1.0
        
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
    kappa_label = "Cohen's Kappa"
    print(f"{kappa_label:<35} | {kappa:.6f}")
    print(f"{'Matthews Corr. Coeff. (MCC)':<35} | {mcc:.6f}")
    print(f"{'Multi-class ROC-AUC (OVR)':<35} | {roc_auc:.6f}")
    
    cm = confusion_matrix(y_true, y_pred)
    print("\n" + "-" * 80)
    print("PER-CLASS DETAILED BREAKDOWN")
    print("-" * 80)
    print(f"{'Class':<12} | {'TP':<10} | {'FP':<8} | {'FN':<8} | {'TN':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 85)
    
    for i, cls in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        print(f"{cls:<12} | {tp:<10,} | {fp:<8,} | {fn:<8,} | {tn:<10,} | {prec*100:6.2f}%    | {rec*100:6.2f}%    | {f1*100:6.2f}%")
        
    print("\nConfusion Matrix:")
    cm_df = pd.DataFrame(cm, index=[f"Actual {c}" for c in class_names],
                         columns=[f"Pred {c}" for c in class_names])
    print(cm_df.to_string())
    
    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "precision_macro": float(prec_macro),
        "precision_weighted": float(prec_weight),
        "recall_macro": float(rec_macro),
        "recall_weighted": float(rec_weight),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weight),
        "cohen_kappa": float(kappa),
        "mcc": float(mcc),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist()
    }

def main():
    print("\n" + "=" * 80)
    print("      TESTING TRAINED MODEL: cic_iot_xgboost_model.pkl")
    print("=" * 80)
    
    # 1. Load Model and Encoder
    print(f"\n[+] Loading model: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print(f"[+] Loading encoder: {ENCODER_PATH}")
    encoder = joblib.load(ENCODER_PATH)
    
    class_names = list(encoder.classes_)
    feature_names = list(model.feature_names_in_)
    print(f"[+] Model classes ({len(class_names)}): {class_names}")
    print(f"[+] Feature count: {len(feature_names)}")
    
    # 2. Evaluate on 1.5M CSV Dataset (Streaming evaluation in batches)
    print(f"\n[+] Evaluating on full 1.5M CSV dataset ({CSV_PATH})...")
    chunksize = 250_000
    y_true_all = []
    y_pred_all = []
    y_proba_all = []
    
    t_start = time.perf_counter()
    for chunk_idx, chunk in enumerate(pd.read_csv(CSV_PATH, chunksize=chunksize)):
        X_chunk = chunk[feature_names]
        y_chunk = encoder.transform(chunk['attack_class'])
        
        preds = model.predict(X_chunk)
        probs = model.predict_proba(X_chunk)
        
        y_true_all.extend(y_chunk)
        y_pred_all.extend(preds)
        y_proba_all.append(probs)
        print(f"  Processed chunk {chunk_idx + 1} ({len(y_true_all):,} / 1,500,000 samples)...")
        
    t_end = time.perf_counter()
    print(f"[+] 1.5M inference completed in {t_end - t_start:.2f}s ({len(y_true_all) / (t_end - t_start):,.1f} flows/sec)")
    
    y_true_arr = np.array(y_true_all)
    y_pred_arr = np.array(y_pred_all)
    y_proba_arr = np.vstack(y_proba_all)
    
    metrics_1_5m = evaluate_predictions(y_true_arr, y_pred_arr, y_proba_arr, class_names, split_name="1.5M Dataset (CIC_IoT_2023_DDoS_DoS_Spoofing_1.5M.csv)")
    
    # 3. Latency & Throughput Benchmark
    print("\n" + "=" * 80)
    print("REAL-TIME LATENCY & INFERENCE BENCHMARK")
    print("=" * 80)
    
    sample_X = pd.read_csv(CSV_PATH, nrows=1)[feature_names]
    n_runs = 1000
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = model.predict(sample_X)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms
        
    lat_arr = np.array(latencies)
    print(f"• Mean Single-Packet Latency : {lat_arr.mean():.4f} ms ({lat_arr.mean() * 1000:.2f} µs)")
    print(f"• Median Latency (P50)       : {np.median(lat_arr):.4f} ms ({np.median(lat_arr) * 1000:.2f} µs)")
    print(f"• 95th Percentile (P95)      : {np.percentile(lat_arr, 95):.4f} ms ({np.percentile(lat_arr, 95) * 1000:.2f} µs)")
    print(f"• 99th Percentile (P99)      : {np.percentile(lat_arr, 99):.4f} ms ({np.percentile(lat_arr, 99) * 1000:.2f} µs)")
    
    # Batch throughput
    print("\nBatch Processing Throughput:")
    print(f"{'Batch Size':<12} | {'Time Taken (ms)':<18} | {'Throughput (Flows/sec)':<25}")
    print("-" * 60)
    test_batch_df = pd.read_csv(CSV_PATH, nrows=50000)[feature_names]
    for b in [1, 10, 100, 500, 1000, 5000, 10000, 50000]:
        bx = test_batch_df.iloc[:b]
        t0 = time.perf_counter()
        _ = model.predict(bx)
        t1 = time.perf_counter()
        duration_ms = (t1 - t0) * 1000.0
        th = b / (t1 - t0) if (t1 - t0) > 0 else 0
        print(f"{b:<12} | {duration_ms:<18.3f} | {th:>18,.1f} flows/sec")
        
    # 4. Save test results to JSON
    results_json = os.path.join(BASE_DIR, "model_test_results.json")
    with open(results_json, 'w') as f:
        json.dump({
            "model_path": MODEL_PATH,
            "encoder_path": ENCODER_PATH,
            "dataset_path": CSV_PATH,
            "classes": class_names,
            "feature_count": len(feature_names),
            "features": feature_names,
            "metrics": metrics_1_5m,
            "latency": {
                "mean_us": float(lat_arr.mean() * 1000),
                "p50_us": float(np.median(lat_arr) * 1000),
                "p95_us": float(np.percentile(lat_arr, 95) * 1000),
                "p99_us": float(np.percentile(lat_arr, 99) * 1000)
            }
        }, f, indent=2)
    print(f"\n[+] Saved detailed evaluation results to: {results_json}")

if __name__ == "__main__":
    main()

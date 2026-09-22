"""
aegis_ml/evaluate_model.py
==========================
Independent holdout evaluation script for the retrained AEGIS-IoT model.

PURPOSE:
- Loads the newly retrained model from 'models/aegis/aegis_xgboost_model.pkl'.
- Loads the independent encoder from 'models/aegis/aegis_label_encoder.pkl'.
- Evaluates the model exclusively on the untouched 5,000-record holdout dataset:
  'Testing_data/testing_data.csv'.
- Computes and prints comprehensive holdout classification metrics:
  Accuracy, Balanced Accuracy, Precision, Recall, Macro F1, Weighted F1,
  Confusion Matrix, and Per-Class breakdown.
- Saves detailed metrics to 'models/aegis/evaluation_results.json'.

STRICT RULES:
- The holdout test set is NEVER used for training or tuning.
- Does NOT calibrate using the test set.
- Does NOT modify the test dataset.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aegis_ml.feature_schema import (
    FEATURE_COLUMNS,
    N_FEATURES,
    CLASS_NAMES,
)
from aegis_ml.dataset_builder import AegisDatasetBuilder
from aegis_ml.label_encoding import AegisLabelEncoder

# Artifact Paths
MODEL_PATH = PROJECT_ROOT / "models" / "aegis" / "aegis_xgboost_model.pkl"
ENCODER_PATH = PROJECT_ROOT / "models" / "aegis" / "aegis_label_encoder.pkl"
RESULTS_JSON_PATH = PROJECT_ROOT / "models" / "aegis" / "evaluation_results.json"


def evaluate_aegis_model() -> Dict[str, Any]:
    """
    Evaluates the newly trained AEGIS-IoT XGBoost model on the untouched
    5,000-record test set.
    """
    print("=" * 70)
    print("AEGIS-IoT INDEPENDENT MODEL EVALUATION")
    print("=" * 70)

    # 1. Verify and Load Artifacts
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found at: {MODEL_PATH}")
    if not ENCODER_PATH.exists():
        raise FileNotFoundError(f"Label encoder artifact not found at: {ENCODER_PATH}")

    print("\n[Step 1/3] Loading retrained model and encoder...")
    model = joblib.load(MODEL_PATH)
    encoder = AegisLabelEncoder.load(ENCODER_PATH)
    print(f"  ✓ Loaded model from: {MODEL_PATH}")
    print(f"  ✓ Loaded encoder from: {ENCODER_PATH}")

    # 2. Load Untouched 5K Test Data
    print("\n[Step 2/3] Loading untouched 5,000-record holdout test dataset...")
    builder = AegisDatasetBuilder(encoder=encoder)
    t0 = time.perf_counter()
    X_test, y_test = builder.get_test_data(encode_labels=True)
    t_load = time.perf_counter() - t0
    
    n_samples = len(X_test)
    print(f"  ✓ Test data loaded in {t_load:.3f}s")
    print(f"  ✓ Test Shape: X = {X_test.shape}, y = {y_test.shape}")
    assert n_samples == 5000, f"Expected exactly 5,000 test records, got {n_samples}"
    assert X_test.shape[1] == N_FEATURES, f"Expected {N_FEATURES} features, got {X_test.shape[1]}"

    # Ground truth distribution
    unique_gt, counts_gt = np.unique(y_test, return_counts=True)
    gt_dist = {encoder.inverse_transform([u])[0]: int(c) for u, c in zip(unique_gt, counts_gt)}
    print(f"  ✓ Ground Truth Class Distribution: {gt_dist}")

    # 3. Model Inference
    print("\n[Step 3/3] Running inference on holdout test set...")
    t_inf_start = time.perf_counter()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    t_inf = time.perf_counter() - t_inf_start
    print(f"  ✓ Inference completed in {t_inf:.3f}s ({t_inf/n_samples*1000:.3f} ms/sample)")

    # Predicted class distribution
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    pred_dist = {encoder.inverse_transform([u])[0]: int(c) for u, c in zip(unique_pred, counts_pred)}
    print(f"  ✓ Predicted Class Distribution:    {pred_dist}")

    # 4. Compute Metrics
    acc = float(accuracy_score(y_test, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    macro_prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    weighted_prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    macro_rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    weighted_rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    per_class = {}
    for idx, c_name in enumerate(CLASS_NAMES):
        tp = int(cm[idx, idx])
        fn = int(cm[idx, :].sum() - tp)
        fp = int(cm[:, idx].sum() - tp)
        tn = int(cm.sum() - (tp + fp + fn))
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class[c_name] = {
            "support": int(cm[idx, :].sum()),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "f1_score": round(f1, 6),
        }

    # Summary Report Printout
    print("\n" + "=" * 70)
    print("HOLDOUT EVALUATION RESULTS")
    print("=" * 70)
    print(f"  Number of Test Samples:     {n_samples:,}")
    print(f"  Number of Features:         {N_FEATURES}")
    print(f"  Overall Accuracy:           {acc * 100:.4f}% ({int(acc * n_samples)} / {n_samples})")
    print(f"  Balanced Accuracy:          {bal_acc * 100:.4f}%")
    print(f"  Macro Precision:            {macro_prec * 100:.4f}%")
    print(f"  Weighted Precision:         {weighted_prec * 100:.4f}%")
    print(f"  Macro Recall:               {macro_rec * 100:.4f}%")
    print(f"  Weighted Recall:            {weighted_rec * 100:.4f}%")
    print(f"  Macro F1-Score:             {macro_f1 * 100:.4f}%")
    print(f"  Weighted F1-Score:          {weighted_f1 * 100:.4f}%")

    print("\nConfusion Matrix:")
    print("                     Predicted")
    print("                 DDoS   DoS   Spoofing")
    for idx, c_name in enumerate(CLASS_NAMES):
        row_str = " ".join(f"{cm[idx, j]:6d}" for j in range(3))
        print(f"Actual {c_name:8s}  {row_str}")

    print("\nPer-Class Detailed Breakdown:")
    for c_name, m in per_class.items():
        print(f"  [{c_name}] Support={m['support']:,} | Precision={m['precision']*100:.2f}% | Recall={m['recall']*100:.2f}% | F1={m['f1_score']*100:.2f}%")

    # Save to JSON
    results = {
        "evaluation_name": "AEGIS-IoT Retrained Model Holdout Evaluation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": str(MODEL_PATH),
        "encoder_path": str(ENCODER_PATH),
        "test_dataset": "Testing_data/testing_data.csv",
        "n_test_samples": n_samples,
        "n_features": N_FEATURES,
        "class_names": CLASS_NAMES,
        "ground_truth_distribution": gt_dist,
        "predicted_distribution": pred_dist,
        "metrics": {
            "accuracy": round(acc, 6),
            "balanced_accuracy": round(bal_acc, 6),
            "macro_precision": round(macro_prec, 6),
            "weighted_precision": round(weighted_prec, 6),
            "macro_recall": round(macro_rec, 6),
            "weighted_recall": round(weighted_rec, 6),
            "macro_f1": round(macro_f1, 6),
            "weighted_f1": round(weighted_f1, 6),
        },
        "confusion_matrix": {
            "labels": CLASS_NAMES,
            "matrix": cm.tolist(),
        },
        "per_class": per_class,
    }

    RESULTS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Detailed results saved to: {RESULTS_JSON_PATH}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    evaluate_aegis_model()

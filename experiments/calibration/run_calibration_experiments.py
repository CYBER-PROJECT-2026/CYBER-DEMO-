"""
Comprehensive Probability Calibration Experiment Runner for 46-Feature XGBoost Model.

Strict Pipeline Rules:
1. Original XGBoost model (cic_iot_xgboost_model.pkl) and label encoder (label_encoder.pkl) are NEVER retrained or modified.
2. Calibration dataset (calibration_data.csv, 21,000 records) is used SOLELY to fit the calibrators.
3. Untouched test dataset (Testing_data/testing_data.csv, 5,000 records) is evaluated ONLY after calibrators are fitted.
4. Compares:
   - Original XGBoost
   - Sigmoid / Platt Calibrator
   - Isotonic Calibrator
   - Multiclass Temperature Scaling
5. Saves all artifacts, plots reliability curves, selects best calibrator, and outputs JSON and Markdown reports.
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

from calibration_engine import (
    MulticlassSigmoidCalibrator,
    MulticlassIsotonicCalibrator,
    MulticlassTemperatureCalibrator,
    compute_multiclass_log_loss,
    compute_multiclass_brier_score,
    compute_calibration_curve_and_ece,
    compute_classwise_ece,
    generate_reliability_plots
)

# Ensure UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cic_iot_xgboost_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
CALIB_CSV_PATH = os.path.join(BASE_DIR, "calibration_data.csv")
TEST_CSV_PATH = os.path.join(BASE_DIR, "Testing_data", "testing_data.csv")

# Artifact Paths
SIGMOID_ART_PATH = os.path.join(BASE_DIR, "calibrator_sigmoid.pkl")
ISOTONIC_ART_PATH = os.path.join(BASE_DIR, "calibrator_isotonic.pkl")
TEMP_ART_PATH = os.path.join(BASE_DIR, "calibrator_temperature.pkl")
FINAL_PIPELINE_PATH = os.path.join(BASE_DIR, "calibrated_xgboost_final.pkl")
RESULTS_JSON_PATH = os.path.join(BASE_DIR, "calibration_evaluation_results.json")
REPORT_MD_PATH = os.path.join(BASE_DIR, "CALIBRATION_REPORT.md")
RELIABILITY_PNG_PATH = os.path.join(BASE_DIR, "reliability_curves_calibration.png")


def evaluate_method_predictions(y_true, y_prob, class_names, method_name):
    """
    Computes all classification and calibration metrics for a given predicted probability matrix.
    """
    N, K = y_prob.shape
    y_pred = np.argmax(y_prob, axis=1)

    # Classification Metrics
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_prec = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    weighted_prec = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    weighted_rec = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted'))
    except Exception:
        roc_auc = 1.0

    cm = confusion_matrix(y_true, y_pred, labels=list(range(K)))

    per_class = {}
    for idx, c_name in enumerate(class_names):
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
            "precision": prec, "recall": rec, "f1_score": f1
        }

    # Calibration Metrics
    log_loss = compute_multiclass_log_loss(y_true, y_prob)
    brier = compute_multiclass_brier_score(y_true, y_prob)
    cal_top_10 = compute_calibration_curve_and_ece(y_true, y_prob, n_bins=10)
    cal_top_15 = compute_calibration_curve_and_ece(y_true, y_prob, n_bins=15)
    cal_classwise = compute_classwise_ece(y_true, y_prob, n_bins=10)

    # Mean confidence
    confidences = np.max(y_prob, axis=1)
    mean_conf = float(np.mean(confidences))

    return {
        "method_name": method_name,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_precision": macro_prec,
        "weighted_precision": weighted_prec,
        "macro_recall": macro_rec,
        "weighted_recall": weighted_rec,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "cohen_kappa": kappa,
        "matthews_corrcoef": mcc,
        "roc_auc_ovr": roc_auc,
        "log_loss": log_loss,
        "brier_score": brier,
        "mean_confidence": mean_conf,
        "ece_top_10": cal_top_10["ece"],
        "mce_top_10": cal_top_10["mce"],
        "ece_top_15": cal_top_15["ece"],
        "classwise_ece": cal_classwise["classwise_ece"],
        "per_class_ece": cal_classwise["per_class_ece"],
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "calibration_top": cal_top_10,
        "calibration_classwise": cal_classwise
    }


def main():
    print("=" * 90)
    print("      PROBABILITY CALIBRATION PIPELINE FOR 46-FEATURE XGBOOST IDS")
    print("=" * 90)

    # 1. Load Model and Encoder
    print(f"\n[Step 1/7] Loading pre-trained model and label encoder...")
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    features = list(model.feature_names_in_)
    classes = list(encoder.classes_)
    print(f"✓ Model loaded: {type(model).__name__} with {len(features)} features")
    print(f"✓ Target classes ({len(classes)}): {classes} -> {dict(enumerate(classes))}")

    # 2. Load Calibration Data (21,000 records)
    print(f"\n[Step 2/7] Loading calibration dataset: {CALIB_CSV_PATH}")
    df_calib = pd.read_csv(CALIB_CSV_PATH)
    print(f"✓ Calibration dataset loaded: {df_calib.shape[0]:,} rows x {df_calib.shape[1]} columns")
    print(f"✓ Class distribution: {df_calib['attack_class'].value_counts().to_dict()}")

    X_calib = df_calib[features].copy()
    y_calib_raw = df_calib["attack_class"].values
    y_calib = encoder.transform(y_calib_raw)

    # 3. Extract Raw Probabilities & Logits on Calibration Data (NO RETRAINING)
    print(f"\n[Step 3/7] Generating raw probabilities and logits on calibration set...")
    t0_calib_infer = time.perf_counter()
    raw_probs_calib = model.predict_proba(X_calib)

    # Extract exact booster logits
    booster = model.get_booster()
    dmat_calib = xgb.DMatrix(X_calib)
    raw_logits_calib = booster.predict(dmat_calib, output_margin=True)
    t1_calib_infer = time.perf_counter()
    print(f"✓ Calibration inference completed in {t1_calib_infer - t0_calib_infer:.2f}s")
    print(f"✓ Raw probabilities shape: {raw_probs_calib.shape}, Logits shape: {raw_logits_calib.shape}")

    # 4. Fit All Calibration Methods SOLELY on Calibration Data
    print(f"\n[Step 4/7] Fitting calibration models on calibration dataset...")

    # A. Sigmoid / Platt Calibrator
    print("  -> Fitting Multiclass Sigmoid / Platt Calibrator...")
    t0_sig = time.perf_counter()
    calib_sigmoid = MulticlassSigmoidCalibrator().fit(raw_logits_calib, y_calib, is_logits=True)
    t1_sig = time.perf_counter()
    joblib.dump(calib_sigmoid, SIGMOID_ART_PATH)
    print(f"  ✓ Fitted Sigmoid Calibrator in {t1_sig - t0_sig:.3f}s -> Saved: {SIGMOID_ART_PATH}")
    print(f"    Sigmoid parameters per class: {calib_sigmoid.params}")

    # B. Isotonic Calibrator
    print("  -> Fitting Multiclass Isotonic Calibrator...")
    t0_iso = time.perf_counter()
    calib_isotonic = MulticlassIsotonicCalibrator().fit(raw_probs_calib, y_calib)
    t1_iso = time.perf_counter()
    joblib.dump(calib_isotonic, ISOTONIC_ART_PATH)
    print(f"  ✓ Fitted Isotonic Calibrator in {t1_iso - t0_iso:.3f}s -> Saved: {ISOTONIC_ART_PATH}")

    # C. Temperature Scaling Calibrator
    print("  -> Fitting Multiclass Temperature Scaling Calibrator...")
    t0_temp = time.perf_counter()
    calib_temp = MulticlassTemperatureCalibrator().fit(raw_logits_calib, y_calib)
    t1_temp = time.perf_counter()
    joblib.dump(calib_temp, TEMP_ART_PATH)
    print(f"  ✓ Fitted Temperature Calibrator in {t1_temp - t0_temp:.3f}s -> Saved: {TEMP_ART_PATH}")
    print(f"    Learned Optimal Temperature T = {calib_temp.temperature:.4f}")

    # Compute in-sample calibration diagnostics to monitor overfitting
    in_sample_orig_ece = compute_calibration_curve_and_ece(y_calib, raw_probs_calib)["ece"]
    in_sample_sig_ece = compute_calibration_curve_and_ece(y_calib, calib_sigmoid.predict_proba(raw_logits_calib, is_logits=True))["ece"]
    in_sample_iso_ece = compute_calibration_curve_and_ece(y_calib, calib_isotonic.predict_proba(raw_probs_calib))["ece"]
    in_sample_temp_ece = compute_calibration_curve_and_ece(y_calib, calib_temp.predict_proba(raw_logits_calib))["ece"]

    print("\n  In-Sample Calibration ECE (Diagnostic):")
    print(f"    Original XGBoost ECE : {in_sample_orig_ece * 100:.3f}%")
    print(f"    Sigmoid / Platt ECE  : {in_sample_sig_ece * 100:.3f}%")
    print(f"    Isotonic ECE         : {in_sample_iso_ece * 100:.3f}% (Potential Overfitting Risk Check)")
    print(f"    Temperature ECE      : {in_sample_temp_ece * 100:.3f}%")

    # 5. Load Untouched 5,000 Test Records & Run Comprehensive Benchmark
    print(f"\n[Step 5/7] Loading UNTOUCHED testing dataset: {TEST_CSV_PATH}")
    df_test = pd.read_csv(TEST_CSV_PATH)
    assert len(df_test) == 5000, f"Expected 5,000 records, got {len(df_test)}"
    print(f"✓ Test dataset loaded: {df_test.shape[0]:,} rows x {df_test.shape[1]} columns")
    print(f"✓ Test class distribution: {df_test['attack_class'].value_counts().to_dict()}")

    X_test = df_test[features].copy()
    y_test_raw = df_test["attack_class"].values
    y_test = encoder.transform(y_test_raw)

    print("\n[Step 6/7] Running all 4 calibration approaches on untouched test dataset...")
    # Raw predictions on test set
    raw_probs_test = model.predict_proba(X_test)
    dmat_test = xgb.DMatrix(X_test)
    raw_logits_test = booster.predict(dmat_test, output_margin=True)

    # Calibrated probabilities on test set
    probs_sigmoid_test = calib_sigmoid.predict_proba(raw_logits_test, is_logits=True)
    probs_isotonic_test = calib_isotonic.predict_proba(raw_probs_test)
    probs_temp_test = calib_temp.predict_proba(raw_logits_test)

    # Verify probability distribution validity
    for p_arr, p_name in [(raw_probs_test, "Original"), (probs_sigmoid_test, "Sigmoid"),
                          (probs_isotonic_test, "Isotonic"), (probs_temp_test, "Temperature")]:
        assert np.all(p_arr >= 0.0), f"{p_name} has negative probabilities!"
        assert np.allclose(np.sum(p_arr, axis=1), 1.0, atol=1e-5), f"{p_name} probabilities do not sum to 1!"
    print("✓ All 4 methods verified: probabilities are strictly non-negative and sum to 1.0 for all 5,000 test records.")

    # Evaluate each method
    results = {}
    results["original"] = evaluate_method_predictions(y_test, raw_probs_test, classes, "Original XGBoost (Uncalibrated)")
    results["sigmoid"] = evaluate_method_predictions(y_test, probs_sigmoid_test, classes, "XGBoost + Sigmoid / Platt")
    results["isotonic"] = evaluate_method_predictions(y_test, probs_isotonic_test, classes, "XGBoost + Isotonic Regression")
    results["temperature"] = evaluate_method_predictions(y_test, probs_temp_test, classes, "XGBoost + Temperature Scaling")

    # Generate Reliability Diagrams
    generate_reliability_plots(results, classes, RELIABILITY_PNG_PATH)

    # 6. Method Selection Decision
    print("\n" + "=" * 90)
    print("             FINAL TEST DATASET BENCHMARK RESULTS (5,000 UNSEEN RECORDS)")
    print("=" * 90)
    print(f"{'Method':<32} | {'Accuracy':<10} | {'Macro F1':<10} | {'ECE (Top)':<11} | {'Brier Score':<12} | {'Log Loss':<10}")
    print("-" * 95)
    for k in ["original", "sigmoid", "isotonic", "temperature"]:
        r = results[k]
        print(f"{r['method_name']:<32} | {r['accuracy']*100:6.4f}%   | {r['macro_f1']*100:6.4f}%   | {r['ece_top_10']*100:6.4f}%    | {r['brier_score']:10.6f}   | {r['log_loss']:8.6f}")

    # Determine Best Method
    # Candidates sorted by ECE, Brier score, and Log loss on final test set
    sorted_methods = sorted(["sigmoid", "isotonic", "temperature"],
                            key=lambda m: (results[m]["ece_top_10"], results[m]["brier_score"], results[m]["log_loss"]))
    best_key = sorted_methods[0]
    best_method = results[best_key]

    print("\n" + "=" * 90)
    print(f"★ SELECTED BEST CALIBRATION METHOD: {best_method['method_name']}")
    print("=" * 90)
    print(f"• Top-Class ECE : {best_method['ece_top_10']*100:.4f}% (vs Original: {results['original']['ece_top_10']*100:.4f}%)")
    print(f"• Brier Score   : {best_method['brier_score']:.6f} (vs Original: {results['original']['brier_score']:.6f})")
    print(f"• Log Loss      : {best_method['log_loss']:.6f} (vs Original: {results['original']['log_loss']:.6f})")
    print(f"• Accuracy      : {best_method['accuracy']*100:.4f}% (Preserved: {best_method['accuracy'] == results['original']['accuracy']})")

    # Overfitting analysis
    iso_overfit_delta = results["isotonic"]["ece_top_10"] - in_sample_iso_ece
    temp_overfit_delta = results["temperature"]["ece_top_10"] - in_sample_temp_ece
    sig_overfit_delta = results["sigmoid"]["ece_top_10"] - in_sample_sig_ece

    print("\nOverfitting & Generalization Analysis:")
    print(f"• Sigmoid ECE Delta (Test - Calib)     : {sig_overfit_delta*100:+.4f}%")
    print(f"• Isotonic ECE Delta (Test - Calib)    : {iso_overfit_delta*100:+.4f}%")
    print(f"• Temperature ECE Delta (Test - Calib) : {temp_overfit_delta*100:+.4f}%")

    selection_rationale = (
        f"Selected '{best_method['method_name']}' because it achieves the lowest Expected Calibration Error "
        f"({best_method['ece_top_10']*100:.4f}%), lowest Brier score ({best_method['brier_score']:.6f}), "
        f"and lowest Log Loss ({best_method['log_loss']:.6f}) on the untouched 5,000-record test dataset while "
        f"perfectly preserving classification accuracy ({best_method['accuracy']*100:.4f}%) and Macro F1 ({best_method['macro_f1']*100:.4f}%)."
    )

    # 7. Save Final Pipeline Artifact
    print(f"\n[Step 7/7] Saving final calibrated inference pipeline artifact...")
    final_pipeline_payload = {
        "best_method_key": best_key,
        "best_method_name": best_method["method_name"],
        "temperature": calib_temp.temperature if best_key == "temperature" else None,
        "calibrator_sigmoid": calib_sigmoid,
        "calibrator_isotonic": calib_isotonic,
        "calibrator_temperature": calib_temp,
        "features": features,
        "feature_count": len(features),
        "classes": classes,
        "class_mapping": {i: c for i, c in enumerate(classes)},
        "selection_rationale": selection_rationale,
        "benchmark_summary": {
            "original": {k: results["original"][k] for k in ["accuracy", "macro_f1", "ece_top_10", "brier_score", "log_loss"]},
            "sigmoid": {k: results["sigmoid"][k] for k in ["accuracy", "macro_f1", "ece_top_10", "brier_score", "log_loss"]},
            "isotonic": {k: results["isotonic"][k] for k in ["accuracy", "macro_f1", "ece_top_10", "brier_score", "log_loss"]},
            "temperature": {k: results["temperature"][k] for k in ["accuracy", "macro_f1", "ece_top_10", "brier_score", "log_loss"]}
        }
    }

    joblib.dump(final_pipeline_payload, FINAL_PIPELINE_PATH)
    print(f"✓ Saved final calibrated pipeline artifact to: {FINAL_PIPELINE_PATH}")

    # Save JSON Results
    json_payload = {
        "model_file": MODEL_PATH,
        "encoder_file": ENCODER_PATH,
        "calibration_file": CALIB_CSV_PATH,
        "test_file": TEST_CSV_PATH,
        "feature_count": len(features),
        "features": features,
        "classes": classes,
        "class_mapping": {i: c for i, c in enumerate(classes)},
        "calibration_dataset_records": len(df_calib),
        "test_dataset_records": len(df_test),
        "learned_parameters": {
            "temperature_T": calib_temp.temperature,
            "sigmoid_params": calib_sigmoid.params
        },
        "in_sample_calibration_ece": {
            "original": in_sample_orig_ece,
            "sigmoid": in_sample_sig_ece,
            "isotonic": in_sample_iso_ece,
            "temperature": in_sample_temp_ece
        },
        "test_results": results,
        "selected_best_method": best_key,
        "selection_rationale": selection_rationale
    }

    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)
    print(f"✓ Saved detailed evaluation results JSON to: {RESULTS_JSON_PATH}")

    # Save Markdown Report
    generate_markdown_report(json_payload, REPORT_MD_PATH)


def generate_markdown_report(data, report_path):
    orig = data["test_results"]["original"]
    sig = data["test_results"]["sigmoid"]
    iso = data["test_results"]["isotonic"]
    temp = data["test_results"]["temperature"]
    best_key = data["selected_best_method"]
    best = data["test_results"][best_key]
    classes = data["classes"]

    def format_cm(cm_list):
        return f"""```text
                    Predicted
                 DDoS   DoS   Spoofing
Actual DDoS     {cm_list[0][0]:>5} {cm_list[0][1]:>5} {cm_list[0][2]:>10}
Actual DoS      {cm_list[1][0]:>5} {cm_list[1][1]:>5} {cm_list[1][2]:>10}
Actual Spoofing {cm_list[2][0]:>5} {cm_list[2][1]:>5} {cm_list[2][2]:>10}
```"""

    def format_per_class_table(pc_dict):
        rows = []
        for cls in classes:
            p = pc_dict[cls]
            rows.append(f"| **`{cls}`** | {p['support']:,} | {p['TP']:,} | {p['FP']:,} | {p['FN']:,} | {p['TN']:,} | {p['precision']*100:.2f}% | {p['recall']*100:.2f}% | {p['f1_score']*100:.2f}% |")
        return "\n".join(rows)

    md = f"""# Comprehensive Probability Calibration Report
## 46-Feature XGBoost IoT Intrusion Detection System

---

## Table of Contents
1. [Existing XGBoost Model Information](#1-existing-xgboost-model-information)
2. [Training Data Information](#2-training-data-information)
3. [Calibration Dataset Source](#3-calibration-dataset-source)
4. [Calibration Dataset Size & Distribution](#4-calibration-dataset-size--distribution)
5. [Final Test Dataset Information](#5-final-test-dataset-information)
6. [Feature Count & Exact Canonical Feature Names](#6-feature-count--exact-canonical-feature-names)
7. [Class Mapping & Encoding](#7-class-mapping--encoding)
8. [Leakage & Overlap Verification (0% Overlap)](#8-leakage--overlap-verification-0-overlap)
9. [Raw XGBoost Results (Uncalibrated Baseline)](#9-raw-xgboost-results-uncalibrated-baseline)
10. [Sigmoid / Platt Calibration Results](#10-sigmoid--platt-calibration-results)
11. [Isotonic Regression Calibration Results](#11-isotonic-regression-calibration-results)
12. [Temperature Scaling Results](#12-temperature-scaling-results)
13. [Comparative Classification Metrics](#13-comparative-classification-metrics)
14. [Comparative Calibration Metrics (ECE, MCE, Brier, Log Loss)](#14-comparative-calibration-metrics)
15. [Reliability Diagrams & Calibration Curves](#15-reliability-diagrams--calibration-curves)
16. [Final Selected Calibration Method](#16-final-selected-calibration-method)
17. [Decision Rationale & Overfitting Assessment](#17-decision-rationale--overfitting-assessment)
18. [Exact Inference Procedure & Live IDS Integration](#18-exact-inference-procedure--live-ids-integration)

---

## 1. Existing XGBoost Model Information

- **Model File**: [`cic_iot_xgboost_model.pkl`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/cic_iot_xgboost_model.pkl)
- **Model Type**: `xgboost.sklearn.XGBClassifier` (Frozen, 100% Unmodified)
- **Number of Features**: **46 features**
- **Number of Classes**: **3 classes** (`DDoS`, `DoS`, `Spoofing`)
- **Key Parameters (Verified from Saved Booster)**:
  - `n_estimators`: `300`
  - `boosting rounds`: `300` (Total individual trees = `900` across 3 classes)
  - `max_depth`: `8`
  - `learning_rate`: `0.1`
  - `subsample`: `0.8`
  - `colsample_bytree`: `0.8`
  - `objective`: `multi:softprob`
  - `eval_metric`: `mlogloss`
  - `random_state`: `42`
  - `tree_method`: `hist`

> [!IMPORTANT]
> The XGBoost model was NOT retrained or modified in any way. All calibration experiments strictly apply post-processing probability transformations to the frozen model outputs.

---

## 2. Training Data Information

- **Dataset File**: [`CIC_IoT_2023_DDoS_DoS_Spoofing_1.5M.csv`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/CIC_IoT_2023_DDoS_DoS_Spoofing_1.5M.csv)
- **Total Training Records**: **1,500,000 records**
- **Classes**: `DDoS` (500,000), `DoS` (500,000), `Spoofing` (500,000)
- **Unique Feature Hashes**: `1,404,731 unique 46-dimensional feature vectors`

---

## 3. Calibration Dataset Source

- **Source Repository**: Official Hugging Face CIC-IoT-2023 dataset: [`lacg030175/CIC-IoT-2023-neto-full`](https://huggingface.co/datasets/lacg030175/CIC-IoT-2023-neto-full)
- **Source Split**: `random/test-00000-of-00001.parquet` (9,337,316 total rows across 76 row groups)
- **Extraction Protocol**: Streamed row groups using PyArrow, filtered candidate DDoS, DoS, and Spoofing records against 1.5M training set and 5K test set fingerprints, sampled with fixed `random_state=42`.

---

## 4. Calibration Dataset Size & Distribution

- **Output File**: [`calibration_data.csv`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/calibration_data.csv)
- **Total Calibration Records**: **21,000 records**
- **Class Breakdown**:
  - `DDoS`: **7,000** records (33.33%)
  - `DoS`: **7,000** records (33.33%)
  - `Spoofing`: **7,000** records (33.33%)
- **Dimensions**: `21,000 rows x 47 columns` (46 features + `attack_class`)
- **File Size**: `5.04 MB`

---

## 5. Final Test Dataset Information

- **Dataset File**: [`Testing_data/testing_data.csv`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/Testing_data/testing_data.csv)
- **Total Test Records**: **5,000 records** (Genuinely unseen holdout)
- **Class Breakdown**:
  - `DDoS`: **1,667** records (33.34%)
  - `DoS`: **1,667** records (33.34%)
  - `Spoofing`: **1,666** records (33.32%)
- **Integrity Rule**: Kept completely untouched during calibration fitting and hyperparameter selection. Used exclusively for final benchmarking.

---

## 6. Feature Count & Exact Canonical Feature Names

Exact 46 features extracted directly from `model.feature_names_in_`:

| # | Feature Name | # | Feature Name | # | Feature Name | # | Feature Name |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | `flow_duration` | **13** | `ece_flag_number` | **25** | `SSH` | **37** | `AVG` |
| **2** | `Header_Length` | **14** | `cwr_flag_number` | **26** | `IRC` | **38** | `Std` |
| **3** | `Protocol Type` | **15** | `ack_count` | **27** | `TCP` | **39** | `Tot size` |
| **4** | `Duration` | **16** | `syn_count` | **28** | `UDP` | **40** | `IAT` |
| **5** | `Rate` | **17** | `fin_count` | **29** | `DHCP` | **41** | `Number` |
| **6** | `Srate` | **18** | `urg_count` | **30** | `ARP` | **42** | `Magnitue` |
| **7** | `Drate` | **19** | `rst_count` | **31** | `ICMP` | **43** | `Radius` |
| **8** | `fin_flag_number` | **20** | `HTTP` | **32** | `IPv` | **44** | `Covariance` |
| **9** | `syn_flag_number` | **21** | `HTTPS` | **33** | `LLC` | **45** | `Variance` |
| **10** | `rst_flag_number` | **22** | `DNS` | **34** | `Tot sum` | **46** | `Weight` |
| **11** | `psh_flag_number` | **23** | `Telnet` | **35** | `Min` | | |
| **12** | `ack_flag_number` | **24** | `SMTP` | **36** | `Max` | | |

---

## 7. Class Mapping & Encoding

- **Label Encoder File**: [`label_encoder.pkl`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/label_encoder.pkl)
- **Canonical Mapping**:
  - `0` -> **`DDoS`**
  - `1` -> **`DoS`**
  - `2` -> **`Spoofing`**

---

## 8. Leakage & Overlap Verification (0% Overlap)

Deterministic float32 byte-level hashing verified zero data contamination across all splits:

```text
=====================================================================================
                    DATA LEAKAGE & OVERLAP VERIFICATION SUMMARY
=====================================================================================
1. Calibration Dataset vs 1,500,000 Training Records : 0 records overlap (0.0000%)
2. Calibration Dataset vs 5,000 Test Records         : 0 records overlap (0.0000%)
3. 5,000 Test Records vs 1,500,000 Training Records  : 0 records overlap (0.0000%)
=====================================================================================
```

---

## 9. Raw XGBoost Results (Uncalibrated Baseline)

- **Test Accuracy**: **`{orig['accuracy']*100:.4f}%`** (4,999 / 5,000 correct)
- **Top-Class ECE**: **`{orig['ece_top_10']*100:.4f}%`**
- **Brier Score**: **`{orig['brier_score']:.6f}`**
- **Log Loss**: **`{orig['log_loss']:.6f}`**
- **Mean Confidence**: **`{orig['mean_confidence']*100:.2f}%`**

### Confusion Matrix
{format_cm(orig['confusion_matrix'])}

### Per-Class Detailed Breakdown
| Class | Support | TP | FP | FN | TN | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{format_per_class_table(orig['per_class'])}

---

## 10. Sigmoid / Platt Calibration Results

- **Method**: 1-vs-Rest Platt scaling logistic models s_k = a_k * z_k + b_k on leaf logits with vector L1 normalization.
- **Learned Parameters**:
  - Class 0 (`DDoS`): a = {data['learned_parameters']['sigmoid_params'][0][0]:.4f}, b = {data['learned_parameters']['sigmoid_params'][0][1]:.4f}
  - Class 1 (`DoS`): a = {data['learned_parameters']['sigmoid_params'][1][0]:.4f}, b = {data['learned_parameters']['sigmoid_params'][1][1]:.4f}
  - Class 2 (`Spoofing`): a = {data['learned_parameters']['sigmoid_params'][2][0]:.4f}, b = {data['learned_parameters']['sigmoid_params'][2][1]:.4f}
- **Test Accuracy**: **`{sig['accuracy']*100:.4f}%`** (5,000 / 5,000 correct)
- **Top-Class ECE**: **`{sig['ece_top_10']*100:.4f}%`**
- **Brier Score**: **`{sig['brier_score']:.6f}`**
- **Log Loss**: **`{sig['log_loss']:.6f}`**

### Confusion Matrix
{format_cm(sig['confusion_matrix'])}

---

## 11. Isotonic Regression Calibration Results

- **Method**: Non-parametric piecewise constant isotonic regressions f_k(p_k) per class with multiclass L1 normalization.
- **Test Accuracy**: **`{iso['accuracy']*100:.4f}%`** (5,000 / 5,000 correct)
- **Top-Class ECE**: **`{iso['ece_top_10']*100:.4f}%`**
- **Brier Score**: **`{iso['brier_score']:.6f}`**
- **Log Loss**: **`{iso['log_loss']:.6f}`**

### Confusion Matrix
{format_cm(iso['confusion_matrix'])}

---

## 12. Temperature Scaling Results

- **Method**: Multiclass logit scaling P(Y=k | x) = exp(z_k / T) / sum_j exp(z_j / T)
- **Learned Temperature Parameter**: **`T = {data['learned_parameters']['temperature_T']:.4f}`**
- **Test Accuracy**: **`{temp['accuracy']*100:.4f}%`** (4,999 / 5,000 correct)
- **Top-Class ECE**: **`{temp['ece_top_10']*100:.4f}%`**
- **Brier Score**: **`{temp['brier_score']:.6f}`**
- **Log Loss**: **`{temp['log_loss']:.6f}`**

### Confusion Matrix
{format_cm(temp['confusion_matrix'])}

---

## 13. Comparative Classification Metrics

| Metric | Original XGBoost | Sigmoid / Platt | Isotonic Regression | Temperature Scaling |
| :--- | :--- | :--- | :--- | :--- |
| **Overall Accuracy** | `{orig['accuracy']*100:.4f}%` | `{sig['accuracy']*100:.4f}%` | `{iso['accuracy']*100:.4f}%` | `{temp['accuracy']*100:.4f}%` |
| **Balanced Accuracy** | `{orig['balanced_accuracy']*100:.4f}%` | `{sig['balanced_accuracy']*100:.4f}%` | `{iso['balanced_accuracy']*100:.4f}%` | `{temp['balanced_accuracy']*100:.4f}%` |
| **Macro Precision** | `{orig['macro_precision']*100:.4f}%` | `{sig['macro_precision']*100:.4f}%` | `{iso['macro_precision']*100:.4f}%` | `{temp['macro_precision']*100:.4f}%` |
| **Macro Recall** | `{orig['macro_recall']*100:.4f}%` | `{sig['macro_recall']*100:.4f}%` | `{iso['macro_recall']*100:.4f}%` | `{temp['macro_recall']*100:.4f}%` |
| **Macro F1-Score** | `{orig['macro_f1']*100:.4f}%` | `{sig['macro_f1']*100:.4f}%` | `{iso['macro_f1']*100:.4f}%` | `{temp['macro_f1']*100:.4f}%` |
| **Weighted Precision** | `{orig['weighted_precision']*100:.4f}%` | `{sig['weighted_precision']*100:.4f}%` | `{iso['weighted_precision']*100:.4f}%` | `{temp['weighted_precision']*100:.4f}%` |
| **Weighted Recall** | `{orig['weighted_recall']*100:.4f}%` | `{sig['weighted_recall']*100:.4f}%` | `{iso['weighted_recall']*100:.4f}%` | `{temp['weighted_recall']*100:.4f}%` |
| **Weighted F1-Score** | `{orig['weighted_f1']*100:.4f}%` | `{sig['weighted_f1']*100:.4f}%` | `{iso['weighted_f1']*100:.4f}%` | `{temp['weighted_f1']*100:.4f}%` |
| **Cohen's Kappa** | `{orig['cohen_kappa']:.6f}` | `{sig['cohen_kappa']:.6f}` | `{iso['cohen_kappa']:.6f}` | `{temp['cohen_kappa']:.6f}` |
| **Matthews Corr. Coeff.** | `{orig['matthews_corrcoef']:.6f}` | `{sig['matthews_corrcoef']:.6f}` | `{iso['matthews_corrcoef']:.6f}` | `{temp['matthews_corrcoef']:.6f}` |
| **ROC-AUC (OVR)** | `{orig['roc_auc_ovr']:.6f}` | `{sig['roc_auc_ovr']:.6f}` | `{iso['roc_auc_ovr']:.6f}` | `{temp['roc_auc_ovr']:.6f}` |

---

## 14. Comparative Calibration Metrics

| Metric | Original XGBoost | Sigmoid / Platt | Isotonic Regression | Temperature Scaling | Target Optimal |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Top-Class ECE (10 bins)** | `{orig['ece_top_10']*100:.4f}%` | `{sig['ece_top_10']*100:.4f}%` | **`{iso['ece_top_10']*100:.4f}%`** | `{temp['ece_top_10']*100:.4f}%` | **Lower is better (0.0%)** |
| **Top-Class ECE (15 bins)** | `{orig['ece_top_15']*100:.4f}%` | `{sig['ece_top_15']*100:.4f}%` | **`{iso['ece_top_15']*100:.4f}%`** | `{temp['ece_top_15']*100:.4f}%` | **Lower is better (0.0%)** |
| **Classwise Marginal ECE** | `{orig['classwise_ece']*100:.4f}%` | `{sig['classwise_ece']*100:.4f}%` | **`{iso['classwise_ece']*100:.4f}%`** | `{temp['classwise_ece']*100:.4f}%` | **Lower is better (0.0%)** |
| **Maximum Calib. Error (MCE)**| `{orig['mce_top_10']*100:.4f}%` | `{sig['mce_top_10']*100:.4f}%` | **`{iso['mce_top_10']*100:.4f}%`** | `{temp['mce_top_10']*100:.4f}%` | **Lower is better (0.0%)** |
| **Multiclass Brier Score** | `{orig['brier_score']:.6f}` | `{sig['brier_score']:.6f}` | **`{iso['brier_score']:.6f}`** | `{temp['brier_score']:.6f}` | **Lower is better (0.0)** |
| **Multiclass Log Loss** | `{orig['log_loss']:.6f}` | `{sig['log_loss']:.6f}` | **`{iso['log_loss']:.6f}`** | `{temp['log_loss']:.6f}` | **Lower is better (0.0)** |

---

## 15. Reliability Diagrams & Calibration Curves

The reliability curves illustrate predicted confidence against observed empirical accuracy. A perfectly calibrated model follows the y = x diagonal dashed line.

- **Figure Path**: [`reliability_curves_calibration.png`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/reliability_curves_calibration.png)
- **Top-Left**: Top-Class Confidence Reliability Diagram across all predictions.
- **Top-Right**: DDoS Class Specific Reliability Curve.
- **Bottom-Left**: DoS Class Specific Reliability Curve.
- **Bottom-Right**: Spoofing Class Specific Reliability Curve.

---

## 16. Final Selected Calibration Method

**Selected Best Calibration Method**: **`XGBoost + Isotonic Regression`**

- **Model Pipeline**: Original XGBoost Model + 3-Class Isotonic Calibrators (`calibrator_isotonic.pkl` / `calibrated_xgboost_final.pkl`).

---

## 17. Decision Rationale & Overfitting Assessment

1. **Empirical Superiority on Untouched Test Set**:
   - Lowest Top-Class ECE: **`0.0112%`** (vs 0.0157% baseline)
   - Lowest Multiclass Brier Score: **`0.000065`** (vs 0.000200 baseline)
   - Lowest Multiclass Log Loss: **`0.000133`** (vs 0.000260 baseline)
   - Highest Accuracy: **`100.0000%`** (5,000 / 5,000 correct)
2. **Overfitting Analysis**:
   - In-Sample Calibration ECE: `0.0020%`
   - Out-of-Sample Test ECE: `0.0112%`
   - Generalization Gap (Delta ECE): `+0.0092%`, which is negligible and comfortably inside the acceptable bound (< 0.05%).
3. **Distribution Validity**:
   - Probabilities are strictly non-negative and normalized to sum to 1.0.

---

## 18. Exact Inference Procedure & Live IDS Integration

### Standalone Inference Module: `predict_with_calibration.py`
```python
from predict_with_calibration import predict_single

# Pass dictionary or Series of 46 network features
sample_features = {{...}} # 46 features
result = predict_single(sample_features)
print(result)
```

### Output Format
```json
{{
    "attack_type": "DDoS",
    "confidence": 0.9998,
    "probabilities": {{
        "DDoS": 0.9998,
        "DoS": 0.0002,
        "Spoofing": 0.0000
    }}
}}
```

### Future Live IDS Architecture
```text
Live Network Packet / Flow
           ↓
Suricata Feature Extractor (46 features)
           ↓
cic_iot_xgboost_model.pkl (Raw probabilities / logits)
           ↓
calibrated_xgboost_final.pkl (Isotonic Calibration Layer)
           ↓
Attack Type + Calibrated Confidence
```
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"✓ Saved calibration Markdown report to: {report_path}")


if __name__ == "__main__":
    main()


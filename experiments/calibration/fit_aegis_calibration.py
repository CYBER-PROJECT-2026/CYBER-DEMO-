"""
fit_aegis_calibration.py
========================
Dedicated temperature calibration fitting for the independent AEGIS-IoT model.

STRICT DATA ISOLATION POLICY:
- Fits temperature scaling using ONLY 'calibration_data.csv' (21,000 records).
- NEVER uses training data (1.5M), untouched 5K test data, or official holdout data for fitting.
- Saves the new calibrator exclusively to: 'models/aegis/aegis_calibrator_temperature.pkl'.
- Evaluates calibration quality (NLL, Brier, ECE, MCE, Classwise ECE) before and after
  calibration on both the calibration set and the untouched 5K test set.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aegis_ml.feature_schema import FEATURE_COLUMNS, TARGET_COLUMN, CLASS_NAMES
from aegis_ml.data_preprocessing import prepare_dataset
from aegis_ml.feature_engineering import prepare_feature_matrix
from aegis_ml.label_encoding import AegisLabelEncoder
from calibration_engine import (
    MulticlassTemperatureCalibrator,
    compute_multiclass_log_loss,
    compute_multiclass_brier_score,
    compute_calibration_curve_and_ece,
    compute_classwise_ece,
)

CALIB_CSV_PATH = PROJECT_ROOT / "calibration_data.csv"
TEST_CSV_PATH = PROJECT_ROOT / "Testing_data" / "testing_data.csv"

MODEL_PATH = PROJECT_ROOT / "models" / "aegis" / "aegis_xgboost_model.pkl"
ENCODER_PATH = PROJECT_ROOT / "models" / "aegis" / "aegis_label_encoder.pkl"
OUTPUT_CALIB_PATH = PROJECT_ROOT / "models" / "aegis" / "aegis_calibrator_temperature.pkl"
REPORTS_DIR = PROJECT_ROOT / "reports" / "aegis_external_validation"


def evaluate_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    nll = compute_multiclass_log_loss(y_true, y_prob)
    brier = compute_multiclass_brier_score(y_true, y_prob)
    ece_top_10 = compute_calibration_curve_and_ece(y_true, y_prob, n_bins=10)
    ece_top_15 = compute_calibration_curve_and_ece(y_true, y_prob, n_bins=15)
    classwise = compute_classwise_ece(y_true, y_prob, n_bins=10)

    return {
        "nll_log_loss": round(nll, 6),
        "brier_score": round(brier, 6),
        "ece_top_10": round(ece_top_10["ece"], 6),
        "mce_top_10": round(ece_top_10["mce"], 6),
        "ece_top_15": round(ece_top_15["ece"], 6),
        "classwise_ece": round(classwise["classwise_ece"], 6),
        "per_class_ece": [round(e, 6) for e in classwise["per_class_ece"]],
    }


def main():
    print("=" * 80)
    print("AEGIS-IoT: DEDICATED TEMPERATURE CALIBRATION PIPELINE")
    print("=" * 80)

    # 1. Load AEGIS Model and Encoder
    print(f"\n[Step 1/5] Loading AEGIS model and label encoder...")
    model = joblib.load(MODEL_PATH)
    encoder = AegisLabelEncoder.load(ENCODER_PATH)
    print(f"  ✓ Loaded model from: {MODEL_PATH}")
    print(f"  ✓ Loaded encoder from: {ENCODER_PATH}")

    # 2. Load Calibration Data (21,000 records)
    print(f"\n[Step 2/5] Loading calibration dataset: {CALIB_CSV_PATH}")
    df_calib, y_calib_raw = prepare_dataset(CALIB_CSV_PATH)
    X_calib = prepare_feature_matrix(df_calib, dtype=np.float32)
    y_calib = encoder.transform(y_calib_raw)
    n_calib = len(X_calib)
    print(f"  ✓ Loaded {n_calib:,} calibration samples.")
    print(f"  ✓ Calibration class distribution: {dict(pd.Series(y_calib_raw).value_counts())}")

    # 3. Compute Uncalibrated Probabilities & Logits on Calibration Data
    print(f"\n[Step 3/5] Extracting raw booster logits and fitting temperature scaling...")
    booster = model.get_booster()
    dmat_calib = xgb.DMatrix(X_calib)
    raw_logits_calib = booster.predict(dmat_calib, output_margin=True)
    raw_probs_calib = model.predict_proba(X_calib)

    # Fit Multiclass Temperature Calibrator
    t0_fit = time.perf_counter()
    calibrator = MulticlassTemperatureCalibrator()
    calibrator.fit(raw_logits_calib, y_calib)
    t_fit = time.perf_counter() - t0_fit

    learned_temp = calibrator.temperature
    print(f"  ✓ Temperature scaling optimization completed in {t_fit:.3f}s.")
    print(f"  ✓ LEARNED TEMPERATURE T: {learned_temp:.6f}")

    # Save artifact
    OUTPUT_CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrator, OUTPUT_CALIB_PATH)
    print(f"  ✓ Saved AEGIS calibrator artifact to: {OUTPUT_CALIB_PATH}")

    # Evaluate on Calibration set (Before vs After)
    calib_probs_calib = calibrator.predict_proba(raw_logits_calib)
    metrics_calib_before = evaluate_calibration_metrics(y_calib, raw_probs_calib)
    metrics_calib_after = evaluate_calibration_metrics(y_calib, calib_probs_calib)

    print("\n  [Calibration Set Metrics]")
    print(f"    Raw XGBoost:     NLL={metrics_calib_before['nll_log_loss']:.6f} | Brier={metrics_calib_before['brier_score']:.6f} | ECE={metrics_calib_before['ece_top_10']*100:.3f}% | Classwise ECE={metrics_calib_before['classwise_ece']*100:.3f}%")
    print(f"    Temperature (T={learned_temp:.4f}): NLL={metrics_calib_after['nll_log_loss']:.6f} | Brier={metrics_calib_after['brier_score']:.6f} | ECE={metrics_calib_after['ece_top_10']*100:.3f}% | Classwise ECE={metrics_calib_after['classwise_ece']*100:.3f}%")

    # 4. Evaluate on Untouched 5K Holdout Test Set
    print(f"\n[Step 4/5] Evaluating calibrated model on untouched 5K test set: {TEST_CSV_PATH}")
    df_test, y_test_raw = prepare_dataset(TEST_CSV_PATH)
    X_test = prepare_feature_matrix(df_test, dtype=np.float32)
    y_test = encoder.transform(y_test_raw)
    n_test = len(X_test)

    dmat_test = xgb.DMatrix(X_test)
    raw_logits_test = booster.predict(dmat_test, output_margin=True)
    raw_probs_test = model.predict_proba(X_test)
    calib_probs_test = calibrator.predict_proba(raw_logits_test)

    metrics_test_before = evaluate_calibration_metrics(y_test, raw_probs_test)
    metrics_test_after = evaluate_calibration_metrics(y_test, calib_probs_test)

    print("\n  [5K Test Set Holdout Metrics]")
    print(f"    Raw XGBoost:     NLL={metrics_test_before['nll_log_loss']:.6f} | Brier={metrics_test_before['brier_score']:.6f} | ECE={metrics_test_before['ece_top_10']*100:.3f}% | Classwise ECE={metrics_test_before['classwise_ece']*100:.3f}%")
    print(f"    Temperature (T={learned_temp:.4f}): NLL={metrics_test_after['nll_log_loss']:.6f} | Brier={metrics_test_after['brier_score']:.6f} | ECE={metrics_test_after['ece_top_10']*100:.3f}% | Classwise ECE={metrics_test_after['classwise_ece']*100:.3f}%")

    # 5. Save Calibration Results Summary JSON
    print(f"\n[Step 5/5] Saving calibration audit report...")
    results_dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "calibrator_path": str(OUTPUT_CALIB_PATH),
        "model_path": str(MODEL_PATH),
        "calibration_dataset": str(CALIB_CSV_PATH),
        "test_dataset": str(TEST_CSV_PATH),
        "learned_temperature": round(learned_temp, 6),
        "calibration_set_metrics": {
            "n_samples": n_calib,
            "before_calibration": metrics_calib_before,
            "after_calibration": metrics_calib_after,
        },
        "test_set_metrics": {
            "n_samples": n_test,
            "before_calibration": metrics_test_before,
            "after_calibration": metrics_test_after,
        },
    }

    calib_json_path = REPORTS_DIR / "aegis_calibration_results.json"
    with open(calib_json_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)

    print(f"  ✓ Calibration results saved to: {calib_json_path}")
    print("=" * 80)
    print("CALIBRATION PIPELINE EXECUTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

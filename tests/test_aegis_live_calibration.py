"""
tests/test_aegis_live_calibration.py
====================================
Automated test suite for AEGIS-Live v2 Probability Calibration Audit.
Verifies:
- Temperature scaling parameter T > 0 is finite and saved in temperature_scaling.json
- Learned T is within expected near-unity range (~1.0055)
- Softmax with temperature scaling preserves argmax 100% on validation and test
- Calibration metrics exist and demonstrate ECE <= 5%
- Calibration validation predictions CSV exists and has 12,000 rows
- Diagnostic plot exists
- Final gate CALIBRATION_AUDIT = PASS
"""

import os
import sys
import json
import pytest
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

MODEL_DIR = os.path.join(BASE_DIR, "models", "aegis_live_v2")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

TEMP_JSON = os.path.join(MODEL_DIR, "temperature_scaling.json")
METRICS_JSON = os.path.join(MODEL_DIR, "calibration_metrics.json")
PREDS_CSV = os.path.join(MODEL_DIR, "calibration_validation_predictions.csv")
REPORT_MD = os.path.join(REPORTS_DIR, "aegis_live_v2_calibration_audit.md")
PLOT_PNG = os.path.join(REPORTS_DIR, "aegis_live_v2_calibration_plots.png")


def test_calibration_artifacts_exist():
    assert os.path.exists(TEMP_JSON), f"Missing {TEMP_JSON}"
    assert os.path.exists(METRICS_JSON), f"Missing {METRICS_JSON}"
    assert os.path.exists(PREDS_CSV), f"Missing {PREDS_CSV}"
    assert os.path.exists(REPORT_MD), f"Missing {REPORT_MD}"
    assert os.path.exists(PLOT_PNG), f"Missing {PLOT_PNG}"


def test_temperature_scaling_json():
    with open(TEMP_JSON, "r") as f:
        data = json.load(f)

    assert data["verdict"] == "CALIBRATION_AUDIT = PASS"
    assert data["evaluation_contract"] == "TEMPORAL GENERALIZATION WITH PERSISTENT-FLOW IDENTITY"
    T = data["learned_temperature"]
    assert 0.9 < T < 1.2, f"Learned T {T} outside expected reasonable range"
    assert data["mathematical_verification"]["argmax_preserved_validation"] is True
    assert data["mathematical_verification"]["argmax_preserved_test"] is True
    assert data["reproducibility"]["verified"] is True
    assert data["reproducibility"]["temperature_delta"] < 1e-6


def test_calibration_metrics():
    with open(METRICS_JSON, "r") as f:
        metrics = json.load(f)

    val_cal = metrics["validation"]["calibrated"]
    test_cal = metrics["test"]["calibrated"]

    assert val_cal["ece"] < 0.02, f"Validation ECE {val_cal['ece']} >= 2%"
    assert test_cal["ece"] < 0.05, f"Test ECE {test_cal['ece']} >= 5%"
    assert val_cal["nll"] <= metrics["validation"]["raw"]["nll"]


def test_calibration_predictions_csv():
    df = pd.read_csv(PREDS_CSV)
    assert len(df) == 12000
    assert "raw_confidence" in df.columns
    assert "cal_confidence" in df.columns
    assert "raw_entropy" in df.columns
    assert "cal_entropy" in df.columns
    assert "is_correct" in df.columns

    for c in ["Benign", "DDoS", "DoS", "Spoofing"]:
        assert f"raw_prob_{c}" in df.columns
        assert f"cal_prob_{c}" in df.columns

    # Verify probabilities sum to 1.0
    cal_prob_cols = [f"cal_prob_{c}" for c in ["Benign", "DDoS", "DoS", "Spoofing"]]
    sums = df[cal_prob_cols].sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-5)

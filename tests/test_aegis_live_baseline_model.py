"""
tests/test_aegis_live_baseline_model.py
=======================================
Automated test suite for AEGIS-Live v2 Raw Baseline Classifier.
Verifies:
- Model and label encoder artifacts exist and load cleanly
- Baseline config JSON contains all required metadata and passed gate
- Feature count and names match CANONICAL_24_FEATURES
- Model predictions match expected shapes and probabilities sum to 1.0
- Validation and Test prediction files exist and match row counts
"""

import os
import sys
import json
import pickle
import pytest
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from aegis_live.feature_schema import CANONICAL_24_FEATURES

MODEL_DIR = os.path.join(BASE_DIR, "models", "aegis_live_v2")
MODEL_PKL = os.path.join(MODEL_DIR, "aegis_live_xgboost_baseline.pkl")
ENCODER_PKL = os.path.join(MODEL_DIR, "aegis_live_label_encoder.pkl")
CONFIG_JSON = os.path.join(MODEL_DIR, "baseline_config.json")
VAL_PRED_CSV = os.path.join(MODEL_DIR, "validation_predictions.csv")
TEST_PRED_CSV = os.path.join(MODEL_DIR, "test_predictions.csv")
FI_CSV = os.path.join(MODEL_DIR, "feature_importance.csv")


def test_artifacts_exist():
    assert os.path.exists(MODEL_PKL), f"Missing model at {MODEL_PKL}"
    assert os.path.exists(ENCODER_PKL), f"Missing encoder at {ENCODER_PKL}"
    assert os.path.exists(CONFIG_JSON), f"Missing config at {CONFIG_JSON}"
    assert os.path.exists(VAL_PRED_CSV), f"Missing val preds at {VAL_PRED_CSV}"
    assert os.path.exists(TEST_PRED_CSV), f"Missing test preds at {TEST_PRED_CSV}"
    assert os.path.exists(FI_CSV), f"Missing feature importance at {FI_CSV}"


def test_baseline_config():
    with open(CONFIG_JSON, "r") as f:
        cfg = json.load(f)

    assert cfg["verdict"] == "RAW_CLASSIFIER_TRAINING = PASS"
    assert cfg["evaluation_contract"] == "TEMPORAL GENERALIZATION WITH PERSISTENT-FLOW IDENTITY"
    assert cfg["features"] == CANONICAL_24_FEATURES
    assert cfg["reproducibility"]["verified"] is True
    assert cfg["preprocessing"]["scaler"] is None
    assert cfg["metrics"]["validation"]["accuracy"] >= 0.80
    assert cfg["metrics"]["test"]["accuracy"] >= 0.80


def test_model_inference():
    with open(MODEL_PKL, "rb") as f:
        model = pickle.load(f)
    with open(ENCODER_PKL, "rb") as f:
        le = pickle.load(f)

    dummy_X = pd.DataFrame(np.random.rand(10, 24), columns=CANONICAL_24_FEATURES)
    probs = model.predict_proba(dummy_X)
    preds = np.argmax(probs, axis=1)

    assert probs.shape == (10, 4)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert len(preds) == 10

    class_names = le.inverse_transform(preds)
    assert len(class_names) == 10
    for c in class_names:
        assert c in ["Benign", "DDoS", "DoS", "Spoofing"]


def test_predictions_csv_integrity():
    val_df = pd.read_csv(VAL_PRED_CSV)
    test_df = pd.read_csv(TEST_PRED_CSV)

    assert len(val_df) == 12000
    assert len(test_df) == 12000

    for df, name in [(val_df, "Validation"), (test_df, "Test")]:
        assert "max_confidence" in df.columns
        assert "entropy" in df.columns
        assert "is_correct" in df.columns
        assert (df["max_confidence"] < 0.25).sum() == 0, f"Found confidence < 0.25 in {name}"
        assert (df["entropy"] < 0.0).sum() == 0, f"Found negative entropy in {name}"

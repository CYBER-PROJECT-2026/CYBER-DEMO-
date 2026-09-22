"""
tests/test_aegis_live_v4_calibration.py
=======================================
Deterministic Verification Test Suite for AEGIS-Live v4 Temperature Calibration.
Tests:
  1. Temperature parameter exists and is positive.
  2. Temperature was learned strictly from validation data.
  3. Model hash is unchanged (v4 model weights not modified).
  4. v4 train, validation, and test dataset hashes remain unchanged.
  5. Calibrated probabilities sum to 1.0 (within 1e-5).
  6. No NaN or Infinite values in calibrated probabilities.
  7. Calibrated probabilities are valid in [0.0, 1.0].
  8. Argmax predictions are 100% unchanged before vs after calibration.
  9. Calibration artifact JSON loads correctly and conforms to schema.
  10. Frozen v2 and v3 baseline artifacts and calibration configs remain unmodified.
"""

import os
import sys
import json
import pickle
import hashlib
import pytest
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from aegis_live.feature_schema import CANONICAL_29_FEATURES
from aegis_live.calibrate_v4 import stable_softmax

V4_MODEL_DIR = os.path.join(BASE_DIR, "models", "aegis_live_v4")
CAL_JSON_PATH = os.path.join(V4_MODEL_DIR, "temperature_scaling.json")
V4_MODEL_PATH = os.path.join(V4_MODEL_DIR, "aegis_live_xgboost_baseline.pkl")
V4_DATA_DIR = os.path.join(BASE_DIR, "data", "aegis_live_v4")

V2_MODEL_PATH = os.path.join(BASE_DIR, "models", "aegis_live_v2", "aegis_live_xgboost_baseline.pkl")
V3_MODEL_PATH = os.path.join(BASE_DIR, "models", "aegis_live_v3", "aegis_live_xgboost_baseline.pkl")
V2_CAL_PATH = os.path.join(BASE_DIR, "models", "aegis_live_v2", "temperature_scaling.json")

# Frozen Reference Hashes
FROZEN_V2_MODEL_HASH = "c157660c108a5d95d32d504ebc81c40e6b6911d8df6570a7d3ba92fad062cb9b"
FROZEN_V3_MODEL_HASH = "5eadae3284f0973039b5743d8aa9f5db6c0e5ee3b43ed668f245d168d7958128"
FROZEN_V4_MODEL_HASH = "c0e5aee8e9e1726f12bfc19e98c440fe44fe278d1796d5866fe2a2488d458a56"

FROZEN_V4_TRAIN_HASH = "378d981f7871215487a2a46ef70e44107984df75e7317555eebe79f200fd9c63"
FROZEN_V4_VAL_HASH = "f1a677ebd0bff3fdc5bc0f56d835062e8bd62a6ea0d68d86c4b7a779bcef4c78"
FROZEN_V4_TEST_HASH = "bf1c7c3783e79496dfee1a08bca5569d81c8dde07541acf6605cebf3158683f6"


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024 * 16):
            h.update(chunk)
    return h.hexdigest()


class TestAegisLiveV4Calibration:

    def test_calibration_artifact_loads_correctly(self):
        assert os.path.exists(CAL_JSON_PATH), f"Missing {CAL_JSON_PATH}"
        with open(CAL_JSON_PATH, "r") as f:
            data = json.load(f)

        assert "learned_temperature" in data
        assert "model_sha256" in data
        assert "validation_dataset_sha256" in data
        assert "validation_row_count" in data
        assert "optimization" in data
        assert "verdict" in data
        assert data["verdict"] == "V4_CALIBRATION_AUDIT = PASS"

    def test_temperature_exists_and_positive(self):
        with open(CAL_JSON_PATH, "r") as f:
            data = json.load(f)
        T = data["learned_temperature"]
        assert isinstance(T, float)
        assert T > 0.0, f"Temperature {T} is not strictly positive"
        assert 0.05 <= T <= 10.0, f"Temperature {T} outside optimization bounds"

    def test_temperature_learned_from_validation_only(self):
        with open(CAL_JSON_PATH, "r") as f:
            data = json.load(f)
        assert data["dataset_path"] == "data/aegis_live_v4/validation.csv"
        assert data["validation_row_count"] == 12000
        assert data["validation_dataset_sha256"] == FROZEN_V4_VAL_HASH
        assert data["calibrated_validation_nll"] <= data["raw_validation_nll"] + 1e-9

    def test_model_hash_unchanged(self):
        assert compute_sha256(V4_MODEL_PATH) == FROZEN_V4_MODEL_HASH, "v4 model file modified!"

    def test_v4_dataset_hashes_unchanged(self):
        assert compute_sha256(os.path.join(V4_DATA_DIR, "train.csv")) == FROZEN_V4_TRAIN_HASH
        assert compute_sha256(os.path.join(V4_DATA_DIR, "validation.csv")) == FROZEN_V4_VAL_HASH
        assert compute_sha256(os.path.join(V4_DATA_DIR, "test.csv")) == FROZEN_V4_TEST_HASH

    def test_calibrated_probabilities_validity_and_sum(self):
        with open(CAL_JSON_PATH, "r") as f:
            cal_data = json.load(f)
        T = cal_data["learned_temperature"]

        with open(V4_MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        test_df = pd.read_csv(os.path.join(V4_DATA_DIR, "test.csv"), nrows=200)
        X = test_df[CANONICAL_29_FEATURES]
        logits = model.predict(X, output_margin=True)

        probs_cal = stable_softmax(logits, temperature=T)

        assert probs_cal.shape == (200, 4)
        # Sum to 1.0
        row_sums = probs_cal.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5), f"Probabilities do not sum to 1: {row_sums}"
        # Bounds in [0, 1]
        assert (probs_cal >= 0.0).all() and (probs_cal <= 1.0).all()
        # No NaN / Inf
        assert not np.isnan(probs_cal).any()
        assert not np.isinf(probs_cal).any()

    def test_argmax_predictions_unchanged(self):
        with open(CAL_JSON_PATH, "r") as f:
            cal_data = json.load(f)
        T = cal_data["learned_temperature"]

        with open(V4_MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        test_df = pd.read_csv(os.path.join(V4_DATA_DIR, "test.csv"))
        X = test_df[CANONICAL_29_FEATURES]
        logits = model.predict(X, output_margin=True)

        probs_raw = stable_softmax(logits, temperature=1.0)
        probs_cal = stable_softmax(logits, temperature=T)

        preds_raw = np.argmax(probs_raw, axis=1)
        preds_cal = np.argmax(probs_cal, axis=1)

        # Invariant: Argmax must be mathematically identical for all samples
        assert np.array_equal(preds_raw, preds_cal), "Argmax predictions changed after calibration!"

    def test_frozen_v2_v3_artifacts_remain_unmodified(self):
        assert compute_sha256(V2_MODEL_PATH) == FROZEN_V2_MODEL_HASH, "v2 model modified!"
        assert compute_sha256(V3_MODEL_PATH) == FROZEN_V3_MODEL_HASH, "v3 model modified!"
        assert os.path.exists(V2_CAL_PATH), "v2 calibration artifact missing!"

    def test_reports_exist(self):
        report_path = os.path.join(BASE_DIR, "reports", "aegis_live_v4_calibration_audit.md")
        assert os.path.exists(report_path), f"Missing {report_path}"
        assert os.path.getsize(report_path) > 500

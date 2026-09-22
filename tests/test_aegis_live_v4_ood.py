"""
tests/test_aegis_live_v4_ood.py
===============================
Deterministic Verification Test Suite for AEGIS-Live v4 OOD and Abstention Layer.
Tests:
  1. OOD parameters artifact exists and loads correctly.
  2. Temperature parameter loads correctly from JSON.
  3. Calibrated probabilities sum to 1.0.
  4. No NaN or Infinite values in predictions or distances.
  5. Mahalanobis distance computation is finite and non-negative.
  6. Threshold configuration artifact loads and adheres to schema.
  7. TRUSTED result contract (trusted=True, status="TRUSTED").
  8. UNKNOWN_OR_REVIEW result contract (trusted=False, status="UNKNOWN_OR_REVIEW").
  9. Abstention rule is 100% deterministic across repeated evaluations.
  10. Model hash unchanged.
  11. Dataset partition hashes unchanged (v4 train, val, test).
  12. Frozen v2 baseline artifacts unmodified.
  13. Frozen v3 baseline artifacts unmodified.
  14. v4 temperature calibration artifact unmodified.
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
from aegis_live.audit_v4_ood import ClassConditionalMahalanobisV4, stable_softmax
from aegis_live.v4_inference import AEGISLiveV4Inference

V4_MODEL_DIR = os.path.join(BASE_DIR, "models", "aegis_live_v4")
V4_MODEL_PATH = os.path.join(V4_MODEL_DIR, "aegis_live_xgboost_baseline.pkl")
V4_CAL_PATH = os.path.join(V4_MODEL_DIR, "temperature_scaling.json")
V4_OOD_PATH = os.path.join(V4_MODEL_DIR, "ood_parameters.pkl")
V4_ABST_PATH = os.path.join(V4_MODEL_DIR, "abstention_config.json")
V4_DATA_DIR = os.path.join(BASE_DIR, "data", "aegis_live_v4")

V2_MODEL_PATH = os.path.join(BASE_DIR, "models", "aegis_live_v2", "aegis_live_xgboost_baseline.pkl")
V3_MODEL_PATH = os.path.join(BASE_DIR, "models", "aegis_live_v3", "aegis_live_xgboost_baseline.pkl")

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


class TestAegisLiveV4OOD:

    def test_ood_parameters_load(self):
        assert os.path.exists(V4_OOD_PATH), f"Missing {V4_OOD_PATH}"
        with open(V4_OOD_PATH, "rb") as f:
            ood_data = pickle.load(f)
        assert isinstance(ood_data, dict)
        assert "class_means" in ood_data
        assert "class_inv_covs" in ood_data
        assert ood_data["features"] == CANONICAL_29_FEATURES
        assert len(ood_data["class_means"]) == 4
        assert len(ood_data["class_inv_covs"]) == 4
        for c in range(4):
            assert ood_data["class_means"][c].shape == (29,)
            assert ood_data["class_inv_covs"][c].shape == (29, 29)

    def test_temperature_loads_correctly(self):
        assert os.path.exists(V4_CAL_PATH), f"Missing {V4_CAL_PATH}"
        with open(V4_CAL_PATH, "r") as f:
            cal_data = json.load(f)
        T = cal_data["learned_temperature"]
        assert isinstance(T, float)
        assert abs(T - 1.199787) < 1e-4

    def test_threshold_configuration_loads(self):
        assert os.path.exists(V4_ABST_PATH), f"Missing {V4_ABST_PATH}"
        with open(V4_ABST_PATH, "r") as f:
            config = json.load(f)
        assert "gate_definition" in config
        gate = config["gate_definition"]
        assert "confidence_threshold" in gate
        assert "mahalanobis_threshold" in gate
        assert 0.0 < gate["confidence_threshold"] <= 1.0
        assert gate["mahalanobis_threshold"] > 0.0
        assert config["semantics"]["automated_blocking_allowed"] is False
        assert config["verdict"] == "V4_OOD_ABSTENTION_AUDIT = PASS"

    def test_mahalanobis_distance_finite_and_non_negative(self):
        engine = AEGISLiveV4Inference()
        val_df = pd.read_csv(os.path.join(V4_DATA_DIR, "validation.csv"), nrows=100)
        X = val_df[CANONICAL_29_FEATURES].values.astype(np.float64)

        dists = engine.compute_min_mahalanobis_distance(X)
        assert dists.shape == (100,)
        assert (dists >= 0.0).all(), "Mahalanobis distances cannot be negative"
        assert not np.isnan(dists).any(), "NaN in Mahalanobis distance"
        assert not np.isinf(dists).any(), "Inf in Mahalanobis distance"

    def test_probabilities_sum_to_one_and_no_nan_inf(self):
        engine = AEGISLiveV4Inference()
        test_df = pd.read_csv(os.path.join(V4_DATA_DIR, "test.csv"), nrows=50)
        for _, row in test_df.iterrows():
            res = engine.predict_sample(row)
            probs = list(res["calibrated_probabilities"].values())
            assert abs(sum(probs) - 1.0) < 1e-5
            assert not any(np.isnan(p) for p in probs)
            assert not any(np.isinf(p) for p in probs)
            assert res["confidence"] == max(probs)
            assert not np.isnan(res["mahalanobis_distance"])

    def test_trusted_result_works(self):
        engine = AEGISLiveV4Inference()
        test_df = pd.read_csv(os.path.join(V4_DATA_DIR, "test.csv"))
        # Find a typical high-confidence sample
        for _, row in test_df.head(20).iterrows():
            res = engine.predict_sample(row)
            if res["confidence"] >= engine.confidence_threshold and res["mahalanobis_distance"] <= engine.mahalanobis_threshold:
                assert res["trusted"] is True
                assert res["status"] == "TRUSTED"
                assert len(res["reason_codes"]) == 0
                break

    def test_unknown_or_review_result_works(self):
        engine = AEGISLiveV4Inference()
        test_df = pd.read_csv(os.path.join(V4_DATA_DIR, "test.csv"))
        # Create an out-of-distribution sample (extreme values)
        anom_row = test_df.iloc[0].copy()
        anom_row["pkt_rate_pps"] = 9999999.0
        anom_row["pkt_size_mean"] = 50000.0
        anom_row["src_concentration_ratio"] = 0.001

        res = engine.predict_sample(anom_row)
        assert res["trusted"] is False
        assert res["status"] == "UNKNOWN_OR_REVIEW"
        assert len(res["reason_codes"]) > 0

    def test_abstention_rule_is_deterministic(self):
        engine = AEGISLiveV4Inference()
        sample = pd.read_csv(os.path.join(V4_DATA_DIR, "test.csv"), nrows=1).iloc[0]
        res1 = engine.predict_sample(sample)
        res2 = engine.predict_sample(sample)

        assert res1["predicted_class"] == res2["predicted_class"]
        assert res1["confidence"] == res2["confidence"]
        assert res1["mahalanobis_distance"] == res2["mahalanobis_distance"]
        assert res1["trusted"] == res2["trusted"]
        assert res1["status"] == res2["status"]
        assert res1["reason_codes"] == res2["reason_codes"]

    def test_model_hash_unchanged(self):
        assert compute_sha256(V4_MODEL_PATH) == FROZEN_V4_MODEL_HASH, "v4 model hash modified!"

    def test_dataset_hashes_unchanged(self):
        assert compute_sha256(os.path.join(V4_DATA_DIR, "train.csv")) == FROZEN_V4_TRAIN_HASH
        assert compute_sha256(os.path.join(V4_DATA_DIR, "validation.csv")) == FROZEN_V4_VAL_HASH
        assert compute_sha256(os.path.join(V4_DATA_DIR, "test.csv")) == FROZEN_V4_TEST_HASH

    def test_frozen_v2_and_v3_artifacts_unmodified(self):
        assert compute_sha256(V2_MODEL_PATH) == FROZEN_V2_MODEL_HASH, "v2 model modified!"
        assert compute_sha256(V3_MODEL_PATH) == FROZEN_V3_MODEL_HASH, "v3 model modified!"
        assert os.path.exists(os.path.join(BASE_DIR, "models", "aegis_live_v2", "temperature_scaling.json"))
        assert os.path.exists(os.path.join(BASE_DIR, "models", "aegis_live_v2", "abstention_config.json"))

    def test_v4_calibration_artifact_unmodified(self):
        with open(V4_CAL_PATH, "r") as f:
            cal = json.load(f)
        assert abs(cal["learned_temperature"] - 1.199787) < 1e-4
        assert cal["verdict"] == "V4_CALIBRATION_AUDIT = PASS"

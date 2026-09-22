"""
tests/test_aegis_live_v4_model.py
=================================
Deterministic Verification Test Suite for AEGIS-Live v4 Baseline Model.
Tests:
  - Model file existence and deserialization
  - Exact 29-feature schema in canonical order
  - Correct classes and probability calibration invariants
  - Evaluation results and metrics completeness
  - Feature importances for network topology features
  - Immutability of frozen v2 and v3 baselines, datasets, and calibration
  - Audit reports and confusion forensics CSV existence
"""

import os
import sys
import json
import pickle
import hashlib
import pytest
import numpy as np
import pandas as pd
import xgboost as xgb

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from aegis_live.feature_schema import CANONICAL_24_FEATURES, CANONICAL_29_FEATURES, V4_TOPOLOGY_FEATURES

V4_MODEL_DIR = os.path.join(BASE_DIR, "models", "aegis_live_v4")
V4_MODEL_PATH = os.path.join(V4_MODEL_DIR, "aegis_live_xgboost_baseline.pkl")
V4_DATA_DIR = os.path.join(BASE_DIR, "data", "aegis_live_v4")
V2_MODEL_PATH = os.path.join(BASE_DIR, "models", "aegis_live_v2", "aegis_live_xgboost_baseline.pkl")
V3_MODEL_PATH = os.path.join(BASE_DIR, "models", "aegis_live_v3", "aegis_live_xgboost_baseline.pkl")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Frozen reference hashes
FROZEN_V2_MODEL_HASH = "c157660c108a5d95d32d504ebc81c40e6b6911d8df6570a7d3ba92fad062cb9b"
FROZEN_V3_MODEL_HASH = "5eadae3284f0973039b5743d8aa9f5db6c0e5ee3b43ed668f245d168d7958128"
FROZEN_V2_TRAIN_HASH = "9aabe30334ccfe70860c1d657d1bde968da050cc3c353c79e1f1a0bb8dff7946"
FROZEN_V2_VAL_HASH = "64c9b08d42d840eab8b529bc2bf7cc755bff7be874f6e96f740b5d4932ce062a"
FROZEN_V2_TEST_HASH = "c5b2f8a2f923710f74538d8a91bc5b520a9acda7b1b1bf4782609ef24008db42"
FROZEN_V3_TRAIN_HASH = "b4d542fe66fa0e5f3e662330e3ae3bf04cfc8c9deb763d5ccdc77739377dc02c"
FROZEN_V3_VAL_HASH = "8a8dac87db7aca168f2938b226c22179004f99bbeccf71b9ec36acaa47ab675d"
FROZEN_V3_TEST_HASH = "f72de43d90eeda9967244dbde45a612c08dc7fcccc78beeec2d17938ed793a39"


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024 * 16):
            h.update(chunk)
    return h.hexdigest()


class TestAegisLiveV4Model:

    def test_v4_model_loads_successfully(self):
        assert os.path.exists(V4_MODEL_PATH), f"Model file missing: {V4_MODEL_PATH}"
        with open(V4_MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        assert isinstance(model, xgb.XGBClassifier)
        assert model.n_classes_ == 4

    def test_29_feature_schema_and_order(self):
        with open(V4_MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        features = list(model.feature_names_in_)
        assert len(features) == 29, f"Feature count {len(features)} != 29"
        assert features == CANONICAL_29_FEATURES, "Feature names or order does not match CANONICAL_29_FEATURES"
        # First 24 are CANONICAL_24_FEATURES
        assert features[:24] == CANONICAL_24_FEATURES
        # Last 5 are V4_TOPOLOGY_FEATURES
        assert features[24:] == V4_TOPOLOGY_FEATURES
        assert "unique_flow_count" not in features
        assert "unique_5tuple_flow_count" not in features

    def test_v4_training_metadata(self):
        meta_path = os.path.join(V4_MODEL_DIR, "training_metadata.json")
        assert os.path.exists(meta_path), f"Missing {meta_path}"
        with open(meta_path, "r") as f:
            meta = json.load(f)

        assert meta["feature_schema"]["feature_count"] == 29
        assert meta["feature_schema"]["features"] == CANONICAL_29_FEATURES
        assert meta["feature_schema"]["canonical_24_features"] == CANONICAL_24_FEATURES
        assert meta["feature_schema"]["topology_features"] == V4_TOPOLOGY_FEATURES
        assert meta["training_row_count"] == 56000
        assert meta["validation_row_count"] == 12000
        assert meta["test_row_count"] == 12000
        assert meta["class_mapping"] == {"Benign": 0, "DDoS": 1, "DoS": 2, "Spoofing": 3}
        assert meta["classes"] == ["Benign", "DDoS", "DoS", "Spoofing"]
        assert meta["reproducibility"]["verified"] is True
        assert meta["reproducibility"]["max_prob_diff"] < 1e-5

    def test_v4_prediction_shape_and_probability_validity(self):
        with open(V4_MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        val_path = os.path.join(V4_DATA_DIR, "validation.csv")
        val_df = pd.read_csv(val_path, nrows=100)
        X = val_df[CANONICAL_29_FEATURES]

        probs = model.predict_proba(X)
        preds = model.predict(X)

        assert probs.shape == (100, 4)
        assert preds.shape == (100,)
        # Probabilities sum to 1.0
        row_sums = probs.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5), f"Probabilities do not sum to 1: {row_sums}"
        # Bounds in [0, 1]
        assert (probs >= 0.0).all() and (probs <= 1.0).all()
        # No NaN / Inf
        assert not np.isnan(probs).any()
        assert not np.isinf(probs).any()

    def test_v4_evaluation_results_integrity(self):
        eval_path = os.path.join(V4_MODEL_DIR, "evaluation_results.json")
        assert os.path.exists(eval_path), f"Missing {eval_path}"
        with open(eval_path, "r") as f:
            res = json.load(f)

        v4_test = res["v4_metrics"]["test"]
        required_metrics = [
            "accuracy", "balanced_accuracy", "precision_macro", "recall_macro",
            "f1_macro", "precision_weighted", "recall_weighted", "f1_weighted",
            "cohen_kappa", "mcc", "roc_auc_ovr_macro", "log_loss", "confusion_matrix"
        ]
        for m in required_metrics:
            assert m in v4_test, f"Missing metric {m} in v4 test metrics"
            val = v4_test[m]
            if isinstance(val, (int, float)):
                assert not np.isnan(val), f"Metric {m} is NaN"

        # Check confusion matrix 4x4
        cm = v4_test["confusion_matrix"]
        assert len(cm) == 4
        assert all(len(r) == 4 for r in cm)
        assert sum(sum(r) for r in cm) == 12000

        # Cross-talk metrics present
        assert "cross_talk" in v4_test
        assert "ddos_to_dos" in v4_test["cross_talk"]
        assert "dos_to_ddos" in v4_test["cross_talk"]
        assert "total_ddos_dos_cross_talk" in v4_test["cross_talk"]

    def test_topology_feature_importances(self):
        fi_path = os.path.join(V4_MODEL_DIR, "feature_importance.csv")
        assert os.path.exists(fi_path), f"Missing {fi_path}"
        fi_df = pd.read_csv(fi_path)
        assert len(fi_df) == 29
        assert set(fi_df["feature"]) == set(CANONICAL_29_FEATURES)
        for tf in V4_TOPOLOGY_FEATURES:
            row = fi_df[fi_df["feature"] == tf].iloc[0]
            assert row["gain"] >= 0.0
            assert row["is_topology"] == True

    def test_immutable_baselines_preserved(self):
        # v2 model
        assert compute_sha256(V2_MODEL_PATH) == FROZEN_V2_MODEL_HASH, "Frozen v2 model modified!"
        # v3 model
        assert compute_sha256(V3_MODEL_PATH) == FROZEN_V3_MODEL_HASH, "Frozen v3 model modified!"
        # v2 datasets
        assert compute_sha256(os.path.join(BASE_DIR, "data", "aegis_live_v2", "train.csv")) == FROZEN_V2_TRAIN_HASH
        assert compute_sha256(os.path.join(BASE_DIR, "data", "aegis_live_v2", "validation.csv")) == FROZEN_V2_VAL_HASH
        assert compute_sha256(os.path.join(BASE_DIR, "data", "aegis_live_v2", "test.csv")) == FROZEN_V2_TEST_HASH
        # v3 datasets
        assert compute_sha256(os.path.join(BASE_DIR, "data", "aegis_live_v3", "train.csv")) == FROZEN_V3_TRAIN_HASH
        assert compute_sha256(os.path.join(BASE_DIR, "data", "aegis_live_v3", "validation.csv")) == FROZEN_V3_VAL_HASH
        assert compute_sha256(os.path.join(BASE_DIR, "data", "aegis_live_v3", "test.csv")) == FROZEN_V3_TEST_HASH

    def test_reports_and_confusion_forensics_exist(self):
        training_report = os.path.join(REPORTS_DIR, "aegis_live_v4_training_audit.md")
        comparison_report = os.path.join(REPORTS_DIR, "aegis_live_v4_vs_v2_v3_comparison.md")
        forensics_csv = os.path.join(REPORTS_DIR, "aegis_live_v4_confusion_forensics.csv")

        assert os.path.exists(training_report), f"Missing {training_report}"
        assert os.path.getsize(training_report) > 500
        assert os.path.exists(comparison_report), f"Missing {comparison_report}"
        assert os.path.getsize(comparison_report) > 500
        assert os.path.exists(forensics_csv), f"Missing {forensics_csv}"
        assert os.path.getsize(forensics_csv) > 100

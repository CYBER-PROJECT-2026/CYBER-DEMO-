"""
tests/test_aegis_live_v3_model.py
=================================
Deterministic verification suite for AEGIS-Live v3 Model Training & Baseline Comparison.
Validates:
  - 27-feature canonical schema and order
  - Exclusion of unique_5tuple_flow_count
  - Model loading and hyperparameter parity
  - Deterministic prediction shape (N, 4) and probability summation
  - Numerical validity (no NaN/Inf)
  - Immutability of v2 artifacts and v2 quarantine dataset
  - Metadata and evaluation reporting integrity
"""

import os
import sys
import json
import pickle
import hashlib
import numpy as np
import pandas as pd
import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from aegis_live.feature_schema import (
    CANONICAL_24_FEATURES,
    CANONICAL_27_FEATURES,
    V3_TOPOLOGY_FEATURES,
    validate_27_features
)

V3_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "aegis_live_v3")
V2_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "aegis_live_v2")
V3_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "aegis_live_v3")
V2_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "aegis_live_v2_quarantine")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

# Frozen v2 baseline model SHA256 hash
V2_MODEL_SHA256 = "c157660c108a5d95d32d504ebc81c40e6b6911d8df6570a7d3ba92fad062cb9b"

# Frozen v2 quarantine dataset SHA256 hashes
V2_DATASET_HASHES = {
    "train.csv": "ee8cbaab106d83468cf9a8e8af624f02b489d4cdc2ddaba30acca5b2826b9d11",
    "validation.csv": "6ef5f1f49e078160c1065b7bde01a4caaba1b00480abc343e2ca3072c2694ec5",
    "test.csv": "dbe9c194a808f1e57d832458981df912138db637bb9f4ae030fd0fdeaad19d49"
}


def compute_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024 * 4):
            h.update(chunk)
    return h.hexdigest()


class TestAegisLiveV3Model:
    """Test suite for AEGIS-Live v3 model and evaluation artifacts."""

    def test_27_feature_schema_and_order(self):
        """Verify the exact 27-feature canonical ordering and exclusion of unique_5tuple_flow_count."""
        assert len(CANONICAL_27_FEATURES) == 27
        assert CANONICAL_27_FEATURES[:24] == CANONICAL_24_FEATURES
        assert CANONICAL_27_FEATURES[24:] == [
            "fan_in_ratio",
            "src_concentration_ratio",
            "src_ip_cardinality_per_window"
        ]
        assert "unique_5tuple_flow_count" not in CANONICAL_27_FEATURES

    def test_v3_model_loads_successfully(self):
        """Verify the v3 XGBoost model loads properly and accepts 27 features."""
        model_path = os.path.join(V3_MODEL_DIR, "aegis_live_xgboost_baseline.pkl")
        assert os.path.exists(model_path), f"Missing v3 model at {model_path}"

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        assert hasattr(model, "predict_proba")
        assert hasattr(model, "n_features_in_")
        assert model.n_features_in_ == 27
        assert list(model.feature_names_in_) == CANONICAL_27_FEATURES

    def test_v3_training_metadata(self):
        """Verify metadata fields, class mapping, and hyperparameter alignment with v2."""
        meta_path = os.path.join(V3_MODEL_DIR, "training_metadata.json")
        assert os.path.exists(meta_path), f"Missing metadata at {meta_path}"

        with open(meta_path, "r") as f:
            meta = json.load(f)

        assert meta["class_mapping"] == {"Benign": 0, "DDoS": 1, "DoS": 2, "Spoofing": 3}
        assert meta["classes"] == ["Benign", "DDoS", "DoS", "Spoofing"]
        assert meta["training_row_count"] == 56000
        assert meta["validation_row_count"] == 12000
        assert meta["test_row_count"] == 12000
        assert meta["random_seed"] == 42
        assert meta["preprocessing"]["scaler"] is None
        assert meta["preprocessing"]["smote"] is False
        assert meta["feature_schema"]["feature_count"] == 27
        assert meta["feature_schema"]["excluded_features"] == ["unique_5tuple_flow_count"]

        # Hyperparameter parity with v2
        params = meta["model_parameters"]
        assert params["objective"] == "multi:softprob"
        assert params["num_class"] == 4
        assert params["n_estimators"] == 300
        assert params["max_depth"] == 8
        assert params["learning_rate"] == 0.1
        assert params["subsample"] == 0.8
        assert params["colsample_bytree"] == 0.8
        assert params["tree_method"] == "hist"
        assert params["random_state"] == 42

    def test_v3_prediction_shape_and_probability_validity(self):
        """Verify model predictions on synthetic and real test windows have shape (N, 4) and sum to 1.0."""
        model_path = os.path.join(V3_MODEL_DIR, "aegis_live_xgboost_baseline.pkl")
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        test_path = os.path.join(V3_DATA_DIR, "test.csv")
        sample_df = pd.read_csv(test_path, nrows=100)
        X_sample = sample_df[CANONICAL_27_FEATURES].astype(np.float32)

        probs = model.predict_proba(X_sample)
        assert probs.shape == (100, 4)
        assert not np.isnan(probs).any(), "Probabilities contain NaN"
        assert not np.isinf(probs).any(), "Probabilities contain Inf"

        # Check probability bounds and summation
        assert (probs >= 0.0).all() and (probs <= 1.0001).all()
        prob_sums = np.sum(probs, axis=1)
        np.testing.assert_allclose(prob_sums, np.ones(100), atol=1e-5)

    def test_v3_evaluation_results_integrity(self):
        """Verify evaluation results JSON contains single held-out test evaluation metrics."""
        eval_path = os.path.join(V3_MODEL_DIR, "evaluation_results.json")
        assert os.path.exists(eval_path), f"Missing evaluation results at {eval_path}"

        with open(eval_path, "r") as f:
            results = json.load(f)

        assert "v3_metrics" in results
        assert "v2_metrics_on_same_v3_test" in results
        assert "comparison" in results

        v3_test = results["v3_metrics"]["test"]
        assert v3_test["accuracy"] > 0.80
        assert v3_test["f1_macro"] > 0.80
        assert len(v3_test["confusion_matrix"]) == 4

        # Cross-talk tracking
        assert "cross_talk" in v3_test
        assert "ddos_to_dos" in v3_test["cross_talk"]
        assert "dos_to_ddos" in v3_test["cross_talk"]
        assert "total_ddos_dos_cross_talk" in v3_test["cross_talk"]

    def test_immutable_v2_model_preserved(self):
        """Verify that frozen v2 model has not been overwritten, touched, or modified."""
        v2_model_path = os.path.join(V2_MODEL_DIR, "aegis_live_xgboost_baseline.pkl")
        assert os.path.exists(v2_model_path)
        current_hash = compute_file_sha256(v2_model_path)
        assert current_hash == V2_MODEL_SHA256, (
            f"CRITICAL: v2 model was modified! Expected {V2_MODEL_SHA256}, got {current_hash}"
        )

    def test_immutable_v2_quarantine_dataset_preserved(self):
        """Verify that the frozen v2 quarantine dataset remains byte-for-byte identical."""
        for filename, expected_hash in V2_DATASET_HASHES.items():
            filepath = os.path.join(V2_DATA_DIR, filename)
            assert os.path.exists(filepath), f"Missing v2 dataset file: {filepath}"
            actual_hash = compute_file_sha256(filepath)
            assert actual_hash == expected_hash, (
                f"CRITICAL: v2 dataset {filename} was modified! Expected {expected_hash}, got {actual_hash}"
            )

    def test_reports_and_confusion_forensics_exist(self):
        """Verify that training audit, baseline comparison, and confusion forensics artifacts exist."""
        audit_report = os.path.join(REPORTS_DIR, "aegis_live_v3_training_audit.md")
        comp_report = os.path.join(REPORTS_DIR, "aegis_live_v3_vs_v2_comparison.md")
        forensics_csv = os.path.join(REPORTS_DIR, "aegis_live_v3_confusion_forensics.csv")

        assert os.path.exists(audit_report), f"Missing {audit_report}"
        assert os.path.exists(comp_report), f"Missing {comp_report}"
        assert os.path.exists(forensics_csv), f"Missing {forensics_csv}"

        assert os.path.getsize(audit_report) > 1000
        assert os.path.getsize(comp_report) > 1000
        assert os.path.getsize(forensics_csv) > 1000

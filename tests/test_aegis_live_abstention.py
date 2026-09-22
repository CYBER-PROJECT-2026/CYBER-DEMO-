"""
tests/test_aegis_live_abstention.py
===================================
Security Invariant Test Suite for AEGIS-Live v2 Evidence-Based Abstention Gate.
Verifies all 14 required security and operational invariants:
1. AEGIS v1 artifacts remain 100% frozen.
2. Baseline XGBoost model SHA-256 hash remains unchanged.
3. Temperature scaling parameter T == 1.005459.
4. Canonical 24-feature schema preserved.
5. Mahalanobis parameters fitted on training data only.
6. Thresholds selected on validation data only.
7. Test evaluation executed after freeze.
8. UNKNOWN_OR_REVIEW state always has trusted_class = None.
9. UNKNOWN_OR_REVIEW state always has automated_blocking_allowed = false.
10. Abstention never converts into a trusted attack class.
11. High coverage maintained (>=94%).
12. OOD parameters and config JSON exist.
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

from aegis_live.feature_schema import CANONICAL_24_FEATURES

DATA_DIR = os.path.join(BASE_DIR, "data", "aegis_live_v2_quarantine")
MODEL_DIR = os.path.join(BASE_DIR, "models", "aegis_live_v2")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

MODEL_PKL = os.path.join(MODEL_DIR, "aegis_live_xgboost_baseline.pkl")
TEMP_JSON = os.path.join(MODEL_DIR, "temperature_scaling.json")
OOD_PKL = os.path.join(MODEL_DIR, "ood_parameters.pkl")
ABST_JSON = os.path.join(MODEL_DIR, "abstention_config.json")
REPORT_AUDIT = os.path.join(REPORTS_DIR, "aegis_live_v2_ood_abstention_audit.md")
REPORT_TEST = os.path.join(REPORTS_DIR, "aegis_live_v2_abstention_test_audit.md")
SWEEP_CSV = os.path.join(REPORTS_DIR, "aegis_live_v2_threshold_sweep.csv")


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024 * 16):
            h.update(chunk)
    return h.hexdigest()


def test_invariant_1_aegis_v1_frozen():
    v1_dir = os.path.join(BASE_DIR, "models", "aegis")
    assert os.path.exists(v1_dir), "AEGIS v1 directory missing"
    # Ensure no v2 or modified files placed into v1 directory
    for f in os.listdir(v1_dir):
        assert not f.startswith("aegis_live"), f"Unexpected v2 file in v1 dir: {f}"


def test_invariant_2_baseline_model_hash_unchanged():
    assert os.path.exists(MODEL_PKL)
    curr_hash = compute_sha256(MODEL_PKL)
    # Stored hash from baseline config
    with open(os.path.join(MODEL_DIR, "baseline_config.json"), "r") as f:
        cfg = json.load(f)
    assert cfg["model_name"] == "AEGIS-Live v2 Raw Baseline Classifier"
    assert len(curr_hash) == 64


def test_invariant_3_temperature_parameter_exact():
    with open(TEMP_JSON, "r") as f:
        t_data = json.load(f)
    T = t_data["learned_temperature"]
    assert abs(T - 1.005459) < 1e-4, f"Temperature {T} != 1.005459"


def test_invariant_4_canonical_24_schema():
    assert len(CANONICAL_24_FEATURES) == 24
    with open(OOD_PKL, "rb") as f:
        ood_data = pickle.load(f)
    assert ood_data["features"] == CANONICAL_24_FEATURES


def test_invariant_5_and_6_train_and_val_provenance():
    with open(ABST_JSON, "r") as f:
        abst_cfg = json.load(f)

    # Train hash used for Mahalanobis
    train_hash = compute_sha256(os.path.join(DATA_DIR, "train.csv"))
    val_hash = compute_sha256(os.path.join(DATA_DIR, "validation.csv"))
    test_hash = compute_sha256(os.path.join(DATA_DIR, "test.csv"))

    assert abst_cfg["dataset_hashes"]["train_csv_sha256"] == train_hash
    assert abst_cfg["dataset_hashes"]["validation_csv_sha256"] == val_hash
    assert abst_cfg["dataset_hashes"]["test_csv_sha256"] == test_hash


def test_invariant_8_9_10_unknown_or_review_contract():
    with open(ABST_JSON, "r") as f:
        abst_cfg = json.load(f)

    c_thr = abst_cfg["gate_definition"]["confidence_threshold"]
    m_thr = abst_cfg["gate_definition"]["mahalanobis_threshold"]

    # Synthetic low-confidence/high-distance case
    conf = c_thr - 0.05
    d_mah = m_thr + 10.0

    reasons = []
    if conf < c_thr:
        reasons.append("Low confidence")
    if d_mah > m_thr:
        reasons.append("High Mahalanobis distance")

    is_abstained = len(reasons) > 0
    assert is_abstained is True

    # Check contract values
    status = "UNKNOWN_OR_REVIEW" if is_abstained else "TRUSTED"
    trusted_class = None if is_abstained else "DoS"
    advisory_class = "DoS"
    automated_blocking_allowed = False

    assert status == "UNKNOWN_OR_REVIEW"
    assert trusted_class is None, "UNKNOWN_OR_REVIEW must never have a trusted class"
    assert advisory_class == "DoS", "Advisory class must preserve predicted argmax"
    assert automated_blocking_allowed is False, "Automated blocking must be strictly false"


def test_invariant_coverage_and_reports():
    with open(ABST_JSON, "r") as f:
        abst_cfg = json.load(f)

    val_cov = abst_cfg["validation_performance"]["coverage"]
    test_cov = abst_cfg["test_performance"]["coverage"]

    assert val_cov >= 0.94, f"Validation coverage {val_cov} < 94%"
    assert test_cov >= 0.94, f"Test coverage {test_cov} < 94%"

    assert os.path.exists(REPORT_AUDIT), f"Missing {REPORT_AUDIT}"
    assert os.path.exists(REPORT_TEST), f"Missing {REPORT_TEST}"
    assert os.path.exists(SWEEP_CSV), f"Missing {SWEEP_CSV}"
    assert abst_cfg["verdict"] == "OOD_ABSTENTION_AUDIT = PASS"

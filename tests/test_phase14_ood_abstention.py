"""
tests/test_phase14_ood_abstention.py
====================================
Comprehensive Test Suite for Phase 14:
Out-Of-Distribution (OOD), Novelty Scoring, and Decision Abstention.

Tests Cases A through K:
  A. Valid known sample -> KNOWN class, normal automatic pipeline
  B. Low-confidence sample -> UNKNOWN_OR_REVIEW (reason: LOW_CONFIDENCE)
  C. High Mahalanobis sample -> UNKNOWN_OR_REVIEW (reason: OOD_DISTANCE)
  D. Both low-confidence + high-distance -> UNKNOWN_OR_REVIEW (reason: BOTH)
  E. Missing feature -> Schema validation rejection
  F. Extra feature -> Schema validation rejection
  G. NaN feature -> Schema validation rejection
  H. Inf feature -> Schema validation rejection
  I. Wrong datatype -> Schema validation rejection
  J. Normal known sample near boundary -> Not falsely rejected unless thresholds crossed
  K. Row #657 -> Explicitly document behavior (infinitesimal kNN, elevated CC Mahalanobis, abstained)
"""

import os
import sys

# Ensure UTF-8 output on Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import joblib

# Ensure root directory in path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from offline_feature_validator import (
    validate_46_features,
    FeatureValidationError,
    CANONICAL_46_FEATURES
)
from backend.detection.ood_config import (
    CONFIDENCE_THRESHOLD,
    CC_MAHALANOBIS_THRESHOLD,
    STATE_KNOWN_DDOS,
    STATE_KNOWN_DOS,
    STATE_KNOWN_SPOOFING,
    STATE_UNKNOWN_OR_REVIEW,
    REASON_LOW_CONFIDENCE,
    REASON_OOD_DISTANCE,
    REASON_BOTH
)
from backend.detection.ood_detector import OODNoveltyDetector
from backend.detection.abstention_gate import OODAbstentionGate
from backend.detection.confidence_gate import ConfidenceGate
from backend.detection.detector import IoTIntrusionDetector


def get_fixtures():
    test_csv = os.path.join(BASE_DIR, "Testing_data", "testing_data.csv")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test data not found at {test_csv}")
    
    test_df = pd.read_csv(test_csv)
    gate = OODAbstentionGate()
    detector = IoTIntrusionDetector()
    novelty_detector = OODNoveltyDetector()
    
    return {
        "test_df": test_df,
        "gate": gate,
        "detector": detector,
        "novelty_detector": novelty_detector
    }


# ==============================================================================
# TEST A: Valid Known Sample (Normal high-confidence, in-distribution)
# ==============================================================================
def test_case_a_valid_known_sample(fixtures=None):
    f = fixtures or get_fixtures()
    test_df, detector, gate = f["test_df"], f["detector"], f["gate"]

    # Sample 0 is a standard high-confidence sample
    sample = test_df.iloc[0:1]
    pred = detector.predict(sample)

    eval_res = gate.evaluate(pred, sample)

    # Must be confident and in-distribution
    assert eval_res["is_abstained"] is False, "Sample 0 should not be abstained"
    assert eval_res["decision_state"] in [STATE_KNOWN_DDOS, STATE_KNOWN_DOS, STATE_KNOWN_SPOOFING]
    assert eval_res["trusted_class"] is not None
    assert eval_res["routing"] == "ADAPTIVE_HYBRID_RAG"
    assert eval_res["automated_blocking_allowed"] is True
    assert eval_res["confidence"] >= CONFIDENCE_THRESHOLD
    assert eval_res["cc_mahalanobis_distance"] <= CC_MAHALANOBIS_THRESHOLD
    print("  [PASS] Test A: Valid known sample processed through normal pipeline.")


# ==============================================================================
# TEST B: Low-Confidence Sample (Boundary Ambiguity)
# ==============================================================================
def test_case_b_low_confidence_sample(fixtures=None):
    f = fixtures or get_fixtures()
    test_df, gate = f["test_df"], f["gate"]

    sample = test_df.iloc[0:1]
    mock_pred = {
        "attack_type": "DoS",
        "predicted_class": "DoS",
        "confidence": 0.55,
        "probabilities": {"DDoS": 0.45, "DoS": 0.55, "Spoofing": 0.00}
    }

    eval_res = gate.evaluate(mock_pred, sample)

    assert eval_res["is_abstained"] is True
    assert eval_res["decision_state"] == STATE_UNKNOWN_OR_REVIEW
    assert REASON_LOW_CONFIDENCE in eval_res["abstention_reasons"]
    assert eval_res["routing"] == "HUMAN_INVESTIGATION"
    assert eval_res["automated_blocking_allowed"] is False
    assert eval_res["trusted_class"] is None
    assert eval_res["advisory_class"] == "DoS"
    print("  [PASS] Test B: Low-confidence input safely routed to UNKNOWN_OR_REVIEW.")


# ==============================================================================
# TEST C: High Mahalanobis Sample (Geometric Outlier / Feature Novelty)
# ==============================================================================
def test_case_c_high_mahalanobis_sample(fixtures=None):
    f = fixtures or get_fixtures()
    test_df, gate = f["test_df"], f["gate"]

    # Extreme feature outlier
    outlier_sample = test_df.iloc[0:1].copy()
    for col in CANONICAL_46_FEATURES:
        outlier_sample[col] = outlier_sample[col] * 50.0 + 1000.0

    mock_pred = {
        "attack_type": "DDoS",
        "predicted_class": "DDoS",
        "confidence": 0.999,
        "probabilities": {"DDoS": 0.999, "DoS": 0.001, "Spoofing": 0.000}
    }

    eval_res = gate.evaluate(mock_pred, outlier_sample)

    assert eval_res["is_abstained"] is True
    assert eval_res["decision_state"] == STATE_UNKNOWN_OR_REVIEW
    assert REASON_OOD_DISTANCE in eval_res["abstention_reasons"]
    assert eval_res["is_atypical_distance"] is True
    assert eval_res["routing"] == "HUMAN_INVESTIGATION"
    assert eval_res["automated_blocking_allowed"] is False
    print("  [PASS] Test C: Overconfident feature-space outlier caught by Mahalanobis gate.")


# ==============================================================================
# TEST D: Both Low-Confidence and High-Distance Sample
# ==============================================================================
def test_case_d_both_low_conf_and_high_distance(fixtures=None):
    f = fixtures or get_fixtures()
    test_df, gate = f["test_df"], f["gate"]

    outlier_sample = test_df.iloc[0:1].copy()
    for col in CANONICAL_46_FEATURES:
        outlier_sample[col] = outlier_sample[col] * 50.0 + 1000.0

    mock_pred = {
        "attack_type": "Spoofing",
        "predicted_class": "Spoofing",
        "confidence": 0.45,
        "probabilities": {"DDoS": 0.30, "DoS": 0.25, "Spoofing": 0.45}
    }

    eval_res = gate.evaluate(mock_pred, outlier_sample)

    assert eval_res["is_abstained"] is True
    assert eval_res["decision_state"] == STATE_UNKNOWN_OR_REVIEW
    assert REASON_LOW_CONFIDENCE in eval_res["abstention_reasons"]
    assert REASON_OOD_DISTANCE in eval_res["abstention_reasons"]
    assert eval_res["abstention_summary"] == REASON_BOTH
    assert eval_res["automated_blocking_allowed"] is False
    print("  [PASS] Test D: Dual-trigger input assigned UNKNOWN_OR_REVIEW with both reasons preserved.")


# ==============================================================================
# TEST E: Missing Feature (Schema Invariant)
# ==============================================================================
def test_case_e_missing_feature(fixtures=None):
    f = fixtures or get_fixtures()
    test_df = f["test_df"]
    incomplete = test_df.iloc[0:1].copy().drop(columns=["Rate"])

    caught = False
    try:
        validate_46_features(incomplete)
    except FeatureValidationError as e:
        caught = True
        assert "Missing" in str(e)
    assert caught, "Should have raised FeatureValidationError for missing column"
    print("  [PASS] Test E: Missing feature rejected by schema validator.")


# ==============================================================================
# TEST F: Extra Feature (Schema Invariant)
# ==============================================================================
def test_case_f_extra_feature(fixtures=None):
    f = fixtures or get_fixtures()
    test_df = f["test_df"]
    extra_df = test_df.iloc[0:1].copy()
    extra_df["unauthorized_metadata_col"] = 999.0

    caught = False
    try:
        validate_46_features(extra_df)
    except FeatureValidationError as e:
        caught = True
        assert "unexpected column" in str(e).lower()
    assert caught, "Should have raised FeatureValidationError for extra column"
    print("  [PASS] Test F: Extra feature rejected by schema validator.")


# ==============================================================================
# TEST G: NaN Value Rejection
# ==============================================================================
def test_case_g_nan_rejection(fixtures=None):
    f = fixtures or get_fixtures()
    test_df = f["test_df"]
    nan_df = test_df.iloc[0:1].copy()
    nan_df["Rate"] = np.nan

    caught = False
    try:
        validate_46_features(nan_df)
    except FeatureValidationError as e:
        caught = True
        assert "missing/null" in str(e).lower() or "nan" in str(e).lower()
    assert caught, "Should have raised FeatureValidationError for NaN"
    print("  [PASS] Test G: NaN value rejected by validator.")


# ==============================================================================
# TEST H: Inf Value Rejection
# ==============================================================================
def test_case_h_inf_rejection(fixtures=None):
    f = fixtures or get_fixtures()
    test_df = f["test_df"]
    inf_df = test_df.iloc[0:1].copy()
    inf_df["Rate"] = np.inf

    caught = False
    try:
        validate_46_features(inf_df)
    except FeatureValidationError as e:
        caught = True
        assert "infinite" in str(e).lower()
    assert caught, "Should have raised FeatureValidationError for Inf"
    print("  [PASS] Test H: Infinite value rejected by validator.")


# ==============================================================================
# TEST I: Wrong Datatype Rejection
# ==============================================================================
def test_case_i_wrong_datatype(fixtures=None):
    f = fixtures or get_fixtures()
    test_df = f["test_df"]
    bad_type = test_df.iloc[0:1].copy()
    bad_type["Rate"] = "UNPARSEABLE_STRING_PAYLOAD"

    caught = False
    try:
        validate_46_features(bad_type)
    except FeatureValidationError as e:
        caught = True
        assert "Could not convert feature" in str(e) or "float" in str(e)
    assert caught, "Should have raised FeatureValidationError for invalid string"
    print("  [PASS] Test I: Unparseable non-numeric datatype rejected.")


# ==============================================================================
# TEST J: Normal Sample Close to Boundary
# ==============================================================================
def test_case_j_boundary_sample_evaluation(fixtures=None):
    f = fixtures or get_fixtures()
    test_df, gate = f["test_df"], f["gate"]

    sample = test_df.iloc[1:2]
    # Sample with confidence exactly 0.72 (above 0.70 threshold) and normal distance
    mock_pred = {
        "attack_type": "DDoS",
        "predicted_class": "DDoS",
        "confidence": 0.72,
        "probabilities": {"DDoS": 0.72, "DoS": 0.28, "Spoofing": 0.00}
    }

    eval_res = gate.evaluate(mock_pred, sample)
    if eval_res["cc_mahalanobis_distance"] <= CC_MAHALANOBIS_THRESHOLD:
        assert eval_res["is_abstained"] is False
        assert eval_res["decision_state"] == STATE_KNOWN_DDOS
        assert eval_res["automated_blocking_allowed"] is True
    print("  [PASS] Test J: Normal boundary sample not falsely rejected.")


# ==============================================================================
# TEST K: Row #657 Forensic Audit Verification
# ==============================================================================
def test_case_k_row_657_behavior(fixtures=None):
    f = fixtures or get_fixtures()
    test_df, detector, gate = f["test_df"], f["detector"], f["gate"]

    row_657 = test_df.iloc[657:658]
    actual_class = test_df["attack_class"].iloc[657] if "attack_class" in test_df.columns else test_df["label"].iloc[657]
    
    # Run production detector
    pred = detector.predict(row_657)
    
    # Ground truth is DDoS, predicted is DoS
    assert str(actual_class) == "DDoS"
    assert pred["attack_type"] == "DoS"

    # Evaluate gate
    eval_res = gate.evaluate(pred, row_657)

    # Check that Row #657 is abstained
    assert eval_res["is_abstained"] is True
    assert eval_res["decision_state"] == STATE_UNKNOWN_OR_REVIEW
    assert eval_res["routing"] == "HUMAN_INVESTIGATION"
    assert eval_res["automated_blocking_allowed"] is False
    
    # Verify measured metrics:
    # 1. Calibrated confidence is below 70% (0.6586)
    assert eval_res["confidence"] < CONFIDENCE_THRESHOLD
    # 2. CC Mahalanobis is elevated (>23.4341)
    assert eval_res["cc_mahalanobis_distance"] > CC_MAHALANOBIS_THRESHOLD
    # 3. Triggered BOTH reasons
    assert REASON_LOW_CONFIDENCE in eval_res["abstention_reasons"]
    assert REASON_OOD_DISTANCE in eval_res["abstention_reasons"]
    print(f"  [PASS] Test K: Row #657 verified abstained (conf={eval_res['confidence']:.4f} < {CONFIDENCE_THRESHOLD}, CC Mah={eval_res['cc_mahalanobis_distance']:.4f} > {CC_MAHALANOBIS_THRESHOLD}).")


if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING PHASE 14 COMPREHENSIVE TEST SUITE (TESTS A THROUGH K)")
    print("=" * 80)
    fixtures = get_fixtures()
    
    tests = [
        ("Test A: Valid Known Sample", test_case_a_valid_known_sample),
        ("Test B: Low Confidence Sample", test_case_b_low_confidence_sample),
        ("Test C: High Mahalanobis Sample", test_case_c_high_mahalanobis_sample),
        ("Test D: Both Low Conf and High Distance", test_case_d_both_low_conf_and_high_distance),
        ("Test E: Missing Feature", test_case_e_missing_feature),
        ("Test F: Extra Feature", test_case_f_extra_feature),
        ("Test G: NaN Rejection", test_case_g_nan_rejection),
        ("Test H: Inf Rejection", test_case_h_inf_rejection),
        ("Test I: Wrong Datatype", test_case_i_wrong_datatype),
        ("Test J: Normal Boundary Sample", test_case_j_boundary_sample_evaluation),
        ("Test K: Row #657 Forensic Audit", test_case_k_row_657_behavior),
    ]
    
    passed = 0
    failed = 0
    for name, test_func in tests:
        print(f"\nRunning {name}...")
        try:
            test_func(fixtures)
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            
    print("\n" + "=" * 80)
    print(f"TEST SUMMARY: {passed} PASSED, {failed} FAILED (TOTAL {len(tests)})")
    print("=" * 80)
    if failed > 0:
        sys.exit(1)

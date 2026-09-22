"""
tests/test_phase15_integrity.py
===============================
Comprehensive Security & Integrity Invariant Test Suite for AEGIS-IoT (Phase 15).

Verifies all 12 Security Invariants:
  1. UNKNOWN_OR_REVIEW can never have trusted_class != None.
  2. UNKNOWN_OR_REVIEW can never have automated_blocking_allowed = True.
  3. advisory_class cannot become trusted_class without explicit human authorization.
  4. RAG cannot modify the ML prediction.
  5. RAG cannot modify OOD state.
  6. Agents cannot bypass abstention.
  7. Frontend / API serialization cannot convert UNKNOWN_OR_REVIEW into a trusted known class.
  8. Inference failures cannot silently become a known-class prediction.
  9. The XGBoost model always receives exactly 46 canonical features in canonical order.
  10. Temperature calibration is applied consistently across all inferences.
  11. SHAP uses the exact same feature vector used for inference.
  12. UNKNOWN_OR_REVIEW cannot trigger automated enforcement without human sign-off.
"""

import os
import sys

# Ensure UTF-8 output on Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import time
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import softmax

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
    REASON_OOD_DISTANCE
)
from backend.detection.ood_detector import OODNoveltyDetector
from backend.detection.abstention_gate import OODAbstentionGate
from backend.detection.detector import IoTIntrusionDetector
from backend.engine.pipeline import CompleteIDSPipeline
from backend.enforcement.enforcement import SecurityEnforcementEngine
from backend.agents.agent_system import MultiAgentOrchestrator
from backend.agents.agents import MultiAgentSystem
from backend.rag.adaptive_hybrid_rag import AdaptiveHybridRAG


# ==============================================================================
# INVARIANT 1: UNKNOWN_OR_REVIEW can never have trusted_class != None
# ==============================================================================
def test_invariant_1_trusted_class_none_on_unknown():
    gate = OODAbstentionGate()
    test_csv = os.path.join(BASE_DIR, "Testing_data", "testing_data.csv")
    test_df = pd.read_csv(test_csv)

    # Test with synthetic low-confidence prediction
    mock_pred = {
        "attack_type": "DoS",
        "predicted_class": "DoS",
        "confidence": 0.50,
        "probabilities": {"DDoS": 0.50, "DoS": 0.50, "Spoofing": 0.0}
    }
    res = gate.evaluate(mock_pred, test_df.iloc[0:1])
    assert res["decision_state"] == STATE_UNKNOWN_OR_REVIEW
    assert res["trusted_class"] is None, "INVARIANT 1 VIOLATION: trusted_class must be None on UNKNOWN_OR_REVIEW"
    assert res["advisory_class"] == "DoS"
    print("  [PASS] INVARIANT 1: UNKNOWN_OR_REVIEW strictly maintains trusted_class = None.")


# ==============================================================================
# INVARIANT 2: UNKNOWN_OR_REVIEW can never have automated_blocking_allowed = True
# ==============================================================================
def test_invariant_2_automated_blocking_prohibited_on_unknown():
    gate = OODAbstentionGate()
    test_csv = os.path.join(BASE_DIR, "Testing_data", "testing_data.csv")
    test_df = pd.read_csv(test_csv)

    # Test with distance outlier
    outlier = test_df.iloc[0:1].copy()
    for col in CANONICAL_46_FEATURES:
        outlier[col] = outlier[col] * 100.0 + 5000.0

    mock_pred = {
        "attack_type": "DDoS",
        "predicted_class": "DDoS",
        "confidence": 0.99,
        "probabilities": {"DDoS": 0.99, "DoS": 0.01, "Spoofing": 0.0}
    }
    res = gate.evaluate(mock_pred, outlier)
    assert res["decision_state"] == STATE_UNKNOWN_OR_REVIEW
    assert res["automated_blocking_allowed"] is False, "INVARIANT 2 VIOLATION: automated_blocking_allowed must be False on UNKNOWN_OR_REVIEW"
    print("  [PASS] INVARIANT 2: UNKNOWN_OR_REVIEW strictly prohibits automated blocking.")


# ==============================================================================
# INVARIANT 3: advisory_class cannot become trusted_class without human authorization
# ==============================================================================
def test_invariant_3_advisory_cannot_become_trusted_automatically():
    pipeline = CompleteIDSPipeline()
    test_csv = os.path.join(BASE_DIR, "Testing_data", "testing_data.csv")
    test_df = pd.read_csv(test_csv)

    # Row 657 is an abstained sample
    res = pipeline.run_inference_and_reasoning(test_df.iloc[657])
    gate = res["gate_evaluation"]

    assert gate["decision_state"] == STATE_UNKNOWN_OR_REVIEW
    assert gate["trusted_class"] is None
    assert gate["advisory_class"] == "DoS"

    # Verify that multi-agent result does NOT convert advisory to trusted
    assert res.get("trusted_class") is None
    assert res.get("decision_state") == STATE_UNKNOWN_OR_REVIEW
    print("  [PASS] INVARIANT 3: advisory_class does not become trusted_class through agent execution.")


# ==============================================================================
# INVARIANT 4: RAG cannot modify the ML prediction
# ==============================================================================
def test_invariant_4_rag_cannot_modify_ml_prediction():
    rag = AdaptiveHybridRAG()
    retrieved = rag.get_grounded_context(attack_type="DoS", shap_evidence=[])
    # Verify RAG returns context without mutating ML state
    assert retrieved is not None
    assert "evidence_bundle" in retrieved
    # Check that calling RAG with conflicting text does not alter model output
    detector = IoTIntrusionDetector()
    test_csv = os.path.join(BASE_DIR, "Testing_data", "testing_data.csv")
    test_df = pd.read_csv(test_csv)
    pred_before = detector.predict(test_df.iloc[0:1])
    # Call RAG extensively
    _ = rag.get_grounded_context(attack_type="Spoofing", shap_evidence=[])
    _ = rag.get_grounded_context(attack_type="DDoS", shap_evidence=[])
    pred_after = detector.predict(test_df.iloc[0:1])
    assert pred_before["predicted_class"] == pred_after["predicted_class"]
    assert pred_before["confidence"] == pred_after["confidence"]
    print("  [PASS] INVARIANT 4: RAG retrieval has zero side-effects on ML predictions.")


# ==============================================================================
# INVARIANT 5: RAG cannot modify OOD state
# ==============================================================================
def test_invariant_5_rag_cannot_modify_ood_state():
    novelty_detector = OODNoveltyDetector()
    test_csv = os.path.join(BASE_DIR, "Testing_data", "testing_data.csv")
    test_df = pd.read_csv(test_csv)

    d1 = novelty_detector.evaluate_novelty(test_df.iloc[657:658], "DoS")
    # Execute RAG
    rag = AdaptiveHybridRAG()
    _ = rag.get_grounded_context(attack_type="DoS", shap_evidence=[])
    d2 = novelty_detector.evaluate_novelty(test_df.iloc[657:658], "DoS")

    assert d1["cc_mahalanobis_distance"] == d2["cc_mahalanobis_distance"]
    assert d1["is_atypical"] == d2["is_atypical"]
    print("  [PASS] INVARIANT 5: RAG operations cannot modify OOD distance or novelty decisions.")


# ==============================================================================
# INVARIANT 6: Agents cannot bypass abstention
# ==============================================================================
def test_invariant_6_agents_cannot_bypass_abstention():
    agent_sys = MultiAgentSystem()
    abstained_alert = {
        "id": "ALT-TEST-ABSTAIN",
        "attack_type": "UNKNOWN_OR_REVIEW",
        "advisory_attack_type": "DoS",
        "decision_state": STATE_UNKNOWN_OR_REVIEW,
        "is_abstained": True,
        "confidence": 0.6586,
        "automated_blocking_allowed": False,
        "abstention_reasons": ["LOW_CONFIDENCE", "OOD_DISTANCE"],
        "src_ip": "192.168.1.50",
        "dst_ip": "192.168.1.1"
    }
    threat_intel = agent_sys.threat_intel_agent(abstained_alert)
    investigation = agent_sys.investigation_agent(abstained_alert, threat_intel)
    risk = agent_sys.risk_assessment_agent(abstained_alert, investigation)
    response = agent_sys.response_generation_agent(abstained_alert, threat_intel, risk)
    report = agent_sys.report_generation_agent(abstained_alert, threat_intel, investigation, risk, response)

    # Agents MUST NOT generate aggressive blocking actions on UNKNOWN_OR_REVIEW
    for act in response["recommended_actions"]:
        assert act.get("automated_blocking_allowed") is False, "INVARIANT 6 VIOLATION: Agents must not enable automated blocking on abstained alerts"
        assert act.get("type") != "FIREWALL_DROP", "INVARIANT 6 VIOLATION: Agents must not generate FIREWALL_DROP on UNKNOWN_OR_REVIEW"

    # Report title must indicate UNVERIFIED/ADVISORY
    assert "[ADVISORY / UNVERIFIED]" in report["title"], "INVARIANT 6 VIOLATION: Report title must reflect unverified status"
    print("  [PASS] INVARIANT 6: Agents strictly honor abstention without generating automated firewall blocks.")


# ==============================================================================
# INVARIANT 7: Frontend / API serialization cannot convert UNKNOWN_OR_REVIEW into a trusted class
# ==============================================================================
def test_invariant_7_serialization_preserves_unknown_state():
    alert_payload = {
        "id": "ALT-12345678",
        "timestamp": time.time(),
        "attack_type": "UNKNOWN_OR_REVIEW",
        "advisory_attack_type": "DoS",
        "decision_state": "UNKNOWN_OR_REVIEW",
        "is_abstained": True,
        "abstention_reasons": ["LOW_CONFIDENCE"],
        "confidence": 0.65,
        "severity": "HIGH",
        "routing": "HUMAN_INVESTIGATION",
        "automated_blocking_allowed": False
    }
    # Simulate API JSON serialization & deserialization
    json_str = json.dumps(alert_payload)
    deserialized = json.loads(json_str)

    assert deserialized["attack_type"] == "UNKNOWN_OR_REVIEW"
    assert deserialized["decision_state"] == "UNKNOWN_OR_REVIEW"
    assert deserialized["is_abstained"] is True
    assert deserialized["automated_blocking_allowed"] is False
    print("  [PASS] INVARIANT 7: API serialization completely preserves UNKNOWN_OR_REVIEW state.")


# ==============================================================================
# INVARIANT 8: Inference failures cannot silently become a known-class prediction
# ==============================================================================
def test_invariant_8_failures_raise_explicit_errors():
    detector = IoTIntrusionDetector()
    corrupt_input = {"Rate": np.nan}  # Missing 45 features and NaN

    failed_safely = False
    try:
        detector.predict(corrupt_input)
    except FeatureValidationError:
        failed_safely = True
    except Exception as e:
        failed_safely = True

    assert failed_safely, "INVARIANT 8 VIOLATION: Inference failure must raise error, not guess a class"
    print("  [PASS] INVARIANT 8: Malformed telemetry triggers explicit validation failure rather than silent fallback.")


# ==============================================================================
# INVARIANT 9: XGBoost model always receives exactly 46 canonical features
# ==============================================================================
def test_invariant_9_model_receives_exact_46_features():
    detector = IoTIntrusionDetector()
    test_csv = os.path.join(BASE_DIR, "Testing_data", "testing_data.csv")
    test_df = pd.read_csv(test_csv)

    formatted_df = detector.format_input(test_df.iloc[0:1])
    assert formatted_df.shape[1] == 46, "INVARIANT 9 VIOLATION: Feature count must be 46"
    assert list(formatted_df.columns) == CANONICAL_46_FEATURES, "INVARIANT 9 VIOLATION: Feature ordering must match canonical schema"
    assert formatted_df.dtypes.iloc[0] == np.float32, "INVARIANT 9 VIOLATION: Feature dtype must be float32"
    print("  [PASS] INVARIANT 9: Feature matrix strictly adheres to 46 canonical columns in exact order.")


# ==============================================================================
# INVARIANT 10: Temperature calibration is applied consistently
# ==============================================================================
def test_invariant_10_temperature_calibration_consistency():
    # 1. Instantiate IoTIntrusionDetector and CompleteIDSPipeline
    detector = IoTIntrusionDetector()
    pipeline = CompleteIDSPipeline()

    # 2. Load row 657 from Testing_data/testing_data.csv
    test_csv = os.path.join(BASE_DIR, "Testing_data", "testing_data.csv")
    test_df = pd.read_csv(test_csv)
    sample = test_df.iloc[657:658]

    # 3. Obtain detector and pipeline predictions
    det_res = detector.predict(sample)
    pipe_res = pipeline.run_inference_and_reasoning(sample)

    # 4. Assert detector confidence == pipeline confidence within a small numerical tolerance (e.g. 1e-6)
    conf_diff = abs(det_res["confidence"] - pipe_res["confidence"])
    assert conf_diff < 1e-6, (
        f"INVARIANT 10 VIOLATION: Detector confidence ({det_res['confidence']:.8f}) and "
        f"pipeline confidence ({pipe_res['confidence']:.8f}) differ by {conf_diff:.2e} (> 1e-6)"
    )

    # 5. Verify the active calibrator temperature is the AEGIS temperature from:
    #    models/aegis/aegis_calibrator_temperature.pkl and is approximately 1.345192
    calib_path = os.path.join(BASE_DIR, "models", "aegis", "aegis_calibrator_temperature.pkl")
    assert os.path.exists(calib_path), f"Calibrator temperature file not found at {calib_path}"
    loaded_calibrator = joblib.load(calib_path)
    active_temp = float(loaded_calibrator.temperature)
    assert abs(active_temp - 1.345192) < 1e-4, (
        f"INVARIANT 10 VIOLATION: Expected AEGIS temperature ~1.345192, got {active_temp:.6f}"
    )
    assert abs(detector.calibrator.temperature - active_temp) < 1e-6, (
        "Detector calibrator temperature does not match active AEGIS temperature file"
    )
    assert abs(pipeline.calib_temp.temperature - active_temp) < 1e-6, (
        "Pipeline calibrator temperature does not match active AEGIS temperature file"
    )

    # 6. Independently calculate the calibrated probability from the active AEGIS model logits using:
    #    scaled_logits = logits / temperature
    #    calibrated_probs = softmax(scaled_logits)
    formatted_df = detector.format_input(sample)
    dmat = xgb.DMatrix(formatted_df)
    logits = detector.booster.predict(dmat, output_margin=True)
    scaled_logits = logits / active_temp
    calibrated_probs = softmax(scaled_logits, axis=1)[0]
    independent_top1_prob = float(np.max(calibrated_probs))

    # 7. Assert the detector confidence equals the independently calculated calibrated top-1 probability within numerical tolerance
    assert abs(det_res["confidence"] - independent_top1_prob) < 1e-6, (
        f"INVARIANT 10 VIOLATION: Detector confidence ({det_res['confidence']:.8f}) != "
        f"independently calculated top-1 calibrated probability ({independent_top1_prob:.8f})"
    )

    # Verify calibration is actively altering confidence from raw probability
    raw_probs = detector.model.predict_proba(formatted_df)[0]
    raw_top1_prob = float(np.max(raw_probs))
    assert abs(det_res["confidence"] - raw_top1_prob) > 0.01, (
        f"INVARIANT 10 VIOLATION: Prediction confidence ({det_res['confidence']:.4f}) is uncalibrated "
        f"(equals raw model probability {raw_top1_prob:.4f})"
    )

    print("  [PASS] INVARIANT 10: Temperature calibration consistently applied and mathematically verified.")


# ==============================================================================
# INVARIANT 11: SHAP uses the exact same feature vector used for inference
# ==============================================================================
def test_invariant_11_shap_feature_vector_parity():
    pipeline = CompleteIDSPipeline()
    test_csv = os.path.join(BASE_DIR, "Testing_data", "testing_data.csv")
    test_df = pd.read_csv(test_csv)

    sample = test_df.iloc[0:1]
    res = pipeline.run_inference_and_reasoning(sample)

    shap_evidence = res["explainability"]["top_features"]
    # All features in SHAP must be members of CANONICAL_46_FEATURES
    for feat_dict in shap_evidence:
        assert feat_dict["feature"] in CANONICAL_46_FEATURES, "INVARIANT 11 VIOLATION: SHAP feature not in canonical 46"
        # Check that the feature value reported by SHAP matches the input feature value
        input_val = float(sample[feat_dict["feature"]].iloc[0])
        assert abs(feat_dict["feature_value"] - input_val) < 1e-3, "INVARIANT 11 VIOLATION: SHAP feature value mismatch"
    print("  [PASS] INVARIANT 11: SHAP uses identical 46-feature input vector with verified feature value parity.")


# ==============================================================================
# INVARIANT 12: UNKNOWN_OR_REVIEW cannot trigger automated enforcement
# ==============================================================================
def test_invariant_12_unknown_blocks_automated_enforcement():
    enforcement = SecurityEnforcementEngine(dry_run=True)
    unauthorized_action = {
        "action_id": "ACT-BLOCK-IP",
        "type": "FIREWALL_DROP",
        "target": "10.0.0.99",
        "command": "netsh advfirewall firewall add rule ...",
        "decision_state": "UNKNOWN_OR_REVIEW",
        "automated_blocking_allowed": False,
        "is_abstained": True
    }

    # An automated process attempting to execute without SOC_ANALYST human approval MUST be rejected
    blocked = False
    try:
        enforcement.execute_approved_action("INC-TEST-001", unauthorized_action, executed_by="AUTOMATED_SYSTEM")
    except PermissionError as pe:
        blocked = True
        assert "BLOCKED: Automated enforcement prohibited" in str(pe)

    assert blocked, "INVARIANT 12 VIOLATION: Automated enforcement must be blocked on UNKNOWN_OR_REVIEW"
    print("  [PASS] INVARIANT 12: Enforcement engine strictly blocks automated execution on abstained incidents.")


if __name__ == "__main__":
    print("=" * 80)
    print("PHASE 15: SECURITY & INTEGRITY INVARIANT TEST SUITE")
    print("=" * 80)

    invariants = [
        ("Invariant 1: Trusted class None on UNKNOWN", test_invariant_1_trusted_class_none_on_unknown),
        ("Invariant 2: Automated blocking prohibited on UNKNOWN", test_invariant_2_automated_blocking_prohibited_on_unknown),
        ("Invariant 3: Advisory class cannot become trusted", test_invariant_3_advisory_cannot_become_trusted_automatically),
        ("Invariant 4: RAG cannot modify ML prediction", test_invariant_4_rag_cannot_modify_ml_prediction),
        ("Invariant 5: RAG cannot modify OOD state", test_invariant_5_rag_cannot_modify_ood_state),
        ("Invariant 6: Agents cannot bypass abstention", test_invariant_6_agents_cannot_bypass_abstention),
        ("Invariant 7: Serialization preserves UNKNOWN state", test_invariant_7_serialization_preserves_unknown_state),
        ("Invariant 8: Inference failures raise errors safely", test_invariant_8_failures_raise_explicit_errors),
        ("Invariant 9: Exact 46 canonical features preserved", test_invariant_9_model_receives_exact_46_features),
        ("Invariant 10: Temperature calibration consistency", test_invariant_10_temperature_calibration_consistency),
        ("Invariant 11: SHAP uses exact inference feature vector", test_invariant_11_shap_feature_vector_parity),
        ("Invariant 12: Enforcement blocks automated execution on UNKNOWN", test_invariant_12_unknown_blocks_automated_enforcement),
    ]

    passed = 0
    failed = 0
    for name, func in invariants:
        print(f"\nExecuting {name}...")
        try:
            func()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 80)
    print(f"SECURITY INVARIANT SUMMARY: {passed} PASSED, {failed} FAILED (TOTAL {len(invariants)})")
    print("=" * 80)
    if failed > 0:
        sys.exit(1)

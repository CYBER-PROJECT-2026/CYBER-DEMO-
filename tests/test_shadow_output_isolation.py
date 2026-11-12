from types import SimpleNamespace

from aegis_live.v4_runtime import (
    AegisLiveV4Pipeline,
    DetectionResultV4,
)
from backend.detection.incident_correlator import (
    IncidentClassificationContract,
    IncidentCorrelationEngine,
)
from backend.detection.output_isolation import (
    dispatch_production_output,
    is_production_operational_output,
)


def _detection(output_mode="PRODUCTION", evaluation_id=None):
    return DetectionResultV4(
        predicted_class="DDoS",
        calibrated_confidence=0.9,
        confidence=0.9,
        mahalanobis_distance=1.0,
        trusted=True,
        decision="TRUSTED",
        status="TRUSTED",
        trusted_class="DDoS",
        advisory_class="DDoS",
        automated_blocking_allowed=False,
        reasons=[],
        reason_codes=[],
        top2_margin=0.5,
        entropy=0.1,
        feature_vector={"duration_ms": 1.0},
        calibrated_probabilities={"DDoS": 0.9},
        output_mode=output_mode,
        evaluation_id=evaluation_id,
    )


def test_shadow_detection_serializes_with_explicit_nonproduction_identity():
    record = _detection("SHADOW_EVALUATION", "eval-123").to_dict()
    assert record["output_mode"] == "SHADOW_EVALUATION"
    assert record["evaluation_id"] == "eval-123"
    assert record["automated_blocking_allowed"] is False
    assert not is_production_operational_output(record)


def test_shadow_records_are_not_dispatched_to_production_notifications():
    calls = []
    shadow = {"output_mode": "SHADOW_EVALUATION", "evaluation_id": "eval-123"}
    assert dispatch_production_output(shadow, [calls.append]) is False
    assert calls == []

    production = {"output_mode": "PRODUCTION", "predicted_class": "DDoS"}
    assert dispatch_production_output(production, [calls.append]) is True
    assert calls == [production]


def test_shadow_evaluation_id_fails_closed_even_if_mode_is_production():
    assert not is_production_operational_output(
        {"output_mode": "PRODUCTION", "evaluation_id": "eval-123"}
    )
    assert not is_production_operational_output({"output_mode": "UNKNOWN"})
    assert is_production_operational_output({"predicted_class": "DDoS"})


def test_correlator_suppresses_shadow_before_incident_persistence():
    classification = IncidentClassificationContract.evaluate_classification(
        {"output_mode": "SHADOW_EVALUATION", "evaluation_id": "eval-123", "predicted_class": "DDoS"}
    )
    assert classification["classification_type"] == "SHADOW_EVALUATION"
    assert classification["incident_prefix"] is None
    assert classification["automated_blocking_allowed"] is False

    correlator = IncidentCorrelationEngine.__new__(IncidentCorrelationEngine)
    incident, is_new = correlator.correlate_or_create(
        {"output_mode": "SHADOW_EVALUATION", "evaluation_id": "eval-123"},
        orchestrator=None,
    )
    assert incident is None
    assert is_new is False


def test_partial_windows_remain_ineligible_and_evaluation_tagged():
    pipeline = AegisLiveV4Pipeline.__new__(AegisLiveV4Pipeline)
    pipeline.output_mode = "SHADOW_EVALUATION"
    pipeline.evaluation_id = "eval-123"
    pipeline.min_packets = 2
    pipeline.interface_name = "test"
    pipeline.incomplete_window_records = []
    pipeline.windows_rejected = 0
    pipeline.window_rejection_reasons = {}
    pipeline._extract_combined_29_features = lambda window: {"duration_ms": 1.0}

    window = SimpleNamespace(
        packets=[SimpleNamespace(timestamp=1.0) for _ in range(49)],
        flow_key=("10.0.0.1", "10.0.0.2", 6),
        last_topology_snapshot={"window_end_ts": 1.0},
        affected_by_ooo_rejection=False,
        ooo_rejected_count=0,
    )
    result = pipeline._process_completed_window(window)
    partial = pipeline.incomplete_window_records[0]

    assert result is None
    assert partial.packet_count == 49
    assert partial.status == "INCOMPLETE_WINDOW"
    assert partial.v4_eligible is False
    assert partial.advisory_eligible is True
    assert partial.output_mode == "SHADOW_EVALUATION"
    assert partial.evaluation_id == "eval-123"
    assert pipeline.windows_rejected == 1

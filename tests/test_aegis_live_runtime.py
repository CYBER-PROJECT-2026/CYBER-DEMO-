"""
tests/test_aegis_live_runtime.py
================================
Regression test suite for AEGIS-Live v2 Live Packet Integration (Detection-Only Mode).

Verifies:
1. Live packet -> feature vector (physical 24-feature extraction)
2. Live IAT calculation (zero-initialized, accurate inter-packet delta)
3. 2-second timeout boundary (packets with >= 2.0s gap start new window)
4. 50-packet boundary (window emits immediately on 50th packet)
5. Missing feature rejection (incomplete vectors quarantined)
6. Calibrated probability calculation (softmax(logits / T) sums to 1.0)
7. Temperature preservation (frozen T = 1.005459)
8. Mahalanobis distance calculation (class-conditional distances non-negative & finite)
9. Exact abstention thresholds (confidence < 0.55 OR Mahalanobis > 35.0)
10. UNKNOWN_OR_REVIEW contract (trusted_class=None, advisory_class preserved, reasons populated)
11. automated_blocking_allowed always False (strictly detection-only)
12. Frozen model hash (XGBoost baseline SHA256 matches)
13. Frozen calibration parameter (temperature_scaling.json SHA256 matches)
14. Frozen OOD thresholds (ood_parameters.pkl and abstention_config.json SHA256 match)
"""

import os
import sys
import json
import pickle
import hashlib
import pytest
import numpy as np
import dpkt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from aegis_live.feature_schema import CANONICAL_24_FEATURES, validate_24_features
from aegis_live.packet_window import PacketRecord, FlowWindow
from aegis_live.live_capture import AegisLivePipeline, LivePipelineExtractor, WindowProvenanceRecord
from aegis_live.live_inference import (
    LiveInferenceEngine,
    DetectionResult,
    FROZEN_MODEL_SHA256,
    FROZEN_TEMPERATURE,
    FROZEN_CONFIDENCE_THRESHOLD,
    FROZEN_MAHALANOBIS_THRESHOLD
)

MODEL_DIR = os.path.join(BASE_DIR, "models", "aegis_live_v2")
MODEL_PKL = os.path.join(MODEL_DIR, "aegis_live_xgboost_baseline.pkl")
TEMP_JSON = os.path.join(MODEL_DIR, "temperature_scaling.json")
OOD_PKL = os.path.join(MODEL_DIR, "ood_parameters.pkl")
ABST_JSON = os.path.join(MODEL_DIR, "abstention_config.json")

EXPECTED_TEMP_SHA256 = "3c98e97bf27eda1533d4173a9cda1f647624dd1c0d995ff8885da15ff304c83a"
EXPECTED_OOD_SHA256 = "4498f97fe83fb74cddfde9b37962bb69a8004967df6686b643593d3d391685ac"
EXPECTED_ABST_SHA256 = "48d4341fbe96cc51f174806f21e4167ea7da08d584384e2de9ee67a66d0c5aab"


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def create_synthetic_ethernet_ip_tcp_frame(src_ip: str = "192.168.1.10",
                                           dst_ip: str = "192.168.1.20",
                                           src_port: int = 12345,
                                           dst_port: int = 80,
                                           tcp_flags: int = 0x02,
                                           payload: bytes = b"X" * 40) -> bytes:
    """Constructs a valid Ethernet / IP / TCP frame bytes."""
    tcp = dpkt.tcp.TCP(
        sport=src_port,
        dport=dst_port,
        flags=tcp_flags,
        seq=1000,
        data=payload
    )
    ip = dpkt.ip.IP(
        src=bytes(map(int, src_ip.split('.'))),
        dst=bytes(map(int, dst_ip.split('.'))),
        p=dpkt.ip.IP_PROTO_TCP,
        ttl=64,
        data=tcp
    )
    ip.len = len(ip)
    eth = dpkt.ethernet.Ethernet(
        src=b"\x00\x11\x22\x33\x44\x55",
        dst=b"\x66\x77\x88\x99\xaa\xbb",
        type=dpkt.ethernet.ETH_TYPE_IP,
        data=ip
    )
    return bytes(eth)


# -----------------------------------------------------------------------------
# 1. Live packet -> feature vector
# -----------------------------------------------------------------------------
def test_live_packet_to_feature_vector():
    pipeline = AegisLivePipeline(interface_name="test_iface")
    raw_frame = create_synthetic_ethernet_ip_tcp_frame()
    
    t0 = 1000.0
    dets1 = pipeline.ingest_packet(t0, raw_frame)
    assert len(dets1) == 0  # 1 packet does not trigger window emission

    # Second packet arrives 0.05s later -> flush to emit window
    dets2 = pipeline.ingest_packet(t0 + 0.05, raw_frame)
    flushed = pipeline.flush(current_time=t0 + 3.0, force_all=True)
    all_dets = dets2 + flushed

    assert len(all_dets) == 1
    det = all_dets[0]
    assert det.provenance.status == "VALID"
    assert len(det.provenance.features) == 24
    assert det.provenance.packet_count == 2
    for f in CANONICAL_24_FEATURES:
        assert f in det.provenance.features
        assert np.isfinite(det.provenance.features[f])


# -----------------------------------------------------------------------------
# 2. Live IAT calculation
# -----------------------------------------------------------------------------
def test_live_iat_calculation():
    extractor = LivePipelineExtractor()
    raw_frame = create_synthetic_ethernet_ip_tcp_frame()

    t0 = 2000.0
    # Add 3 packets with 50ms and 150ms IATs: mean IAT = 100ms
    extractor.process_raw_packet_with_provenance(t0, raw_frame)
    extractor.process_raw_packet_with_provenance(t0 + 0.05, raw_frame)
    extractor.process_raw_packet_with_provenance(t0 + 0.20, raw_frame)

    flushed = extractor.flush_active_windows_with_provenance(current_time=t0 + 3.0, force_all=True)
    assert len(flushed) == 1
    w = flushed[0]
    feats = w.features

    assert abs(feats["duration_ms"] - 200.0) < 1e-3
    assert abs(feats["iat_mean_ms"] - 100.0) < 1e-3


# -----------------------------------------------------------------------------
# 3. 2-second timeout boundary
# -----------------------------------------------------------------------------
def test_2_second_timeout_boundary():
    extractor = LivePipelineExtractor(window_timeout_sec=2.0)
    raw_frame = create_synthetic_ethernet_ip_tcp_frame()

    t0 = 3000.0
    # Packet 1 & 2 within 0.1s
    extractor.process_raw_packet_with_provenance(t0, raw_frame)
    extractor.process_raw_packet_with_provenance(t0 + 0.1, raw_frame)

    # Packet 3 arrives after 2.1s (gap >= 2.0s)
    emitted = extractor.process_raw_packet_with_provenance(t0 + 2.2, raw_frame)
    assert len(emitted) == 1
    old_window = emitted[0]
    assert old_window.packet_count == 2
    assert old_window.window_end_ts == t0 + 0.1

    # Flush fresh window containing packet 3 + packet 4
    extractor.process_raw_packet_with_provenance(t0 + 2.3, raw_frame)
    flushed = extractor.flush_active_windows_with_provenance(current_time=t0 + 5.0, force_all=True)
    assert len(flushed) == 1
    fresh_window = flushed[0]
    assert fresh_window.packet_count == 2
    assert fresh_window.window_start_ts == t0 + 2.2


# -----------------------------------------------------------------------------
# 4. 50-packet boundary
# -----------------------------------------------------------------------------
def test_50_packet_boundary():
    extractor = LivePipelineExtractor(max_packets_per_window=50, window_timeout_sec=2.0)
    raw_frame = create_synthetic_ethernet_ip_tcp_frame()

    t0 = 4000.0
    all_emitted = []
    for i in range(50):
        # Arrive every 10ms (total duration 490ms < 2.0s)
        recs = extractor.process_raw_packet_with_provenance(t0 + i * 0.01, raw_frame)
        all_emitted.extend(recs)

    # Must emit exactly 1 window upon packet 50 arrival
    assert len(all_emitted) == 1
    w = all_emitted[0]
    assert w.packet_count == 50
    assert len(extractor.active_windows) == 0  # Completed window was removed from active


# -----------------------------------------------------------------------------
# 5. Missing feature rejection
# -----------------------------------------------------------------------------
def test_missing_feature_rejection():
    engine = LiveInferenceEngine()
    incomplete = {c: 1.0 for c in CANONICAL_24_FEATURES[:15]}  # Missing 9 features

    res = engine.predict(incomplete)
    assert res.status == "REJECTED_QUARANTINE"
    assert res.trusted_class is None
    assert res.advisory_class is None
    assert res.automated_blocking_allowed is False
    assert any("feature_validation_failed" in r for r in res.reasons)


# -----------------------------------------------------------------------------
# 6. Calibrated probability calculation
# -----------------------------------------------------------------------------
def test_calibrated_probability_calculation():
    engine = LiveInferenceEngine()
    # Construct a valid benign-like feature vector
    valid_features = {
        "duration_ms": 100.0, "pkt_rate_pps": 20.0, "byte_rate_bps": 20000.0,
        "iat_mean_ms": 50.0, "iat_std_ms": 5.0, "pkt_size_mean": 100.0,
        "pkt_size_std": 10.0, "pkt_size_min": 60.0, "pkt_size_max": 200.0,
        "syn_ratio": 0.0, "ack_ratio": 1.0, "rst_ratio": 0.0,
        "fin_ratio": 0.0, "urg_ratio": 0.0, "psh_ratio": 0.0,
        "ece_ratio": 0.0, "cwr_ratio": 0.0, "ip_ttl_mean": 64.0,
        "ip_ttl_std": 0.0, "tcp_ratio": 1.0, "udp_ratio": 0.0,
        "icmp_ratio": 0.0, "inbound_pkt_ratio": 0.5, "outbound_pkt_ratio": 0.5
    }

    res = engine.predict(valid_features)
    assert res.status in ("TRUSTED", "UNKNOWN_OR_REVIEW")
    prob_sum = sum(res.calibrated_probs.values())
    assert abs(prob_sum - 1.0) < 1e-6
    for p in res.calibrated_probs.values():
        assert 0.0 <= p <= 1.0


# -----------------------------------------------------------------------------
# 7. Temperature preservation
# -----------------------------------------------------------------------------
def test_temperature_preservation():
    engine = LiveInferenceEngine()
    assert abs(engine.temperature - FROZEN_TEMPERATURE) < 1e-4
    assert abs(engine.temperature - 1.005459) < 1e-4


# -----------------------------------------------------------------------------
# 8. Mahalanobis distance calculation
# -----------------------------------------------------------------------------
def test_mahalanobis_calculation():
    engine = LiveInferenceEngine()
    vec = np.zeros(24, dtype=np.float64)
    dists = engine._compute_mahalanobis(vec)

    assert len(dists) == 4
    for c in ["Benign", "DDoS", "DoS", "Spoofing"]:
        assert c in dists
        assert dists[c] >= 0.0
        assert np.isfinite(dists[c])


# -----------------------------------------------------------------------------
# 9. Exact abstention thresholds
# -----------------------------------------------------------------------------
def test_exact_abstention_thresholds():
    engine = LiveInferenceEngine()
    assert engine.confidence_threshold == 0.55
    assert engine.mahalanobis_threshold == 35.0


# -----------------------------------------------------------------------------
# 10. UNKNOWN_OR_REVIEW contract
# -----------------------------------------------------------------------------
def test_unknown_or_review_contract():
    engine = LiveInferenceEngine()
    # High-uncertainty / extreme vector designed to trigger abstention
    extreme_features = {
        "duration_ms": 1.0, "pkt_rate_pps": 1000.0, "byte_rate_bps": 50000000.0,
        "iat_mean_ms": 0.01, "iat_std_ms": 0.0, "pkt_size_mean": 1500.0,
        "pkt_size_std": 0.0, "pkt_size_min": 1500.0, "pkt_size_max": 1500.0,
        "syn_ratio": 0.5, "ack_ratio": 0.5, "rst_ratio": 0.0,
        "fin_ratio": 0.0, "urg_ratio": 0.0, "psh_ratio": 0.0,
        "ece_ratio": 0.0, "cwr_ratio": 0.0, "ip_ttl_mean": 255.0,
        "ip_ttl_std": 0.0, "tcp_ratio": 1.0, "udp_ratio": 0.0,
        "icmp_ratio": 0.0, "inbound_pkt_ratio": 0.0, "outbound_pkt_ratio": 1.0
    }

    res = engine.predict(extreme_features)
    if res.confidence < 0.55 or res.predicted_class_distance > 35.0:
        assert res.status == "UNKNOWN_OR_REVIEW"
        assert res.trusted_class is None
        assert res.advisory_class is not None
        assert res.advisory_class == res.raw_predicted_class
        assert len(res.reasons) > 0
    else:
        assert res.status == "TRUSTED"
        assert res.trusted_class == res.raw_predicted_class


# -----------------------------------------------------------------------------
# 11. automated_blocking_allowed always False
# -----------------------------------------------------------------------------
def test_automated_blocking_allowed_always_false():
    engine = LiveInferenceEngine()
    test_vec = {c: 0.5 for c in CANONICAL_24_FEATURES}
    test_vec["syn_ratio"] = 0.0
    test_vec["ack_ratio"] = 1.0
    test_vec["inbound_pkt_ratio"] = 1.0
    test_vec["outbound_pkt_ratio"] = 0.0
    test_vec["tcp_ratio"] = 1.0
    test_vec["udp_ratio"] = 0.0
    test_vec["icmp_ratio"] = 0.0

    res = engine.predict(test_vec)
    assert res.automated_blocking_allowed is False


# -----------------------------------------------------------------------------
# 12. Frozen model hash
# -----------------------------------------------------------------------------
def test_frozen_model_hash():
    actual_hash = compute_sha256(MODEL_PKL)
    assert actual_hash == FROZEN_MODEL_SHA256


# -----------------------------------------------------------------------------
# 13. Frozen calibration parameter
# -----------------------------------------------------------------------------
def test_frozen_calibration_parameter():
    actual_hash = compute_sha256(TEMP_JSON)
    assert actual_hash == EXPECTED_TEMP_SHA256
    with open(TEMP_JSON, "r") as f:
        data = json.load(f)
    assert abs(data["learned_temperature"] - 1.005459) < 1e-4


# -----------------------------------------------------------------------------
# 14. Frozen OOD thresholds
# -----------------------------------------------------------------------------
def test_frozen_ood_thresholds():
    ood_hash = compute_sha256(OOD_PKL)
    abst_hash = compute_sha256(ABST_JSON)
    assert ood_hash == EXPECTED_OOD_SHA256
    assert abst_hash == EXPECTED_ABST_SHA256

    with open(ABST_JSON, "r") as f:
        abst_data = json.load(f)
    g = abst_data["gate_definition"]
    assert g["confidence_threshold"] == 0.55
    assert g["mahalanobis_threshold"] == 35.0

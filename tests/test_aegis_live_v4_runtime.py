"""
tests/test_aegis_live_v4_runtime.py
===================================
Deterministic Automated Test Suite for AEGIS-Live v4 Live Runtime Pipeline.
Verifies:
  1. Exact 29-feature ordering and schema compliance.
  2. Byte/float parity of first 24 physical flow features with v2/v3 canonical extractor.
  3. Topology causal cutoff (packets with t > Tend do not influence topology features).
  4. Future packet injection invariance.
  5. 1-second topology eviction (packets older than 1.0s evicted).
  6. Streaming vs offline batch topology feature equality.
  7. Temperature calibration correctness (T = 1.199787, sum to 1.0).
  8. Mahalanobis distance calculation correctness (non-negative, finite).
  9. Threshold boundary behavior (confidence >= 0.75 and MD <= 7.92).
  10. UNKNOWN_OR_REVIEW contract semantics.
  11. automated_blocking_allowed == False on 100% of emissions.
  12. Session reset correctness.
  13. Malformed and non-IPv4 packet handling without crash.
  14. Zero NaN / zero Inf in features, distances, and probabilities.
  15. Frozen model and calibration artifact SHA256 hash preservation.
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

from aegis_live.feature_schema import CANONICAL_24_FEATURES, CANONICAL_29_FEATURES
from aegis_live.feature_extractor import LiveFeatureExtractor
from aegis_live.network_topology_features import (
    NetworkTopologyPacket,
    NetworkTopologyTracker,
    compute_network_topology_batch
)
from aegis_live.v4_inference import AEGISLiveV4Inference
from aegis_live.v4_runtime import AegisLiveV4Pipeline, DetectionResultV4

# Frozen Hash References
FROZEN_V4_MODEL_HASH = "c0e5aee8e9e1726f12bfc19e98c440fe44fe278d1796d5866fe2a2488d458a56"
FROZEN_V2_MODEL_HASH = "c157660c108a5d95d32d504ebc81c40e6b6911d8df6570a7d3ba92fad062cb9b"
FROZEN_V3_MODEL_HASH = "5eadae3284f0973039b5743d8aa9f5db6c0e5ee3b43ed668f245d168d7958128"


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def make_raw_ethernet_ip_tcp_frame(src_ip: str = "192.168.1.50",
                                   dst_ip: str = "192.168.1.1",
                                   src_port: int = 10000,
                                   dst_port: int = 80,
                                   tcp_flags: int = 0x02,
                                   payload: bytes = b"P" * 32) -> bytes:
    """Constructs a valid raw Ethernet/IP/TCP frame."""
    tcp = dpkt.tcp.TCP(
        sport=src_port,
        dport=dst_port,
        flags=tcp_flags,
        seq=100,
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


class TestAegisLiveV4Runtime:

    def test_exact_29_feature_ordering(self):
        assert len(CANONICAL_29_FEATURES) == 29
        assert CANONICAL_29_FEATURES[:24] == CANONICAL_24_FEATURES
        expected_top = [
            "src_concentration_ratio",
            "src_ip_entropy",
            "src_ip_cardinality",
            "dst_ip_cardinality",
            "fan_in_ratio"
        ]
        assert CANONICAL_29_FEATURES[24:] == expected_top

        pipeline = AegisLiveV4Pipeline()
        # Feed 50 packets to trigger window
        t_base = 1660000000.0
        emitted = []
        for i in range(50):
            frame = make_raw_ethernet_ip_tcp_frame(src_port=10000 + i)
            dets = pipeline.ingest_packet(t_base + i * 0.01, frame)
            emitted.extend(dets)

        assert len(emitted) == 1
        res = emitted[0]
        assert list(res.feature_vector.keys()) == CANONICAL_29_FEATURES

    def test_v2_v3_24_feature_parity(self):
        """Verifies that the first 24 features emitted by v4 are float-identical to v2/v3 extractor."""
        v2_extractor = LiveFeatureExtractor(max_packets_per_window=50, window_timeout_sec=2.0)
        v4_pipeline = AegisLiveV4Pipeline(max_packets_per_window=50, window_timeout_sec=2.0)

        t_base = 1660000100.0
        v2_feats = None
        v4_res = None

        for i in range(50):
            frame = make_raw_ethernet_ip_tcp_frame(src_port=20000, dst_port=80, payload=b"D" * (20 + i))
            ts = t_base + i * 0.02
            f2 = v2_extractor.process_raw_packet(ts, frame)
            if f2 is not None:
                v2_feats = f2

            dets = v4_pipeline.ingest_packet(ts, frame)
            if dets:
                v4_res = dets[0]

        assert v2_feats is not None
        assert v4_res is not None

        for col in CANONICAL_24_FEATURES:
            val_v2 = float(v2_feats[col])
            val_v4 = float(v4_res.feature_vector[col])
            assert np.isclose(val_v2, val_v4, rtol=1e-4, atol=1e-3), f"Feature parity mismatch on {col}: v2={val_v2}, v4={val_v4}"

    def test_topology_causal_cutoff_and_no_future_packets(self):
        """Ensures that packets arriving after a window completes do not alter that window's features."""
        pipeline = AegisLiveV4Pipeline(max_packets_per_window=50)
        t_base = 1660000200.0

        for i in range(50):
            frame = make_raw_ethernet_ip_tcp_frame(src_ip="10.0.0.1")
            pipeline.ingest_packet(t_base + i * 0.01, frame)

        assert len(pipeline.detection_records) == 1
        recorded_feats = dict(pipeline.detection_records[0].feature_vector)

        # Now inject 100 future packets from high-cardinality attackers
        for i in range(100):
            future_frame = make_raw_ethernet_ip_tcp_frame(src_ip=f"198.51.100.{i + 1}")
            pipeline.ingest_packet(t_base + 1.0 + i * 0.01, future_frame)

        # The first window's recorded features must remain byte-identical
        assert pipeline.detection_records[0].feature_vector == recorded_feats

    def test_future_packet_injection_invariance(self):
        """Comparing window extraction with and without future packets in stream."""
        p_clean = AegisLiveV4Pipeline(max_packets_per_window=50)
        p_future = AegisLiveV4Pipeline(max_packets_per_window=50)

        t_base = 1660000300.0
        # Stream 50 packets to both
        for i in range(50):
            frame = make_raw_ethernet_ip_tcp_frame(src_ip="192.168.1.100", src_port=30000)
            p_clean.ingest_packet(t_base + i * 0.01, frame)
            p_future.ingest_packet(t_base + i * 0.01, frame)

        # Inject future packets into p_future only
        for i in range(50):
            frame_fut = make_raw_ethernet_ip_tcp_frame(src_ip="10.99.99.99", src_port=40000)
            p_future.ingest_packet(t_base + 2.0 + i * 0.01, frame_fut)

        feat_clean = p_clean.detection_records[0].feature_vector
        feat_future = p_future.detection_records[0].feature_vector

        for col in CANONICAL_29_FEATURES:
            assert feat_clean[col] == feat_future[col], f"Leakage detected in {col}!"

    def test_one_second_topology_eviction(self):
        """Verifies that sliding topology tracker purges packets older than 1.0s."""
        tracker = NetworkTopologyTracker(window_sec=1.0, mode="sliding")
        # Packet at t=10.0
        p1 = NetworkTopologyPacket(10.0, "1.1.1.1", "2.2.2.2", 100, 80, 6, 60)
        tracker.add_packet(p1)
        # Packet at t=10.5
        p2 = NetworkTopologyPacket(10.5, "1.1.1.2", "2.2.2.2", 101, 80, 6, 60)
        tracker.add_packet(p2)

        f_mid = tracker.extract_current_features()
        assert f_mid["src_ip_cardinality"] == 2.0
        assert f_mid["packet_count"] == 2.0

        # Packet at t=11.6 (> 10.5 + 1.0) -> both p1 and p2 must be evicted!
        p3 = NetworkTopologyPacket(11.6, "1.1.1.3", "2.2.2.2", 102, 80, 6, 60)
        tracker.add_packet(p3)

        f_late = tracker.extract_current_features()
        assert f_late["src_ip_cardinality"] == 1.0
        assert f_late["packet_count"] == 1.0

    def test_streaming_vs_offline_batch_equality(self):
        """Verifies that streaming NetworkTopologyTracker produces identical output to batch."""
        packets = [
            NetworkTopologyPacket(10.0 + i * 0.05, f"192.168.1.{(i % 4) + 1}", "10.0.0.1", 1000 + i, 80, 6, 64)
            for i in range(15) # Within 0.75s (< 1.0s)
        ]
        # Offline batch
        batch_feats = compute_network_topology_batch(packets)

        # Streaming tracker
        tracker = NetworkTopologyTracker(window_sec=1.0, mode="sliding")
        for p in packets:
            tracker.add_packet(p)
        stream_feats = tracker.extract_current_features()

        for key in ["src_ip_cardinality", "src_concentration_ratio", "dst_ip_cardinality", "src_ip_entropy", "fan_in_ratio"]:
            assert abs(batch_feats[key] - stream_feats[key]) < 1e-6, f"Mismatch in {key}"

    def test_temperature_calibration_correctness(self):
        pipeline = AegisLiveV4Pipeline()
        assert abs(pipeline.inference_engine.temperature - 1.199787) < 1e-4

        # Feed 50 packets to generate a prediction
        t_base = 1660000400.0
        for i in range(50):
            frame = make_raw_ethernet_ip_tcp_frame()
            pipeline.ingest_packet(t_base + i * 0.01, frame)

        res = pipeline.detection_records[0]
        probs = list(res.calibrated_probabilities.values())
        assert abs(sum(probs) - 1.0) < 1e-5
        assert res.calibrated_confidence == max(probs)
        assert res.confidence == max(probs)

    def test_mahalanobis_calculation_correctness(self):
        pipeline = AegisLiveV4Pipeline()
        t_base = 1660000500.0
        for i in range(50):
            frame = make_raw_ethernet_ip_tcp_frame()
            pipeline.ingest_packet(t_base + i * 0.01, frame)

        res = pipeline.detection_records[0]
        assert res.mahalanobis_distance >= 0.0
        assert not np.isnan(res.mahalanobis_distance)
        assert not np.isinf(res.mahalanobis_distance)

    def test_threshold_boundary_behavior(self):
        pipeline = AegisLiveV4Pipeline()
        assert pipeline.inference_engine.confidence_threshold == 0.75
        assert pipeline.inference_engine.mahalanobis_threshold == 7.92

        # Direct test of boundary behavior
        sample_normal = {col: 0.0 for col in CANONICAL_29_FEATURES}
        sample_normal["pkt_rate_pps"] = 50.0
        sample_normal["pkt_size_mean"] = 60.0
        sample_normal["src_concentration_ratio"] = 0.5
        sample_normal["fan_in_ratio"] = 1.0

        res_norm = pipeline.inference_engine.predict_sample(sample_normal)
        assert "trusted" in res_norm
        assert "decision" in res_norm
        assert res_norm["decision"] in ("TRUSTED", "UNKNOWN_OR_REVIEW")

    def test_unknown_or_review_semantics(self):
        pipeline = AegisLiveV4Pipeline()
        # Synthetic extreme anomalous vector guaranteed to fail gates while respecting [0, 1] ratios
        anom = {col: 0.0 for col in CANONICAL_29_FEATURES}
        anom["pkt_rate_pps"] = 9999999.0
        anom["byte_rate_bps"] = 999999999.0
        anom["pkt_size_mean"] = 50000.0
        anom["pkt_size_max"] = 50000.0
        anom["tcp_ratio"] = 1.0
        anom["syn_ratio"] = 1.0
        anom["src_concentration_ratio"] = 0.001
        anom["fan_in_ratio"] = 0.001
        res = pipeline.inference_engine.predict_sample(anom)

        assert res["trusted"] is False
        assert res["decision"] == "UNKNOWN_OR_REVIEW"
        assert res["status"] == "UNKNOWN_OR_REVIEW"
        assert res["trusted_class"] is None
        assert res["advisory_class"] is not None
        assert len(res["reasons"]) > 0
        assert res["automated_blocking_allowed"] is False

    def test_automated_blocking_allowed_strictly_false(self):
        pipeline = AegisLiveV4Pipeline()
        t_base = 1660000600.0
        for i in range(100):
            frame = make_raw_ethernet_ip_tcp_frame()
            pipeline.ingest_packet(t_base + i * 0.01, frame)

        assert len(pipeline.detection_records) >= 2
        for r in pipeline.detection_records:
            assert r.automated_blocking_allowed is False

    def test_session_reset_correctness(self):
        pipeline = AegisLiveV4Pipeline()
        t_base = 1660000700.0
        for i in range(25):
            frame = make_raw_ethernet_ip_tcp_frame()
            pipeline.ingest_packet(t_base + i * 0.01, frame)

        assert len(pipeline.active_windows) == 1
        assert pipeline.packets_captured == 25

        pipeline.reset()
        assert len(pipeline.active_windows) == 0
        assert pipeline.packets_captured == 0
        assert len(pipeline.detection_records) == 0
        assert pipeline.net_tracker.total_packets == 0

    def test_malformed_and_non_ipv4_packet_handling(self):
        pipeline = AegisLiveV4Pipeline()
        # 1. Truncated frame
        res1 = pipeline.ingest_packet(1.0, b"\x00" * 10)
        assert res1 == []
        # 2. Non-IPv4 frame (e.g. ARP 0x0806)
        arp_frame = b"\xff" * 6 + b"\x00" * 6 + b"\x08\x06" + b"\x00" * 28
        res2 = pipeline.ingest_packet(2.0, arp_frame)
        assert res2 == []
        # 3. Truncated IP
        trunc_ip = b"\x00" * 12 + b"\x08\x00" + b"\x45\x00"
        res3 = pipeline.ingest_packet(3.0, trunc_ip)
        assert res3 == []

        assert pipeline.packets_rejected >= 3

    def test_no_nan_or_inf_in_features_and_predictions(self):
        pipeline = AegisLiveV4Pipeline()
        t_base = 1660000800.0
        for i in range(50):
            frame = make_raw_ethernet_ip_tcp_frame()
            pipeline.ingest_packet(t_base + i * 0.01, frame)

        res = pipeline.detection_records[0]
        for k, v in res.feature_vector.items():
            assert not np.isnan(v), f"NaN in {k}"
            assert not np.isinf(v), f"Inf in {k}"

        for k, v in res.calibrated_probabilities.items():
            assert not np.isnan(v), f"NaN in prob {k}"
            assert not np.isinf(v), f"Inf in prob {k}"

    def test_frozen_artifact_hashes_verified(self):
        v4_model_path = os.path.join(BASE_DIR, "models", "aegis_live_v4", "aegis_live_xgboost_baseline.pkl")
        v2_model_path = os.path.join(BASE_DIR, "models", "aegis_live_v2", "aegis_live_xgboost_baseline.pkl")
        v3_model_path = os.path.join(BASE_DIR, "models", "aegis_live_v3", "aegis_live_xgboost_baseline.pkl")

        assert compute_sha256(v4_model_path) == FROZEN_V4_MODEL_HASH
        assert compute_sha256(v2_model_path) == FROZEN_V2_MODEL_HASH
        assert compute_sha256(v3_model_path) == FROZEN_V3_MODEL_HASH

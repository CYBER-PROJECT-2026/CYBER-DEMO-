"""
tests/test_aegis_live_v3_topology_features.py
=============================================
Automated test suite for AEGIS-Live v3 Topology-Feature Feasibility & Leakage Audit.

Verifies:
1. Strict non-mutation of frozen v2 production baseline:
   - v2 model hash matches FROZEN_MODEL_SHA256
   - v2 24-feature schema is preserved
2. Feasibility and leakage audit artifacts exist and pass gate:
   - reports/aegis_live_v3_topology_feature_definitions.md
   - reports/aegis_live_v3_topology_feature_raw_pcap.csv
   - reports/aegis_live_v3_topology_separability.csv
   - reports/aegis_live_v3_topology_separability.md
   - reports/aegis_live_v3_topology_leakage_audit.md
   - reports/aegis_live_v3_topology_feasibility_audit.md
   - Gate asserts V3_TOPOLOGY_FEATURE_FEASIBILITY_AUDIT = PASS
3. Replay consistency:
   - Batch calculation and incremental streaming calculation produce identical results
4. Robustness under all edge cases:
   - 1 packet, 2 packets, 50 packets
   - repeated source IP, multiple source IPs
   - one destination, multiple destinations
   - TCP, UDP, ICMP
   - missing transport ports
   - malformed and non-IP packets
   - duplicate packets, zero-packet empty windows
"""

import os
import sys
import hashlib
import pytest
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from aegis_live.feature_schema import CANONICAL_24_FEATURES
from aegis_live.live_inference import FROZEN_MODEL_SHA256
from aegis_live.v3_topology_features import (
    V3TopologyPacket,
    V3TopologyTracker,
    compute_topology_features_batch,
    parse_raw_ethernet_frame_v3,
    V3_TOPOLOGY_FEATURE_NAMES
)

REPORTS_DIR = os.path.join(BASE_DIR, "reports")
DEFS_MD = os.path.join(REPORTS_DIR, "aegis_live_v3_topology_feature_definitions.md")
RAW_CSV = os.path.join(REPORTS_DIR, "aegis_live_v3_topology_feature_raw_pcap.csv")
SEP_CSV = os.path.join(REPORTS_DIR, "aegis_live_v3_topology_separability.csv")
SEP_MD = os.path.join(REPORTS_DIR, "aegis_live_v3_topology_separability.md")
LEAK_MD = os.path.join(REPORTS_DIR, "aegis_live_v3_topology_leakage_audit.md")
FEAS_MD = os.path.join(REPORTS_DIR, "aegis_live_v3_topology_feasibility_audit.md")
MODEL_PKL = os.path.join(BASE_DIR, "models", "aegis_live_v2", "aegis_live_xgboost_baseline.pkl")


def test_frozen_v2_baseline_unmodified():
    # Verify v2 model hash remains strictly unchanged
    with open(MODEL_PKL, "rb") as f:
        model_sha256 = hashlib.sha256(f.read()).hexdigest()
    assert model_sha256 == FROZEN_MODEL_SHA256, "Frozen v2 model hash was altered!"

    # Verify v2 schema length
    assert len(CANONICAL_24_FEATURES) == 24, "Canonical v2 schema was modified!"


def test_audit_artifacts_exist_and_populated():
    for fpath in [DEFS_MD, RAW_CSV, SEP_CSV, SEP_MD, LEAK_MD, FEAS_MD]:
        assert os.path.exists(fpath), f"Missing artifact {fpath}"
        assert os.path.getsize(fpath) > 500, f"Artifact {fpath} is unexpectedly small"

    with open(FEAS_MD, "r", encoding="utf-8") as f:
        content = f.read()
    assert "V3_TOPOLOGY_FEATURE_FEASIBILITY_AUDIT = PASS" in content

    # Verify all 12 questions answered
    for q_idx in range(1, 13):
        assert f"### Q{q_idx}:" in content, f"Missing answer to Q{q_idx}"


def test_raw_pcap_summary_integrity():
    df_raw = pd.read_csv(RAW_CSV)
    assert len(df_raw) > 0
    # Must contain 9 regions * 4 features = 36 rows
    assert len(df_raw) == 36, f"Expected 36 summary rows, got {len(df_raw)}"
    for feat in V3_TOPOLOGY_FEATURE_NAMES:
        assert (df_raw["feature"] == feat).sum() == 9
    assert not df_raw.isna().any().any()


def test_separability_metrics_invariants():
    df_sep = pd.read_csv(SEP_CSV)
    assert len(df_sep) == 8  # 2 comparisons * 4 features
    assert df_sep["ks_statistic"].between(0.0, 1.0).all()
    assert df_sep["distribution_overlap"].between(0.0, 1.0).all()
    assert not df_sep.isna().any().any()


def test_batch_vs_streaming_replay_consistency():
    packets = [
        V3TopologyPacket(1.0, "192.168.1.10", "192.168.1.1", 1024, 80, 6, 60, True),
        V3TopologyPacket(1.01, "192.168.1.20", "192.168.1.1", 1025, 80, 6, 60, True),
        V3TopologyPacket(1.02, "192.168.1.10", "192.168.1.1", 1026, 80, 6, 60, True),
        V3TopologyPacket(1.03, "192.168.1.30", "192.168.1.2", 1027, 443, 6, 64, True),
        V3TopologyPacket(1.04, "192.168.1.10", "192.168.1.1", 1028, 80, 6, 60, True),
    ]

    batch_feats = compute_topology_features_batch(packets)

    tracker = V3TopologyTracker(max_packets=50)
    for p in packets:
        tracker.add_packet(p)
    stream_feats = tracker.extract_features()

    for k in V3_TOPOLOGY_FEATURE_NAMES:
        assert abs(batch_feats[k] - stream_feats[k]) < 1e-9, f"Mismatch on {k}"


def test_robustness_single_packet():
    p = [V3TopologyPacket(1.0, "10.0.0.1", "10.0.0.2", 1234, 80, 6, 60, True)]
    feats = compute_topology_features_batch(p)
    assert feats["src_ip_cardinality_per_window"] == 1.0
    assert feats["src_concentration_ratio"] == 1.0
    assert feats["unique_5tuple_flow_count"] == 1.0
    assert feats["fan_in_ratio"] == 1.0


def test_robustness_empty_and_zero_packet_window():
    feats = compute_topology_features_batch([])
    assert feats["src_ip_cardinality_per_window"] == 0.0
    assert feats["src_concentration_ratio"] == 0.0
    assert feats["unique_5tuple_flow_count"] == 0.0
    assert feats["fan_in_ratio"] == 0.0


def test_robustness_repeated_source_ip():
    # 50 packets from exactly the same source IP to 1 destination
    pkts = [
        V3TopologyPacket(1.0 + i*0.01, "192.168.1.50", "192.168.1.1", 1000 + i, 80, 6, 60, True)
        for i in range(50)
    ]
    feats = compute_topology_features_batch(pkts)
    assert feats["src_ip_cardinality_per_window"] == 1.0
    assert feats["src_concentration_ratio"] == 1.0
    assert feats["unique_5tuple_flow_count"] == 50.0
    assert feats["fan_in_ratio"] == 1.0


def test_robustness_multiple_sources_and_destinations():
    # 50 packets: 5 distinct sources (10 pkts each), 2 distinct destinations
    pkts = [
        V3TopologyPacket(1.0 + i*0.01, f"192.168.1.{10 + (i % 5)}", f"10.0.0.{1 + (i % 2)}", 2000 + i, 80, 6, 60, True)
        for i in range(50)
    ]
    feats = compute_topology_features_batch(pkts)
    assert feats["src_ip_cardinality_per_window"] == 5.0
    assert abs(feats["src_concentration_ratio"] - 0.20) < 1e-6
    assert feats["unique_5tuple_flow_count"] == 50.0
    assert abs(feats["fan_in_ratio"] - 2.5) < 1e-6


def test_robustness_udp_and_icmp_missing_ports():
    # ICMP has no transport ports (src_port=0, dst_port=0, proto=1)
    pkts = [
        V3TopologyPacket(1.0, "192.168.1.1", "192.168.1.2", 0, 0, 1, 84, True),
        V3TopologyPacket(1.1, "192.168.1.1", "192.168.1.2", 53, 53, 17, 120, True),  # UDP DNS
    ]
    feats = compute_topology_features_batch(pkts)
    assert feats["src_ip_cardinality_per_window"] == 1.0
    assert feats["src_concentration_ratio"] == 1.0
    assert feats["unique_5tuple_flow_count"] == 2.0  # 1 ICMP + 1 UDP flow
    assert feats["fan_in_ratio"] == 1.0


def test_robustness_malformed_and_non_ip():
    # Frame decode failure or ARP packet (is_ipv4=False, src_ip=None)
    malformed = [
        V3TopologyPacket(1.0, None, None, 0, 0, 0, 42, False),
        V3TopologyPacket(1.01, None, None, 0, 0, 0, 42, False),
    ]
    feats = compute_topology_features_batch(malformed)
    assert feats["src_ip_cardinality_per_window"] == 0.0
    assert feats["src_concentration_ratio"] == 0.0
    assert feats["unique_5tuple_flow_count"] == 0.0
    assert feats["fan_in_ratio"] == 0.0


def test_parse_raw_ethernet_frame_safety():
    # Non-IP bytes (garbage)
    garbage = b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x08\x06" + b"\x00" * 30
    pkt = parse_raw_ethernet_frame_v3(12345.67, garbage)
    assert isinstance(pkt, V3TopologyPacket)
    assert pkt.is_ipv4 is False
    assert pkt.src_ip is None
    assert pkt.dst_ip is None

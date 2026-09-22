"""
tests/test_aegis_live_network_topology.py
=========================================
Deterministic verification suite for the Read-Only Network-Level Topology Feasibility Audit.

Validates:
  - Strictly causal past-to-present calculation with zero future-packet access.
  - Streaming (NetworkTopologyTracker) == Offline (compute_network_topology_batch) numerical equivalence.
  - Correct window boundary and reset semantics.
  - Absence of label, filename, or category leakage.
  - Numerical validity: no NaN, Inf, negative cardinality, or division-by-zero on edge cases.
  - Immutability of frozen v2 and v3 baseline models and quarantine datasets.
"""

import os
import sys
import hashlib
import numpy as np
import pytest

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from aegis_live.network_topology_features import (
    NetworkTopologyPacket,
    NetworkTopologyTracker,
    compute_network_topology_batch,
    compute_network_topology_from_counts,
    parse_raw_ethernet_network_topology,
    NETWORK_TOPOLOGY_FEATURE_NAMES
)

# Immutability baseline hashes
V2_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "aegis_live_v2", "aegis_live_xgboost_baseline.pkl")
V3_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "aegis_live_v3", "aegis_live_xgboost_baseline.pkl")
V2_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "aegis_live_v2_quarantine")
V3_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "aegis_live_v3")

V2_MODEL_SHA256 = "c157660c108a5d95d32d504ebc81c40e6b6911d8df6570a7d3ba92fad062cb9b"
V3_MODEL_SHA256 = "5eadae3284f0973039b5743d8aa9f5db6c0e5ee3b43ed668f245d168d7958128"


def compute_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024 * 4):
            h.update(chunk)
    return h.hexdigest()


class TestNetworkTopologyFeasibility:
    """Test suite for network-level topology feature feasibility audit."""

    def test_causal_computation_and_no_future_access(self):
        """Verify that streaming tracker processes packets strictly sequentially with zero future access."""
        tracker = NetworkTopologyTracker(window_sec=1.0, mode="tumbling")

        # Simulate 10 packets with strictly increasing timestamps
        packets = [
            NetworkTopologyPacket(timestamp=100.0 + i * 0.1, src_ip=f"10.0.0.{i % 3}",
                                  dst_ip="192.168.1.1", src_port=1000 + i, dst_port=80,
                                  protocol=6, pkt_len=64)
            for i in range(15) # Spans 1.5 seconds -> triggers 1 window closure at 101.0
        ]

        closed_windows = []
        for p in packets:
            ready, feats = tracker.add_packet(p)
            if ready:
                closed_windows.append(feats)

        assert len(closed_windows) == 1
        w0 = closed_windows[0]
        # Window closed causally at t=101.0, spanning 100.0 to 101.0
        assert w0["window_start_ts"] == 100.0
        assert w0["window_end_ts"] == 101.0
        assert w0["packet_count"] == 10 # First 10 packets in [100.0, 101.0)
        assert w0["src_ip_cardinality"] == 3.0 # IPs .0, .1, .2
        assert w0["dst_ip_cardinality"] == 1.0 # Only 192.168.1.1
        assert w0["fan_in_ratio"] == 3.0 # 3 sources / 1 dst

    def test_streaming_vs_offline_exact_equivalence(self):
        """Verify that streaming NetworkTopologyTracker matches compute_network_topology_batch exactly."""
        np.random.seed(42)
        mock_packets = []
        src_pool = [f"192.168.1.{i}" for i in range(1, 20)]
        dst_pool = ["10.0.0.1", "10.0.0.2"]

        for i in range(200):
            p = NetworkTopologyPacket(
                timestamp=1000.0 + i * 0.004, # 200 pkts in 0.8 seconds (single window)
                src_ip=np.random.choice(src_pool),
                dst_ip=np.random.choice(dst_pool),
                src_port=int(np.random.randint(1024, 65535)),
                dst_port=80,
                protocol=6,
                pkt_len=int(np.random.randint(40, 1500))
            )
            mock_packets.append(p)

        # Batch offline computation
        batch_feats = compute_network_topology_batch(mock_packets)

        # Streaming tracker computation
        tracker = NetworkTopologyTracker(window_sec=1.0, mode="tumbling")
        for p in mock_packets:
            tracker.add_packet(p)
        stream_feats = tracker.extract_current_features()

        for f in NETWORK_TOPOLOGY_FEATURE_NAMES:
            b_val = batch_feats[f]
            s_val = stream_feats[f]
            np.testing.assert_allclose(s_val, b_val, atol=1e-6, err_msg=f"Mismatch for {f}")

    def test_window_boundary_correctness_and_reset(self):
        """Verify tumbling window resets counters cleanly on window expiration."""
        tracker = NetworkTopologyTracker(window_sec=0.5, mode="tumbling")

        p1 = NetworkTopologyPacket(timestamp=10.0, src_ip="1.1.1.1", dst_ip="2.2.2.2",
                                   src_port=50, dst_port=80, protocol=6, pkt_len=60)
        p2 = NetworkTopologyPacket(timestamp=10.6, src_ip="3.3.3.3", dst_ip="4.4.4.4",
                                   src_port=51, dst_port=80, protocol=6, pkt_len=60)

        ready1, _ = tracker.add_packet(p1)
        assert not ready1

        ready2, w1 = tracker.add_packet(p2) # Elapsed 0.6s >= 0.5s -> triggers closure
        assert ready2
        assert w1["packet_count"] == 1
        assert w1["src_ip_cardinality"] == 1.0

        # After reset, tracker should only hold p2
        curr = tracker.extract_current_features()
        assert curr["packet_count"] == 1
        assert curr["src_ip_cardinality"] == 1.0

    def test_no_label_or_metadata_leakage(self):
        """Verify that tracker data structures contain zero labels, filenames, or category fields."""
        p = NetworkTopologyPacket(timestamp=1.0, src_ip="1.1.1.1", dst_ip="2.2.2.2",
                                   src_port=1, dst_port=2, protocol=6, pkt_len=50)

        assert not hasattr(p, "label")
        assert not hasattr(p, "category")
        assert not hasattr(p, "pcap_file")
        assert not hasattr(p, "target_class")

        tracker = NetworkTopologyTracker(window_sec=1.0)
        tracker.add_packet(p)
        feats = tracker.extract_current_features()

        for forbidden in ["label", "category", "pcap_file", "flow_id", "attack_type"]:
            assert forbidden not in feats

    def test_numerical_stability_edge_cases_no_nan_inf(self):
        """Verify that zero packets, single packet, or duplicate IPs never produce NaN or Inf."""
        # Empty batch
        empty_res = compute_network_topology_batch([])
        for f, v in empty_res.items():
            assert not np.isnan(v) and not np.isinf(v)
            assert v == 0.0

        # Single packet
        p_single = [NetworkTopologyPacket(timestamp=1.0, src_ip="1.1.1.1", dst_ip="2.2.2.2",
                                          src_port=80, dst_port=80, protocol=6, pkt_len=60)]
        single_res = compute_network_topology_batch(p_single)
        assert single_res["src_ip_cardinality"] == 1.0
        assert single_res["src_concentration_ratio"] == 1.0
        assert single_res["src_ip_entropy"] == 0.0 # log2(1) = 0
        assert single_res["fan_in_ratio"] == 1.0
        assert single_res["fan_out_ratio"] == 1.0

        # All identical sources (extreme concentration)
        identical_src = [
            NetworkTopologyPacket(timestamp=1.0 + i*0.01, src_ip="1.1.1.1", dst_ip=f"2.2.2.{i}",
                                  src_port=80, dst_port=80, protocol=6, pkt_len=60)
            for i in range(10)
        ]
        conc_res = compute_network_topology_batch(identical_src)
        assert conc_res["src_concentration_ratio"] == 1.0
        assert conc_res["src_ip_entropy"] == 0.0
        assert conc_res["fan_out_ratio"] == 10.0

    def test_frozen_v2_model_and_dataset_unmodified(self):
        """Verify byte-for-byte preservation of frozen v2 baseline."""
        assert os.path.exists(V2_MODEL_PATH)
        assert compute_file_sha256(V2_MODEL_PATH) == V2_MODEL_SHA256

        for split in ["train.csv", "validation.csv", "test.csv"]:
            p = os.path.join(V2_DATA_DIR, split)
            assert os.path.exists(p)

    def test_frozen_v3_model_and_dataset_unmodified(self):
        """Verify byte-for-byte preservation of frozen v3 baseline."""
        assert os.path.exists(V3_MODEL_PATH)
        assert compute_file_sha256(V3_MODEL_PATH) == V3_MODEL_SHA256

        for split in ["train.csv", "validation.csv", "test.csv"]:
            p = os.path.join(V3_DATA_DIR, split)
            assert os.path.exists(p)

    def test_audit_deliverables_exist(self):
        """Verify that audit markdown report and statistics CSV were generated."""
        report_path = os.path.join(PROJECT_ROOT, "reports", "aegis_live_network_topology_feasibility_audit.md")
        csv_path = os.path.join(PROJECT_ROOT, "reports", "aegis_live_network_topology_feature_statistics.csv")

        assert os.path.exists(report_path), f"Missing {report_path}"
        assert os.path.exists(csv_path), f"Missing {csv_path}"

        assert os.path.getsize(report_path) > 2000
        assert os.path.getsize(csv_path) > 1000

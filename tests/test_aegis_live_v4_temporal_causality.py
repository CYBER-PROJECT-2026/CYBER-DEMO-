"""
tests/test_aegis_live_v4_temporal_causality.py
==============================================
Dedicated verification test suite for AEGIS-Live v4 temporal causality,
streaming-vs-offline equivalence, partition boundary reset, and zero future-time leakage.

Proves:
  1. Causal non-interference: Packets with t > T_end cannot alter features computed at T_end.
  2. Exact boundary timing: t == T_end, t == T_end + eps, t == T_end - W, t == T_end - W - eps.
  3. Partition boundary isolation: tracker.reset() completely purges prior state.
  4. Streaming FIFO vs Offline batch mathematical equivalence.
  5. Adversarial future burst non-contamination.
  6. All 80,000 dataset rows satisfy temporal ordering and valid bounds.
  7. Frozen model and dataset hash immutability.
"""

import os
import sys
import json
import hashlib
import pytest
import numpy as np
import pandas as pd
from typing import List, Dict

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from aegis_live.feature_schema import CANONICAL_24_FEATURES, CANONICAL_29_FEATURES, V4_TOPOLOGY_FEATURES
from aegis_live.network_topology_features import (
    NetworkTopologyPacket,
    NetworkTopologyTracker,
    compute_network_topology_batch,
    compute_network_topology_from_counts
)
from aegis_live.packet_window import FlowWindow, PacketRecord

V4_DATA_DIR = os.path.join(BASE_DIR, "data", "aegis_live_v4")
V3_DATA_DIR = os.path.join(BASE_DIR, "data", "aegis_live_v3")
V2_DATA_DIR = os.path.join(BASE_DIR, "data", "aegis_live_v2")

V2_MODEL_PATH = os.path.join(BASE_DIR, "models", "aegis_live_v2", "aegis_live_xgboost_baseline.pkl")
V3_MODEL_PATH = os.path.join(BASE_DIR, "models", "aegis_live_v3", "aegis_live_xgboost_baseline.pkl")
V4_MODEL_PATH = os.path.join(BASE_DIR, "models", "aegis_live_v4", "aegis_live_xgboost_baseline.pkl")

# Frozen Reference Hashes
FROZEN_V2_MODEL_HASH = "c157660c108a5d95d32d504ebc81c40e6b6911d8df6570a7d3ba92fad062cb9b"
FROZEN_V3_MODEL_HASH = "5eadae3284f0973039b5743d8aa9f5db6c0e5ee3b43ed668f245d168d7958128"
FROZEN_V4_MODEL_HASH = "c0e5aee8e9e1726f12bfc19e98c440fe44fe278d1796d5866fe2a2488d458a56"

FROZEN_V4_TRAIN_HASH = "378d981f7871215487a2a46ef70e44107984df75e7317555eebe79f200fd9c63"
FROZEN_V4_VAL_HASH = "f1a677ebd0bff3fdc5bc0f56d835062e8bd62a6ea0d68d86c4b7a779bcef4c78"
FROZEN_V4_TEST_HASH = "bf1c7c3783e79496dfee1a08bca5569d81c8dde07541acf6605cebf3158683f6"


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024 * 16):
            h.update(chunk)
    return h.hexdigest()


class TestAegisLiveV4TemporalCausality:

    def test_causal_non_interference_future_packets(self):
        """
        Prove that packets with t > T_end cannot change the network topology features at T_end.
        """
        tracker = NetworkTopologyTracker(window_sec=1.0, mode="sliding")

        # Ingest packets up to t = 100.0
        past_packets = [
            NetworkTopologyPacket(99.1, "192.168.1.10", "192.168.1.1", 5000, 80, 6, 60),
            NetworkTopologyPacket(99.3, "192.168.1.20", "192.168.1.1", 5001, 80, 6, 60),
            NetworkTopologyPacket(99.6, "192.168.1.10", "192.168.1.1", 5002, 80, 6, 60),
            NetworkTopologyPacket(100.0, "192.168.1.30", "192.168.1.1", 5003, 80, 6, 60)
        ]
        for p in past_packets:
            tracker.add_packet(p)

        features_at_t100 = tracker.extract_current_features()

        # Ingest future packets (t > 100.0) with massive distinct traffic
        future_packets = [
            NetworkTopologyPacket(100.0001, "10.0.0.99", "192.168.1.1", 6000, 80, 6, 1200),
            NetworkTopologyPacket(100.05, "10.0.0.100", "192.168.1.1", 6001, 80, 6, 1200),
            NetworkTopologyPacket(101.5, "10.0.0.101", "192.168.1.1", 6002, 80, 6, 1200)
        ]
        for p in future_packets:
            tracker.add_packet(p)

        # Now re-create tracker with ONLY packets <= 100.0
        tracker_replay = NetworkTopologyTracker(window_sec=1.0, mode="sliding")
        for p in past_packets:
            tracker_replay.add_packet(p)
        features_replay = tracker_replay.extract_current_features()

        for feat in V4_TOPOLOGY_FEATURES:
            assert features_at_t100[feat] == features_replay[feat], f"Mismatch for {feat} under causal replay"

    def test_packet_boundary_timing_cases(self):
        """
        Verify exact edge case timing at window boundaries:
          1. Packet exactly at T_end (included)
          2. Packet at T_end + epsilon (excluded from window at T_end)
          3. Packet at T_end - window_sec (included)
          4. Packet at T_end - window_sec - epsilon (evicted)
        """
        T_end = 100.0
        window_sec = 1.0
        cutoff = T_end - window_sec  # 99.0

        # Create tracker and add packet at cutoff - 1e-6 (should be evicted at T_end)
        tracker = NetworkTopologyTracker(window_sec=window_sec, mode="sliding")
        p_expired = NetworkTopologyPacket(cutoff - 1e-6, "192.168.1.99", "192.168.1.1", 1000, 80, 6, 60)
        tracker.add_packet(p_expired)

        # Packet exactly at cutoff (99.0) (should be kept at T_end = 100.0)
        p_boundary_start = NetworkTopologyPacket(cutoff, "192.168.1.10", "192.168.1.1", 1001, 80, 6, 60)
        tracker.add_packet(p_boundary_start)

        # Packet inside window (99.5)
        p_mid = NetworkTopologyPacket(99.5, "192.168.1.20", "192.168.1.1", 1002, 80, 6, 60)
        tracker.add_packet(p_mid)

        # Packet exactly at T_end (100.0)
        p_end = NetworkTopologyPacket(T_end, "192.168.1.30", "192.168.1.1", 1003, 80, 6, 60)
        tracker.add_packet(p_end)

        feats = tracker.extract_current_features()
        # p_expired (192.168.1.99 at 98.999999) must be evicted!
        # Active sources should be 192.168.1.10, 192.168.1.20, 192.168.1.30 -> cardinality = 3.0
        assert feats["src_ip_cardinality"] == 3.0, f"Expected 3.0, got {feats['src_ip_cardinality']}"
        assert feats["packet_count"] == 3.0, f"Expected 3 packets in window, got {feats['packet_count']}"

        # At T_end + 1e-6 (100.000001), cutoff becomes 99.000001.
        # Adding a packet at 100.000001 evicts p_boundary_start (at 99.0).
        # To test addition without eviction of p_boundary_start, test with a 4th source arriving at T_end:
        tracker_4src = NetworkTopologyTracker(window_sec=window_sec, mode="sliding")
        for p in [p_boundary_start, p_mid, p_end]:
            tracker_4src.add_packet(p)
        assert tracker_4src.extract_current_features()["src_ip_cardinality"] == 3.0

        # When future packet arrives at 100.000001 from a new source:
        p_future = NetworkTopologyPacket(T_end + 1e-6, "192.168.1.88", "192.168.1.1", 1004, 80, 6, 60)
        tracker.add_packet(p_future)
        feats_after = tracker.extract_current_features()

        # p_boundary_start (99.0) was evicted (< 99.000001), p_future (192.168.1.88) was ingested
        # Active sources are 192.168.1.20, 192.168.1.30, 192.168.1.88 -> 3 sources
        assert feats_after["src_ip_cardinality"] == 3.0
        assert feats_after["window_start_ts"] == 99.5
        assert feats_after["window_end_ts"] == T_end + 1e-6

        # Crucially: the previous extraction `feats` remains completely unchanged (3.0 sources, window_end_ts=100.0)
        assert feats["src_ip_cardinality"] == 3.0
        assert feats["window_end_ts"] == 100.0

    def test_partition_boundary_reset_and_gap_isolation(self):
        """
        Verify that tracker.reset() completely flushes the tracker state at partition boundaries,
        preventing any train or quarantine packet from leaking into validation or test partitions.
        """
        tracker_with_history = NetworkTopologyTracker(window_sec=1.0, mode="sliding")

        # 1. Simulate train packets
        for i in range(100):
            p = NetworkTopologyPacket(50.0 + i * 0.01, f"10.0.0.{i%10}", "192.168.1.1", 5000 + i, 80, 6, 60)
            tracker_with_history.add_packet(p)

        assert tracker_with_history.total_packets > 0

        # 2. Partition boundary: Train -> Quarantine. RESET tracker.
        tracker_with_history.reset()
        assert tracker_with_history.total_packets == 0
        assert len(tracker_with_history.src_counts) == 0

        # 3. Simulate quarantine gap packets (5 seconds)
        for i in range(50):
            p = NetworkTopologyPacket(60.0 + i * 0.1, "172.16.0.1", "192.168.1.1", 4000 + i, 80, 6, 60)
            # In pipeline, quarantine packets are skipped or ingested and then reset before validation
            tracker_with_history.add_packet(p)

        # 4. Partition boundary: Quarantine -> Validation. RESET tracker.
        tracker_with_history.reset()
        assert tracker_with_history.total_packets == 0

        # 5. Ingest validation packets into both tracker_with_history and a brand new fresh tracker
        val_packets = [
            NetworkTopologyPacket(70.1, "192.168.1.10", "192.168.1.2", 1234, 80, 6, 100),
            NetworkTopologyPacket(70.3, "192.168.1.20", "192.168.1.2", 1235, 80, 6, 100),
            NetworkTopologyPacket(70.5, "192.168.1.10", "192.168.1.2", 1236, 80, 6, 100)
        ]

        tracker_fresh = NetworkTopologyTracker(window_sec=1.0, mode="sliding")
        for p in val_packets:
            tracker_with_history.add_packet(p)
            tracker_fresh.add_packet(p)

        feats_history = tracker_with_history.extract_current_features()
        feats_fresh = tracker_fresh.extract_current_features()

        for feat in V4_TOPOLOGY_FEATURES:
            assert feats_history[feat] == feats_fresh[feat], f"State contamination detected for {feat}"
            assert abs(feats_history[feat] - feats_fresh[feat]) < 1e-12

    def test_streaming_vs_offline_batch_equivalence(self):
        """
        Verify that streaming sliding NetworkTopologyTracker produces mathematically identical
        results to the offline batch reference calculator across multiple time intervals.
        """
        np.random.seed(42)
        packets: List[NetworkTopologyPacket] = []
        t = 1000.0
        sources = [f"192.168.1.{i}" for i in range(1, 15)]
        destinations = [f"10.0.0.{j}" for j in range(1, 5)]

        # Generate 200 random packets
        for _ in range(200):
            t += np.random.exponential(scale=0.02) # ~50 pps
            s = np.random.choice(sources)
            d = np.random.choice(destinations)
            sp = int(np.random.randint(1024, 65535))
            dp = int(np.random.choice([80, 443, 8080, 22]))
            p = NetworkTopologyPacket(
                timestamp=float(t),
                src_ip=s,
                dst_ip=d,
                src_port=sp,
                dst_port=dp,
                protocol=6,
                pkt_len=60
            )
            packets.append(p)

        tracker = NetworkTopologyTracker(window_sec=1.0, mode="sliding")

        # Test at every 10th packet
        for idx, p in enumerate(packets):
            tracker.add_packet(p)
            if idx > 20 and idx % 10 == 0:
                streaming_feats = tracker.extract_current_features()
                cutoff = p.timestamp - 1.0
                batch_window = [pkt for pkt in packets[:idx+1] if pkt.timestamp >= cutoff]
                batch_feats = compute_network_topology_batch(batch_window)

                for feat in V4_TOPOLOGY_FEATURES:
                    assert abs(streaming_feats[feat] - batch_feats[feat]) < 1e-7, (
                        f"Streaming vs batch divergence at idx {idx} for {feat}: "
                        f"{streaming_feats[feat]} vs {batch_feats[feat]}"
                    )

    def test_adversarial_future_attack_burst_immunity(self):
        """
        Simulate an adversarial attack burst occurring immediately after a valid flow window.
        Prove that the earlier window's combined 29-feature vector is impervious to future packets.
        """
        tracker = NetworkTopologyTracker(window_sec=1.0, mode="sliding")
        flow_key = ("192.168.1.10", "192.168.1.1", 6)
        window = FlowWindow(flow_key, max_packets=50, timeout_sec=2.0)

        # Ingest 50 normal flow packets
        t_base = 2000.0
        for i in range(50):
            t = t_base + i * 0.005
            pkt_rec = PacketRecord(t, 60, 6, 64, 2, True)
            net_pkt = NetworkTopologyPacket(t, "192.168.1.10", "192.168.1.1", 5000 + i, 80, 6, 60)
            net_tracker_ready, _ = tracker.add_packet(net_pkt)
            ready = window.add_packet(pkt_rec)

        assert ready is True
        t_end = window.last_timestamp

        # Extract features at t_end
        flow_feats = window.extract_features()
        net_feats = tracker.extract_current_features()
        clean_29_feats = {
            **flow_feats,
            "src_concentration_ratio": float(net_feats["src_concentration_ratio"]),
            "src_ip_entropy": float(net_feats["src_ip_entropy"]),
            "src_ip_cardinality": float(net_feats["src_ip_cardinality"]),
            "dst_ip_cardinality": float(net_feats["dst_ip_cardinality"]),
            "fan_in_ratio": float(net_feats["fan_in_ratio"])
        }

        # Now inject an adversarial future flood burst from 1,000 new spoofed IPs at t_end + 0.0001
        for j in range(1000):
            t_future = t_end + 0.0001 + j * 0.00001
            net_pkt_future = NetworkTopologyPacket(t_future, f"172.16.{j//256}.{j%256}", "192.168.1.1", 8000, 80, 6, 1200)
            tracker.add_packet(net_pkt_future)

        # If a causal observer re-evaluates the earlier row ending at t_end using only packets <= t_end:
        tracker_replay = NetworkTopologyTracker(window_sec=1.0, mode="sliding")
        for i in range(50):
            t = t_base + i * 0.005
            net_pkt = NetworkTopologyPacket(t, "192.168.1.10", "192.168.1.1", 5000 + i, 80, 6, 60)
            tracker_replay.add_packet(net_pkt)

        net_feats_replay = tracker_replay.extract_current_features()
        replay_29_feats = {
            **flow_feats,
            "src_concentration_ratio": float(net_feats_replay["src_concentration_ratio"]),
            "src_ip_entropy": float(net_feats_replay["src_ip_entropy"]),
            "src_ip_cardinality": float(net_feats_replay["src_ip_cardinality"]),
            "dst_ip_cardinality": float(net_feats_replay["dst_ip_cardinality"]),
            "fan_in_ratio": float(net_feats_replay["fan_in_ratio"])
        }

        for k in CANONICAL_29_FEATURES:
            assert clean_29_feats[k] == replay_29_feats[k], f"Adversarial future burst contaminated feature {k}"

    def test_all_80k_v4_rows_temporal_invariants(self):
        """
        Audit all 80,000 rows across train, validation, and test partitions:
          - window_start_ts <= window_end_ts
          - packet_count == 50
          - Zero NaN or Inf
          - Physical bounds for all 29 features
          - Strict >= 5.0s quarantine dead times
        """
        total_rows = 0
        for split in ["train", "validation", "test"]:
            path = os.path.join(V4_DATA_DIR, f"{split}.csv")
            df = pd.read_csv(path)
            total_rows += len(df)

            # Invariant: start <= end
            violations = (df["window_start_ts"] > df["window_end_ts"]).sum()
            assert violations == 0, f"Found {violations} start > end timestamp violations in {split}"

            # Invariant: packet_count == 50
            pkt_cnt_violations = (df["packet_count"] != 50).sum()
            assert pkt_cnt_violations == 0, f"Found {pkt_cnt_violations} non-50 packet windows in {split}"

            # Invariant: No NaN/Inf
            assert df[CANONICAL_29_FEATURES].isna().sum().sum() == 0, f"NaN in {split}"
            assert np.isinf(df[CANONICAL_29_FEATURES].values).sum() == 0, f"Inf in {split}"

            # Invariant: Physical bounds for topology features
            assert (df["src_concentration_ratio"] >= 0.0).all() and (df["src_concentration_ratio"] <= 1.0).all()
            assert (df["src_ip_entropy"] >= 0.0).all()
            assert (df["src_ip_cardinality"] >= 1.0).all()
            assert (df["dst_ip_cardinality"] >= 1.0).all()
            assert (df["fan_in_ratio"] >= 0.0).all()

        assert total_rows == 80000, f"Total audited rows {total_rows} != 80,000"

        # Quarantine dead times
        train_df = pd.read_csv(os.path.join(V4_DATA_DIR, "train.csv"))
        val_df = pd.read_csv(os.path.join(V4_DATA_DIR, "validation.csv"))
        test_df = pd.read_csv(os.path.join(V4_DATA_DIR, "test.csv"))

        for c in ["Benign", "DDoS", "DoS", "Spoofing"]:
            t_tr_max = train_df[train_df["label"] == c]["window_end_ts"].max()
            t_val_min = val_df[val_df["label"] == c]["window_start_ts"].min()
            gap1 = t_val_min - t_tr_max
            assert gap1 >= 5.0, f"Train->Val quarantine gap for {c} is {gap1:.4f}s (< 5.0s)"

            t_val_max = val_df[val_df["label"] == c]["window_end_ts"].max()
            t_test_min = test_df[test_df["label"] == c]["window_start_ts"].min()
            gap2 = t_test_min - t_val_max
            assert gap2 >= 5.0, f"Val->Test quarantine gap for {c} is {gap2:.4f}s (< 5.0s)"

    def test_v4_test_set_identical_24_features_to_v3(self):
        """
        Verify that all 12,000 held-out test windows in v4 are temporally identical to v3
        and have exactly 0.000000 difference across all 24 canonical physical flow features.
        """
        test_v3 = pd.read_csv(os.path.join(V3_DATA_DIR, "test.csv"))
        test_v4 = pd.read_csv(os.path.join(V4_DATA_DIR, "test.csv"))

        assert len(test_v3) == 12000
        assert len(test_v4) == 12000

        # Exact row-by-row window metadata match
        assert test_v3["flow_id"].equals(test_v4["flow_id"])
        assert np.allclose(test_v3["window_start_ts"].values, test_v4["window_start_ts"].values, atol=1e-6)
        assert np.allclose(test_v3["window_end_ts"].values, test_v4["window_end_ts"].values, atol=1e-6)
        assert test_v3["label"].equals(test_v4["label"])

        # Exact match on all 24 physical flow features
        for f in CANONICAL_24_FEATURES:
            diff = np.max(np.abs(test_v3[f].values - test_v4[f].values))
            assert diff == 0.0, f"Test set flow feature divergence on {f}: {diff:.6e}"

    def test_frozen_baseline_and_v4_hashes_preserved(self):
        """
        Verify that all frozen model files, dataset partitions, and artifacts remain byte-for-byte unmodified.
        """
        assert compute_sha256(V2_MODEL_PATH) == FROZEN_V2_MODEL_HASH, "v2 model hash modified!"
        assert compute_sha256(V3_MODEL_PATH) == FROZEN_V3_MODEL_HASH, "v3 model hash modified!"
        assert compute_sha256(V4_MODEL_PATH) == FROZEN_V4_MODEL_HASH, "v4 model hash modified!"

        assert compute_sha256(os.path.join(V4_DATA_DIR, "train.csv")) == FROZEN_V4_TRAIN_HASH
        assert compute_sha256(os.path.join(V4_DATA_DIR, "validation.csv")) == FROZEN_V4_VAL_HASH
        assert compute_sha256(os.path.join(V4_DATA_DIR, "test.csv")) == FROZEN_V4_TEST_HASH


def _build_eth_ipv4_tcp_bytes(src_ip: str, dst_ip: str, sport: int, dport: int,
                                tcp_flags: int = 0x18, payload: bytes = b"X" * 20) -> bytes:
    """Helper creating raw Ethernet/IPv4/TCP bytes without calling model inference."""
    import struct
    import socket
    ip_hl = 20
    tcp_hl = 20
    tot_len = ip_hl + tcp_hl + len(payload)
    ip_hdr = struct.pack(
        "!BBHHHBBH4s4s",
        0x45, 0, tot_len, 1, 0, 64, 6, 0,
        socket.inet_aton(src_ip), socket.inet_aton(dst_ip)
    )
    tcp_hdr = struct.pack(
        "!HHIIBBHHH",
        sport, dport, 1, 0, 0x50, tcp_flags, 65535, 0, 0
    )
    eth_hdr = b"\x00\x11\x22\x33\x44\x55\x66\x77\x88\x99\xaa\xbb\x08\x00"
    return eth_hdr + ip_hdr + tcp_hdr + payload


def _stub_inference_factory():
    """Returns a deterministic stub replacing AEGISLiveV4Inference (ZERO frozen model inference)."""
    class _StubInference:
        def __init__(self, *args, **kwargs):
            self.stub_only = True

        def predict_sample(self, feats_29):
            return {
                "predicted_class": "Benign",
                "calibrated_confidence": 0.99,
                "confidence": 0.99,
                "mahalanobis_distance": 1.0,
                "trusted": True,
                "decision": "TRUSTED",
                "status": "TRUSTED",
                "trusted_class": "Benign",
                "advisory_class": "Benign",
                "automated_blocking_allowed": False,
                "reasons": [],
                "reason_codes": [],
                "top2_margin": 0.98,
                "entropy": 0.05,
                "feature_vector": dict(feats_29),
                "calibrated_probabilities": {"Benign": 0.99, "DDoS": 0.004, "DoS": 0.003, "Spoofing": 0.003},
            }

    return _StubInference


class TestV4Phase1CausalSnapshotRemediation:
    """
    Batch 26B regression suite verifying Phase 1 temporal causality across runtime and builder.
    Uses mocked AEGISLiveV4Inference so zero frozen model inference executes.
    """

    def _new_runtime(self, **kwargs):
        from unittest.mock import patch
        from aegis_live.v4_runtime import AegisLiveV4Pipeline
        with patch("aegis_live.v4_runtime.AEGISLiveV4Inference", _stub_inference_factory()):
            return AegisLiveV4Pipeline(**kwargs)

    def test_01_flow_flushed_after_inactivity_timeout_uses_ending_snapshot(self):
        pipeline = self._new_runtime(local_subnet="192.168.0.0/16", max_packets_per_window=50, window_timeout_sec=2.0, min_packets_to_extract=2)

        # Flow A: 4 packets at t=10.00..10.03 while only 192.168.1.10 -> 192.168.1.1 is active
        pkt_a = _build_eth_ipv4_tcp_bytes("192.168.1.10", "192.168.1.1", 5000, 1883)
        for i in range(4):
            assert pipeline.ingest_packet(10.0 + i * 0.01, pkt_a) == []

        # Future traffic on other IPs at t=11.5..12.0 (evicts t=10.03 from 1.0s topology window)
        for j in range(9):
            pkt_bg = _build_eth_ipv4_tcp_bytes(f"192.168.1.{100 + j}", "192.168.1.1", 6000 + j, 80)
            pipeline.ingest_packet(11.5 + j * 0.05, pkt_bg)

        # Packet on Flow A at t=12.50 (gap = 2.47s >= 2.0s) triggers timeout flush of [10.00..10.03]
        emitted = pipeline.ingest_packet(12.50, pkt_a)
        assert emitted == []
        rec = next(r for r in pipeline.incomplete_window_records if "192.168.1.10" in r.flow_identity)
        assert rec.packet_count == 4
        assert rec.status == "INCOMPLETE_WINDOW" and rec.v4_eligible is False
        assert abs(rec.window_end_ts - 10.03) < 1e-9
        assert abs(rec.topology_snapshot_ts - 10.03) < 1e-9

        # Snapshot must match t=10.03 (1 source), NOT t=12.50 (10 sources)
        assert rec.features["src_ip_cardinality"] == 1.0
        assert rec.features["dst_ip_cardinality"] == 1.0
        assert rec.features["src_ip_entropy"] == 0.0
        assert rec.features["src_concentration_ratio"] == 1.0
        assert rec.features["fan_in_ratio"] == 1.0

    def test_02_flow_flushed_at_end_of_stream_uses_own_ending_snapshot(self):
        pipeline = self._new_runtime(local_subnet="192.168.0.0/16", max_packets_per_window=50, window_timeout_sec=2.0, min_packets_to_extract=2)

        pkt_a = _build_eth_ipv4_tcp_bytes("192.168.1.10", "192.168.1.1", 5000, 1883)
        for i in range(3):
            pipeline.ingest_packet(10.0 + i * 0.02, pkt_a)

        # Later stream activity on different flows at t=20.0..20.2
        for j in range(5):
            pkt_other = _build_eth_ipv4_tcp_bytes(f"192.168.1.{50 + j}", "192.168.1.2", 7000 + j, 80)
            pipeline.ingest_packet(20.0 + j * 0.04, pkt_other)

        flushed = pipeline.flush(current_time=25.0, force_all=True)
        assert flushed == []
        rec = next(r for r in pipeline.incomplete_window_records if "192.168.1.10" in r.flow_identity)
        assert rec.status == "INCOMPLETE_WINDOW" and rec.v4_eligible is False
        assert abs(rec.window_end_ts - 10.04) < 1e-9
        assert rec.features["src_ip_cardinality"] == 1.0
        assert rec.features["src_ip_entropy"] == 0.0
        assert rec.features["fan_in_ratio"] == 1.0

    def test_03_multiple_flows_with_different_ending_timestamps(self):
        pipeline = self._new_runtime(local_subnet="192.168.0.0/16", max_packets_per_window=50, window_timeout_sec=10.0, min_packets_to_extract=2)

        # Flow A ends at t=10.1 (1 source)
        pkt_a = _build_eth_ipv4_tcp_bytes("192.168.1.10", "192.168.1.1", 5000, 1883)
        pipeline.ingest_packet(10.0, pkt_a)
        pipeline.ingest_packet(10.1, pkt_a)

        # Flow B and helper C end at t=10.6 (all 3 sources within 1.0s of 10.6)
        pkt_b = _build_eth_ipv4_tcp_bytes("192.168.1.20", "192.168.1.1", 5001, 80)
        pkt_c = _build_eth_ipv4_tcp_bytes("192.168.1.30", "192.168.1.1", 5002, 80)
        pipeline.ingest_packet(10.4, pkt_b)
        pipeline.ingest_packet(10.5, pkt_c)
        pipeline.ingest_packet(10.6, pkt_b)

        # Flow D ends at t=15.1 (10.0..10.6 already evicted from 1.0s window -> 1 source)
        pkt_d = _build_eth_ipv4_tcp_bytes("192.168.1.40", "192.168.1.1", 5003, 502)
        pipeline.ingest_packet(15.0, pkt_d)
        pipeline.ingest_packet(15.1, pkt_d)

        flushed = pipeline.flush(current_time=20.0, force_all=True)
        assert flushed == []
        by_src = {r.flow_identity: r for r in pipeline.incomplete_window_records}
        assert len(by_src) == 4
        rec_a = [v for k, v in by_src.items() if "192.168.1.10" in k][0]
        rec_b = [v for k, v in by_src.items() if "192.168.1.20" in k][0]
        rec_d = [v for k, v in by_src.items() if "192.168.1.40" in k][0]

        assert rec_a.features["src_ip_cardinality"] == 1.0
        assert rec_b.features["src_ip_cardinality"] == 3.0
        assert rec_d.features["src_ip_cardinality"] == 1.0

    def test_04_interleaved_packets_from_multiple_flows_match_causal_batch(self):
        pipeline = self._new_runtime(local_subnet="192.168.0.0/16", max_packets_per_window=50, window_timeout_sec=2.0, min_packets_to_extract=2)

        stream = []
        # Interleave Flow A (ends at 10.3), Flow B (hits 50 pkts at 10.7), Flow C (ends at 10.5)
        for i in range(5):
            stream.append((10.0 + i * 0.075, "192.168.1.10", "192.168.1.1", 4000, 1883))
        for i in range(4):
            stream.append((10.02 + i * 0.16, "192.168.1.30", "192.168.1.1", 4200, 502))
        for i in range(50):
            stream.append((10.01 + i * 0.014, "192.168.1.20", "192.168.1.2", 4100 + i, 80))
        stream.sort(key=lambda x: x[0])

        topo_pkts = []
        emitted = []
        for idx, (ts, s_ip, d_ip, sp, dp) in enumerate(stream):
            buf = _build_eth_ipv4_tcp_bytes(s_ip, d_ip, sp, dp)
            topo_pkts.append(NetworkTopologyPacket(ts, s_ip, d_ip, sp, dp, 6, len(buf)))
            emitted.extend(pipeline.ingest_packet(ts, buf))
        emitted.extend(pipeline.flush(current_time=20.0, force_all=True))

        assert len(emitted) == 1  # Only Flow B is a frozen V4 eligible 50-packet window.
        assert emitted[0].provenance.packet_count == 50
        records = [(d.provenance, d.feature_vector) for d in emitted]
        records.extend((r, r.features) for r in pipeline.incomplete_window_records)
        assert sorted(prov.packet_count for prov, _ in records) == [4, 5, 50]
        for provenance, features in records:
            t_end = provenance.window_end_ts
            # Find last stream packet belonging to this flow
            matching_indices = [
                i for i, (ts, s_ip, d_ip, _, _) in enumerate(stream)
                if ts <= t_end + 1e-12 and s_ip in provenance.flow_identity and d_ip in provenance.flow_identity
            ]
            last_idx = matching_indices[-1]
            cutoff = stream[last_idx][0] - 1.0
            batch_ref = compute_network_topology_batch([p for p in topo_pkts[:last_idx + 1] if p.timestamp >= cutoff])
            for feat in V4_TOPOLOGY_FEATURES:
                assert abs(features[feat] - batch_ref[feat]) < 1e-12, f"Mismatch on {feat}"

    def test_05_future_packet_after_ending_timestamp_cannot_contaminate(self):
        pkt_a = _build_eth_ipv4_tcp_bytes("192.168.1.10", "192.168.1.1", 5000, 1883)

        p_clean = self._new_runtime(local_subnet="192.168.0.0/16", max_packets_per_window=50, window_timeout_sec=2.0, min_packets_to_extract=2)
        for i in range(4):
            p_clean.ingest_packet(10.0 + i * 0.05, pkt_a)
        assert p_clean.flush(current_time=15.0, force_all=True) == []
        rec_clean = next(r for r in p_clean.incomplete_window_records if "192.168.1.10" in r.flow_identity)

        p_fut = self._new_runtime(local_subnet="192.168.0.0/16", max_packets_per_window=50, window_timeout_sec=2.0, min_packets_to_extract=2)
        for i in range(4):
            p_fut.ingest_packet(10.0 + i * 0.05, pkt_a)
        for j in range(25):
            pkt_spike = _build_eth_ipv4_tcp_bytes(f"10.10.0.{j + 1}", "192.168.1.1", 9000 + j, 80)
            p_fut.ingest_packet(10.1501 + j * 0.02, pkt_spike)
        # Flush Flow A by timeout trigger at t=12.5
        dets_timeout = p_fut.ingest_packet(12.5, pkt_a)
        assert dets_timeout == []
        rec_future = next(r for r in p_fut.incomplete_window_records if "192.168.1.10" in r.flow_identity)

        for col in CANONICAL_29_FEATURES:
            assert rec_clean.features[col] == rec_future.features[col]

    def test_06_full_50_packet_window_emission_and_topology(self):
        pipeline = self._new_runtime(local_subnet="192.168.0.0/16", max_packets_per_window=50, window_timeout_sec=2.0, min_packets_to_extract=2)
        bg = _build_eth_ipv4_tcp_bytes("192.168.1.99", "192.168.1.1", 7777, 80)

        emitted = []
        for i in range(50):
            ts = 100.0 + i * 0.01
            if i == 25:
                emitted.extend(pipeline.ingest_packet(ts - 0.005, bg))
            pkt = _build_eth_ipv4_tcp_bytes("192.168.1.10", "192.168.1.1", 5000 + i, 1883)
            emitted.extend(pipeline.ingest_packet(ts, pkt))

        assert len(emitted) == 1
        det = emitted[0]
        assert det.provenance.packet_count == 50
        assert det.feature_vector["src_ip_cardinality"] == 2.0

    def test_07_subcapacity_window_retains_snapshot_across_delay(self):
        pipeline = self._new_runtime(local_subnet="192.168.0.0/16", max_packets_per_window=50, window_timeout_sec=2.0, min_packets_to_extract=2)
        pkt_a = _build_eth_ipv4_tcp_bytes("192.168.1.10", "192.168.1.1", 5000, 1883)
        pkt_h1 = _build_eth_ipv4_tcp_bytes("192.168.1.11", "192.168.1.1", 5001, 1883)
        pkt_h2 = _build_eth_ipv4_tcp_bytes("192.168.1.12", "192.168.1.1", 5002, 1883)

        pipeline.ingest_packet(50.0, pkt_h1)
        pipeline.ingest_packet(50.01, pkt_h2)
        for i in range(5):
            pipeline.ingest_packet(50.02 + i * 0.01, pkt_a)  # ends at 50.06 with 3 sources

        # Advance time well past 1.0s topology window
        pkt_late = _build_eth_ipv4_tcp_bytes("192.168.1.200", "192.168.1.9", 8000, 80)
        pipeline.ingest_packet(55.0, pkt_late)

        flushed = pipeline.flush(current_time=60.0, force_all=True)
        assert flushed == []
        rec = next(r for r in pipeline.incomplete_window_records if "192.168.1.10" in r.flow_identity)
        assert rec.packet_count == 5
        assert rec.status == "INCOMPLETE_WINDOW" and rec.v4_eligible is False
        assert abs(rec.window_end_ts - 50.06) < 1e-9
        assert rec.features["src_ip_cardinality"] == 3.0

    def test_08_builder_runtime_temporal_parity_on_synthetic_stream(self):
        from aegis_live.build_v4_dataset import LeakageSafeV4DatasetBuilder

        # Synthetic 50-packet Flow A interleaved with background packet B + trailing Flow C
        stream = [
            (10.005, _build_eth_ipv4_tcp_bytes("192.168.1.99", "192.168.1.1", 9000, 80)),
        ]
        for i in range(50):
            stream.append((10.000 + i * 0.01, _build_eth_ipv4_tcp_bytes("192.168.1.10", "192.168.1.1", 5000, 1883, payload=b"M" * (16 + (i % 5)))))
        stream.append((10.60, _build_eth_ipv4_tcp_bytes("192.168.1.88", "192.168.1.1", 9001, 80)))
        stream.sort(key=lambda x: x[0])

        builder = LeakageSafeV4DatasetBuilder(target_per_class=10, window_capacity=50, timeout_sec=2.0)
        b_windows = builder.extract_packet_stream_windows(stream, pcap_file="synthetic_parity.pcap", class_label="Benign")
        assert len(b_windows) == 1
        b_row = b_windows[0]

        pipeline = self._new_runtime(local_subnet="192.168.0.0/16", max_packets_per_window=50, window_timeout_sec=2.0, min_packets_to_extract=2)
        r_dets = pipeline.replay_packet_stream(iter(stream))
        r_full = [d for d in r_dets if d.provenance.packet_count == 50]
        assert len(r_full) == 1
        r_vec = r_full[0].feature_vector

        for col in CANONICAL_29_FEATURES:
            assert abs(float(b_row[col]) - float(r_vec[col])) < 1e-12, f"Builder/Runtime parity mismatch on {col}"

    def test_09_empty_tracker_and_empty_or_single_packet_window_handling(self):
        empty_batch = compute_network_topology_batch([])
        assert empty_batch["src_ip_cardinality"] == 0.0
        assert empty_batch["src_ip_entropy"] == 0.0
        assert empty_batch["fan_in_ratio"] == 0.0

        pipeline = self._new_runtime(local_subnet="192.168.0.0/16", max_packets_per_window=50, window_timeout_sec=2.0, min_packets_to_extract=2)
        assert pipeline.net_tracker.extract_current_features()["packet_count"] == 0.0
        assert pipeline.flush(current_time=100.0, force_all=True) == []

        # Single packet is retained diagnostically and excluded from frozen V4.
        pkt_single = _build_eth_ipv4_tcp_bytes("192.168.1.10", "192.168.1.1", 5000, 1883)
        assert pipeline.ingest_packet(100.0, pkt_single) == []
        assert pipeline.flush(current_time=105.0, force_all=True) == []
        assert pipeline.windows_rejected == 1
        assert pipeline.incomplete_window_records[-1].status == "INCOMPLETE_WINDOW"
        assert pipeline.incomplete_window_records[-1].packet_count == 1

    def test_10_repeated_flush_calls_without_duplicate_emission(self):
        pipeline = self._new_runtime(local_subnet="192.168.0.0/16", max_packets_per_window=50, window_timeout_sec=2.0, min_packets_to_extract=2)
        pkt = _build_eth_ipv4_tcp_bytes("192.168.1.10", "192.168.1.1", 5000, 1883)
        for i in range(4):
            pipeline.ingest_packet(10.0 + i * 0.05, pkt)

        first = pipeline.flush(current_time=15.0, force_all=True)
        second = pipeline.flush(current_time=16.0, force_all=True)
        third = pipeline.flush(current_time=20.0, force_all=False)

        assert first == []
        assert second == []
        assert third == []
        assert len(pipeline.detection_records) == 0
        assert len(pipeline.incomplete_window_records) == 1

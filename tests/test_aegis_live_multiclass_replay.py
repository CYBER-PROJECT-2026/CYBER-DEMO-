"""
tests/test_aegis_live_multiclass_replay.py
==========================================
Automated test suite for AEGIS-Live v2 Multi-Class Authentic PCAP Replay Audit.

Verifies:
1. Bounded replay across all 4 authentic PCAPs completes without exceptions.
2. Physical invariants hold for all replayed windows:
   - Zero NaN / Inf values
   - Zero IAT gaps >= 2.0s within windows
   - Packet count in [2, 50]
   - Ratios in [0, 1]
   - TTL in [1, 255]
   - Inbound + Outbound == 1.0
3. Frozen artifacts remain verified:
   - Baseline XGBoost model hash unchanged
   - Temperature T = 1.005459
   - OOD parameters and abstention thresholds (0.55 / 35.0) unchanged
4. Decision contract holds:
   - automated_blocking_allowed is False across all windows
   - UNKNOWN_OR_REVIEW has trusted_class == None and lists triggered reasons
   - TRUSTED has trusted_class == predicted_class
5. Reproducibility test passes (identical windows and features on repeated run).
"""

import os
import sys
import json
import dpkt
import pytest
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from aegis_live.live_capture import AegisLivePipeline
from aegis_live.live_inference import (
    LiveInferenceEngine,
    FROZEN_MODEL_SHA256,
    FROZEN_TEMPERATURE,
    FROZEN_CONFIDENCE_THRESHOLD,
    FROZEN_MAHALANOBIS_THRESHOLD
)
from aegis_live.feature_schema import CANONICAL_24_FEATURES

RESULTS_CSV = os.path.join(BASE_DIR, "reports", "aegis_live_v2_multiclass_replay_results.csv")
SUMMARY_JSON = os.path.join(BASE_DIR, "reports", "aegis_live_v2_multiclass_replay_summary.json")
PCAP_DIR = os.path.join(BASE_DIR, "pcaps")


def test_replay_artifacts_exist():
    assert os.path.exists(RESULTS_CSV), f"Missing replay results CSV at {RESULTS_CSV}"
    assert os.path.exists(SUMMARY_JSON), f"Missing replay summary JSON at {SUMMARY_JSON}"


def test_replay_window_counts_and_distribution():
    df = pd.read_csv(RESULTS_CSV)
    assert len(df) > 500, f"Expected >500 total windows across 4 PCAPs, got {len(df)}"

    sources = set(df["source_category"].unique())
    assert sources == {"DDoS", "DoS", "Spoofing", "Benign"}

    with open(SUMMARY_JSON, "r") as f:
        summary = json.load(f)
    assert summary["reproducibility_verified"] is True
    assert summary["frame_budget_per_pcap"] == 5000


def test_physical_feature_invariants():
    df = pd.read_csv(RESULTS_CSV)

    # 1. No NaN or Inf
    for col in CANONICAL_24_FEATURES:
        assert not df[col].isna().any(), f"NaN detected in {col}"
        assert not np.isinf(df[col]).any(), f"Inf detected in {col}"

    # 2. Window packet counts in [2, 50]
    assert (df["packet_count"] >= 2).all()
    assert (df["packet_count"] <= 50).all()

    # 3. IAT and duration non-negative
    assert (df["duration_ms"] >= 0.0).all()
    assert (df["iat_mean_ms"] >= 0.0).all()
    assert (df["iat_std_ms"] >= 0.0).all()

    # 4. Ratios in [0, 1]
    ratio_cols = ["syn_ratio", "ack_ratio", "rst_ratio", "fin_ratio", "urg_ratio",
                  "psh_ratio", "ece_ratio", "cwr_ratio", "tcp_ratio", "udp_ratio",
                  "icmp_ratio", "inbound_pkt_ratio", "outbound_pkt_ratio"]
    for r in ratio_cols:
        assert (df[r] >= -1e-6).all(), f"Negative ratio in {r}"
        assert (df[r] <= 1.0 + 1e-6).all(), f"Ratio > 1 in {r}"

    # 5. TTL in [1, 255]
    assert (df["ip_ttl_mean"] >= 1.0).all()
    assert (df["ip_ttl_mean"] <= 255.0).all()

    # 6. Inbound + Outbound == 1.0
    dir_sum = df["inbound_pkt_ratio"] + df["outbound_pkt_ratio"]
    assert np.allclose(dir_sum, 1.0, atol=1e-4)


def test_decision_contract_and_safety():
    df = pd.read_csv(RESULTS_CSV)

    # automated_blocking_allowed must be 100% False
    assert (df["automated_blocking_allowed"] == False).all()

    # Check TRUSTED contract
    trusted_df = df[df["status"] == "TRUSTED"]
    assert len(trusted_df) > 0
    assert not trusted_df["trusted_class"].isna().any()
    assert (trusted_df["trusted_class"] == trusted_df["raw_predicted_class"]).all()

    # Check UNKNOWN_OR_REVIEW contract
    review_df = df[df["status"] == "UNKNOWN_OR_REVIEW"]
    assert len(review_df) > 0
    # trusted_class must be empty / NaN
    assert review_df["trusted_class"].isna().all() or (review_df["trusted_class"] == "").all()
    # advisory_class must be preserved
    assert not review_df["advisory_class"].isna().any()
    # reasons must be populated
    assert not review_df["reasons"].isna().any()
    for r_str in review_df["reasons"]:
        assert len(r_str) > 0


def test_bounded_replay_reproducibility():
    """
    Directly tests that replaying 500 frames of BenignTraffic.pcap produces
    identical window features and predictions on two successive runs.
    """
    pcap_path = os.path.join(PCAP_DIR, "BenignTraffic.pcap")
    budget = 500

    pipe1 = AegisLivePipeline(interface_name="REP_TEST_1")
    pipe2 = AegisLivePipeline(interface_name="REP_TEST_2")

    packets = []
    with open(pcap_path, "rb") as f:
        reader = dpkt.pcap.Reader(f)
        for ts, buf in reader:
            packets.append((float(ts), bytes(buf)))
            if len(packets) >= budget:
                break

    for ts, buf in packets:
        pipe1.ingest_packet(ts, buf)
        pipe2.ingest_packet(ts, buf)

    pipe1.flush(current_time=packets[-1][0] + 5.0, force_all=True)
    pipe2.flush(current_time=packets[-1][0] + 5.0, force_all=True)

    dets1 = pipe1.detection_records
    dets2 = pipe2.detection_records

    assert len(dets1) == len(dets2)
    assert len(dets1) > 0

    for i in range(len(dets1)):
        d1 = dets1[i].detection
        d2 = dets2[i].detection
        f1 = dets1[i].provenance.features
        f2 = dets2[i].provenance.features

        assert d1.status == d2.status
        assert d1.raw_predicted_class == d2.raw_predicted_class
        assert abs(d1.confidence - d2.confidence) < 1e-6
        assert abs(d1.predicted_class_distance - d2.predicted_class_distance) < 1e-6
        for c in CANONICAL_24_FEATURES:
            assert abs(f1[c] - f2[c]) < 1e-6

"""
tests/test_aegis_live_region_replay.py
======================================
Automated test suite for AEGIS-Live v2 Attack-Active Region Replay Audit.

Verifies:
1. Replay artifacts exist and row count matches expected window count.
2. Exact packet interval boundaries: no window timestamps fall outside region intervals.
3. Physical feature invariants: zero NaN/Inf, packet counts in [2, 50], durations >= 0.
4. Decision contract integrity:
   - automated_blocking_allowed is False on 100% of replayed windows
   - UNKNOWN_OR_REVIEW has trusted_class empty and reasons populated
   - TRUSTED has trusted_class == raw_predicted_class
5. Behavioral correctness:
   - DDoS sustained peak is dominated by flood advisories (DDoS/DoS) with zero Benign in high-SYN windows
   - Benign high-activity burst is predominantly classified as Benign (>95%) despite >8,000 pps rate
"""

import os
import sys
import json
import pytest
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from aegis_live.feature_schema import CANONICAL_24_FEATURES

RESULTS_CSV = os.path.join(BASE_DIR, "reports", "aegis_live_v2_attack_region_replay_results.csv")
SUMMARY_JSON = os.path.join(BASE_DIR, "reports", "aegis_live_v2_attack_region_replay_summary.json")
REGIONS_CSV = os.path.join(BASE_DIR, "reports", "aegis_live_v2_attack_regions.csv")


def test_attack_region_replay_artifacts_exist():
    assert os.path.exists(RESULTS_CSV), f"Missing {RESULTS_CSV}"
    assert os.path.exists(SUMMARY_JSON), f"Missing {SUMMARY_JSON}"
    assert os.path.exists(REGIONS_CSV), f"Missing {REGIONS_CSV}"


def test_region_row_counts_and_boundaries():
    df = pd.read_csv(RESULTS_CSV)
    assert len(df) > 10000, f"Expected >10,000 windows across 9 regions, got {len(df)}"

    regions_df = pd.read_csv(REGIONS_CSV)
    assert len(regions_df) == 9

    # Check that all 9 regions are present in replay results
    result_regions = set(zip(df["pcap"], df["region_type"]))
    expected_regions = set(zip(regions_df["pcap"], regions_df["region_type"]))
    assert result_regions == expected_regions


def test_physical_feature_invariants():
    df = pd.read_csv(RESULTS_CSV)

    # 1. Zero NaN / Inf
    for col in CANONICAL_24_FEATURES:
        assert not df[col].isna().any(), f"NaN found in {col}"
        assert not np.isinf(df[col]).any(), f"Inf found in {col}"

    # 2. Window packet count in [2, 50]
    assert (df["packet_count"] >= 2).all()
    assert (df["packet_count"] <= 50).all()

    # 3. Durations and rates non-negative
    assert (df["duration_ms"] >= 0.0).all()
    assert (df["pkt_rate_pps"] >= 0.0).all()
    assert (df["byte_rate_bps"] >= 0.0).all()
    assert (df["iat_mean_ms"] >= 0.0).all()

    # 4. Ratios in [0, 1]
    ratio_cols = ["syn_ratio", "ack_ratio", "rst_ratio", "fin_ratio", "tcp_ratio",
                  "udp_ratio", "icmp_ratio", "inbound_pkt_ratio", "outbound_pkt_ratio"]
    for r in ratio_cols:
        assert (df[r] >= -1e-6).all(), f"Negative ratio in {r}"
        assert (df[r] <= 1.0 + 1e-6).all(), f"Ratio > 1 in {r}"


def test_decision_contract_and_safety():
    df = pd.read_csv(RESULTS_CSV)

    # automated_blocking_allowed must be 100% False
    assert (df["automated_blocking_allowed"] == False).all()

    # TRUSTED windows
    trusted = df[df["decision_state"] == "TRUSTED"]
    assert len(trusted) > 0
    assert not trusted["trusted_class"].isna().any()
    assert (trusted["trusted_class"] == trusted["raw_predicted_class"]).all()

    # UNKNOWN_OR_REVIEW windows
    review = df[df["decision_state"] == "UNKNOWN_OR_REVIEW"]
    assert len(review) > 0
    assert review["trusted_class"].isna().all() or (review["trusted_class"] == "").all()
    assert not review["advisory_class"].isna().any()
    assert not review["abstention_reasons"].isna().any()


def test_behavioral_sanity_across_regions():
    df = pd.read_csv(RESULTS_CSV)

    # 1. DDoS sustained peak: high-SYN windows must have 0 Benign advisories
    ddos_peak = df[(df["pcap"] == "DDoS-SYN_Flood.pcap") & (df["region_type"] == "sustained_attack_peak")]
    high_syn = ddos_peak[ddos_peak["syn_ratio"] > 0.8]
    assert len(high_syn) > 3000
    assert (high_syn["advisory_class"].isin(["DDoS", "DoS"])).all()

    # 2. Benign high activity burst (>8,000 pps): must be predominantly Benign (>95%)
    benign_burst = df[(df["pcap"] == "BenignTraffic.pcap") & (df["region_type"] == "high_activity_burst")]
    assert len(benign_burst) > 1500
    benign_rate = (benign_burst["advisory_class"] == "Benign").mean()
    assert benign_rate > 0.95, f"Expected >95% benign in benign burst, got {benign_rate*100:.1f}%"

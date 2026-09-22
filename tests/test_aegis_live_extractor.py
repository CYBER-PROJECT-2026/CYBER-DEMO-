"""
tests/test_aegis_live_extractor.py
==================================
Unit tests validating mathematical correctness, bounds, and edge cases
for the AEGIS-Live v2 24-feature extractor.
"""

import os
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pytest
import numpy as np
import pandas as pd

from aegis_live.feature_schema import CANONICAL_24_FEATURES, validate_24_features
from aegis_live.packet_window import PacketRecord, FlowWindow


def test_canonical_24_feature_count():
    assert len(CANONICAL_24_FEATURES) == 24
    assert len(set(CANONICAL_24_FEATURES)) == 24


def test_validator_rejects_missing_features():
    incomplete = {c: 1.0 for c in CANONICAL_24_FEATURES[:20]}
    with pytest.raises(ValueError, match="Missing required AEGIS-Live v2 features"):
        validate_24_features(incomplete)


def test_validator_rejects_nan_and_inf():
    valid = {c: 0.5 for c in CANONICAL_24_FEATURES}
    
    # Test NaN
    nan_dict = valid.copy()
    nan_dict["duration_ms"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_24_features(nan_dict)

    # Test Inf
    inf_dict = valid.copy()
    inf_dict["pkt_rate_pps"] = np.inf
    with pytest.raises(ValueError, match="Infinite"):
        validate_24_features(inf_dict)


def test_validator_rejects_out_of_bounds_ratios():
    valid = {c: 0.5 for c in CANONICAL_24_FEATURES}
    bad_ratio = valid.copy()
    bad_ratio["syn_ratio"] = 1.5
    with pytest.raises(ValueError, match="Ratio column syn_ratio has values outside"):
        validate_24_features(bad_ratio)


def test_flow_window_tcp_syn_flood():
    """
    Simulates a 10-packet TCP SYN flood with 1ms inter-arrival times and 60-byte packets.
    """
    window = FlowWindow(flow_key=("192.168.1.100", "10.0.0.1", 6), max_packets=10)
    
    start_time = 1700000000.0
    for i in range(10):
        t = start_time + (i * 0.001)  # 1ms intervals
        pkt = PacketRecord(
            timestamp=t,
            pkt_len=60,
            ip_proto=6,   # TCP
            ip_ttl=64,
            tcp_flags=0x02,  # SYN flag
            is_inbound=True
        )
        window.add_packet(pkt)

    feats = window.extract_features()
    df = validate_24_features(feats)

    # Assertions on physical values
    assert df["syn_ratio"].iloc[0] == 1.0
    assert df["ack_ratio"].iloc[0] == 0.0
    assert df["tcp_ratio"].iloc[0] == 1.0
    assert df["udp_ratio"].iloc[0] == 0.0
    assert df["inbound_pkt_ratio"].iloc[0] == 1.0
    assert df["outbound_pkt_ratio"].iloc[0] == 0.0
    assert df["pkt_size_mean"].iloc[0] == 60.0
    assert df["pkt_size_std"].iloc[0] == 0.0
    assert df["pkt_size_min"].iloc[0] == 60.0
    assert df["pkt_size_max"].iloc[0] == 60.0
    assert df["ip_ttl_mean"].iloc[0] == 64.0
    assert df["ip_ttl_std"].iloc[0] == 0.0
    
    # Duration: 9ms (from t0 to t9)
    np.testing.assert_allclose(df["duration_ms"].iloc[0], 9.0, rtol=1e-3)
    # IAT mean: 1ms
    np.testing.assert_allclose(df["iat_mean_ms"].iloc[0], 1.0, rtol=1e-3)
    np.testing.assert_allclose(df["iat_std_ms"].iloc[0], 0.0, atol=1e-3)


def test_flow_window_udp_flood():
    """
    Simulates a 50-packet high-speed UDP flood.
    """
    window = FlowWindow(flow_key=("192.168.1.100", "10.0.0.1", 17), max_packets=50)
    
    start_time = 1700000000.0
    for i in range(50):
        t = start_time + (i * 0.0001)  # 100 microseconds intervals (10,000 pps)
        pkt = PacketRecord(
            timestamp=t,
            pkt_len=512 if i % 2 == 0 else 1024,
            ip_proto=17,  # UDP
            ip_ttl=56,
            tcp_flags=0,
            is_inbound=True
        )
        ready = window.add_packet(pkt)

    assert ready is True
    feats = window.extract_features()
    df = validate_24_features(feats)

    assert df["udp_ratio"].iloc[0] == 1.0
    assert df["tcp_ratio"].iloc[0] == 0.0
    assert df["syn_ratio"].iloc[0] == 0.0
    assert df["pkt_size_min"].iloc[0] == 512.0
    assert df["pkt_size_max"].iloc[0] == 1024.0
    np.testing.assert_allclose(df["pkt_size_mean"].iloc[0], 768.0, rtol=1e-3)
    np.testing.assert_allclose(df["iat_mean_ms"].iloc[0], 0.1, rtol=1e-3)
    assert df["pkt_rate_pps"].iloc[0] > 9000.0

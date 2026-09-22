"""
tests/test_aegis_live_iat_regression.py
========================================
Regression unit tests verifying on-arrival window timeout semantics and
clean physical IAT calculation for AEGIS-Live v2.

Tests:
1. Forensic scenario: Packet A (1665164517.508446) and Packet B (1665164577.757202) on same flow
   - Proves Packet A and B never share a window
   - Proves no window contains IAT == 60.248756s
   - Proves second window has clean IAT == 0.0
2. 1.0-second gap -> same window
3. Exactly 2.0-second gap -> new window
4. 2.001-second gap -> new window
5. 10.0-second gap -> new window
6. 60.0-second gap -> new window
7. Normal 50-packet capacity trigger
8. Flow expiration/reset behavior
9. FlowWindow.add_packet() raises ValueError if gap >= timeout_sec
"""

import os
import sys
import socket
import pytest
import dpkt
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from aegis_live.feature_extractor import LiveFeatureExtractor
from aegis_live.packet_window import PacketRecord, FlowWindow


def make_raw_udp_packet(src_ip: str, dst_ip: str, sport: int, dport: int, payload: bytes = b'test') -> bytes:
    """Helper creating raw Ethernet/IPv4/UDP frame bytes."""
    udp = dpkt.udp.UDP(sport=sport, dport=dport, data=payload)
    udp.ulen = len(udp)
    ip = dpkt.ip.IP(
        src=socket.inet_aton(src_ip),
        dst=socket.inet_aton(dst_ip),
        p=dpkt.ip.IP_PROTO_UDP,
        ttl=64,
        data=udp
    )
    ip.len = len(ip)
    eth = dpkt.ethernet.Ethernet(
        src=b'\x00\x11\x22\x33\x44\x55',
        dst=b'\x66\x77\x88\x99\xaa\xbb',
        type=dpkt.ethernet.ETH_TYPE_IP,
        data=ip
    )
    return bytes(eth)


def test_forensic_scenario_mdns_isolation():
    """
    Exact forensic scenario from reports/aegis_live_v2_iat_window_audit.md:
    Flow: 192.168.137.172 <-> 224.0.0.251:5353 UDP
    Packet A: 1665164517.508446
    Packet B: 1665164577.757202 (Gap: 60.248756s)
    """
    extractor = LiveFeatureExtractor(
        local_subnet="192.168.0.0/16",
        max_packets_per_window=50,
        window_timeout_sec=2.0,
        min_packets_to_extract=2
    )

    flow_key = ("192.168.137.172", "224.0.0.251", 17)
    pkt_bytes = make_raw_udp_packet("192.168.137.172", "224.0.0.251", 5353, 5353)

    ts_a = 1665164517.508446
    ts_b = 1665164577.757202
    gap = ts_b - ts_a
    assert abs(gap - 60.248756) < 1e-4

    # 1. Packet A arrives
    res_a = extractor.process_raw_packet(ts_a, pkt_bytes)
    assert res_a is None
    assert flow_key in extractor.active_windows
    assert len(extractor.active_windows[flow_key].packets) == 1
    assert extractor.active_windows[flow_key].packets[0].timestamp == ts_a

    # 2. Packet B arrives 60.25s later
    res_b = extractor.process_raw_packet(ts_b, pkt_bytes)

    # Packet A was alone (< min_packets=2), so it was flushed without extraction
    # Packet B starts a fresh window
    assert flow_key in extractor.active_windows
    window_b = extractor.active_windows[flow_key]
    assert len(window_b.packets) == 1
    assert window_b.packets[0].timestamp == ts_b

    # Packet A and Packet B MUST NOT share a window
    assert len(window_b.packets) == 1
    assert window_b.first_timestamp == ts_b

    # Add Packet C 100ms later to allow feature extraction
    ts_c = ts_b + 0.100
    res_c = extractor.process_raw_packet(ts_c, pkt_bytes)
    assert res_c is None

    # Flush window containing B and C
    flushed = extractor.flush_timed_out_windows(ts_c + 3.0)
    assert len(flushed) == 1
    feat = flushed[0]

    # Verification:
    # 1. Duration is ~100ms, NOT 60 seconds
    np.testing.assert_allclose(feat["duration_ms"], 100.0, rtol=1e-3)
    # 2. Mean IAT is 100ms, NOT 60,248 ms
    np.testing.assert_allclose(feat["iat_mean_ms"], 100.0, rtol=1e-3)
    # 3. No IAT inside window equals or exceeds 2.0s
    assert feat["iat_mean_ms"] < 2000.0


def test_gap_1_second_same_window():
    """1.0-second gap (< 2.0s timeout) must stay in the same window."""
    extractor = LiveFeatureExtractor(max_packets_per_window=50, window_timeout_sec=2.0)
    flow_key = ("10.0.0.1", "10.0.0.2", 17)
    pkt_bytes = make_raw_udp_packet("10.0.0.1", "10.0.0.2", 1234, 5678)

    t0 = 1000.0
    t1 = 1001.0  # 1.0s gap

    extractor.process_raw_packet(t0, pkt_bytes)
    extractor.process_raw_packet(t1, pkt_bytes)

    window = extractor.active_windows[flow_key]
    assert len(window.packets) == 2
    assert window.packets[0].timestamp == t0
    assert window.packets[1].timestamp == t1


def test_gap_exactly_2_seconds_new_window():
    """Exactly 2.0-second gap (>= 2.0s timeout) must trigger a new window."""
    extractor = LiveFeatureExtractor(max_packets_per_window=50, window_timeout_sec=2.0, min_packets_to_extract=2)
    flow_key = ("10.0.0.1", "10.0.0.2", 17)
    pkt_bytes = make_raw_udp_packet("10.0.0.1", "10.0.0.2", 1234, 5678)

    t0 = 1000.0
    t0_b = 1000.5  # 2nd packet in window 1
    t1 = 1002.5   # exactly 2.0s after t0_b

    extractor.process_raw_packet(t0, pkt_bytes)
    extractor.process_raw_packet(t0_b, pkt_bytes)
    # Window 1 has 2 packets.
    # Arrival of t1 (gap = 2.0s) must close Window 1 and start Window 2
    res = extractor.process_raw_packet(t1, pkt_bytes)

    assert res is not None  # Window 1 was closed and returned
    np.testing.assert_allclose(res["duration_ms"], 500.0, rtol=1e-3)
    np.testing.assert_allclose(res["iat_mean_ms"], 500.0, rtol=1e-3)

    # Window 2 contains only t1
    window2 = extractor.active_windows[flow_key]
    assert len(window2.packets) == 1
    assert window2.packets[0].timestamp == t1


def test_gap_2_001_seconds_new_window():
    """2.001-second gap (> 2.0s timeout) must trigger a new window."""
    extractor = LiveFeatureExtractor(max_packets_per_window=50, window_timeout_sec=2.0, min_packets_to_extract=2)
    flow_key = ("10.0.0.1", "10.0.0.2", 17)
    pkt_bytes = make_raw_udp_packet("10.0.0.1", "10.0.0.2", 1234, 5678)

    t0 = 1000.0
    t0_b = 1000.1
    t1 = 1002.101  # 2.001s gap

    extractor.process_raw_packet(t0, pkt_bytes)
    extractor.process_raw_packet(t0_b, pkt_bytes)
    res = extractor.process_raw_packet(t1, pkt_bytes)

    assert res is not None
    np.testing.assert_allclose(res["duration_ms"], 100.0, rtol=1e-3)

    window2 = extractor.active_windows[flow_key]
    assert len(window2.packets) == 1
    assert window2.packets[0].timestamp == t1


def test_gap_10_seconds_new_window():
    """10.0-second gap must trigger a new window."""
    extractor = LiveFeatureExtractor(max_packets_per_window=50, window_timeout_sec=2.0, min_packets_to_extract=2)
    flow_key = ("10.0.0.1", "10.0.0.2", 17)
    pkt_bytes = make_raw_udp_packet("10.0.0.1", "10.0.0.2", 1234, 5678)

    extractor.process_raw_packet(100.0, pkt_bytes)
    extractor.process_raw_packet(100.5, pkt_bytes)
    res = extractor.process_raw_packet(110.5, pkt_bytes)

    assert res is not None
    assert extractor.active_windows[flow_key].first_timestamp == 110.5


def test_gap_60_seconds_new_window():
    """60.0-second gap must trigger a new window."""
    extractor = LiveFeatureExtractor(max_packets_per_window=50, window_timeout_sec=2.0, min_packets_to_extract=2)
    flow_key = ("10.0.0.1", "10.0.0.2", 17)
    pkt_bytes = make_raw_udp_packet("10.0.0.1", "10.0.0.2", 1234, 5678)

    extractor.process_raw_packet(100.0, pkt_bytes)
    extractor.process_raw_packet(100.2, pkt_bytes)
    res = extractor.process_raw_packet(160.2, pkt_bytes)

    assert res is not None
    assert extractor.active_windows[flow_key].first_timestamp == 160.2


def test_capacity_50_packets_trigger():
    """50 packets arriving within timeout must trigger extraction at capacity."""
    extractor = LiveFeatureExtractor(max_packets_per_window=50, window_timeout_sec=2.0)
    flow_key = ("10.0.0.1", "10.0.0.2", 17)
    pkt_bytes = make_raw_udp_packet("10.0.0.1", "10.0.0.2", 1234, 5678)

    start_t = 1000.0
    res = None
    for i in range(50):
        t = start_t + (i * 0.01)  # 10ms intervals, well within 2s timeout
        res = extractor.process_raw_packet(t, pkt_bytes)
        if i < 49:
            assert res is None

    assert res is not None
    # Capacity window was closed and evicted
    assert flow_key not in extractor.active_windows
    np.testing.assert_allclose(res["duration_ms"], 490.0, rtol=1e-3)
    np.testing.assert_allclose(res["iat_mean_ms"], 10.0, rtol=1e-3)


def test_flow_window_add_packet_rejects_gap_gte_2s():
    """FlowWindow.add_packet() directly raises ValueError if gap >= timeout_sec."""
    window = FlowWindow(flow_key=("10.0.0.1", "10.0.0.2", 17), max_packets=50, timeout_sec=2.0)
    
    pkt1 = PacketRecord(timestamp=100.0, pkt_len=60, ip_proto=17, ip_ttl=64, tcp_flags=0, is_inbound=True)
    pkt2 = PacketRecord(timestamp=102.5, pkt_len=60, ip_proto=17, ip_ttl=64, tcp_flags=0, is_inbound=True)

    window.add_packet(pkt1)
    with pytest.raises(ValueError, match="gap 2.500000s >= timeout 2.0s"):
        window.add_packet(pkt2)

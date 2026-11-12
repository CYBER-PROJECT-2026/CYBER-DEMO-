import socket
import struct
import os
import sys
from collections import Counter
from unittest.mock import patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from aegis_live.feature_schema import CANONICAL_29_FEATURES
from aegis_live.packet_window import parse_ipv4_ethernet_frame, classify_direction
from aegis_live.v4_runtime import AegisLiveV4Pipeline


def frame(proto=6, total_len=40, ihl=5, flags_frag=0, padding=b""):
    eth = b"\x00" * 12 + b"\x08\x00"
    ip = struct.pack("!BBHHHBBH4s4s", 0x40 | ihl, 0, total_len, 1, flags_frag, 64, proto, 0,
                     socket.inet_aton("192.168.1.10"), socket.inet_aton("8.8.8.8"))
    l4 = (struct.pack("!HHIIBBHHH", 1234, 443, 0, 0, 0x50, 0x18, 65535, 0, 0)
          if proto == 6 else b"\x04\xd2\x01\xbb\x00\x08\x00\x00")
    return eth + ip + l4 + padding


def test_shared_parser_validations_and_raw_length():
    good = frame(padding=b"\x00" * 8)
    parsed, error = parse_ipv4_ethernet_frame(good)
    assert error is None and parsed.pkt_len == len(good) and parsed.has_ethernet_padding
    assert parse_ipv4_ethernet_frame(frame(ihl=4))[1] == "invalid_ip_header_length"
    assert parse_ipv4_ethernet_frame(frame(total_len=19))[1] == "invalid_ipv4_total_length"
    assert parse_ipv4_ethernet_frame(frame(total_len=50))[1] == "truncated_ipv4_payload"
    assert parse_ipv4_ethernet_frame(frame(proto=6, total_len=30))[1] == "truncated_tcp_header"
    assert parse_ipv4_ethernet_frame(frame(proto=17, total_len=22))[1] == "truncated_udp_header"
    vlan = b"\x00" * 12 + b"\x81\x00" + b"\x00" * 64
    assert parse_ipv4_ethernet_frame(vlan)[1] == "non_ipv4_vlan_8021q"
    noninitial = frame(total_len=20, flags_frag=1)
    parsed, error = parse_ipv4_ethernet_frame(noninitial)
    assert error is None and parsed.is_non_initial_fragment and parsed.sport == 0


def test_direction_classes_and_frozen_feature_schema():
    assert [classify_direction(a, b) for a, b in [(1,0),(0,1),(1,1),(0,0)]] == [
        "local_to_external", "external_to_local", "local_to_local", "external_to_external"]
    assert len(CANONICAL_29_FEATURES) == 29
    assert CANONICAL_29_FEATURES[22:24] == ["inbound_pkt_ratio", "outbound_pkt_ratio"]


def test_invalid_packet_does_not_mutate_runtime_flow_or_topology():
    with patch("aegis_live.v4_runtime.AEGISLiveV4Inference") as mock:
        mock.return_value = type("Stub", (), {})()
        pipeline = AegisLiveV4Pipeline(reorder_window_sec=0.0)
    before = (len(pipeline.active_windows), len(pipeline.net_tracker.packet_queue), pipeline.timestamp_guard.last_accepted_ts)
    pipeline.ingest_packet(1.0, frame(total_len=19))
    after = (len(pipeline.active_windows), len(pipeline.net_tracker.packet_queue), pipeline.timestamp_guard.last_accepted_ts)
    assert after == before
    assert pipeline.packet_rejection_reasons["invalid_ipv4_total_length"] == 1

import math
import os
import socket
import struct
import sys
from collections import Counter
from unittest.mock import patch

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from aegis_live.feature_schema import CANONICAL_29_FEATURES
from aegis_live.v4_runtime import AegisLiveV4Pipeline
from aegis_live.build_v4_dataset import LeakageSafeV4DatasetBuilder
from aegis_live.packet_window import ReorderBufferOverflowError


def _tcp(src="192.168.1.10", dst="192.168.1.1", sport=5000, dport=1883, payload=b"T" * 8):
    eth = b"\x00" * 12 + b"\x08\x00"
    ip_len = 20 + 20 + len(payload)
    ip = struct.pack("!BBHHHBBH4s4s", 0x45, 0, ip_len, 7, 0, 64, 6, 0,
                     socket.inet_aton(src), socket.inet_aton(dst))
    tcp = struct.pack("!HHIIBBHHH", sport, dport, 1, 0, 0x50, 0x18, 65535, 0, 0)
    return eth + ip + tcp + payload


def _udp(src="8.8.8.8", dst="192.168.1.1", sport=53000, dport=53, payload=b"U" * 4):
    eth = b"\x00" * 12 + b"\x08\x00"
    ip_len = 20 + 8 + len(payload)
    ip = struct.pack("!BBHHHBBH4s4s", 0x45, 0, ip_len, 9, 0, 48, 17, 0,
                     socket.inet_aton(src), socket.inet_aton(dst))
    udp = struct.pack("!HHHH", sport, dport, 8 + len(payload), 0)
    return eth + ip + udp + payload


def _bad_total_length(buf=None):
    frame = bytearray(buf or _tcp())
    frame[16:18] = b"\x00\x13"
    return bytes(frame)


class _InferenceStub:
    def __init__(self, *args, **kwargs):
        self.calls = []

    def predict_sample(self, feats):
        self.calls.append(dict(feats))
        return {
            "predicted_class": "Benign", "calibrated_confidence": 0.99, "confidence": 0.99,
            "mahalanobis_distance": 1.0, "trusted": True, "decision": "TRUSTED", "status": "TRUSTED",
            "trusted_class": "Benign", "advisory_class": "Benign", "reasons": [], "reason_codes": [],
            "top2_margin": 0.98, "entropy": 0.05, "feature_vector": dict(feats),
            "calibrated_probabilities": {"Benign": 0.99},
        }


def _runtime(**kwargs):
    with patch("aegis_live.v4_runtime.AEGISLiveV4Inference", _InferenceStub):
        pipeline = AegisLiveV4Pipeline(**kwargs)
    return pipeline, pipeline.inference_engine


@pytest.mark.parametrize("placement", ["before", "between", "after"])
@pytest.mark.parametrize("bad_ts", [9.9, 10.05, 100.0, math.nan, math.inf])
def test_bad_frames_do_not_affect_offline_ordering_or_capacity(placement, bad_ts):
    valid = _tcp()
    bad = _bad_total_length(valid)
    ordered = [(10.0, valid), (10.1, valid), (10.2, valid)]
    index = {"before": 0, "between": 1, "after": 3}[placement]
    ordered.insert(index, (bad_ts, bad))

    pipeline, stub = _runtime(reorder_window_sec=0.25, max_reorder_packets=3)
    pipeline.replay_packet_stream(iter(ordered), reorder_window_sec=0.25, max_reorder_packets=3)

    assert pipeline.packet_rejection_reasons == {"invalid_ipv4_total_length": 1}
    assert pipeline.reorder_late_rejected == 0
    assert pipeline.timestamp_guard.accepted_count == 3
    assert pipeline.timestamp_guard.rejected_non_finite_count == 0
    assert pipeline.net_tracker.total_packets == 3
    assert not pipeline.failed_state
    assert stub.calls == []
    assert len(pipeline.incomplete_window_records) == 1
    assert pipeline.incomplete_window_records[0].packet_count == 3
    assert pipeline.incomplete_window_records[0].features["iat_mean_ms"] == pytest.approx(100.0)


def test_builder_filters_full_parse_errors_before_timestamp_guard_and_reorder():
    valid = _tcp()
    stream = [(10.0, valid), (100.0, _bad_total_length(valid)), (10.1, valid), (10.2, valid)]
    builder = LeakageSafeV4DatasetBuilder(target_per_class=10, reorder_window_sec=0.25, max_reorder_packets=3)
    builder.exclusion_counts["synthetic"] = Counter()
    result = list(builder._iter_ordered_packets(stream, "synthetic"))
    assert [ts for ts, _ in result] == [10.0, 10.1, 10.2]
    assert builder.exclusion_counts["synthetic"]["invalid_ipv4_total_length"] == 1
    assert builder.exclusion_counts["synthetic"].get("out_of_order_timestamp", 0) == 0
    assert builder.failed_state is False


def test_unsupported_and_truncated_frames_do_not_use_reorder_capacity():
    tcp = _tcp()
    udp = _udp()
    vlan = b"\x00" * 12 + b"\x81\x00" + b"\x00" * 64
    short_ethernet = b"\x00" * 13
    short_ipv4 = b"\x00" * 12 + b"\x08\x00" + b"\x45" * 10
    invalid_ihl = bytearray(tcp)
    invalid_ihl[14] = 0x44
    truncated_ip = bytearray(tcp)
    truncated_ip[16:18] = struct.pack("!H", 100)
    truncated_tcp = bytearray(tcp)
    truncated_tcp[16:18] = struct.pack("!H", 30)
    truncated_udp = bytearray(udp)
    truncated_udp[16:18] = struct.pack("!H", 24)
    invalid_frames = [short_ethernet, vlan, short_ipv4, bytes(invalid_ihl),
                      _bad_total_length(tcp), bytes(truncated_ip), bytes(truncated_tcp), bytes(truncated_udp)]
    stream = [(10.0, tcp)]
    stream.extend((20.0 + i, frame) for i, frame in enumerate(invalid_frames))
    stream.extend([(10.1, tcp), (10.2, tcp)])

    pipeline, _ = _runtime(reorder_window_sec=0.25, max_reorder_packets=3)
    pipeline.replay_packet_stream(iter(stream), reorder_window_sec=0.25, max_reorder_packets=3)
    assert pipeline.timestamp_guard.accepted_count == 3
    assert pipeline.reorder_overflows == 0
    assert pipeline.net_tracker.total_packets == 3
    assert pipeline.packet_rejection_reasons == {
        "ethernet_frame_too_short": 1,
        "non_ipv4_vlan_8021q": 1,
        "ipv4_packet_too_short": 1,
        "invalid_ip_header_length": 1,
        "invalid_ipv4_total_length": 1,
        "truncated_ipv4_payload": 1,
        "truncated_tcp_header": 1,
        "truncated_udp_header": 1,
    }

    builder = LeakageSafeV4DatasetBuilder(target_per_class=10, reorder_window_sec=0.25, max_reorder_packets=3)
    builder.exclusion_counts["invalid-mix"] = Counter()
    ordered = list(builder._iter_ordered_packets(stream, "invalid-mix"))
    assert [ts for ts, _ in ordered] == [10.0, 10.1, 10.2]
    assert builder.exclusion_counts["invalid-mix"]["reorder_buffer_overflow"] == 0
    assert builder.exclusion_counts["invalid-mix"]["out_of_order_timestamp"] == 0


def test_full_50_packet_tcp_and_udp_builder_runtime_parity():
    stream = []
    tcp = _tcp()
    udp = _udp()
    for i in range(50):
        ts = 20.0 + i * 0.01
        stream.extend([(ts, tcp), (ts, udp)])
    builder = LeakageSafeV4DatasetBuilder(target_per_class=10, window_capacity=50, reorder_window_sec=0.25)
    builder_rows = builder.extract_packet_stream_windows(iter(stream), "synthetic", "Benign")
    pipeline, stub = _runtime(max_packets_per_window=50, window_timeout_sec=2.0, reorder_window_sec=0.25)
    results = pipeline.replay_packet_stream(iter(stream), reorder_window_sec=0.25)

    assert len(builder_rows) == 2
    assert len(results) == 2
    assert len(stub.calls) == 2
    by_proto = {int(row["flow_id"].rsplit(":", 1)[1]): row for row in builder_rows}
    for detection in results:
        proto = int(detection.provenance.flow_identity.rsplit("proto=", 1)[1])
        row = by_proto[proto]
        assert detection.provenance.packet_count == row["packet_count"] == 50
        assert detection.provenance.status == "VALID"
        assert detection.provenance.v4_eligible is True
        assert detection.automated_blocking_allowed is False
        assert list(detection.feature_vector) == CANONICAL_29_FEATURES
        assert max(abs(float(detection.feature_vector[k]) - float(row[k])) for k in CANONICAL_29_FEATURES) == 0.0


@pytest.mark.parametrize("count", [1, 49])
def test_eof_partial_windows_are_diagnostic_and_never_classified(count):
    pipeline, stub = _runtime(max_packets_per_window=50, window_timeout_sec=2.0)
    frame = _tcp()
    for i in range(count):
        pipeline.ingest_packet(30.0 + i * 0.01, frame)
    results = pipeline.flush(current_time=40.0, force_all=True)
    assert results == []
    assert stub.calls == []
    assert len(pipeline.incomplete_window_records) == 1
    record = pipeline.incomplete_window_records[0]
    assert record.packet_count == count
    assert record.status == "INCOMPLETE_WINDOW"
    assert record.v4_eligible is False
    assert record.rejection_reason == "incomplete_window_requires_50_packets"
    assert record.advisory_eligible is (count >= 2)
    assert pipeline.get_summary_statistics()["incomplete_windows"][0]["status"] == "INCOMPLETE_WINDOW"


def test_timeout_and_capacity_partial_windows_are_ineligible():
    frame = _tcp()
    timed, timed_stub = _runtime(max_packets_per_window=50, window_timeout_sec=2.0)
    for i in range(49):
        timed.ingest_packet(50.0 + i * 0.01, frame)
    assert timed.ingest_packet(53.0, frame) == []
    assert timed.incomplete_window_records[0].packet_count == 49
    assert timed.incomplete_window_records[0].status == "INCOMPLETE_WINDOW"
    assert timed.incomplete_window_records[0].rejection_reason == "incomplete_window_requires_50_packets"
    timed.flush(current_time=60.0, force_all=True)
    assert [r.packet_count for r in timed.incomplete_window_records] == [49, 1]
    assert timed_stub.calls == []

    capacity, cap_stub = _runtime(max_packets_per_window=49)
    for i in range(49):
        capacity.ingest_packet(70.0 + i * 0.01, frame)
    assert capacity.detection_records == []
    assert cap_stub.calls == []
    assert capacity.incomplete_window_records[0].packet_count == 49
    assert capacity.incomplete_window_records[0].status == "INCOMPLETE_WINDOW"


def test_valid_packet_reorder_overflow_remains_fail_closed():
    valid_stream = [(10.0, _tcp()), (10.1, _tcp())]
    pipeline, _ = _runtime(reorder_window_sec=0.25, max_reorder_packets=1)
    with pytest.raises(ReorderBufferOverflowError):
        pipeline.replay_packet_stream(iter(valid_stream), reorder_window_sec=0.25, max_reorder_packets=1)
    assert pipeline.failed_state is True
    assert pipeline.reorder_overflows == 1
    assert pipeline.last_operation_status == "FAILED_REORDER_OVERFLOW"

    builder = LeakageSafeV4DatasetBuilder(target_per_class=10, reorder_window_sec=0.25, max_reorder_packets=1)
    builder.exclusion_counts["overflow"] = Counter()
    with pytest.raises(ReorderBufferOverflowError):
        list(builder._iter_ordered_packets(valid_stream, "overflow"))
    assert builder.failed_state is True
    assert builder.last_operation_status == "FAILED_REORDER_OVERFLOW"
    assert builder.exclusion_counts["overflow"]["reorder_buffer_overflow"] == 1

"""
tests/test_aegis_live_v4_timestamp_ordering.py
=============================================
Batch 26D deterministic regression suite verifying:
  - Shared MonotonicTimestampGuard strict chronological core
  - Live zero-buffer Policy A (reorder_window_sec=0.0)
  - Offline bounded reorder Policy B (lateness_sec=0.25, max_packets=10000)
  - Fail-closed ReorderBufferOverflowError
  - Builder/runtime ordering parity & Batch 26B snapshot causality preservation
  - ZERO frozen model inference (AEGISLiveV4Inference mocked cleanly)
"""

import os
import sys
import struct
import socket
import pytest
from unittest.mock import patch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from aegis_live.feature_schema import CANONICAL_29_FEATURES
from aegis_live.packet_window import (
    PacketRecord,
    FlowWindow,
    MonotonicTimestampGuard,
    BoundedReorderBuffer,
    ReorderBufferOverflowError,
)
from aegis_live.network_topology_features import NetworkTopologyPacket, NetworkTopologyTracker
from aegis_live.v4_runtime import AegisLiveV4Pipeline, PipelineFailedError
from aegis_live.build_v4_dataset import LeakageSafeV4DatasetBuilder


def _make_tcp_frame(src_ip: str = "192.168.1.10", dst_ip: str = "192.168.1.1",
                    sport: int = 5000, dport: int = 1883, payload: bytes = b"P" * 20) -> bytes:
    ip_hdr = struct.pack(
        "!BBHHHBBH4s4s",
        0x45, 0, 40 + len(payload), 1, 0, 64, 6, 0,
        socket.inet_aton(src_ip), socket.inet_aton(dst_ip),
    )
    tcp_hdr = struct.pack("!HHIIBBHHH", sport, dport, 1, 0, 0x50, 0x18, 65535, 0, 0)
    eth_hdr = b"\x00\x11\x22\x33\x44\x55\x66\x77\x88\x99\xaa\xbb\x08\x00"
    return eth_hdr + ip_hdr + tcp_hdr + payload


def _stub_inference_cls():
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


def _new_runtime(**kwargs):
    with patch("aegis_live.v4_runtime.AEGISLiveV4Inference", _stub_inference_cls()):
        return AegisLiveV4Pipeline(**kwargs)


class TestV4TimestampOrderingBatch26D:

    def test_01_chronological_packets(self):
        p = _new_runtime(local_subnet="192.168.0.0/16", max_packets_per_window=50, window_timeout_sec=2.0, min_packets_to_extract=2)
        pkt = _make_tcp_frame()
        stream = [(10.0 + i * 0.1, pkt) for i in range(5)]

        dets = p.replay_packet_stream(iter(stream))
        assert dets == []
        rec = p.incomplete_window_records[0]
        assert rec.packet_count == 5 and rec.status == "INCOMPLETE_WINDOW" and not rec.v4_eligible
        assert abs(rec.window_start_ts - 10.0) < 1e-9
        assert abs(rec.window_end_ts - 10.4) < 1e-9
        assert abs(rec.features["duration_ms"] - 400.0) < 1e-6
        assert abs(rec.features["iat_mean_ms"] - 100.0) < 1e-6
        assert p.packets_rejected == 0
        assert p.timestamp_guard.accepted_count == 5

    def test_02_equal_timestamps_stable_arrival_ordering(self):
        buf = BoundedReorderBuffer(lateness_sec=0.25, max_packets=100)
        items = [(10.0, "seq0"), (10.0, "seq1"), (10.0, "seq2"), (10.5, "seq3")]
        emitted = []
        for ts, tag in items:
            ok, reason, ready = buf.push(ts, tag)
            assert ok is True and reason is None
            emitted.extend(ready)
        emitted.extend(buf.drain())
        assert [t for _, t in emitted] == ["seq0", "seq1", "seq2", "seq3"]

        # Runtime accepts equal timestamps monotonically without dropping
        p = _new_runtime(local_subnet="192.168.0.0/16")
        pkt = _make_tcp_frame()
        dets = p.replay_packet_stream(iter([(10.0, pkt), (10.0, pkt), (10.0, pkt)]))
        assert dets == []
        assert p.incomplete_window_records[0].packet_count == 3
        assert p.incomplete_window_records[0].status == "INCOMPLETE_WINDOW"
        assert p.timestamp_guard.rejected_ooo_count == 0

    def test_03_mild_50ms_jitter_offline_vs_live(self):
        pkt = _make_tcp_frame()
        jittered = [(10.0, pkt), (10.1, pkt), (10.2, pkt), (10.15, pkt), (10.3, pkt)]

        # Offline reorder_window_sec=0.25 recovers 10.15 cleanly
        p_off = _new_runtime(local_subnet="192.168.0.0/16")
        dets_off = p_off.replay_packet_stream(iter(jittered), reorder_window_sec=0.25)
        assert dets_off == []
        assert p_off.incomplete_window_records[0].packet_count == 5
        assert abs(p_off.incomplete_window_records[0].features["iat_mean_ms"] - 75.0) < 1e-6
        assert p_off.packets_rejected == 0

        # Live mode (default reorder_window_sec=0.0) rejects 10.15 cleanly without corrupting IAT
        p_live = _new_runtime(local_subnet="192.168.0.0/16")
        dets_live = p_live.replay_packet_stream(iter(jittered))
        assert dets_live == []
        assert p_live.incomplete_window_records[0].packet_count == 4
        assert abs(p_live.incomplete_window_records[0].features["iat_mean_ms"] - 100.0) < 1e-6
        assert p_live.packet_rejection_reasons.get("out_of_order_timestamp") == 1
        assert p_live.incomplete_window_records[0].affected_by_ooo_rejection is True

    def test_04_multiple_packets_within_250ms_lateness_interval(self):
        pkt = _make_tcp_frame()
        stream = [
            (10.00, pkt),
            (10.24, pkt),
            (10.05, pkt),
            (10.18, pkt),
            (10.10, pkt),
            (10.40, pkt),
        ]
        p = _new_runtime(local_subnet="192.168.0.0/16")
        dets = p.replay_packet_stream(iter(stream), reorder_window_sec=0.25)
        assert dets == []
        rec = p.incomplete_window_records[0]
        assert rec.packet_count == 6
        assert abs(rec.window_start_ts - 10.00) < 1e-9
        assert abs(rec.window_end_ts - 10.40) < 1e-9
        assert abs(rec.features["iat_mean_ms"] - 80.0) < 1e-6  # 400ms / 5 intervals
        assert p.packets_rejected == 0

    def test_05_regression_larger_than_250ms_rejected(self):
        pkt = _make_tcp_frame()
        # 10.50 advances watermark to 10.25; then 10.10 (-400ms behind max_seen) must be rejected
        stream = [(10.00, pkt), (10.10, pkt), (10.20, pkt), (10.50, pkt), (10.10, pkt), (10.60, pkt)]
        p = _new_runtime(local_subnet="192.168.0.0/16")
        dets = p.replay_packet_stream(iter(stream), reorder_window_sec=0.25)
        assert dets == []
        assert p.incomplete_window_records[0].packet_count == 5
        assert p.incomplete_window_records[0].status == "INCOMPLETE_WINDOW"
        assert p.reorder_late_rejected == 1
        assert p.packet_rejection_reasons.get("out_of_order_timestamp") == 1

    def test_06_packet_arriving_behind_emitted_watermark_rejected(self):
        buf = BoundedReorderBuffer(lateness_sec=0.25, max_packets=100)
        assert buf.push(10.00, "p1")[2] == []
        assert buf.push(10.20, "p2")[2] == []
        # 10.55 sets watermark=10.30 and emits p1(10.00) and p2(10.20)
        ok, _, ready = buf.push(10.55, "p3")
        assert ok is True
        assert [t for _, t in ready] == ["p1", "p2"]
        assert abs(buf.last_emitted_ts - 10.20) < 1e-9

        # Arrival at 10.15 is both earlier than last_emitted_ts (10.20) and behind watermark (10.30)
        ok_late, reason, ready_late = buf.push(10.15, "late")
        assert ok_late is False
        assert reason == "earlier_than_last_emitted"
        assert ready_late == []
        assert buf.rejected_late_count == 1
        assert len(buf.rejected_events) == 1

    def test_07_end_of_stream_buffer_draining(self):
        buf = BoundedReorderBuffer(lateness_sec=0.25, max_packets=100)
        # Push 4 packets all within 200ms (none reach watermark)
        for ts, tag in [(10.20, "d"), (10.05, "b"), (10.15, "c"), (10.00, "a")]:
            ok, _, ready = buf.push(ts, tag)
            assert ok and ready == []
        assert buf.emitted_count == 0

        drained = buf.drain()
        assert [t for _, t in drained] == ["a", "b", "c", "d"]
        assert buf.emitted_count == 4

    def test_08_exactly_10000_buffered_packets_succeeds(self):
        buf = BoundedReorderBuffer(lateness_sec=0.25, max_packets=10000)
        # All 10,000 packets inside [10.0, 10.20] so all stay buffered concurrently
        for i in range(10000):
            ts = 10.0 + ((9999 - i) / 10000.0) * 0.20
            ok, reason, ready = buf.push(ts, i)
            assert ok is True and reason is None and ready == []

        assert buf.max_buffer_occupancy == 10000
        drained = buf.drain()
        assert len(drained) == 10000
        # Verify strictly monotonic non-decreasing drained timestamps
        timestamps = [ts for ts, _ in drained]
        assert all(timestamps[k] <= timestamps[k + 1] for k in range(len(timestamps) - 1))

    def test_09_buffer_overflow_fails_closed(self):
        # 1. Direct buffer overflow on 10,001st packet within 250ms window
        buf = BoundedReorderBuffer(lateness_sec=0.25, max_packets=10000)
        for i in range(10000):
            buf.push(10.0 + (i / 10000.0) * 0.20, i)
        with pytest.raises(ReorderBufferOverflowError, match="overflow"):
            buf.push(10.21, 10000)
        assert buf.overflow_count == 1

        # 2. Runtime and Builder both propagate fail-closed overflow without swallowing
        pkt = _make_tcp_frame()
        overflow_stream = [(10.0 + (i / 20.0) * 0.10, pkt) for i in range(12)]
        p = _new_runtime(local_subnet="192.168.0.0/16")
        with pytest.raises(ReorderBufferOverflowError):
            p.replay_packet_stream(iter(overflow_stream), reorder_window_sec=0.25, max_reorder_packets=10)
        assert p.reorder_overflows == 1

        builder = LeakageSafeV4DatasetBuilder(target_per_class=10, reorder_window_sec=0.25, max_reorder_packets=10)
        with pytest.raises(ReorderBufferOverflowError):
            builder.extract_packet_stream_windows(iter(overflow_stream), "overflow.pcap", "Benign")
        assert builder.exclusion_counts["overflow.pcap"]["reorder_buffer_overflow"] == 1

    def test_10_rejected_packet_never_reaches_topology_tracking(self):
        p = _new_runtime(local_subnet="192.168.0.0/16")
        p1 = _make_tcp_frame("192.168.1.10", "192.168.1.1", 5000, 1883)
        p_late = _make_tcp_frame("10.99.99.99", "192.168.1.1", 7000, 80)

        p.ingest_packet(10.0, p1)
        p.ingest_packet(10.5, p1)
        snap_before = dict(p.net_tracker.extract_current_features())

        # Reject late packet from spoofed IP 10.99.99.99 at t=9.0
        assert p.ingest_packet(9.0, p_late) == []
        snap_after = dict(p.net_tracker.extract_current_features())

        assert snap_after == snap_before
        assert "10.99.99.99" not in p.net_tracker.src_counts
        assert p.net_tracker.total_packets == 2
        assert abs(p.net_tracker.last_ts - 10.5) < 1e-9

    def test_11_rejected_packet_never_enters_flow_window_and_metadata_only(self):
        p = _new_runtime(local_subnet="192.168.0.0/16")
        pkt = _make_tcp_frame("192.168.1.10", "192.168.1.1", 5000, 1883)

        p.ingest_packet(10.0, pkt)
        p.ingest_packet(10.2, pkt)
        p.ingest_packet(10.1, pkt)  # out of order on same flow
        p.ingest_packet(10.4, pkt)

        dets = p.flush(current_time=15.0, force_all=True)
        assert dets == []
        rec = p.incomplete_window_records[0]
        assert rec.packet_count == 3 and rec.status == "INCOMPLETE_WINDOW" and not rec.v4_eligible
        assert rec.affected_by_ooo_rejection is True
        assert rec.ooo_rejected_count == 1
        assert list(rec.features.keys()) == CANONICAL_29_FEATURES
        assert "affected_by_ooo_rejection" not in rec.features

    def test_12_strict_core_rejects_direct_out_of_order_input(self):
        guard = MonotonicTimestampGuard()
        assert guard.check_and_accept(10.0)[0] is True
        assert guard.check_and_accept(10.0)[0] is True
        ok, evt = guard.check_and_accept(9.5)
        assert ok is False
        assert evt["rejected_ts"] == 9.5 and evt["previous_accepted_ts"] == 10.0

        fw = FlowWindow(("192.168.1.10", "192.168.1.1", 6), max_packets=50, timeout_sec=2.0)
        fw.add_packet(PacketRecord(10.0, 60, 6, 64, 0x18, True, 5000, 1883))
        with pytest.raises(ValueError, match="Out-of-order"):
            fw.add_packet(PacketRecord(9.9, 60, 6, 64, 0x18, True, 5000, 1883))
        assert len(fw.packets) == 1

        tracker = NetworkTopologyTracker(window_sec=1.0, mode="sliding")
        tracker.add_packet(NetworkTopologyPacket(10.0, "192.168.1.10", "192.168.1.1", 5000, 1883, 6, 60))
        with pytest.raises(ValueError, match="Out-of-order"):
            tracker.add_packet(NetworkTopologyPacket(9.9, "192.168.1.20", "192.168.1.1", 5001, 80, 6, 60))
        assert tracker.total_packets == 1

    def test_13_live_mode_zero_buffering_immediate_emission(self):
        p = _new_runtime(local_subnet="192.168.0.0/16", max_packets_per_window=50, window_timeout_sec=2.0)
        assert p.reorder_window_sec == 0.0
        pkt = _make_tcp_frame()

        for i in range(49):
            assert p.ingest_packet(10.0 + i * 0.01, pkt) == []
        # 50th packet emits immediately, and only now meets frozen V4 eligibility.
        dets = p.ingest_packet(10.49, pkt)
        assert len(dets) == 1
        assert dets[0].provenance.packet_count == 50
        assert dets[0].provenance.v4_eligible is True

    def test_14_builder_runtime_parity_on_jittered_synthetic_stream(self):
        stream = [
            (10.005, _make_tcp_frame("192.168.1.99", "192.168.1.1", 9000, 80)),
        ]
        # Build 50-packet Flow A with mild jitter (packets 10 and 11 swapped within 250ms)
        # plus 1 severe -0.45s out-of-order regression packet that both must reject
        for i in range(50):
            ts = 10.000 + i * 0.01
            if i == 10:
                ts = 10.000 + 11 * 0.01
            elif i == 11:
                ts = 10.000 + 10 * 0.01
            stream.append((ts, _make_tcp_frame("192.168.1.10", "192.168.1.1", 5000, 1883, payload=b"M" * (16 + (i % 5)))))
        # Late regression (> 250ms behind max_seen 10.49) at index 50
        stream.append((10.05, _make_tcp_frame("192.168.1.77", "192.168.1.1", 9002, 80)))
        stream.append((10.60, _make_tcp_frame("192.168.1.88", "192.168.1.1", 9001, 80)))

        builder = LeakageSafeV4DatasetBuilder(target_per_class=10, window_capacity=50, timeout_sec=2.0, reorder_window_sec=0.25)
        b_wins = builder.extract_packet_stream_windows(iter(stream), pcap_file="jitter_parity.pcap", class_label="Benign")
        assert len(b_wins) == 1
        assert builder.exclusion_counts["jitter_parity.pcap"]["out_of_order_timestamp"] == 1
        b_row = b_wins[0]

        p = _new_runtime(local_subnet="192.168.0.0/16", max_packets_per_window=50, window_timeout_sec=2.0, min_packets_to_extract=2)
        r_dets = p.replay_packet_stream(iter(stream), reorder_window_sec=0.25)
        assert p.packet_rejection_reasons.get("out_of_order_timestamp") == 1
        r_full = [d for d in r_dets if d.provenance.packet_count == 50]
        assert len(r_full) == 1
        rd = r_full[0]

        assert b_row["window_start_ts"] == rd.provenance.window_start_ts
        assert b_row["window_end_ts"] == rd.provenance.window_end_ts
        assert b_row["packet_count"] == rd.provenance.packet_count
        assert list(rd.feature_vector.keys()) == CANONICAL_29_FEATURES
        for col in CANONICAL_29_FEATURES:
            assert abs(float(b_row[col]) - float(rd.feature_vector[col])) < 1e-12

    def test_15_temporal_snapshot_causality_preserved_after_reordering(self):
        pa = _make_tcp_frame("192.168.1.10", "192.168.1.1", 5000, 1883)
        pb = _make_tcp_frame("192.168.1.20", "192.168.1.1", 5001, 80)
        pc = _make_tcp_frame("192.168.1.30", "192.168.1.2", 5002, 502)

        # Flow A arrives slightly jittered around t=10.0..10.10 while Flow B overlaps at 10.05;
        # then Flow C surges later at 11.20..11.50; Flow A times out at 12.50.
        stream = [
            (10.00, pa),
            (10.10, pa),
            (10.05, pa),   # jittered within 250ms -> reordered between 10.00 and 10.10!
            (10.06, pb),   # concurrent Flow B at 10.06 (cardinality at 10.10 must be 2.0)
            (11.30, pc),
            (11.40, pc),
            (12.50, pa),   # triggers Flow A timeout
        ]

        p = _new_runtime(local_subnet="192.168.0.0/16", max_packets_per_window=50, window_timeout_sec=2.0, min_packets_to_extract=2)
        dets = p.replay_packet_stream(iter(stream), reorder_window_sec=0.25)

        assert dets == []
        det_a_timeout = next(r for r in p.incomplete_window_records if "192.168.1.10" in r.flow_identity and r.packet_count == 3)
        assert det_a_timeout.status == "INCOMPLETE_WINDOW" and det_a_timeout.v4_eligible is False
        assert abs(det_a_timeout.window_start_ts - 10.00) < 1e-9
        assert abs(det_a_timeout.window_end_ts - 10.10) < 1e-9
        assert abs(det_a_timeout.topology_snapshot_ts - 10.10) < 1e-9
        # Reordered 10.05 + 10.06 give exact cardinality=2.0 and clean IAT=50ms (NOT leaked pc at 11.30!)
        assert det_a_timeout.features["src_ip_cardinality"] == 2.0
        assert abs(det_a_timeout.features["iat_mean_ms"] - 50.0) < 1e-6

    def test_16_non_finite_timestamps_rejected_without_state_mutation(self):
        invalid_ts_list = [float("nan"), float("inf"), float("-inf")]
        pkt = _make_tcp_frame()

        # 1. Shared MonotonicTimestampGuard
        guard = MonotonicTimestampGuard()
        assert guard.check_and_accept(10.0)[0] is True
        for bad_ts in invalid_ts_list:
            ok, evt = guard.check_and_accept(bad_ts)
            assert ok is False
            assert evt["reason"] == "non_finite_timestamp"
        assert guard.last_accepted_ts == 10.0
        assert guard.accepted_count == 1
        assert guard.rejected_non_finite_count == 3
        assert guard.rejected_ooo_count == 0
        # Equal timestamp still accepted cleanly afterward
        assert guard.check_and_accept(10.0)[0] is True
        assert guard.accepted_count == 2

        # 2. BoundedReorderBuffer
        buf = BoundedReorderBuffer(lateness_sec=0.25, max_packets=10)
        assert buf.push(10.0, "a") == (True, None, [])
        for bad_ts in invalid_ts_list:
            ok, reason, ready = buf.push(bad_ts, "bad")
            assert ok is False and reason == "non_finite_timestamp" and ready == []
        assert buf.max_seen_ts == 10.0
        assert buf.rejected_non_finite_count == 3
        assert buf.rejected_late_count == 0
        assert buf.push(10.0, "b") == (True, None, [])
        assert [t for _, t in buf.drain()] == ["a", "b"]

        # 3. Direct FlowWindow & NetworkTopologyTracker core rejection
        fw = FlowWindow(("192.168.1.10", "192.168.1.1", 6))
        fw.add_packet(PacketRecord(10.0, 60, 6, 64, 0x18, True, 5000, 1883))
        for bad_ts in invalid_ts_list:
            with pytest.raises(ValueError, match="Non-finite"):
                fw.add_packet(PacketRecord(bad_ts, 60, 6, 64, 0x18, True, 5000, 1883))
        assert len(fw.packets) == 1 and fw.last_timestamp == 10.0

        tracker = NetworkTopologyTracker(window_sec=1.0, mode="sliding")
        tracker.add_packet(NetworkTopologyPacket(10.0, "192.168.1.10", "192.168.1.1", 5000, 1883, 6, 60))
        for bad_ts in invalid_ts_list:
            with pytest.raises(ValueError, match="Non-finite"):
                tracker.add_packet(NetworkTopologyPacket(bad_ts, "10.0.0.99", "192.168.1.1", 7000, 80, 6, 60))
        assert tracker.last_ts == 10.0 and tracker.total_packets == 1

        # 4. Runtime (live & offline) and Dataset Builder
        p_live = _new_runtime(local_subnet="192.168.0.0/16")
        p_live.ingest_packet(10.0, pkt)
        for bad_ts in invalid_ts_list:
            assert p_live.ingest_packet(bad_ts, pkt) == []
        p_live.ingest_packet(10.0, pkt)  # equal timestamp preserved
        dets_live = p_live.flush(current_time=15.0, force_all=True)
        assert dets_live == [] and p_live.incomplete_window_records[0].packet_count == 2
        assert p_live.incomplete_window_records[0].status == "INCOMPLETE_WINDOW"
        assert p_live.packet_rejection_reasons.get("non_finite_timestamp") == 3
        assert p_live.timestamp_guard.rejected_non_finite_count == 3
        assert p_live.incomplete_window_records[0].affected_by_ooo_rejection is False

        p_off = _new_runtime(local_subnet="192.168.0.0/16")
        stream_off = [(10.0, pkt), (float("nan"), pkt), (float("inf"), pkt), (10.0, pkt)]
        dets_off = p_off.replay_packet_stream(iter(stream_off), reorder_window_sec=0.25)
        assert dets_off == [] and p_off.incomplete_window_records[0].packet_count == 2
        assert p_off.incomplete_window_records[0].status == "INCOMPLETE_WINDOW"
        assert p_off.reorder_non_finite_rejected == 2
        assert p_off.packet_rejection_reasons.get("non_finite_timestamp") == 2

        builder = LeakageSafeV4DatasetBuilder(target_per_class=10, window_capacity=50, reorder_window_sec=0.25)
        b_wins = builder.extract_packet_stream_windows(iter(stream_off), "nan_inf.pcap", "Benign")
        assert b_wins == [] and builder.exclusion_counts["nan_inf.pcap"]["non_finite_timestamp"] == 2

    def test_17_overflow_transaction_safety_after_completed_windows_runtime(self):
        pkt = _make_tcp_frame()
        p = _new_runtime(local_subnet="192.168.0.0/16", max_packets_per_window=50, window_timeout_sec=2.0, min_packets_to_extract=2)

        # Prior baseline session produces valid evidence before the failing replay
        pre_dets = p.replay_packet_stream(iter([(1.0 + i * 0.01, pkt) for i in range(50)]), reorder_window_sec=0.0)
        assert len(pre_dets) == 1

        # Second replay completes Window 2 cleanly first, then hits reorder buffer overflow (cap=4)
        stream = [(5.0 + i * 0.1, pkt) for i in range(50)]  # <=3 packets buffered per 250ms; full window completes.
        stream.extend([(10.0, pkt), (10.2, pkt)])  # advances watermark through packet 50 before overflow
        # Next burst puts 5 packets strictly inside [10.26, 10.30] (> cap=4) without advancing watermark enough
        stream += [(10.30 - i * 0.01, pkt) for i in range(5)]

        with pytest.raises(ReorderBufferOverflowError):
            p.replay_packet_stream(iter(stream), reorder_window_sec=0.25, max_reorder_packets=4)

        # Pipeline marked failed, diagnostic partial evidence preserved (not discarded silently)
        stats = p.get_summary_statistics()
        assert p.failed_state is True
        assert stats["failed_state"] is True
        assert stats["last_operation_status"] == "FAILED_REORDER_OVERFLOW"
        assert stats["partial_results_warning"] is True
        assert stats["last_overflow_diagnostics"]["pre_existing_detections"] == 1
        assert stats["last_overflow_diagnostics"]["partial_detections_in_failed_op"] >= 1
        assert stats["total_detections"] == len(p.detection_records) >= 2

        # Accidental reuse blocked until explicit reset()
        with pytest.raises(PipelineFailedError, match="reset"):
            p.ingest_packet(20.0, pkt)
        with pytest.raises(PipelineFailedError, match="reset"):
            p.flush(20.0, force_all=True)
        with pytest.raises(PipelineFailedError, match="reset"):
            p.replay_packet_stream(iter([(20.0, pkt)]))
        with pytest.raises(PipelineFailedError, match="reset"):
            p.replay_pcap("nonexistent.pcap")

        # Explicit reset restores clean reusable operation
        p.reset()
        assert p.failed_state is False
        assert p.last_operation_status == "OK"
        assert p.partial_results_warning is False
        assert p.detection_records == []
        resumed = p.replay_packet_stream(iter([(30.0 + i * 0.1, pkt) for i in range(50)]), reorder_window_sec=0.25)
        assert len(resumed) == 1

    def test_18_overflow_transaction_safety_builder_produces_no_output(self, tmp_path):
        pkt = _make_tcp_frame()
        stream = [(5.0 + i * 0.1, pkt) for i in range(5)]
        stream.append((6.0, pkt))
        stream += [(10.10 - i * 0.01, pkt) for i in range(6)]  # exceeds cap=4

        builder = LeakageSafeV4DatasetBuilder(
            target_per_class=10,
            window_capacity=5,
            timeout_sec=2.0,
            output_dir=str(tmp_path),
            reorder_window_sec=0.25,
            max_reorder_packets=4,
        )
        with pytest.raises(ReorderBufferOverflowError):
            builder.extract_packet_stream_windows(iter(stream), "partial_ovf.pcap", "Benign")

        assert builder.failed_state is True
        assert builder.last_operation_status == "FAILED_REORDER_OVERFLOW"
        assert builder.exclusion_counts["partial_ovf.pcap"]["reorder_buffer_overflow"] == 1
        assert list(tmp_path.iterdir()) == []

        with pytest.raises(RuntimeError, match="reset"):
            builder.extract_packet_stream_windows(iter([(20.0, pkt)]), "next.pcap", "Benign")
        with pytest.raises(RuntimeError, match="reset"):
            builder.build_dataset()

        builder.reset()
        assert builder.failed_state is False and builder.last_operation_status == "OK"

    def test_19_non_ipv4_pre_reorder_filtering_telemetry_and_capacity(self):
        tcp_pkt = _make_tcp_frame("192.168.1.10", "192.168.1.1", 5000, 1883)
        arp_pkt = b"\xff" * 6 + b"\x11\x22\x33\x44\x55\x66" + b"\x08\x06" + b"\x00" * 46
        ipv6_pkt = b"\xff" * 6 + b"\x11\x22\x33\x44\x55\x66" + b"\x86\xdd" + b"\x00" * 50
        short_eth = b"\x00" * 10
        short_ip = b"\xff" * 12 + b"\x08\x00" + b"\x45" + b"\x00" * 10

        # Interleave 6 non-IPv4 / short frames (including an out-of-order ARP at 9.0) with 4 IPv4 frames
        # inside [10.00, 10.10] while max_reorder_packets=4: must NOT overflow or mislabel non-IPv4 as OOO.
        mixed_stream = [
            (10.00, tcp_pkt),
            (10.01, arp_pkt),
            (10.02, ipv6_pkt),
            (10.06, tcp_pkt),
            (10.03, arp_pkt),
            (10.04, tcp_pkt),  # mild -20ms jitter
            (9.00, arp_pkt),   # out-of-order timestamp on ARP -> non_ipv4_packet, NOT out_of_order_timestamp!
            (10.05, short_eth),
            (10.07, short_ip),
            (10.08, tcp_pkt),
        ]
        ipv4_only_stream = [
            (10.00, tcp_pkt),
            (10.06, tcp_pkt),
            (10.04, tcp_pkt),
            (10.08, tcp_pkt),
        ]

        p_mixed = _new_runtime(local_subnet="192.168.0.0/16", max_packets_per_window=50, window_timeout_sec=2.0)
        dets_mixed = p_mixed.replay_packet_stream(iter(mixed_stream), reorder_window_sec=0.25, max_reorder_packets=4)
        p_clean = _new_runtime(local_subnet="192.168.0.0/16", max_packets_per_window=50, window_timeout_sec=2.0)
        dets_clean = p_clean.replay_packet_stream(iter(ipv4_only_stream), reorder_window_sec=0.25, max_reorder_packets=4)

        assert dets_mixed == [] and dets_clean == []
        assert p_mixed.incomplete_window_records[0].packet_count == 4
        assert p_clean.incomplete_window_records[0].packet_count == 4
        assert p_mixed.packet_rejection_reasons.get("out_of_order_timestamp", 0) == 0
        assert p_mixed.reorder_late_rejected == 0
        assert p_mixed.packet_rejection_reasons["non_ipv4_packet"] == 4
        assert p_mixed.packet_rejection_reasons["ethernet_frame_too_short"] == 1
        assert p_mixed.packet_rejection_reasons["ipv4_packet_too_short"] == 1
        for col in CANONICAL_29_FEATURES:
            assert p_mixed.incomplete_window_records[0].features[col] == p_clean.incomplete_window_records[0].features[col]

        # Dataset builder parity on identical mixed stream
        builder = LeakageSafeV4DatasetBuilder(target_per_class=10, window_capacity=50, timeout_sec=2.0, reorder_window_sec=0.25, max_reorder_packets=4)
        b_wins = builder.extract_packet_stream_windows(iter(mixed_stream), "mixed.pcap", "Benign")
        assert b_wins == []
        assert builder.exclusion_counts["mixed.pcap"]["out_of_order_timestamp"] == 0
        assert builder.exclusion_counts["mixed.pcap"]["non_ipv4_packet"] == 4
        assert builder.exclusion_counts["mixed.pcap"]["ethernet_frame_too_short"] == 1
        assert builder.exclusion_counts["mixed.pcap"]["ipv4_packet_too_short"] == 1

    def test_20_effective_configuration_telemetry_all_modes(self):
        import io
        import dpkt
        pkt = _make_tcp_frame()

        # 1. Default live configuration
        p = _new_runtime(local_subnet="192.168.0.0/16")
        tg = p.get_summary_statistics()["timestamp_guard"]
        assert tg["configured_reorder_window_sec"] == 0.0
        assert tg["reorder_window_sec"] == 0.0
        assert tg["max_reorder_packets"] == 10000
        assert tg["operation_mode"] == "live_zero_buffer"

        # 2. Explicit live override (both replay override to 0.0 and direct ingest_packet)
        p_init_off = _new_runtime(local_subnet="192.168.0.0/16", reorder_window_sec=0.25, max_reorder_packets=5000)
        p_init_off.replay_packet_stream(iter([(10.0, pkt), (10.1, pkt)]), reorder_window_sec=0.0, max_reorder_packets=250)
        tg_over = p_init_off.get_summary_statistics()["timestamp_guard"]
        assert tg_over["configured_reorder_window_sec"] == 0.25
        assert tg_over["reorder_window_sec"] == 0.0
        assert tg_over["max_reorder_packets"] == 250
        assert tg_over["operation_mode"] == "live_zero_buffer"

        p_init_off.ingest_packet(10.2, pkt)
        tg_direct = p_init_off.get_summary_statistics()["timestamp_guard"]
        assert tg_direct["reorder_window_sec"] == 0.0
        assert tg_direct["operation_mode"] == "live_zero_buffer"

        # 3. Offline replay default (mocked dpkt reader in memory — NO PCAP file scanned)
        buf_io = io.BytesIO()
        writer = dpkt.pcap.Writer(buf_io)
        writer.writepkt(pkt, ts=10.0)
        writer.writepkt(pkt, ts=10.1)
        raw_pcap_bytes = buf_io.getvalue()
        with patch("aegis_live.v4_runtime.os.path.exists", return_value=True), \
             patch("builtins.open", return_value=io.BytesIO(raw_pcap_bytes)):
            p.replay_pcap("synthetic_in_memory.pcap")
        tg_pcap_def = p.get_summary_statistics()["timestamp_guard"]
        assert tg_pcap_def["configured_reorder_window_sec"] == 0.0
        assert tg_pcap_def["reorder_window_sec"] == 0.25
        assert tg_pcap_def["max_reorder_packets"] == 10000
        assert tg_pcap_def["operation_mode"] == "offline_bounded_reorder"

        # 4. Offline replay override
        p.replay_packet_stream(iter([(20.0, pkt), (20.1, pkt)]), reorder_window_sec=0.15, max_reorder_packets=2048)
        tg_pcap_over = p.get_summary_statistics()["timestamp_guard"]
        assert tg_pcap_over["reorder_window_sec"] == 0.15
        assert tg_pcap_over["max_reorder_packets"] == 2048
        assert tg_pcap_over["operation_mode"] == "offline_bounded_reorder"

        # 5. Dataset builder configuration
        b_def = LeakageSafeV4DatasetBuilder()
        cfg_def = b_def.get_ordering_configuration()
        assert cfg_def["reorder_window_sec"] == 0.25
        assert cfg_def["max_reorder_packets"] == 10000
        assert cfg_def["operation_mode"] == "offline_bounded_reorder"
        assert cfg_def["failed_state"] is False



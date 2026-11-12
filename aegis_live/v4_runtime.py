"""
aegis_live/v4_runtime.py
========================
AEGIS-Live v4 Live Packet Ingestion, 29-Feature Fusion, and Inference Pipeline.
STRICTLY DETECTION-ONLY MODE: No automated mitigation, no firewall actions, no blocking.

FROZEN ARTIFACTS LOADED:
- Model: models/aegis_live_v4/aegis_live_xgboost_baseline.pkl
- Calibration: models/aegis_live_v4/temperature_scaling.json (T = 1.199787)
- OOD Parameters: models/aegis_live_v4/ood_parameters.pkl
- Abstention Config: models/aegis_live_v4/abstention_config.json

DECISION CONTRACT:
- If trusted:
    status = "TRUSTED"
    decision = "TRUSTED"
    trusted = True
    trusted_class = predicted_class
    advisory_class = predicted_class
    automated_blocking_allowed = False
- If abstained:
    status = "UNKNOWN_OR_REVIEW"
    decision = "UNKNOWN_OR_REVIEW"
    trusted = False
    trusted_class = None
    advisory_class = predicted_class
    automated_blocking_allowed = False
    reasons = [triggered_reasons]
"""

import os
import sys
import math
import time
import socket
import ipaddress
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Generator, Union

import numpy as np
import pandas as pd
import dpkt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from aegis_live.feature_schema import CANONICAL_24_FEATURES, CANONICAL_29_FEATURES, validate_29_features
from aegis_live.packet_window import (
    PacketRecord,
    FlowWindow,
    MonotonicTimestampGuard,
    BoundedReorderBuffer,
    ReorderBufferOverflowError,
    parse_ipv4_ethernet_frame,
    classify_direction,
)
from aegis_live.network_topology_features import NetworkTopologyPacket, NetworkTopologyTracker
from aegis_live.v4_inference import AEGISLiveV4Inference


class PipelineFailedError(RuntimeError):
    """Raised when an operation is attempted on a pipeline in a failed overflow state prior to reset()."""


V4_TRAINING_WINDOW_PACKETS = 50


@dataclass
class WindowProvenanceRecordV4:
    interface: str
    window_start_ts: float
    window_end_ts: float
    packet_count: int
    flow_identity: str
    features: Dict[str, float]
    status: str                         # "VALID", "INCOMPLETE_WINDOW", or "INELIGIBLE_WINDOW"
    rejection_reason: Optional[str] = None
    v4_eligible: bool = True
    advisory_eligible: bool = False
    topology_snapshot_ts: Optional[float] = None
    affected_by_ooo_rejection: bool = False
    ooo_rejected_count: int = 0
    output_mode: str = "PRODUCTION"
    evaluation_id: Optional[str] = None


@dataclass
class DetectionResultV4:
    predicted_class: str
    calibrated_confidence: float
    confidence: float
    mahalanobis_distance: float
    trusted: bool
    decision: str                       # "TRUSTED" or "UNKNOWN_OR_REVIEW"
    status: str                         # "TRUSTED" or "UNKNOWN_OR_REVIEW"
    trusted_class: Optional[str]
    advisory_class: str
    automated_blocking_allowed: bool    # IMMUTABLY FALSE
    reasons: List[str]
    reason_codes: List[str]
    top2_margin: float
    entropy: float
    feature_vector: Dict[str, float]
    calibrated_probabilities: Dict[str, float]
    provenance: Optional[WindowProvenanceRecordV4] = None
    emitted_at_utc: float = field(default_factory=time.time)
    service_port: Optional[int] = None
    ports_seen: Optional[List[int]] = None
    output_mode: str = "PRODUCTION"
    evaluation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_class": self.predicted_class,
            "calibrated_confidence": self.calibrated_confidence,
            "confidence": self.confidence,
            "mahalanobis_distance": self.mahalanobis_distance,
            "trusted": self.trusted,
            "decision": self.decision,
            "status": self.status,
            "trusted_class": self.trusted_class,
            "advisory_class": self.advisory_class,
            "automated_blocking_allowed": self.automated_blocking_allowed,
            "reasons": self.reasons,
            "reason_codes": self.reason_codes,
            "top2_margin": self.top2_margin,
            "entropy": self.entropy,
            "feature_vector": self.feature_vector,
            "calibrated_probabilities": self.calibrated_probabilities,
            "emitted_at_utc": self.emitted_at_utc,
            "service_port": self.service_port,
            "ports_seen": self.ports_seen or [],
            "output_mode": self.output_mode,
            "evaluation_id": self.evaluation_id,
        }


class AegisLiveV4Pipeline:
    """
    AEGIS-Live v4 Production Runtime Pipeline.
    
    Streams raw packets through:
      1. Ethernet & IPv4 parsing with strict packet validation.
      2. Causal 1.0-second network-level topology tracker (sliding FIFO with strict eviction).
      3. Per-flow 50-packet sliding windows (with 2.0s inactivity timeout).
      4. Exact 29-feature fusion (24 physical flow features + 5 network topology features).
      5. Frozen v4 XGBoost classifier + temperature scaling + Mahalanobis OOD gating.
    """

    def __init__(self,
                 local_subnet: str = "192.168.0.0/16",
                 max_packets_per_window: int = 50,
                 window_timeout_sec: float = 2.0,
                 min_packets_to_extract: int = 2,
                 net_window_sec: float = 1.0,
                 interface_name: str = "v4_live_pipeline",
                 model_dir: Optional[str] = None,
                 reorder_window_sec: float = 0.0,
                 max_reorder_packets: int = 10000,
                 output_mode: str = "PRODUCTION",
                 evaluation_id: Optional[str] = None):
        self.local_network = ipaddress.ip_network(local_subnet)
        self.max_packets = max_packets_per_window
        self.timeout_sec = window_timeout_sec
        self.min_packets = min_packets_to_extract
        self.net_window_sec = net_window_sec
        self.interface_name = interface_name
        self.output_mode = str(output_mode).upper()
        if self.output_mode not in {"PRODUCTION", "SHADOW_EVALUATION"}:
            raise ValueError("output_mode must be PRODUCTION or SHADOW_EVALUATION")
        if self.output_mode == "SHADOW_EVALUATION" and not evaluation_id:
            raise ValueError("evaluation_id is required for SHADOW_EVALUATION output")
        if self.output_mode == "PRODUCTION" and evaluation_id is not None:
            raise ValueError("evaluation_id is only valid for SHADOW_EVALUATION output")
        self.evaluation_id = str(evaluation_id) if evaluation_id is not None else None
        rw = float(reorder_window_sec)
        if not math.isfinite(rw) or rw < 0.0:
            raise ValueError("reorder_window_sec must be finite and non-negative")
        mr = int(max_reorder_packets)
        if mr < 1:
            raise ValueError("max_reorder_packets must be >= 1")
        self.reorder_window_sec = rw
        self.max_reorder_packets = mr
        self.effective_reorder_window_sec: float = rw
        self.effective_max_reorder_packets: int = mr
        self.last_operation_mode: str = "offline_bounded_reorder" if rw > 0.0 else "live_zero_buffer"
        self._in_replay_stream: bool = False

        # State components
        self.active_windows: Dict[Tuple[str, str, int], FlowWindow] = {}
        self.net_tracker = NetworkTopologyTracker(window_sec=self.net_window_sec, mode="sliding")
        self.timestamp_guard = MonotonicTimestampGuard()
        self.inference_engine = AEGISLiveV4Inference(model_dir=model_dir)

        # Telemetry & Diagnostics
        self.packets_captured = 0
        self.packets_rejected = 0
        self.packet_rejection_reasons: Dict[str, int] = {}
        self.windows_generated = 0
        self.windows_rejected = 0
        self.window_rejection_reasons: Dict[str, int] = {}
        self.detection_records: List[DetectionResultV4] = []
        self.incomplete_window_records: List[WindowProvenanceRecordV4] = []
        self.reorder_late_rejected = 0
        self.reorder_non_finite_rejected = 0
        self.reorder_overflows = 0
        self.directionality_counts = {k: 0 for k in ("local_to_external", "external_to_local", "local_to_local", "external_to_external")}

        # Overflow / Transaction Safety State
        self.failed_state: bool = False
        self.failure_reason: Optional[str] = None
        self.last_operation_status: str = "OK"
        self.partial_results_warning: bool = False
        self.last_overflow_diagnostics: Optional[Dict[str, Any]] = None

    def _assert_not_failed(self, operation: str) -> None:
        if self.failed_state:
            raise PipelineFailedError(
                f"Cannot execute {operation}: pipeline is in failed state "
                f"({self.last_operation_status}: {self.failure_reason}). Call reset() before reusing."
            )

    def reset(self):
        """
        Completely resets all active flow windows, network topology tracking buffers,
        counters, failed transaction state, and emitted records for an independent capture or replay session.
        """
        self.active_windows.clear()
        self.net_tracker.reset()
        self.timestamp_guard.reset()
        self.packets_captured = 0
        self.packets_rejected = 0
        self.packet_rejection_reasons.clear()
        self.windows_generated = 0
        self.windows_rejected = 0
        self.window_rejection_reasons.clear()
        self.detection_records.clear()
        self.incomplete_window_records.clear()
        self.reorder_late_rejected = 0
        self.reorder_non_finite_rejected = 0
        self.reorder_overflows = 0
        for key in self.directionality_counts:
            self.directionality_counts[key] = 0
        self.failed_state = False
        self.failure_reason = None
        self.last_operation_status = "OK"
        self.partial_results_warning = False
        self.last_overflow_diagnostics = None
        self.effective_reorder_window_sec = self.reorder_window_sec
        self.effective_max_reorder_packets = self.max_reorder_packets
        self.last_operation_mode = "offline_bounded_reorder" if self.reorder_window_sec > 0.0 else "live_zero_buffer"
        self._in_replay_stream = False

    def _is_local(self, ip_str: str) -> bool:
        try:
            ip = ipaddress.ip_address(ip_str)
            if ip in self.local_network:
                return True
            if hasattr(self, "additional_local_networks"):
                for net in self.additional_local_networks:
                    if ip in net:
                        return True
            return False
        except ValueError:
            return False

    def add_local_subnet(self, subnet_cidr: str):
        """Allows registering supplementary subnets (e.g. simulated IoT lab network) as local."""
        if not hasattr(self, "additional_local_networks"):
            self.additional_local_networks = []
        try:
            net = ipaddress.ip_network(subnet_cidr, strict=False)
            if net not in self.additional_local_networks:
                self.additional_local_networks.append(net)
        except Exception:
            pass

    def _record_packet_rejection(self, reason: str):
        self.packets_rejected += 1
        self.packet_rejection_reasons[reason] = self.packet_rejection_reasons.get(reason, 0) + 1

    def _record_window_rejection(self, reason: str):
        self.windows_rejected += 1
        self.window_rejection_reasons[reason] = self.window_rejection_reasons.get(reason, 0) + 1

    def _record_incomplete_window(self, window: FlowWindow, trigger_reason: str) -> WindowProvenanceRecordV4:
        """Retains non-50-packet windows for diagnostics without invoking frozen V4 inference."""
        packet_count = len(window.packets)
        eligibility_reason = (
            "incomplete_window_requires_50_packets"
            if packet_count < V4_TRAINING_WINDOW_PACKETS
            else "window_packet_count_not_50"
        )
        features = self._extract_combined_29_features(window)
        snapshot = getattr(window, "last_topology_snapshot", None)
        record = WindowProvenanceRecordV4(
            interface=self.interface_name,
            window_start_ts=float(window.packets[0].timestamp),
            window_end_ts=float(window.packets[-1].timestamp),
            packet_count=packet_count,
            flow_identity=f"{window.flow_key[0]} <-> {window.flow_key[1]} proto={window.flow_key[2]}",
            features=features,
            status="INCOMPLETE_WINDOW" if packet_count < V4_TRAINING_WINDOW_PACKETS else "INELIGIBLE_WINDOW",
            rejection_reason=eligibility_reason,
            v4_eligible=False,
            advisory_eligible=packet_count >= self.min_packets,
            topology_snapshot_ts=(float(snapshot["window_end_ts"]) if snapshot and "window_end_ts" in snapshot else float(window.packets[-1].timestamp)),
            affected_by_ooo_rejection=bool(getattr(window, "affected_by_ooo_rejection", False)),
            ooo_rejected_count=int(getattr(window, "ooo_rejected_count", 0)),
            output_mode=self.output_mode,
            evaluation_id=self.evaluation_id,
        )
        self.incomplete_window_records.append(record)
        self.windows_rejected += 1
        self.window_rejection_reasons[trigger_reason] = self.window_rejection_reasons.get(trigger_reason, 0) + 1
        return record

    def _extract_combined_29_features(self, window: FlowWindow) -> Dict[str, float]:
        """
        Extracts the exact 29-feature vector at the window's completion timestamp.
        Guarantees:
        - 24 features from FlowWindow are 100% equivalent to canonical v2/v3 extractor.
        - 5 features from NetworkTopologyTracker come from this window's causal snapshot
          captured at window.last_timestamp (packets with t <= window.last_timestamp
          and t >= window.last_timestamp - 1.0s), avoiding future leakage on timeout/flush.
        """
        feats_24 = window.extract_features()
        if getattr(window, "last_topology_snapshot", None) is not None:
            net_feats = window.last_topology_snapshot
        else:
            net_feats = self.net_tracker.extract_current_features()

        return {
            **feats_24,
            "src_concentration_ratio": float(net_feats["src_concentration_ratio"]),
            "src_ip_entropy": float(net_feats["src_ip_entropy"]),
            "src_ip_cardinality": float(net_feats["src_ip_cardinality"]),
            "dst_ip_cardinality": float(net_feats["dst_ip_cardinality"]),
            "fan_in_ratio": float(net_feats["fan_in_ratio"])
        }

    def _process_completed_window(self, window: FlowWindow) -> Optional[DetectionResultV4]:
        """Processes a completed FlowWindow and runs v4 inference and OOD gating."""
        if len(window.packets) != V4_TRAINING_WINDOW_PACKETS:
            self._record_incomplete_window(window, "window_size_not_50_at_classification_gate")
            return None
        flow_id = f"{window.flow_key[0]} <-> {window.flow_key[1]} proto={window.flow_key[2]}"
        start_ts = float(window.packets[0].timestamp)
        end_ts = float(window.packets[-1].timestamp)
        pkt_cnt = len(window.packets)

        feats_29 = self._extract_combined_29_features(window)
        self.windows_generated += 1

        snapshot = getattr(window, "last_topology_snapshot", None)
        snap_ts = float(snapshot["window_end_ts"]) if snapshot and "window_end_ts" in snapshot else end_ts

        prov = WindowProvenanceRecordV4(
            interface=self.interface_name,
            window_start_ts=start_ts,
            window_end_ts=end_ts,
            packet_count=pkt_cnt,
            flow_identity=flow_id,
            features=feats_29,
            status="VALID",
            v4_eligible=True,
            advisory_eligible=True,
            topology_snapshot_ts=snap_ts,
            affected_by_ooo_rejection=bool(getattr(window, "affected_by_ooo_rejection", False)),
            ooo_rejected_count=int(getattr(window, "ooo_rejected_count", 0)),
            output_mode=self.output_mode,
            evaluation_id=self.evaluation_id
        )

        pred = self.inference_engine.predict_sample(feats_29)

        ports_seen = sorted(list(window.ports_seen)) if hasattr(window, "ports_seen") and window.ports_seen else []
        svc_port = None
        try:
            from backend.detection.incident_correlator import resolve_service_port
            resolved = resolve_service_port(ports_seen)
            if resolved > 0:
                svc_port = resolved
        except Exception:
            svc_port = None

        det = DetectionResultV4(
            predicted_class=pred["predicted_class"],
            calibrated_confidence=pred["calibrated_confidence"],
            confidence=pred["confidence"],
            mahalanobis_distance=pred["mahalanobis_distance"],
            trusted=pred["trusted"],
            decision=pred["decision"],
            status=pred["status"],
            trusted_class=pred["trusted_class"],
            advisory_class=pred["advisory_class"],
            automated_blocking_allowed=False,
            reasons=pred["reasons"],
            reason_codes=pred["reason_codes"],
            top2_margin=pred["top2_margin"],
            entropy=pred["entropy"],
            feature_vector=pred["feature_vector"],
            calibrated_probabilities=pred["calibrated_probabilities"],
            provenance=prov,
            service_port=svc_port,
            ports_seen=ports_seen,
            output_mode=self.output_mode,
            evaluation_id=self.evaluation_id
        )
        self.detection_records.append(det)
        return det

    def ingest_packet(self, ts: float, buf: bytes) -> List[DetectionResultV4]:
        """
        Ingests a single raw Ethernet packet into the v4 live pipeline.
        Returns any detection results emitted as a consequence of window completion.
        """
        self._assert_not_failed("ingest_packet")
        if not self._in_replay_stream:
            self.effective_reorder_window_sec = 0.0
            self.effective_max_reorder_packets = self.max_reorder_packets
            self.last_operation_mode = "live_zero_buffer"

        self.packets_captured += 1
        # Fully validate the frame before converting or accepting its timestamp.
        parsed, parse_err = parse_ipv4_ethernet_frame(buf)
        if parse_err is not None:
            self._record_packet_rejection(parse_err)
            return []
        ts = float(ts)
        if not math.isfinite(ts):
            self.timestamp_guard.check_and_accept(ts)
            self._record_packet_rejection("non_finite_timestamp")
            return []
        src_ip, dst_ip = parsed.src_ip, parsed.dst_ip
        ip_proto, ttl = parsed.ip_proto, parsed.ip_ttl
        sport, dport, tcp_flags = parsed.sport, parsed.dport, parsed.tcp_flags

        # 3. Strict Chronological Core Guard (reject out-of-order packets before mutating topology or flow state)
        ip_pair = tuple(sorted([src_ip, dst_ip]))
        flow_key = (ip_pair[0], ip_pair[1], ip_proto)
        accepted_ts, _ = self.timestamp_guard.check_and_accept(ts)
        if not accepted_ts:
            self._record_packet_rejection("out_of_order_timestamp")
            if flow_key in self.active_windows:
                self.active_windows[flow_key].mark_ooo_rejection()
            return []

        # 4. Update Causal Network Topology Tracker (strictly at packet ts <= Tend)
        net_pkt = NetworkTopologyPacket(
            timestamp=ts,
            src_ip=src_ip,
            dst_ip=dst_ip,
            src_port=sport,
            dst_port=dport,
            protocol=ip_proto,
            pkt_len=parsed.pkt_len
        )
        self.net_tracker.add_packet(net_pkt)

        # 5. Construct Flow Packet Record
        is_inbound = self._is_local(dst_ip)
        direction = classify_direction(self._is_local(src_ip), is_inbound)
        self.directionality_counts[direction] += 1
        pkt_rec = PacketRecord(
            timestamp=ts,
            pkt_len=len(buf),
            ip_proto=ip_proto,
            ip_ttl=ttl,
            tcp_flags=tcp_flags,
            is_inbound=is_inbound,
            sport=sport,
            dport=dport
        )

        emitted: List[DetectionResultV4] = []

        if flow_key in self.active_windows:
            window = self.active_windows[flow_key]
            gap = (pkt_rec.timestamp - window.last_timestamp) if window.last_timestamp is not None else 0.0

            if gap >= self.timeout_sec:
                # Only the frozen 50-packet contract may enter classification.
                if len(window.packets) == V4_TRAINING_WINDOW_PACKETS:
                    det = self._process_completed_window(window)
                    if det is not None:
                        emitted.append(det)
                else:
                    self._record_incomplete_window(window, "subcapacity_window_timeout")

                # Fresh window for new packet
                fresh_window = FlowWindow(flow_key, self.max_packets, self.timeout_sec)
                ready = fresh_window.add_packet(pkt_rec)
                fresh_window.set_topology_snapshot(self.net_tracker.extract_current_features())
                if ready:
                    det = self._process_completed_window(fresh_window)
                    if det is not None:
                        emitted.append(det)
                    del self.active_windows[flow_key]
                else:
                    self.active_windows[flow_key] = fresh_window
            else:
                ready = window.add_packet(pkt_rec)
                window.set_topology_snapshot(self.net_tracker.extract_current_features())
                if ready:
                    det = self._process_completed_window(window)
                    if det is not None:
                        emitted.append(det)
                    del self.active_windows[flow_key]
        else:
            fresh_window = FlowWindow(flow_key, self.max_packets, self.timeout_sec)
            ready = fresh_window.add_packet(pkt_rec)
            fresh_window.set_topology_snapshot(self.net_tracker.extract_current_features())
            if ready:
                det = self._process_completed_window(fresh_window)
                if det is not None:
                    emitted.append(det)
            else:
                self.active_windows[flow_key] = fresh_window

        return emitted

    def flush(self, current_time: float, force_all: bool = False) -> List[DetectionResultV4]:
        """
        Flushes inactive flow windows exceeding timeout_sec, or force all on end-of-stream.
        """
        self._assert_not_failed("flush")
        current_time = float(current_time)
        emitted: List[DetectionResultV4] = []
        keys_to_remove = []

        for flow_key, window in list(self.active_windows.items()):
            gap = (current_time - window.last_timestamp) if window.last_timestamp is not None else float("inf")
            if force_all or gap >= self.timeout_sec:
                if len(window.packets) == V4_TRAINING_WINDOW_PACKETS:
                    det = self._process_completed_window(window)
                    if det is not None:
                        emitted.append(det)
                else:
                    self._record_incomplete_window(window, "subcapacity_flushed_window")
                keys_to_remove.append(flow_key)

        for k in keys_to_remove:
            if k in self.active_windows:
                del self.active_windows[k]

        return emitted

    def _mark_flow_ooo_from_bytes(self, buf: bytes) -> None:
        """Marks an active flow window as touched by a rejected out-of-order packet (metadata only)."""
        if len(buf) >= 34 and ((buf[12] << 8) | buf[13]) == 0x0800:
            ip_proto = buf[23]
            src_ip = socket.inet_ntoa(buf[26:30])
            dst_ip = socket.inet_ntoa(buf[30:34])
            ip_pair = tuple(sorted([src_ip, dst_ip]))
            flow_key = (ip_pair[0], ip_pair[1], ip_proto)
            if flow_key in self.active_windows:
                self.active_windows[flow_key].mark_ooo_rejection()

    def replay_packet_stream(self,
                             packet_stream: Generator[Tuple[float, bytes], None, None],
                             max_windows: Optional[int] = None,
                             reorder_window_sec: Optional[float] = None,
                             max_reorder_packets: Optional[int] = None) -> List[DetectionResultV4]:
        """
        Replays an arbitrary generator of (ts, raw_bytes) tuples.
        Defaults to self.reorder_window_sec (0.0s zero-buffer Policy A for live capture).
        Fully validates Ethernet/IPv4/L4 frames before timestamp conversion or bounded
        (ts, seq) reordering, then applies the strict core guard.
        Fails closed with ReorderBufferOverflowError and transitions pipeline into failed_state.
        """
        self._assert_not_failed("replay_packet_stream")
        lateness = self.reorder_window_sec if reorder_window_sec is None else float(reorder_window_sec)
        if not math.isfinite(lateness) or lateness < 0.0:
            raise ValueError("reorder_window_sec must be finite and non-negative")
        cap = self.max_reorder_packets if max_reorder_packets is None else int(max_reorder_packets)
        if cap < 1:
            raise ValueError("max_reorder_packets must be >= 1")

        self.effective_reorder_window_sec = lateness
        self.effective_max_reorder_packets = cap
        self.last_operation_mode = "offline_bounded_reorder" if lateness > 0.0 else "live_zero_buffer"

        new_detections: List[DetectionResultV4] = []
        pre_replay_detections = len(self.detection_records)
        stopped_early = False

        if lateness <= 0.0:
            self._in_replay_stream = True
            try:
                for ts, buf in packet_stream:
                    dets = self.ingest_packet(ts, buf)
                    new_detections.extend(dets)
                    if max_windows is not None and len(self.detection_records) >= max_windows:
                        stopped_early = True
                        break
            finally:
                self._in_replay_stream = False
        else:
            reorder_buf = BoundedReorderBuffer(lateness_sec=lateness, max_packets=cap)
            self._in_replay_stream = True
            try:
                for ts, buf in packet_stream:
                    _, parse_err = parse_ipv4_ethernet_frame(buf)
                    if parse_err is not None:
                        self.packets_captured += 1
                        self._record_packet_rejection(parse_err)
                        continue
                    ts_f = float(ts)
                    if not math.isfinite(ts_f):
                        reorder_buf.push(ts_f, buf)
                        self.packets_captured += 1
                        self._record_packet_rejection("non_finite_timestamp")
                        self.reorder_non_finite_rejected += 1
                        continue

                    accepted, _, ready = reorder_buf.push(ts_f, buf)
                    if not accepted:
                        self.packets_captured += 1
                        self._record_packet_rejection("out_of_order_timestamp")
                        self.reorder_late_rejected += 1
                        self._mark_flow_ooo_from_bytes(buf)
                        continue
                    for emit_ts, emit_buf in ready:
                        dets = self.ingest_packet(emit_ts, emit_buf)
                        new_detections.extend(dets)
                        if max_windows is not None and len(self.detection_records) >= max_windows:
                            stopped_early = True
                            break
                    if stopped_early:
                        break
            except ReorderBufferOverflowError as exc:
                self.reorder_overflows += 1
                self.failed_state = True
                self.failure_reason = str(exc)
                self.last_operation_status = "FAILED_REORDER_OVERFLOW"
                if len(self.detection_records) > 0:
                    self.partial_results_warning = True
                self.last_overflow_diagnostics = {
                    "pre_existing_detections": pre_replay_detections,
                    "partial_detections_in_failed_op": len(new_detections),
                    "total_retained_detections": len(self.detection_records),
                    "partial_evidence_retained": True,
                }
                raise
            finally:
                self._in_replay_stream = False

            if not stopped_early:
                self._in_replay_stream = True
                try:
                    for emit_ts, emit_buf in reorder_buf.drain():
                        dets = self.ingest_packet(emit_ts, emit_buf)
                        new_detections.extend(dets)
                        if max_windows is not None and len(self.detection_records) >= max_windows:
                            break
                finally:
                    self._in_replay_stream = False

        # Final flush on end of stream using highest accepted chronological timestamp
        flush_base_ts = self.timestamp_guard.last_accepted_ts or 0.0
        flushed = self.flush(current_time=flush_base_ts + 5.0, force_all=True)
        new_detections.extend(flushed)
        self.last_operation_status = "OK"
        return new_detections

    def replay_pcap(self,
                    pcap_path: str,
                    max_packets: Optional[int] = None,
                    max_windows: Optional[int] = None,
                    packet_range: Optional[Tuple[int, int]] = None,
                    reorder_window_sec: float = 0.25,
                    max_reorder_packets: int = 10000) -> List[DetectionResultV4]:
        """
        Replays an authentic PCAP capture file through the exact v4 pipeline.
        Uses bounded 250ms (max 10,000 packets) offline reordering by default before
        passing monotonic packets into the strict core guard.
        """
        self._assert_not_failed("replay_pcap")
        if not os.path.exists(pcap_path):
            raise FileNotFoundError(f"PCAP not found: {pcap_path}")

        prev_iface = self.interface_name
        self.interface_name = f"PCAP_REPLAY:{os.path.basename(pcap_path)}"
        start_idx = packet_range[0] if packet_range else 0
        end_idx = packet_range[1] if packet_range else None

        try:
            with open(pcap_path, "rb") as f:
                try:
                    reader = dpkt.pcap.Reader(f)
                except ValueError:
                    f.seek(0)
                    reader = dpkt.pcapng.Reader(f)

                def _pcap_slice():
                    pkt_idx = 0
                    for ts, buf in reader:
                        if pkt_idx < start_idx:
                            pkt_idx += 1
                            continue
                        if end_idx is not None and pkt_idx > end_idx:
                            break
                        yield float(ts), buf
                        pkt_idx += 1
                        if max_packets is not None and (pkt_idx - start_idx) >= max_packets:
                            break

                return self.replay_packet_stream(
                    _pcap_slice(),
                    max_windows=max_windows,
                    reorder_window_sec=reorder_window_sec,
                    max_reorder_packets=max_reorder_packets,
                )
        finally:
            self.interface_name = prev_iface

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Summarizes packet, window, configuration, failure, and detection statistics."""
        trusted_cnt = sum(1 for r in self.detection_records if r.status == "TRUSTED")
        unknown_cnt = sum(1 for r in self.detection_records if r.status == "UNKNOWN_OR_REVIEW")

        class_dist: Dict[str, int] = {}
        for r in self.detection_records:
            adv = r.advisory_class
            class_dist[adv] = class_dist.get(adv, 0) + 1

        confs = [r.confidence for r in self.detection_records]
        mahs = [r.mahalanobis_distance for r in self.detection_records]

        return {
            "packets_captured": self.packets_captured,
            "packets_rejected": self.packets_rejected,
            "packet_rejection_reasons": self.packet_rejection_reasons,
            "directionality_counts": dict(self.directionality_counts),
            "directionality_proportions": {k: v / max(1, sum(self.directionality_counts.values())) for k, v in self.directionality_counts.items()},
            "windows_generated": self.windows_generated,
            "windows_rejected": self.windows_rejected,
            "window_rejection_reasons": self.window_rejection_reasons,
            "incomplete_window_count": len(self.incomplete_window_records),
            "incomplete_windows": [
                {
                    "output_mode": r.output_mode,
                    "evaluation_id": r.evaluation_id,
                    "status": r.status,
                    "v4_eligible": r.v4_eligible,
                    "advisory_eligible": r.advisory_eligible,
                    "rejection_reason": r.rejection_reason,
                    "packet_count": r.packet_count,
                    "flow_identity": r.flow_identity,
                    "window_start_ts": r.window_start_ts,
                    "window_end_ts": r.window_end_ts,
                }
                for r in self.incomplete_window_records
            ],
            "trusted_count": trusted_cnt,
            "unknown_or_review_count": unknown_cnt,
            "trusted_pct": (trusted_cnt / len(self.detection_records) * 100.0) if self.detection_records else 0.0,
            "unknown_or_review_pct": (unknown_cnt / len(self.detection_records) * 100.0) if self.detection_records else 0.0,
            "total_detections": len(self.detection_records),
            "class_distribution": class_dist,
            "failed_state": self.failed_state,
            "failure_reason": self.failure_reason,
            "last_operation_status": self.last_operation_status,
            "partial_results_warning": self.partial_results_warning,
            "last_overflow_diagnostics": self.last_overflow_diagnostics,
            "timestamp_guard": {
                "configured_reorder_window_sec": self.reorder_window_sec,
                "configured_max_reorder_packets": self.max_reorder_packets,
                "reorder_window_sec": self.effective_reorder_window_sec,
                "max_reorder_packets": self.effective_max_reorder_packets,
                "operation_mode": self.last_operation_mode,
                "accepted_count": self.timestamp_guard.accepted_count,
                "rejected_ooo_count": self.timestamp_guard.rejected_ooo_count,
                "rejected_non_finite_count": self.timestamp_guard.rejected_non_finite_count,
                "max_regression_sec": self.timestamp_guard.max_regression_sec,
                "last_rejected_event": self.timestamp_guard.last_rejected_event,
                "reorder_late_rejected": self.reorder_late_rejected,
                "reorder_non_finite_rejected": self.reorder_non_finite_rejected,
                "reorder_overflows": self.reorder_overflows,
                "failed_state": self.failed_state,
                "failure_reason": self.failure_reason,
                "last_operation_status": self.last_operation_status,
                "partial_results_warning": self.partial_results_warning,
            },
            "confidence_summary": {
                "mean": float(np.mean(confs)) if confs else 0.0,
                "median": float(np.median(confs)) if confs else 0.0,
                "min": float(np.min(confs)) if confs else 0.0,
                "max": float(np.max(confs)) if confs else 0.0
            },
            "mahalanobis_summary": {
                "mean": float(np.mean(mahs)) if mahs else 0.0,
                "median": float(np.median(mahs)) if mahs else 0.0,
                "min": float(np.min(mahs)) if mahs else 0.0,
                "max": float(np.max(mahs)) if mahs else 0.0
            }
        }

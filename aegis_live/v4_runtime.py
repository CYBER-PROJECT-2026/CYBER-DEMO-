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
from aegis_live.packet_window import PacketRecord, FlowWindow
from aegis_live.network_topology_features import NetworkTopologyPacket, NetworkTopologyTracker
from aegis_live.v4_inference import AEGISLiveV4Inference


@dataclass
class WindowProvenanceRecordV4:
    interface: str
    window_start_ts: float
    window_end_ts: float
    packet_count: int
    flow_identity: str
    features: Dict[str, float]
    status: str                         # "VALID" or "REJECTED"
    rejection_reason: Optional[str] = None


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
            "emitted_at_utc": self.emitted_at_utc
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
                 model_dir: Optional[str] = None):
        self.local_network = ipaddress.ip_network(local_subnet)
        self.max_packets = max_packets_per_window
        self.timeout_sec = window_timeout_sec
        self.min_packets = min_packets_to_extract
        self.net_window_sec = net_window_sec
        self.interface_name = interface_name

        # State components
        self.active_windows: Dict[Tuple[str, str, int], FlowWindow] = {}
        self.net_tracker = NetworkTopologyTracker(window_sec=self.net_window_sec, mode="sliding")
        self.inference_engine = AEGISLiveV4Inference(model_dir=model_dir)

        # Telemetry & Diagnostics
        self.packets_captured = 0
        self.packets_rejected = 0
        self.packet_rejection_reasons: Dict[str, int] = {}
        self.windows_generated = 0
        self.windows_rejected = 0
        self.window_rejection_reasons: Dict[str, int] = {}
        self.detection_records: List[DetectionResultV4] = []

    def reset(self):
        """
        Completely resets all active flow windows, network topology tracking buffers,
        counters, and emitted records for an independent capture or replay session.
        """
        self.active_windows.clear()
        self.net_tracker.reset()
        self.packets_captured = 0
        self.packets_rejected = 0
        self.packet_rejection_reasons.clear()
        self.windows_generated = 0
        self.windows_rejected = 0
        self.window_rejection_reasons.clear()
        self.detection_records.clear()

    def _is_local(self, ip_str: str) -> bool:
        try:
            return ipaddress.ip_address(ip_str) in self.local_network
        except ValueError:
            return False

    def _record_packet_rejection(self, reason: str):
        self.packets_rejected += 1
        self.packet_rejection_reasons[reason] = self.packet_rejection_reasons.get(reason, 0) + 1

    def _record_window_rejection(self, reason: str):
        self.windows_rejected += 1
        self.window_rejection_reasons[reason] = self.window_rejection_reasons.get(reason, 0) + 1

    def _extract_combined_29_features(self, window: FlowWindow) -> Dict[str, float]:
        """
        Extracts the exact 29-feature vector at the window's completion timestamp.
        Guarantees:
        - 24 features from FlowWindow are 100% equivalent to canonical v2/v3 extractor.
        - 5 features from NetworkTopologyTracker represent packets with t <= window.last_timestamp
          and t >= window.last_timestamp - 1.0s.
        """
        feats_24 = window.extract_features()
        net_feats = self.net_tracker.extract_current_features()

        return {
            **feats_24,
            "src_concentration_ratio": float(net_feats["src_concentration_ratio"]),
            "src_ip_entropy": float(net_feats["src_ip_entropy"]),
            "src_ip_cardinality": float(net_feats["src_ip_cardinality"]),
            "dst_ip_cardinality": float(net_feats["dst_ip_cardinality"]),
            "fan_in_ratio": float(net_feats["fan_in_ratio"])
        }

    def _process_completed_window(self, window: FlowWindow) -> DetectionResultV4:
        """Processes a completed FlowWindow and runs v4 inference and OOD gating."""
        flow_id = f"{window.flow_key[0]} <-> {window.flow_key[1]} proto={window.flow_key[2]}"
        start_ts = float(window.packets[0].timestamp)
        end_ts = float(window.packets[-1].timestamp)
        pkt_cnt = len(window.packets)

        feats_29 = self._extract_combined_29_features(window)
        self.windows_generated += 1

        prov = WindowProvenanceRecordV4(
            interface=self.interface_name,
            window_start_ts=start_ts,
            window_end_ts=end_ts,
            packet_count=pkt_cnt,
            flow_identity=flow_id,
            features=feats_29,
            status="VALID"
        )

        pred = self.inference_engine.predict_sample(feats_29)

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
            provenance=prov
        )
        self.detection_records.append(det)
        return det

    def ingest_packet(self, ts: float, buf: bytes) -> List[DetectionResultV4]:
        """
        Ingests a single raw Ethernet packet into the v4 live pipeline.
        Returns any detection results emitted as a consequence of window completion.
        """
        self.packets_captured += 1
        ts = float(ts)

        # 1. Parse Ethernet Frame
        if len(buf) < 14:
            self._record_packet_rejection("ethernet_frame_too_short")
            return []

        eth_type = (buf[12] << 8) | buf[13]
        if eth_type != 0x0800:
            self._record_packet_rejection("non_ipv4_packet")
            return []

        if len(buf) < 34:
            self._record_packet_rejection("ipv4_packet_too_short")
            return []

        # 2. Parse IPv4 Header
        ihl = (buf[14] & 0x0F) * 4
        if ihl < 20 or len(buf) < 14 + ihl:
            self._record_packet_rejection("invalid_ip_header_length")
            return []

        ttl = buf[22]
        ip_proto = buf[23]
        src_ip = socket.inet_ntoa(buf[26:30])
        dst_ip = socket.inet_ntoa(buf[30:34])

        sport = 0
        dport = 0
        tcp_flags = 0
        l4_offset = 14 + ihl

        if ip_proto == 6:  # TCP
            if len(buf) >= l4_offset + 14:
                sport = (buf[l4_offset] << 8) | buf[l4_offset + 1]
                dport = (buf[l4_offset + 2] << 8) | buf[l4_offset + 3]
                tcp_flags = buf[l4_offset + 13]
            else:
                self._record_packet_rejection("truncated_tcp_header")
                return []
        elif ip_proto == 17:  # UDP
            if len(buf) >= l4_offset + 4:
                sport = (buf[l4_offset] << 8) | buf[l4_offset + 1]
                dport = (buf[l4_offset + 2] << 8) | buf[l4_offset + 3]
            else:
                self._record_packet_rejection("truncated_udp_header")
                return []

        # 3. Update Causal Network Topology Tracker (strictly at packet ts <= Tend)
        net_pkt = NetworkTopologyPacket(
            timestamp=ts,
            src_ip=src_ip,
            dst_ip=dst_ip,
            src_port=sport,
            dst_port=dport,
            protocol=ip_proto,
            pkt_len=len(buf)
        )
        self.net_tracker.add_packet(net_pkt)

        # 4. Construct Flow Packet Record
        is_inbound = self._is_local(dst_ip)
        pkt_rec = PacketRecord(
            timestamp=ts,
            pkt_len=len(buf),
            ip_proto=ip_proto,
            ip_ttl=ttl,
            tcp_flags=tcp_flags,
            is_inbound=is_inbound
        )

        ip_pair = tuple(sorted([src_ip, dst_ip]))
        flow_key = (ip_pair[0], ip_pair[1], ip_proto)

        emitted: List[DetectionResultV4] = []

        if flow_key in self.active_windows:
            window = self.active_windows[flow_key]
            gap = (pkt_rec.timestamp - window.last_timestamp) if window.last_timestamp is not None else 0.0

            if gap >= self.timeout_sec:
                # Flush existing window if it contains sufficient packets
                if len(window.packets) >= self.min_packets:
                    det = self._process_completed_window(window)
                    emitted.append(det)
                else:
                    self._record_window_rejection("subcapacity_window_timeout")

                # Fresh window for new packet
                fresh_window = FlowWindow(flow_key, self.max_packets, self.timeout_sec)
                ready = fresh_window.add_packet(pkt_rec)
                if ready:
                    det = self._process_completed_window(fresh_window)
                    emitted.append(det)
                    del self.active_windows[flow_key]
                else:
                    self.active_windows[flow_key] = fresh_window
            else:
                ready = window.add_packet(pkt_rec)
                if ready:
                    det = self._process_completed_window(window)
                    emitted.append(det)
                    del self.active_windows[flow_key]
        else:
            fresh_window = FlowWindow(flow_key, self.max_packets, self.timeout_sec)
            ready = fresh_window.add_packet(pkt_rec)
            if ready:
                det = self._process_completed_window(fresh_window)
                emitted.append(det)
            else:
                self.active_windows[flow_key] = fresh_window

        return emitted

    def flush(self, current_time: float, force_all: bool = False) -> List[DetectionResultV4]:
        """
        Flushes inactive flow windows exceeding timeout_sec, or force all on end-of-stream.
        """
        current_time = float(current_time)
        emitted: List[DetectionResultV4] = []
        keys_to_remove = []

        for flow_key, window in list(self.active_windows.items()):
            gap = (current_time - window.last_timestamp) if window.last_timestamp is not None else float("inf")
            if force_all or gap >= self.timeout_sec:
                if len(window.packets) >= self.min_packets:
                    det = self._process_completed_window(window)
                    emitted.append(det)
                else:
                    self._record_window_rejection("subcapacity_flushed_window")
                keys_to_remove.append(flow_key)

        for k in keys_to_remove:
            if k in self.active_windows:
                del self.active_windows[k]

        return emitted

    def replay_packet_stream(self,
                             packet_stream: Generator[Tuple[float, bytes], None, None],
                             max_windows: Optional[int] = None) -> List[DetectionResultV4]:
        """Replays an arbitrary generator of (ts, raw_bytes) tuples."""
        new_detections: List[DetectionResultV4] = []
        last_ts = 0.0

        for ts, buf in packet_stream:
            last_ts = float(ts)
            dets = self.ingest_packet(last_ts, buf)
            new_detections.extend(dets)
            if max_windows is not None and len(self.detection_records) >= max_windows:
                break

        # Final flush on end of stream
        flushed = self.flush(current_time=last_ts + 5.0, force_all=True)
        new_detections.extend(flushed)
        return new_detections

    def replay_pcap(self,
                    pcap_path: str,
                    max_packets: Optional[int] = None,
                    max_windows: Optional[int] = None,
                    packet_range: Optional[Tuple[int, int]] = None) -> List[DetectionResultV4]:
        """
        Replays an authentic PCAP capture file through the exact v4 pipeline.
        
        Parameters:
          - pcap_path: Path to .pcap file
          - max_packets: Maximum raw packets to ingest
          - max_windows: Maximum windows to emit
          - packet_range: Optional (start_idx, end_idx) 0-indexed packet interval
        """
        if not os.path.exists(pcap_path):
            raise FileNotFoundError(f"PCAP not found: {pcap_path}")

        prev_iface = self.interface_name
        self.interface_name = f"PCAP_REPLAY:{os.path.basename(pcap_path)}"
        new_detections: List[DetectionResultV4] = []
        last_ts = 0.0

        start_idx = packet_range[0] if packet_range else 0
        end_idx = packet_range[1] if packet_range else None

        with open(pcap_path, "rb") as f:
            try:
                reader = dpkt.pcap.Reader(f)
            except ValueError:
                f.seek(0)
                reader = dpkt.pcapng.Reader(f)

            pkt_idx = 0
            for ts, buf in reader:
                if pkt_idx < start_idx:
                    pkt_idx += 1
                    continue

                if end_idx is not None and pkt_idx > end_idx:
                    break

                last_ts = float(ts)
                dets = self.ingest_packet(last_ts, buf)
                new_detections.extend(dets)
                pkt_idx += 1

                if max_packets is not None and (pkt_idx - start_idx) >= max_packets:
                    break
                if max_windows is not None and len(self.detection_records) >= max_windows:
                    break

        # Flush any remaining completed windows
        flushed = self.flush(current_time=last_ts + 5.0, force_all=True)
        new_detections.extend(flushed)

        self.interface_name = prev_iface
        return new_detections

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Summarizes packet, window, and detection statistics."""
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
            "windows_generated": self.windows_generated,
            "windows_rejected": self.windows_rejected,
            "window_rejection_reasons": self.window_rejection_reasons,
            "trusted_count": trusted_cnt,
            "unknown_or_review_count": unknown_cnt,
            "trusted_pct": (trusted_cnt / len(self.detection_records) * 100.0) if self.detection_records else 0.0,
            "unknown_or_review_pct": (unknown_cnt / len(self.detection_records) * 100.0) if self.detection_records else 0.0,
            "total_detections": len(self.detection_records),
            "class_distribution": class_dist,
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

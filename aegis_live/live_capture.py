"""
aegis_live/live_capture.py
==========================
Real-Time Network Packet Ingestion & Feature Provenance Pipeline for AEGIS-Live v2.
STRICTLY DETECTION-ONLY MODE: Zero network mutation, zero packet transmission, zero firewall modifications.

Reuses the existing canonical FlowWindow and PacketRecord implementation from aegis_live.packet_window.
Emits verified 24-feature records with physical provenance metadata.
"""

import os
import sys
import time
import socket
import ipaddress
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Callable

import numpy as np
import dpkt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from aegis_live.feature_schema import CANONICAL_24_FEATURES, validate_24_features
from aegis_live.packet_window import PacketRecord, FlowWindow
from aegis_live.feature_extractor import LiveFeatureExtractor
from aegis_live.live_inference import LiveInferenceEngine, DetectionResult


@dataclass
class WindowProvenanceRecord:
    interface: str
    window_start_ts: float
    window_end_ts: float
    packet_count: int
    flow_identity: str
    features: Dict[str, float]
    status: str                         # "VALID" or "REJECTED"
    rejection_reason: Optional[str] = None


@dataclass
class LiveDetectionRecord:
    provenance: WindowProvenanceRecord
    detection: DetectionResult
    emitted_at_utc: float = field(default_factory=time.time)


class LivePipelineExtractor(LiveFeatureExtractor):
    """
    Subclasses LiveFeatureExtractor to capture physical provenance metadata
    without duplicating any feature-calculation math.
    """
    def __init__(self,
                 local_subnet: str = "192.168.0.0/16",
                 max_packets_per_window: int = 50,
                 window_timeout_sec: float = 2.0,
                 min_packets_to_extract: int = 2):
        super().__init__(
            local_subnet=local_subnet,
            max_packets_per_window=max_packets_per_window,
            window_timeout_sec=window_timeout_sec,
            min_packets_to_extract=min_packets_to_extract
        )
        self.packets_captured = 0
        self.packets_rejected = 0
        self.packet_rejection_reasons: Dict[str, int] = {}
        self.windows_generated = 0
        self.windows_rejected = 0
        self.window_rejection_reasons: Dict[str, int] = {}

    def _record_packet_rejection(self, reason: str):
        self.packets_rejected += 1
        self.packet_rejection_reasons[reason] = self.packet_rejection_reasons.get(reason, 0) + 1

    def _record_window_rejection(self, reason: str):
        self.windows_rejected += 1
        self.window_rejection_reasons[reason] = self.window_rejection_reasons.get(reason, 0) + 1

    def _create_provenance_record(self,
                                  window: FlowWindow,
                                  interface: str) -> WindowProvenanceRecord:
        start_ts = window.packets[0].timestamp
        end_ts = window.packets[-1].timestamp
        pkt_cnt = len(window.packets)
        flow_id = f"{window.flow_key[0]} <-> {window.flow_key[1]} proto={window.flow_key[2]}"

        try:
            feats = window.extract_features()
            validate_24_features(feats)
            self.windows_generated += 1
            return WindowProvenanceRecord(
                interface=interface,
                window_start_ts=start_ts,
                window_end_ts=end_ts,
                packet_count=pkt_cnt,
                flow_identity=flow_id,
                features=feats,
                status="VALID",
                rejection_reason=None
            )
        except Exception as e:
            self._record_window_rejection(str(e))
            return WindowProvenanceRecord(
                interface=interface,
                window_start_ts=start_ts,
                window_end_ts=end_ts,
                packet_count=pkt_cnt,
                flow_identity=flow_id,
                features={},
                status="REJECTED",
                rejection_reason=str(e)
            )

    def process_raw_packet_with_provenance(self,
                                           ts: float,
                                           buf: bytes,
                                           interface: str = "live") -> List[WindowProvenanceRecord]:
        """
        Parses a single raw Ethernet frame, enforces canonical window rules,
        and returns any completed window records with complete physical provenance.
        """
        self.packets_captured += 1
        completed_records: List[WindowProvenanceRecord] = []

        try:
            eth = dpkt.ethernet.Ethernet(buf)
        except Exception:
            self._record_packet_rejection("ethernet_decode_failed")
            return completed_records

        # Check for IP payload
        if not isinstance(eth.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
            self._record_packet_rejection("non_ip_packet")
            return completed_records

        ip_pkt = eth.data
        ip_proto = ip_pkt.p
        ip_ttl = getattr(ip_pkt, 'ttl', getattr(ip_pkt, 'hlim', 64))
        pkt_len = len(buf)

        try:
            if isinstance(ip_pkt, dpkt.ip.IP):
                src_ip = socket.inet_ntoa(ip_pkt.src)
                dst_ip = socket.inet_ntoa(ip_pkt.dst)
            else:
                src_ip = socket.inet_ntop(socket.AF_INET6, ip_pkt.src)
                dst_ip = socket.inet_ntop(socket.AF_INET6, ip_pkt.dst)
        except Exception:
            self._record_packet_rejection("ip_address_decode_failed")
            return completed_records

        # TCP flags
        tcp_flags = 0
        if ip_proto == dpkt.ip.IP_PROTO_TCP and hasattr(ip_pkt, 'data') and isinstance(ip_pkt.data, dpkt.tcp.TCP):
            tcp_flags = ip_pkt.data.flags

        is_inbound = self._is_local(dst_ip)

        # Canonical bidirectional flow key: sorted IP endpoints + protocol
        ip_pair = tuple(sorted([src_ip, dst_ip]))
        flow_key = (ip_pair[0], ip_pair[1], ip_proto)

        pkt_rec = PacketRecord(
            timestamp=float(ts),
            pkt_len=pkt_len,
            ip_proto=ip_proto,
            ip_ttl=ip_ttl,
            tcp_flags=tcp_flags,
            is_inbound=is_inbound
        )

        if flow_key in self.active_windows:
            window = self.active_windows[flow_key]
            gap = (pkt_rec.timestamp - window.last_timestamp) if window.last_timestamp is not None else 0.0

            if gap >= self.timeout_sec:
                # 2-second timeout boundary hit: flush stale window if >= min_packets
                if len(window.packets) >= self.min_packets:
                    rec = self._create_provenance_record(window, interface)
                    completed_records.append(rec)

                # Reset stale flow and start fresh window
                fresh_window = FlowWindow(flow_key, self.max_packets, self.timeout_sec)
                ready = fresh_window.add_packet(pkt_rec)
                if ready:
                    rec = self._create_provenance_record(fresh_window, interface)
                    completed_records.append(rec)
                    del self.active_windows[flow_key]
                else:
                    self.active_windows[flow_key] = fresh_window
            else:
                # Normal packet arrival within 2-second window
                ready = window.add_packet(pkt_rec)
                if ready:
                    rec = self._create_provenance_record(window, interface)
                    completed_records.append(rec)
                    del self.active_windows[flow_key]
        else:
            fresh_window = FlowWindow(flow_key, self.max_packets, self.timeout_sec)
            ready = fresh_window.add_packet(pkt_rec)
            if ready:
                rec = self._create_provenance_record(fresh_window, interface)
                completed_records.append(rec)
            else:
                self.active_windows[flow_key] = fresh_window

        return completed_records

    def flush_active_windows_with_provenance(self,
                                             current_time: float,
                                             interface: str = "live",
                                             force_all: bool = False) -> List[WindowProvenanceRecord]:
        """
        Flushes timed-out or all active flow windows.
        """
        completed: List[WindowProvenanceRecord] = []
        keys_to_del = []

        for flow_key, window in self.active_windows.items():
            if force_all or window.is_timed_out(current_time):
                keys_to_del.append(flow_key)
                if len(window.packets) >= self.min_packets:
                    rec = self._create_provenance_record(window, interface)
                    completed.append(rec)

        for k in keys_to_del:
            del self.active_windows[k]

        return completed


class AegisLivePipeline:
    """
    AEGIS-Live v2 Complete Live Detection Pipeline.
    Connects Packet Capture -> 24-Feature Extractor -> Frozen Inference -> Frozen Calibration -> OOD/Abstention Gate.
    STRICTLY DETECTION-ONLY: automated_blocking_allowed is always False.
    """
    def __init__(self,
                 interface_name: str = "Wi-Fi",
                 local_subnet: str = "192.168.0.0/16",
                 max_packets_per_window: int = 50,
                 window_timeout_sec: float = 2.0,
                 min_packets_to_extract: int = 2):
        self.interface_name = interface_name
        self.extractor = LivePipelineExtractor(
            local_subnet=local_subnet,
            max_packets_per_window=max_packets_per_window,
            window_timeout_sec=window_timeout_sec,
            min_packets_to_extract=min_packets_to_extract
        )
        self.inference_engine = LiveInferenceEngine()
        self.detection_records: List[LiveDetectionRecord] = []

    def ingest_packet(self, ts: float, buf: bytes) -> List[LiveDetectionRecord]:
        """
        Ingests a single packet frame and returns any generated detection records.
        """
        prov_records = self.extractor.process_raw_packet_with_provenance(
            ts=ts,
            buf=buf,
            interface=self.interface_name
        )

        detections = []
        for prov in prov_records:
            if prov.status == "VALID":
                det = self.inference_engine.predict(prov.features)
            else:
                # Quarantined window
                det = DetectionResult(
                    status="REJECTED_QUARANTINE",
                    trusted_class=None,
                    advisory_class=None,
                    raw_predicted_class=None,
                    calibrated_probs={},
                    confidence=0.0,
                    top2_margin=0.0,
                    entropy=0.0,
                    class_distances={},
                    min_mahalanobis_distance=float("inf"),
                    predicted_class_distance=float("inf"),
                    reasons=[f"window_rejected: {prov.rejection_reason}"],
                    automated_blocking_allowed=False
                )
            record = LiveDetectionRecord(provenance=prov, detection=det)
            self.detection_records.append(record)
            detections.append(record)

        return detections

    def flush(self, current_time: float, force_all: bool = False) -> List[LiveDetectionRecord]:
        """
        Flushes inactive flow windows.
        """
        prov_records = self.extractor.flush_active_windows_with_provenance(
            current_time=current_time,
            interface=self.interface_name,
            force_all=force_all
        )

        detections = []
        for prov in prov_records:
            if prov.status == "VALID":
                det = self.inference_engine.predict(prov.features)
            else:
                det = DetectionResult(
                    status="REJECTED_QUARANTINE",
                    trusted_class=None,
                    advisory_class=None,
                    raw_predicted_class=None,
                    calibrated_probs={},
                    confidence=0.0,
                    top2_margin=0.0,
                    entropy=0.0,
                    class_distances={},
                    min_mahalanobis_distance=float("inf"),
                    predicted_class_distance=float("inf"),
                    reasons=[f"window_rejected: {prov.rejection_reason}"],
                    automated_blocking_allowed=False
                )
            record = LiveDetectionRecord(provenance=prov, detection=det)
            self.detection_records.append(record)
            detections.append(record)

        return detections

    def replay_pcap(self, pcap_path: str, max_windows: Optional[int] = None) -> List[LiveDetectionRecord]:
        """
        Replays an authentic PCAP capture file through the EXACT SAME pipeline.
        Ensures 1-to-1 equivalence between replay and offline extraction.
        """
        if not os.path.exists(pcap_path):
            raise FileNotFoundError(f"PCAP not found: {pcap_path}")

        prev_iface = self.interface_name
        self.interface_name = f"PCAP_REPLAY:{os.path.basename(pcap_path)}"
        new_detections = []
        last_ts = 0.0

        with open(pcap_path, "rb") as f:
            pcap = dpkt.pcap.Reader(f)
            for ts, buf in pcap:
                last_ts = float(ts)
                dets = self.ingest_packet(last_ts, buf)
                new_detections.extend(dets)
                if max_windows is not None and len(self.detection_records) >= max_windows:
                    break

        # Flush any remaining completed windows
        flushed = self.flush(current_time=last_ts + 5.0, force_all=True)
        new_detections.extend(flushed)

        self.interface_name = prev_iface
        return new_detections

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Summarizes packet, window, and detection statistics.
        """
        trusted_cnt = sum(1 for r in self.detection_records if r.detection.status == "TRUSTED")
        unknown_cnt = sum(1 for r in self.detection_records if r.detection.status == "UNKNOWN_OR_REVIEW")
        rejected_cnt = sum(1 for r in self.detection_records if r.detection.status == "REJECTED_QUARANTINE")

        class_dist: Dict[str, int] = {}
        for r in self.detection_records:
            adv = r.detection.advisory_class
            if adv:
                class_dist[adv] = class_dist.get(adv, 0) + 1

        confs = [r.detection.confidence for r in self.detection_records if r.detection.status in ("TRUSTED", "UNKNOWN_OR_REVIEW")]
        mahs = [r.detection.predicted_class_distance for r in self.detection_records if r.detection.status in ("TRUSTED", "UNKNOWN_OR_REVIEW")]

        return {
            "packets_captured": self.extractor.packets_captured,
            "packets_rejected": self.extractor.packets_rejected,
            "packet_rejection_reasons": self.extractor.packet_rejection_reasons,
            "windows_generated": self.extractor.windows_generated,
            "windows_rejected": self.extractor.windows_rejected,
            "window_rejection_reasons": self.extractor.window_rejection_reasons,
            "trusted_count": trusted_cnt,
            "unknown_or_review_count": unknown_cnt,
            "rejected_quarantine_count": rejected_cnt,
            "total_detections": len(self.detection_records),
            "class_distribution": class_dist,
            "confidence_summary": {
                "mean": float(np.mean(confs)) if confs else 0.0,
                "min": float(np.min(confs)) if confs else 0.0,
                "max": float(np.max(confs)) if confs else 0.0
            },
            "mahalanobis_summary": {
                "mean": float(np.mean(mahs)) if mahs else 0.0,
                "min": float(np.min(mahs)) if mahs else 0.0,
                "max": float(np.max(mahs)) if mahs else 0.0
            }
        }

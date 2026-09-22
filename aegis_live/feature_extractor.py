"""
aegis_live/feature_extractor.py
===============================
Real-Time & PCAP Packet Feature Extractor for AEGIS-Live v2.
Parses packet streams via dpkt (with Scapy fallback) and emits validated 24-feature records.
"""

import os
import sys
import socket
import ipaddress
from typing import Generator, Dict, Any, Optional, Tuple, List
import dpkt
import pandas as pd

from aegis_live.feature_schema import CANONICAL_24_FEATURES, validate_24_features
from aegis_live.packet_window import PacketRecord, FlowWindow


class LiveFeatureExtractor:
    """
    Extracts 24 clean features from PCAP captures or live network interfaces.
    """
    def __init__(self,
                 local_subnet: str = "192.168.0.0/16",
                 max_packets_per_window: int = 50,
                 window_timeout_sec: float = 2.0,
                 min_packets_to_extract: int = 2):
        self.local_network = ipaddress.ip_network(local_subnet)
        self.max_packets = max_packets_per_window
        self.timeout_sec = window_timeout_sec
        self.min_packets = min_packets_to_extract
        self.active_windows: Dict[Tuple[str, str, int], FlowWindow] = {}

    def _is_local(self, ip_str: str) -> bool:
        try:
            return ipaddress.ip_address(ip_str) in self.local_network
        except ValueError:
            return False

    def process_raw_packet(self, ts: float, buf: bytes) -> Optional[Dict[str, float]]:
        """
        Parses a single raw Ethernet frame.
        Returns a 24-feature dict if a flow window reaches capacity, otherwise None.
        """
        try:
            eth = dpkt.ethernet.Ethernet(buf)
        except Exception:
            return None

        # Check for IP
        if not isinstance(eth.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
            return None

        ip_pkt = eth.data
        ip_proto = ip_pkt.p
        ip_ttl = getattr(ip_pkt, 'ttl', getattr(ip_pkt, 'hlim', 64))
        pkt_len = len(buf)

        try:
            src_ip = socket.inet_ntoa(ip_pkt.src) if isinstance(ip_pkt, dpkt.ip.IP) else socket.inet_ntop(socket.AF_INET6, ip_pkt.src)
            dst_ip = socket.inet_ntoa(ip_pkt.dst) if isinstance(ip_pkt, dpkt.ip.IP) else socket.inet_ntop(socket.AF_INET6, ip_pkt.dst)
        except Exception:
            return None

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

        expired_features = None

        if flow_key in self.active_windows:
            window = self.active_windows[flow_key]
            gap = (pkt_rec.timestamp - window.last_timestamp) if window.last_timestamp is not None else 0.0

            if gap >= self.timeout_sec:
                # 1. Finalize/flush existing window if it contains enough packets
                if len(window.packets) >= self.min_packets:
                    expired_features = window.extract_features()

                # 2. Reset/remove stale flow window and create fresh window
                fresh_window = FlowWindow(flow_key, self.max_packets, self.timeout_sec)
                ready = fresh_window.add_packet(pkt_rec)
                if ready:
                    features = fresh_window.extract_features()
                    del self.active_windows[flow_key]
                    return expired_features if expired_features is not None else features
                else:
                    self.active_windows[flow_key] = fresh_window
                    return expired_features
            else:
                # Gap < timeout_sec: append packet normally
                ready = window.add_packet(pkt_rec)
                if ready:
                    features = window.extract_features()
                    del self.active_windows[flow_key]
                    return features
                return None
        else:
            fresh_window = FlowWindow(flow_key, self.max_packets, self.timeout_sec)
            ready = fresh_window.add_packet(pkt_rec)
            if ready:
                features = fresh_window.extract_features()
                return features
            self.active_windows[flow_key] = fresh_window
            return None

    def flush_timed_out_windows(self, current_time: float) -> List[Dict[str, float]]:
        """
        Flushes and extracts features from any windows exceeding the timeout.
        """
        extracted = []
        keys_to_del = []

        for flow_key, window in self.active_windows.items():
            if window.is_timed_out(current_time):
                keys_to_del.append(flow_key)
                if len(window.packets) >= self.min_packets:
                    extracted.append(window.extract_features())

        for k in keys_to_del:
            del self.active_windows[k]

        return extracted

    def extract_from_pcap(self, pcap_path: str, max_windows: Optional[int] = None) -> pd.DataFrame:
        """
        Extracts 24-feature vectors from an entire PCAP file.
        """
        if not os.path.exists(pcap_path):
            raise FileNotFoundError(f"PCAP not found: {pcap_path}")

        records = []
        last_ts = 0.0

        with open(pcap_path, 'rb') as f:
            try:
                pcap_reader = dpkt.pcap.Reader(f)
            except ValueError:
                # Try pcapng
                f.seek(0)
                pcap_reader = dpkt.pcapng.Reader(f)

            for ts, buf in pcap_reader:
                last_ts = ts
                feat = self.process_raw_packet(ts, buf)
                if feat is not None:
                    records.append(feat)
                    if max_windows is not None and len(records) >= max_windows:
                        break

        # Flush any remaining completed windows at end of PCAP
        remaining = self.flush_timed_out_windows(last_ts + self.timeout_sec + 1.0)
        records.extend(remaining)

        if not records:
            return pd.DataFrame(columns=CANONICAL_24_FEATURES)

        df = pd.DataFrame(records)
        return validate_24_features(df)

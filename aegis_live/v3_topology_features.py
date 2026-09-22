"""
aegis_live/v3_topology_features.py
==================================
Feasibility Prototype: Candidate Topology Features for AEGIS-Live v3.

STRICT NON-MUTATION CONTRACT:
- This module is strictly an experimental prototype for feasibility and leakage auditing.
- Does NOT modify, import, or alter the frozen AEGIS-Live v2 baseline pipeline or 24-feature schema.
- Zero v2 inference mutation, zero retraining, zero dataset modifications.

Candidate Features Implemented:
1. src_ip_cardinality_per_window
2. src_concentration_ratio
3. unique_5tuple_flow_count
4. fan_in_ratio
"""

import socket
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import dpkt


V3_TOPOLOGY_FEATURE_NAMES = [
    "src_ip_cardinality_per_window",
    "src_concentration_ratio",
    "unique_5tuple_flow_count",
    "fan_in_ratio"
]


@dataclass
class V3TopologyPacket:
    """Lightweight packet representation for topology tracking."""
    timestamp: float
    src_ip: Optional[str]
    dst_ip: Optional[str]
    src_port: int
    dst_port: int
    ip_proto: int
    pkt_len: int
    is_ipv4: bool = True


def parse_raw_ethernet_frame_v3(ts: float, buf: bytes) -> V3TopologyPacket:
    """
    Safely parses an Ethernet frame into a V3TopologyPacket.
    Never raises an exception; returns fallback values for malformed or non-IP packets.
    """
    pkt_len = len(buf)
    try:
        eth = dpkt.ethernet.Ethernet(buf)
    except Exception:
        return V3TopologyPacket(
            timestamp=float(ts),
            src_ip=None,
            dst_ip=None,
            src_port=0,
            dst_port=0,
            ip_proto=0,
            pkt_len=pkt_len,
            is_ipv4=False
        )

    # Check for IPv4 payload
    if not isinstance(eth.data, dpkt.ip.IP):
        return V3TopologyPacket(
            timestamp=float(ts),
            src_ip=None,
            dst_ip=None,
            src_port=0,
            dst_port=0,
            ip_proto=0,
            pkt_len=pkt_len,
            is_ipv4=False
        )

    ip_pkt = eth.data
    try:
        src_ip = socket.inet_ntoa(ip_pkt.src)
        dst_ip = socket.inet_ntoa(ip_pkt.dst)
    except Exception:
        return V3TopologyPacket(
            timestamp=float(ts),
            src_ip=None,
            dst_ip=None,
            src_port=0,
            dst_port=0,
            ip_proto=0,
            pkt_len=pkt_len,
            is_ipv4=False
        )

    ip_proto = int(ip_pkt.p)
    src_port = 0
    dst_port = 0

    if ip_proto == dpkt.ip.IP_PROTO_TCP and hasattr(ip_pkt, "data") and isinstance(ip_pkt.data, dpkt.tcp.TCP):
        src_port = int(ip_pkt.data.sport)
        dst_port = int(ip_pkt.data.dport)
    elif ip_proto == dpkt.ip.IP_PROTO_UDP and hasattr(ip_pkt, "data") and isinstance(ip_pkt.data, dpkt.udp.UDP):
        src_port = int(ip_pkt.data.sport)
        dst_port = int(ip_pkt.data.dport)

    return V3TopologyPacket(
        timestamp=float(ts),
        src_ip=src_ip,
        dst_ip=dst_ip,
        src_port=src_port,
        dst_port=dst_port,
        ip_proto=ip_proto,
        pkt_len=pkt_len,
        is_ipv4=True
    )


def compute_topology_features_batch(packets: List[V3TopologyPacket]) -> Dict[str, float]:
    """
    Direct batch calculation of the 4 candidate topology features
    over a list of V3TopologyPacket instances.
    """
    if not packets:
        return {
            "src_ip_cardinality_per_window": 0.0,
            "src_concentration_ratio": 0.0,
            "unique_5tuple_flow_count": 0.0,
            "fan_in_ratio": 0.0
        }

    src_counts: Counter = Counter()
    dst_set: Set[str] = set()
    flows_set: Set[Tuple[str, str, int, int, int]] = set()
    total_ip_packets = 0

    for pkt in packets:
        if pkt.is_ipv4 and pkt.src_ip is not None and pkt.dst_ip is not None:
            src_counts[pkt.src_ip] += 1
            dst_set.add(pkt.dst_ip)
            flows_set.add((pkt.src_ip, pkt.dst_ip, pkt.src_port, pkt.dst_port, pkt.ip_proto))
            total_ip_packets += 1

    unique_src = len(src_counts)
    unique_dst = len(dst_set)

    # 1. src_ip_cardinality_per_window
    src_cardinality = float(unique_src)

    # 2. src_concentration_ratio
    if total_ip_packets > 0:
        top_count = max(src_counts.values())
        src_concentration = float(top_count) / float(total_ip_packets)
    else:
        src_concentration = 0.0

    # 3. unique_5tuple_flow_count
    flow_count = float(len(flows_set))

    # 4. fan_in_ratio
    if unique_dst > 0:
        fan_in = float(unique_src) / float(unique_dst)
    else:
        fan_in = 0.0

    return {
        "src_ip_cardinality_per_window": src_cardinality,
        "src_concentration_ratio": src_concentration,
        "unique_5tuple_flow_count": flow_count,
        "fan_in_ratio": fan_in
    }


class V3TopologyTracker:
    """
    Incremental O(1) streaming tracker for candidate topology features.
    Maintains sets and frequency counters incrementally on every packet arrival.
    """
    def __init__(self, max_packets: int = 50, timeout_sec: float = 2.0):
        self.max_packets = max_packets
        self.timeout_sec = timeout_sec
        self.reset()

    def reset(self):
        self.src_counts: Counter = Counter()
        self.dst_set: Set[str] = set()
        self.flows_set: Set[Tuple[str, str, int, int, int]] = set()
        self.total_ip_packets: int = 0
        self.total_packets: int = 0
        self.packets: List[V3TopologyPacket] = []
        self.first_timestamp: Optional[float] = None
        self.last_timestamp: Optional[float] = None

    def add_packet(self, pkt: V3TopologyPacket) -> bool:
        """
        Adds a packet incrementally.
        Returns True if window reaches capacity (self.max_packets).
        """
        if self.first_timestamp is None:
            self.first_timestamp = pkt.timestamp
            self.last_timestamp = pkt.timestamp
        else:
            self.last_timestamp = max(self.last_timestamp, pkt.timestamp)

        self.total_packets += 1
        self.packets.append(pkt)

        if pkt.is_ipv4 and pkt.src_ip is not None and pkt.dst_ip is not None:
            self.src_counts[pkt.src_ip] += 1
            self.dst_set.add(pkt.dst_ip)
            self.flows_set.add((pkt.src_ip, pkt.dst_ip, pkt.src_port, pkt.dst_port, pkt.ip_proto))
            self.total_ip_packets += 1

        return self.total_packets >= self.max_packets

    def extract_features(self) -> Dict[str, float]:
        """
        Extracts candidate topology features directly from incremental state in O(1) time.
        """
        unique_src = len(self.src_counts)
        unique_dst = len(self.dst_set)

        src_cardinality = float(unique_src)

        if self.total_ip_packets > 0:
            top_count = max(self.src_counts.values())
            src_concentration = float(top_count) / float(self.total_ip_packets)
        else:
            src_concentration = 0.0

        flow_count = float(len(self.flows_set))

        if unique_dst > 0:
            fan_in = float(unique_src) / float(unique_dst)
        else:
            fan_in = 0.0

        return {
            "src_ip_cardinality_per_window": src_cardinality,
            "src_concentration_ratio": src_concentration,
            "unique_5tuple_flow_count": flow_count,
            "fan_in_ratio": fan_in
        }

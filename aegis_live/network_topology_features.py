"""
aegis_live/network_topology_features.py
======================================
Network-Level Topology Feature Extraction for AEGIS-Live.

Investigates extracting topology features across a common network-level time window
aggregating all active endpoints/flows (as opposed to isolated symmetric bidirectional 3-tuple flow windows).

Candidate Features (7):
  1. src_ip_cardinality: |{src_ip_i}|
  2. src_concentration_ratio: max_s count(s) / |W|
  3. dst_ip_cardinality: |{dst_ip_i}|
  4. unique_flow_count: |{(src_ip, dst_ip, sport, dport, proto)}|
  5. src_ip_entropy: -sum p(s) log2 p(s)
  6. fan_in_ratio: |{src_ip}| / |{dst_ip}|
  7. fan_out_ratio: |{dst_ip}| / |{src_ip}|

Implements both streaming (O(1) incremental) and offline batch calculators
with verified numerical equivalence, strictly causal past-to-present operation,
and zero access to labels, filenames, or future packets.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import Counter, deque
import math
import socket
import numpy as np
import dpkt

NETWORK_TOPOLOGY_FEATURE_NAMES: List[str] = [
    "src_ip_cardinality",
    "src_concentration_ratio",
    "dst_ip_cardinality",
    "unique_flow_count",
    "src_ip_entropy",
    "fan_in_ratio",
    "fan_out_ratio"
]


@dataclass(slots=True)
class NetworkTopologyPacket:
    """Lightweight representation of an observed packet for network topology tracking."""
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    pkt_len: int

    @property
    def canonical_flow_key(self) -> Tuple[str, str, int, int, int]:
        """Canonical bidirectional 5-tuple key."""
        if (self.src_ip, self.src_port) <= (self.dst_ip, self.dst_port):
            return (self.src_ip, self.dst_ip, self.src_port, self.dst_port, self.protocol)
        return (self.dst_ip, self.src_ip, self.dst_port, self.src_port, self.protocol)


def parse_raw_ethernet_network_topology(ts: float, raw_bytes: bytes) -> Optional[NetworkTopologyPacket]:
    """
    Parses a raw Ethernet frame into a NetworkTopologyPacket causally.
    Returns None for non-IPv4 frames or parsing failures.
    """
    try:
        eth = dpkt.ethernet.Ethernet(raw_bytes)
    except Exception:
        return None

    if eth.type != dpkt.ethernet.ETH_TYPE_IP:
        return None

    ip = eth.data
    if not isinstance(ip, dpkt.ip.IP):
        return None

    try:
        src_ip = socket.inet_ntoa(ip.src)
        dst_ip = socket.inet_ntoa(ip.dst)
    except Exception:
        return None

    proto = int(ip.p)
    src_port, dst_port = 0, 0
    if proto == dpkt.ip.IP_PROTO_TCP and isinstance(ip.data, dpkt.tcp.TCP):
        src_port = int(ip.data.sport)
        dst_port = int(ip.data.dport)
    elif proto == dpkt.ip.IP_PROTO_UDP and isinstance(ip.data, dpkt.udp.UDP):
        src_port = int(ip.data.sport)
        dst_port = int(ip.data.dport)

    return NetworkTopologyPacket(
        timestamp=float(ts),
        src_ip=src_ip,
        dst_ip=dst_ip,
        src_port=src_port,
        dst_port=dst_port,
        protocol=proto,
        pkt_len=int(len(raw_bytes))
    )


def compute_network_topology_from_counts(
    src_counts: Counter,
    dst_counts: Counter,
    flow_counts: Counter,
    total_packets: int
) -> Dict[str, float]:
    """
    Computes exact network topology features from pre-aggregated frequency counts.
    Ensures zero NaN, zero Inf, and strict numerical stability.
    """
    if total_packets == 0 or len(src_counts) == 0:
        return {
            "src_ip_cardinality": 0.0,
            "src_concentration_ratio": 0.0,
            "dst_ip_cardinality": 0.0,
            "unique_flow_count": 0.0,
            "src_ip_entropy": 0.0,
            "fan_in_ratio": 0.0,
            "fan_out_ratio": 0.0
        }

    src_card = float(len(src_counts))
    dst_card = float(len(dst_counts))
    flow_card = float(len(flow_counts))

    # 2. src_concentration_ratio: proportion of packets from top source
    top_src_count = max(src_counts.values())
    src_concentration = float(top_src_count) / float(total_packets)

    # 5. src_ip_entropy: -sum p(s) log2 p(s)
    entropy = 0.0
    for count in src_counts.values():
        p = count / total_packets
        if p > 0.0:
            entropy -= p * math.log2(p)

    # 6. fan_in_ratio: |src| / |dst|
    if dst_card > 0.0:
        fan_in = src_card / dst_card
    else:
        fan_in = 0.0

    # 7. fan_out_ratio: |dst| / |src|
    if src_card > 0.0:
        fan_out = dst_card / src_card
    else:
        fan_out = 0.0

    return {
        "src_ip_cardinality": src_card,
        "src_concentration_ratio": float(np.clip(src_concentration, 0.0, 1.0)),
        "dst_ip_cardinality": dst_card,
        "unique_flow_count": flow_card,
        "src_ip_entropy": float(max(0.0, entropy)),
        "fan_in_ratio": float(max(0.0, fan_in)),
        "fan_out_ratio": float(max(0.0, fan_out))
    }


def compute_network_topology_batch(packets: List[NetworkTopologyPacket]) -> Dict[str, float]:
    """
    Offline batch computation for a list of packets in an observation window.
    Acts as the golden ground truth for equivalence validation.
    """
    if not packets:
        return compute_network_topology_from_counts(Counter(), Counter(), Counter(), 0)

    src_counts = Counter(p.src_ip for p in packets)
    dst_counts = Counter(p.dst_ip for p in packets)
    flow_counts = Counter(p.canonical_flow_key for p in packets)
    return compute_network_topology_from_counts(src_counts, dst_counts, flow_counts, len(packets))


class NetworkTopologyTracker:
    """
    Streaming sliding/tumbling network-level time window tracker.
    
    Supports:
      - Tumbling mode: window advances by window_sec when elapsed time >= window_sec.
      - Sliding FIFO mode: window continuously evicts packets older than (current_ts - window_sec).
    
    Guarantees:
      - Strictly causal: packets arrive in non-decreasing timestamp order.
      - Incremental state maintenance with zero lookahead.
    """

    def __init__(self, window_sec: float = 1.0, mode: str = "tumbling"):
        if window_sec <= 0.0:
            raise ValueError("window_sec must be positive")
        if mode not in ("tumbling", "sliding"):
            raise ValueError("mode must be 'tumbling' or 'sliding'")

        self.window_sec: float = float(window_sec)
        self.mode: str = mode

        # Streaming state
        self.window_start_ts: Optional[float] = None
        self.last_ts: Optional[float] = None

        # Frequency counters
        self.src_counts: Counter = Counter()
        self.dst_counts: Counter = Counter()
        self.flow_counts: Counter = Counter()
        self.total_packets: int = 0

        # For sliding FIFO mode
        self.packet_queue: deque[NetworkTopologyPacket] = deque()

    def reset(self):
        """Clears all streaming window state."""
        self.window_start_ts = None
        self.last_ts = None
        self.src_counts.clear()
        self.dst_counts.clear()
        self.flow_counts.clear()
        self.total_packets = 0
        self.packet_queue.clear()

    def add_packet(self, packet: Optional[NetworkTopologyPacket]) -> Tuple[bool, Optional[Dict[str, float]]]:
        """
        Ingests a packet causally.
        Returns:
          (is_window_ready, optional_features_if_ready)
        """
        if packet is None:
            return False, None

        ts = float(packet.timestamp)
        if not math.isfinite(ts):
            raise ValueError(f"Non-finite topology packet timestamp {ts!r}")
        if self.last_ts is not None and ts < self.last_ts:
            raise ValueError(
                f"Out-of-order topology packet timestamp {ts:.6f} < last_ts {self.last_ts:.6f}"
            )
        self.last_ts = ts

        # Initialize window start on first packet
        if self.window_start_ts is None:
            self.window_start_ts = ts

        if self.mode == "tumbling":
            # Check if current packet triggers window closure
            if (ts - self.window_start_ts) >= self.window_sec:
                # Extract features of completed window
                features = compute_network_topology_from_counts(
                    self.src_counts, self.dst_counts, self.flow_counts, self.total_packets
                )
                features["window_start_ts"] = float(self.window_start_ts)
                features["window_end_ts"] = float(ts)
                features["packet_count"] = float(self.total_packets)

                # Reset state for next tumbling window
                self.window_start_ts = ts
                self.src_counts.clear()
                self.dst_counts.clear()
                self.flow_counts.clear()
                self.total_packets = 0

                # Ingest new packet into new window
                self._ingest_packet(packet)
                return True, features
            else:
                self._ingest_packet(packet)
                return False, None

        else: # Sliding FIFO mode
            self.packet_queue.append(packet)
            self._ingest_packet(packet)

            # Evict expired packets from front of queue
            cutoff_ts = ts - self.window_sec
            while self.packet_queue and self.packet_queue[0].timestamp < cutoff_ts:
                expired = self.packet_queue.popleft()
                self._evict_packet(expired)

            self.window_start_ts = self.packet_queue[0].timestamp if self.packet_queue else ts
            return False, None

    def _ingest_packet(self, p: NetworkTopologyPacket):
        self.src_counts[p.src_ip] += 1
        self.dst_counts[p.dst_ip] += 1
        self.flow_counts[p.canonical_flow_key] += 1
        self.total_packets += 1

    def _evict_packet(self, p: NetworkTopologyPacket):
        self.src_counts[p.src_ip] -= 1
        if self.src_counts[p.src_ip] <= 0:
            del self.src_counts[p.src_ip]

        self.dst_counts[p.dst_ip] -= 1
        if self.dst_counts[p.dst_ip] <= 0:
            del self.dst_counts[p.dst_ip]

        flow = p.canonical_flow_key
        self.flow_counts[flow] -= 1
        if self.flow_counts[flow] <= 0:
            del self.flow_counts[flow]

        self.total_packets -= 1

    def extract_current_features(self) -> Dict[str, float]:
        """Extracts features from the currently accumulated state."""
        feats = compute_network_topology_from_counts(
            self.src_counts, self.dst_counts, self.flow_counts, self.total_packets
        )
        feats["window_start_ts"] = float(self.window_start_ts or 0.0)
        feats["window_end_ts"] = float(self.last_ts or 0.0)
        feats["packet_count"] = float(self.total_packets)
        return feats

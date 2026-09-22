"""
aegis_live/packet_window.py
===========================
Deterministic Sliding Flow-Window Buffer for AEGIS-Live v2.
Aggregates packet streams into bounded temporal or packet-count windows
and extracts clean, physically genuine statistical metrics.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import ipaddress

from aegis_live.feature_schema import CANONICAL_24_FEATURES


@dataclass
class PacketRecord:
    timestamp: float       # Epoch seconds (float)
    pkt_len: int           # Total frame/datagram length in bytes
    ip_proto: int          # IP protocol number (6=TCP, 17=UDP, 1=ICMP)
    ip_ttl: int            # IP Time-To-Live
    tcp_flags: int         # Bitmask of TCP flags (FIN=1, SYN=2, RST=4, PSH=8, ACK=16, URG=32, ECE=64, CWR=128)
    is_inbound: bool       # True if packet destination is local host/network


class FlowWindow:
    """
    Maintains a sliding packet window for a single bidirectional flow key.
    """
    def __init__(self, flow_key: Tuple[str, str, int], max_packets: int = 50, timeout_sec: float = 2.0):
        self.flow_key = flow_key
        self.max_packets = max_packets
        self.timeout_sec = timeout_sec
        self.packets: List[PacketRecord] = []
        self.first_timestamp: Optional[float] = None
        self.last_timestamp: Optional[float] = None

    def add_packet(self, pkt: PacketRecord) -> bool:
        """
        Adds a packet to the window.
        Returns True if the window is ready for extraction (capacity reached).
        Raises ValueError if packet timestamp gap from previous packet >= timeout_sec.
        """
        if self.last_timestamp is not None and (pkt.timestamp - self.last_timestamp) >= self.timeout_sec:
            raise ValueError(
                f"Cannot add packet with timestamp {pkt.timestamp:.6f} to window with last timestamp "
                f"{self.last_timestamp:.6f}: gap {pkt.timestamp - self.last_timestamp:.6f}s >= timeout {self.timeout_sec}s"
            )
        if not self.packets:
            self.first_timestamp = pkt.timestamp
            self.last_timestamp = pkt.timestamp
        else:
            self.last_timestamp = max(self.last_timestamp, pkt.timestamp)
        self.packets.append(pkt)
        return len(self.packets) >= self.max_packets

    def is_timed_out(self, current_time: float) -> bool:
        """
        Returns True if the flow has been inactive longer than timeout_sec.
        """
        if not self.packets or self.last_timestamp is None:
            return False
        return (current_time - self.last_timestamp) >= self.timeout_sec

    def extract_features(self) -> Dict[str, float]:
        """
        Computes the exact 24 features from the accumulated packets.
        Guarantees:
        - Finite float32 values
        - Clean zero-initialized IAT (NO epoch artifacts)
        - Deterministic math
        """
        N = len(self.packets)
        if N == 0:
            raise ValueError("Cannot extract features from empty window")

        # 1. Timing & Duration
        t_start = self.packets[0].timestamp
        t_end = self.packets[-1].timestamp
        duration_sec = max(t_end - t_start, 0.0)
        duration_ms = duration_sec * 1000.0

        # Safe rate denominator (minimum 1 microsecond to prevent zero division)
        safe_duration_sec = max(duration_sec, 1e-6)
        pkt_rate_pps = float(N) / safe_duration_sec

        # 2. Byte Rate & Packet Sizes
        sizes = [p.pkt_len for p in self.packets]
        total_bytes = sum(sizes)
        byte_rate_bps = float(total_bytes) / safe_duration_sec

        pkt_size_mean = float(np.mean(sizes))
        pkt_size_std = float(np.std(sizes, ddof=1)) if N > 1 else 0.0
        pkt_size_min = float(min(sizes))
        pkt_size_max = float(max(sizes))

        # 3. Clean Zero-Initialized IAT (Strictly Inter-Packet Differences)
        if N > 1:
            iats_sec = [self.packets[i].timestamp - self.packets[i - 1].timestamp for i in range(1, N)]
            # Bounded non-negative IAT
            iats_ms = [max(dt, 0.0) * 1000.0 for dt in iats_sec]
            iat_mean_ms = float(np.mean(iats_ms))
            iat_std_ms = float(np.std(iats_ms, ddof=1)) if len(iats_ms) > 1 else 0.0
        else:
            iat_mean_ms = 0.0
            iat_std_ms = 0.0

        # 4. TCP Control Flags
        # Standard bitmask: FIN=0x01, SYN=0x02, RST=0x04, PSH=0x08, ACK=0x10, URG=0x20, ECE=0x40, CWR=0x80
        syn_cnt = sum(1 for p in self.packets if (p.tcp_flags & 0x02))
        ack_cnt = sum(1 for p in self.packets if (p.tcp_flags & 0x10))
        rst_cnt = sum(1 for p in self.packets if (p.tcp_flags & 0x04))
        fin_cnt = sum(1 for p in self.packets if (p.tcp_flags & 0x01))
        urg_cnt = sum(1 for p in self.packets if (p.tcp_flags & 0x20))
        psh_cnt = sum(1 for p in self.packets if (p.tcp_flags & 0x08))
        ece_cnt = sum(1 for p in self.packets if (p.tcp_flags & 0x40))
        cwr_cnt = sum(1 for p in self.packets if (p.tcp_flags & 0x80))

        syn_ratio = float(syn_cnt) / N
        ack_ratio = float(ack_cnt) / N
        rst_ratio = float(rst_cnt) / N
        fin_ratio = float(fin_cnt) / N
        urg_ratio = float(urg_cnt) / N
        psh_ratio = float(psh_cnt) / N
        ece_ratio = float(ece_cnt) / N
        cwr_ratio = float(cwr_cnt) / N

        # 5. IP Layer (TTL & Protocol Ratios)
        ttls = [p.ip_ttl for p in self.packets]
        ip_ttl_mean = float(np.mean(ttls))
        ip_ttl_std = float(np.std(ttls, ddof=1)) if N > 1 else 0.0

        tcp_cnt = sum(1 for p in self.packets if p.ip_proto == 6)
        udp_cnt = sum(1 for p in self.packets if p.ip_proto == 17)
        icmp_cnt = sum(1 for p in self.packets if p.ip_proto == 1)

        tcp_ratio = float(tcp_cnt) / N
        udp_ratio = float(udp_cnt) / N
        icmp_ratio = float(icmp_cnt) / N

        # 6. Directional Ratios
        inbound_cnt = sum(1 for p in self.packets if p.is_inbound)
        outbound_cnt = N - inbound_cnt
        inbound_pkt_ratio = float(inbound_cnt) / N
        outbound_pkt_ratio = float(outbound_cnt) / N

        return {
            "duration_ms": duration_ms,
            "pkt_rate_pps": pkt_rate_pps,
            "byte_rate_bps": byte_rate_bps,
            "iat_mean_ms": iat_mean_ms,
            "iat_std_ms": iat_std_ms,
            "pkt_size_mean": pkt_size_mean,
            "pkt_size_std": pkt_size_std,
            "pkt_size_min": pkt_size_min,
            "pkt_size_max": pkt_size_max,
            "syn_ratio": syn_ratio,
            "ack_ratio": ack_ratio,
            "rst_ratio": rst_ratio,
            "fin_ratio": fin_ratio,
            "urg_ratio": urg_ratio,
            "psh_ratio": psh_ratio,
            "ece_ratio": ece_ratio,
            "cwr_ratio": cwr_ratio,
            "ip_ttl_mean": ip_ttl_mean,
            "ip_ttl_std": ip_ttl_std,
            "tcp_ratio": tcp_ratio,
            "udp_ratio": udp_ratio,
            "icmp_ratio": icmp_ratio,
            "inbound_pkt_ratio": inbound_pkt_ratio,
            "outbound_pkt_ratio": outbound_pkt_ratio
        }

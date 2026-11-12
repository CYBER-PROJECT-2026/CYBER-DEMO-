"""
aegis_live/packet_window.py
===========================
Deterministic Sliding Flow-Window Buffer for AEGIS-Live v2.
Aggregates packet streams into bounded temporal or packet-count windows
and extracts clean, physically genuine statistical metrics.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import heapq
import math
import numpy as np
import ipaddress
import socket

from aegis_live.feature_schema import CANONICAL_24_FEATURES


class ReorderBufferOverflowError(RuntimeError):
    """Raised fail-closed when the bounded reorder buffer exceeds max_packets."""


@dataclass(slots=True)
class ParsedIPv4Frame:
    """Validated Ethernet/IPv4/L4 parse result using V4 raw Ethernet len(buf) semantics."""
    src_ip: str
    dst_ip: str
    ip_proto: int
    ip_ttl: int
    ip_ihl: int
    ip_tot_len: int
    pkt_len: int
    sport: int
    dport: int
    tcp_flags: int
    is_fragment: bool
    is_non_initial_fragment: bool
    has_ethernet_padding: bool


def classify_ipv4_ethernet_frame(buf: bytes) -> Optional[str]:
    """
    Pre-reorder L2/L3 frame filter shared by runtime and dataset builder.
    Returns an explicit rejection reason for short/VLAN/non-IPv4 frames, or None if IPv4 candidate.
    """
    if len(buf) < 14:
        return "ethernet_frame_too_short"
    eth_type = (buf[12] << 8) | buf[13]
    if eth_type in (0x8100, 0x88A8):
        return "non_ipv4_vlan_8021q"
    if eth_type != 0x0800:
        return "non_ipv4_packet"
    if len(buf) < 34:
        return "ipv4_packet_too_short"
    return None


def parse_ipv4_ethernet_frame(buf: bytes) -> Tuple[Optional[ParsedIPv4Frame], Optional[str]]:
    """
    Shared deterministic Ethernet II + IPv4 + TCP/UDP header validator/parser
    used identically by AegisLiveV4Pipeline and LeakageSafeV4DatasetBuilder.
    Validates L2/L3/L4 bounds before reading fields while preserving V4 pkt_len = len(buf).
    """
    pre_err = classify_ipv4_ethernet_frame(buf)
    if pre_err is not None:
        return None, pre_err

    version = (buf[14] >> 4) & 0x0F
    if version != 4:
        return None, "invalid_ipv4_version"

    ihl = (buf[14] & 0x0F) * 4
    if ihl < 20:
        return None, "invalid_ip_header_length"
    if len(buf) < 14 + ihl:
        return None, "truncated_ipv4_header"

    ip_tot_len = (buf[16] << 8) | buf[17]
    if ip_tot_len < ihl:
        return None, "invalid_ipv4_total_length"
    l3_end = 14 + ip_tot_len
    if len(buf) < l3_end:
        return None, "truncated_ipv4_payload"

    flags_frag = (buf[20] << 8) | buf[21]
    mf_flag = bool(flags_frag & 0x2000)
    frag_offset = (flags_frag & 0x1FFF) * 8
    is_fragment = mf_flag or (frag_offset > 0)
    is_non_initial_fragment = frag_offset > 0
    has_ethernet_padding = len(buf) > l3_end

    ttl = buf[22]
    ip_proto = buf[23]
    src_ip = socket.inet_ntoa(buf[26:30])
    dst_ip = socket.inet_ntoa(buf[30:34])

    sport = 0
    dport = 0
    tcp_flags = 0
    l4_offset = 14 + ihl
    l4_avail = l3_end - l4_offset

    if not is_non_initial_fragment:
        if ip_proto == 6:  # TCP requires standard 20-byte header inside IP payload
            if l4_avail < 20:
                return None, "truncated_tcp_header"
            data_offset = (buf[l4_offset + 12] >> 4) * 4
            if data_offset < 20 or l4_avail < data_offset:
                return None, "truncated_tcp_header"
            sport = (buf[l4_offset] << 8) | buf[l4_offset + 1]
            dport = (buf[l4_offset + 2] << 8) | buf[l4_offset + 3]
            tcp_flags = buf[l4_offset + 13]
        elif ip_proto == 17:  # UDP requires 8-byte header inside IP payload
            if l4_avail < 8:
                return None, "truncated_udp_header"
            sport = (buf[l4_offset] << 8) | buf[l4_offset + 1]
            dport = (buf[l4_offset + 2] << 8) | buf[l4_offset + 3]

    return ParsedIPv4Frame(
        src_ip=src_ip,
        dst_ip=dst_ip,
        ip_proto=ip_proto,
        ip_ttl=ttl,
        ip_ihl=ihl,
        ip_tot_len=ip_tot_len,
        pkt_len=int(len(buf)),
        sport=sport,
        dport=dport,
        tcp_flags=tcp_flags,
        is_fragment=is_fragment,
        is_non_initial_fragment=is_non_initial_fragment,
        has_ethernet_padding=has_ethernet_padding,
    ), None


def classify_direction(src_is_local: bool, dst_is_local: bool) -> str:
    """Diagnostic direction class; canonical V4 direction still uses local(dst)."""
    if src_is_local and dst_is_local:
        return "local_to_local"
    if src_is_local:
        return "local_to_external"
    if dst_is_local:
        return "external_to_local"
    return "external_to_external"


class MonotonicTimestampGuard:
    """
    Shared strict chronological processing core guard.
    Accepts finite ts >= last_accepted_ts (preserving stream order on equal timestamps)
    and rejects non-finite or ts < last_accepted_ts without clamping or silent reordering.
    """

    def __init__(self, max_recorded_events: int = 100):
        self.last_accepted_ts: Optional[float] = None
        self.accepted_count: int = 0
        self.rejected_ooo_count: int = 0
        self.rejected_non_finite_count: int = 0
        self.max_regression_sec: float = 0.0
        self.last_rejected_event: Optional[Dict[str, Any]] = None
        self.max_recorded_events: int = max(0, int(max_recorded_events))
        self.rejected_events: List[Dict[str, Any]] = []

    def reset(self) -> None:
        self.last_accepted_ts = None
        self.accepted_count = 0
        self.rejected_ooo_count = 0
        self.rejected_non_finite_count = 0
        self.max_regression_sec = 0.0
        self.last_rejected_event = None
        self.rejected_events.clear()

    def check_and_accept(self, ts: float) -> Tuple[bool, Optional[Dict[str, Any]]]:
        ts = float(ts)
        if not math.isfinite(ts):
            self.rejected_non_finite_count += 1
            event = {
                "reason": "non_finite_timestamp",
                "rejected_ts": ts,
                "previous_accepted_ts": self.last_accepted_ts,
            }
            self.last_rejected_event = event
            if len(self.rejected_events) < self.max_recorded_events:
                self.rejected_events.append(event)
            return False, event

        if self.last_accepted_ts is not None and ts < self.last_accepted_ts:
            regression = self.last_accepted_ts - ts
            self.rejected_ooo_count += 1
            if regression > self.max_regression_sec:
                self.max_regression_sec = regression
            event = {
                "reason": "out_of_order_timestamp",
                "rejected_ts": ts,
                "previous_accepted_ts": self.last_accepted_ts,
                "regression_sec": regression,
            }
            self.last_rejected_event = event
            if len(self.rejected_events) < self.max_recorded_events:
                self.rejected_events.append(event)
            return False, event

        self.last_accepted_ts = ts
        self.accepted_count += 1
        return True, None


class BoundedReorderBuffer:
    """
    Bounded min-heap packet reorder buffer for offline PCAP replay and dataset building.
    Orders packets stably by (timestamp, arrival_sequence), advancing a sliding
    watermark = max_seen_timestamp - lateness_sec. Fails closed on buffer overflow.
    """

    def __init__(self, lateness_sec: float = 0.25, max_packets: int = 10000):
        lateness_sec = float(lateness_sec)
        if not math.isfinite(lateness_sec) or lateness_sec < 0.0:
            raise ValueError("lateness_sec must be finite and non-negative")
        if max_packets < 1:
            raise ValueError("max_packets must be >= 1")
        self.lateness_sec: float = lateness_sec
        self.max_packets: int = int(max_packets)
        self._heap: List[Tuple[float, int, Any]] = []
        self._seq: int = 0
        self.max_seen_ts: Optional[float] = None
        self.watermark_ts: Optional[float] = None
        self.last_emitted_ts: Optional[float] = None
        self.accepted_count: int = 0
        self.emitted_count: int = 0
        self.rejected_late_count: int = 0
        self.rejected_non_finite_count: int = 0
        self.overflow_count: int = 0
        self.max_buffer_occupancy: int = 0
        self.rejected_events: List[Dict[str, Any]] = []

    def reset(self) -> None:
        self._heap.clear()
        self._seq = 0
        self.max_seen_ts = None
        self.watermark_ts = None
        self.last_emitted_ts = None
        self.accepted_count = 0
        self.emitted_count = 0
        self.rejected_late_count = 0
        self.rejected_non_finite_count = 0
        self.overflow_count = 0
        self.max_buffer_occupancy = 0
        self.rejected_events.clear()

    def _record_rejection(self, ts: float, reason: str, is_late: bool = True) -> None:
        if is_late:
            self.rejected_late_count += 1
        else:
            self.rejected_non_finite_count += 1
        if len(self.rejected_events) < 100:
            self.rejected_events.append({
                "rejected_ts": ts,
                "reason": reason,
                "last_emitted_ts": self.last_emitted_ts,
                "watermark_ts": self.watermark_ts,
                "max_seen_ts": self.max_seen_ts,
            })

    def push(self, ts: float, payload: Any) -> Tuple[bool, Optional[str], List[Tuple[float, Any]]]:
        """
        Submits a (ts, payload) item into the bounded reorder buffer.
        Returns: (accepted, rejection_reason_if_rejected, ready_items_in_order).
        Raises ReorderBufferOverflowError fail-closed if buffer exceeds max_packets.
        """
        ts = float(ts)
        if not math.isfinite(ts):
            self._record_rejection(ts, "non_finite_timestamp", is_late=False)
            return False, "non_finite_timestamp", []

        # Check late arrivals relative to already-emitted timestamps or emitted watermark
        if self.last_emitted_ts is not None and ts < self.last_emitted_ts:
            self._record_rejection(ts, "earlier_than_last_emitted", is_late=True)
            return False, "earlier_than_last_emitted", []
        if self.watermark_ts is not None and ts < (self.watermark_ts - 1e-9):
            self._record_rejection(ts, "behind_emitted_watermark", is_late=True)
            return False, "behind_emitted_watermark", []

        # Advance high-water mark and watermark
        if self.max_seen_ts is None or ts > self.max_seen_ts:
            self.max_seen_ts = ts
        self.watermark_ts = self.max_seen_ts - self.lateness_sec

        # Drain existing eligible items at or before the updated watermark first
        ready: List[Tuple[float, Any]] = []
        while self._heap and self._heap[0][0] <= (self.watermark_ts + 1e-9):
            emit_ts, _, emit_payload = heapq.heappop(self._heap)
            self.last_emitted_ts = emit_ts
            self.emitted_count += 1
            ready.append((emit_ts, emit_payload))

        # Check fail-closed capacity before buffering new item (unless it can emit immediately at <= watermark)
        if ts <= (self.watermark_ts + 1e-9):
            # Can emit after draining all <= watermark? Only if heap is empty or head > ts
            if not self._heap or ts < self._heap[0][0]:
                self.accepted_count += 1
                self.last_emitted_ts = ts
                self.emitted_count += 1
                ready.append((ts, payload))
                return True, None, ready

        if len(self._heap) >= self.max_packets:
            self.overflow_count += 1
            raise ReorderBufferOverflowError(
                f"BoundedReorderBuffer overflow: buffered={len(self._heap)} >= max_packets={self.max_packets} "
                f"(ts={ts:.6f}, watermark={self.watermark_ts:.6f})"
            )

        heapq.heappush(self._heap, (ts, self._seq, payload))
        self._seq += 1
        self.accepted_count += 1
        if len(self._heap) > self.max_buffer_occupancy:
            self.max_buffer_occupancy = len(self._heap)
        return True, None, ready

    def drain(self) -> List[Tuple[float, Any]]:
        """Drains all remaining buffered packets in stable (ts, arrival_seq) order at end of stream."""
        ready: List[Tuple[float, Any]] = []
        while self._heap:
            emit_ts, _, emit_payload = heapq.heappop(self._heap)
            self.last_emitted_ts = emit_ts
            self.emitted_count += 1
            ready.append((emit_ts, emit_payload))
        return ready


@dataclass
class PacketRecord:
    timestamp: float       # Epoch seconds (float)
    pkt_len: int           # Total frame/datagram length in bytes
    ip_proto: int          # IP protocol number (6=TCP, 17=UDP, 1=ICMP)
    ip_ttl: int            # IP Time-To-Live
    tcp_flags: int         # Bitmask of TCP flags (FIN=1, SYN=2, RST=4, PSH=8, ACK=16, URG=32, ECE=64, CWR=128)
    is_inbound: bool       # True if packet destination is local host/network
    sport: int = 0         # Source port (0 if unavailable or ICMP)
    dport: int = 0         # Destination port (0 if unavailable or ICMP)


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
        self.ports_seen: set = set()
        self.last_topology_snapshot: Optional[Dict[str, float]] = None
        self.ooo_rejected_count: int = 0
        self.affected_by_ooo_rejection: bool = False

    def mark_ooo_rejection(self) -> None:
        """Records metadata that an out-of-order packet targeting this flow was rejected."""
        self.ooo_rejected_count += 1
        self.affected_by_ooo_rejection = True

    def set_topology_snapshot(self, snapshot: Optional[Dict[str, float]]) -> None:
        """
        Stores an immutable causal snapshot of network topology features captured
        at this window's latest packet arrival timestamp.
        """
        self.last_topology_snapshot = dict(snapshot) if snapshot is not None else None

    def add_packet(self, pkt: PacketRecord) -> bool:
        """
        Adds a packet to the window.
        Returns True if the window is ready for extraction (capacity reached).
        Raises ValueError if packet timestamp is non-finite, regresses (< last_timestamp), or gap >= timeout_sec.
        """
        ts = float(pkt.timestamp)
        if not math.isfinite(ts):
            raise ValueError(
                f"Non-finite packet timestamp {pkt.timestamp!r} for flow {self.flow_key}"
            )
        if self.last_timestamp is not None and pkt.timestamp < self.last_timestamp:
            raise ValueError(
                f"Out-of-order packet timestamp {pkt.timestamp:.6f} < last timestamp "
                f"{self.last_timestamp:.6f} for flow {self.flow_key}"
            )
        if self.last_timestamp is not None and (pkt.timestamp - self.last_timestamp) >= self.timeout_sec:
            raise ValueError(
                f"Cannot add packet with timestamp {pkt.timestamp:.6f} to window with last timestamp "
                f"{self.last_timestamp:.6f}: gap {pkt.timestamp - self.last_timestamp:.6f}s >= timeout {self.timeout_sec}s"
            )
        if getattr(pkt, "sport", 0):
            self.ports_seen.add(pkt.sport)
        if getattr(pkt, "dport", 0):
            self.ports_seen.add(pkt.dport)
        if not self.packets:
            self.first_timestamp = pkt.timestamp
            self.last_timestamp = pkt.timestamp
        else:
            self.last_timestamp = pkt.timestamp
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

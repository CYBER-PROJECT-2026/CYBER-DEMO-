import time
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple, Callable

class NetworkFlow:
    def __init__(self, key: Tuple, src_ip: str, dst_ip: str, src_port: int, dst_port: int, proto: str, start_time: float):
        self.key = key
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        self.proto = proto
        self.start_time = start_time
        self.last_time = start_time
        
        # Packet & Byte counters
        self.fwd_packets = 0
        self.bwd_packets = 0
        self.fwd_bytes = 0
        self.bwd_bytes = 0
        self.total_bytes = 0
        
        # TCP Flags breakdown
        self.flags = {
            "FIN": 0,
            "SYN": 0,
            "RST": 0,
            "PSH": 0,
            "ACK": 0,
            "URG": 0
        }

    def update(self, packet_len: int, is_forward: bool, flags_dict: Optional[Dict[str, int]], timestamp: float):
        self.last_time = timestamp
        self.total_bytes += packet_len
        if is_forward:
            self.fwd_packets += 1
            self.fwd_bytes += packet_len
        else:
            self.bwd_packets += 1
            self.bwd_bytes += packet_len

        if flags_dict:
            for k, v in flags_dict.items():
                if k in self.flags:
                    self.flags[k] += v

    def to_features(self) -> Dict[str, Any]:
        """Calculates exact 17 features required by the XGBoost detection model"""
        duration = max(self.last_time - self.start_time, 0.0001)
        tot_packets = max(self.fwd_packets + self.bwd_packets, 1)

        rate = tot_packets / duration
        srate = self.fwd_packets / duration
        drate = self.bwd_packets / duration

        proto_type = 6 if self.proto == "TCP" else (17 if self.proto == "UDP" else 1)

        features = {
            "flow_duration": round(duration, 6),
            "Protocol Type": proto_type,
            "Duration": round(duration, 6),
            "Rate": round(rate, 4),
            "Srate": round(srate, 4),
            "Drate": round(drate, 4),
            "fin_flag_number": 1 if self.flags["FIN"] > 0 else 0,
            "syn_flag_number": 1 if self.flags["SYN"] > 0 else 0,
            "rst_flag_number": 1 if self.flags["RST"] > 0 else 0,
            "psh_flag_number": 1 if self.flags["PSH"] > 0 else 0,
            "ack_flag_number": 1 if self.flags["ACK"] > 0 else 0,
            "syn_count": self.flags["SYN"],
            "fin_count": self.flags["FIN"],
            "TCP": 1 if self.proto == "TCP" else 0,
            "UDP": 1 if self.proto == "UDP" else 0,
            "Tot sum": self.total_bytes,
            "Weight": tot_packets
        }
        return features

class RealTimeFlowExtractor:
    def __init__(self, active_timeout: float = 1.0, idle_timeout: float = 0.5, packet_threshold: int = 50, on_flow_ready: Optional[Callable] = None):
        self.active_timeout = active_timeout
        self.idle_timeout = idle_timeout
        self.packet_threshold = packet_threshold
        self.on_flow_ready = on_flow_ready
        self.active_flows: Dict[Tuple, NetworkFlow] = {}

    def _get_flow_key(self, src_ip: str, src_port: int, dst_ip: str, dst_port: int, proto: str) -> Tuple[Tuple, bool]:
        """Creates a bidirectional canonical flow key and returns (key, is_forward)"""
        forward_tuple = ((src_ip, src_port), (dst_ip, dst_port), proto)
        backward_tuple = ((dst_ip, dst_port), (src_ip, src_port), proto)

        if forward_tuple in self.active_flows:
            return forward_tuple, True
        elif backward_tuple in self.active_flows:
            return backward_tuple, False
        else:
            return forward_tuple, True

    def process_packet_raw(self, src_ip: str, src_port: int, dst_ip: str, dst_port: int, proto: str, packet_len: int, flags_dict: Optional[Dict[str, int]] = None, timestamp: Optional[float] = None) -> Optional[Dict[str, Any]]:
        ts = timestamp if timestamp is not None else time.time()
        flow_key, is_forward = self._get_flow_key(src_ip, src_port, dst_ip, dst_port, proto)

        if flow_key not in self.active_flows:
            self.active_flows[flow_key] = NetworkFlow(flow_key, src_ip, dst_ip, src_port, dst_port, proto, ts)

        flow = self.active_flows[flow_key]
        flow.update(packet_len, is_forward, flags_dict, ts)

        # Check for immediate active flush conditions
        duration = flow.last_time - flow.start_time
        tot_packets = flow.fwd_packets + flow.bwd_packets

        if duration >= self.active_timeout or tot_packets >= self.packet_threshold:
            features = flow.to_features()
            flow_metadata = {
                "src_ip": flow.src_ip,
                "src_port": flow.src_port,
                "dst_ip": flow.dst_ip,
                "dst_port": flow.dst_port,
                "proto": flow.proto,
                "features": features
            }
            del self.active_flows[flow_key]
            if self.on_flow_ready:
                self.on_flow_ready(flow_metadata)
            return flow_metadata

        return None

    def flush_expired_flows(self) -> List[Dict[str, Any]]:
        """Flushes flows that exceeded idle timeout"""
        now = time.time()
        expired = []
        keys_to_delete = []

        for key, flow in self.active_flows.items():
            if now - flow.last_time >= self.idle_timeout or now - flow.start_time >= self.active_timeout:
                features = flow.to_features()
                flow_data = {
                    "src_ip": flow.src_ip,
                    "src_port": flow.src_port,
                    "dst_ip": flow.dst_ip,
                    "dst_port": flow.dst_port,
                    "proto": flow.proto,
                    "features": features
                }
                expired.append(flow_data)
                keys_to_delete.append(key)
                if self.on_flow_ready:
                    self.on_flow_ready(flow_data)

        for k in keys_to_delete:
            if k in self.active_flows:
                del self.active_flows[k]

        return expired

import threading
import time
from typing import Optional, Callable, Dict, Any
from .flow_extractor import RealTimeFlowExtractor

try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP, get_if_list
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

class LivePacketSniffer:
    def __init__(self, flow_extractor: RealTimeFlowExtractor, iface: Optional[str] = None):
        self.flow_extractor = flow_extractor
        self.iface = iface
        self.is_running = False
        self.sniffer_thread: Optional[threading.Thread] = None
        self.flush_thread: Optional[threading.Thread] = None
        self.packet_count = 0
        self.byte_count = 0

    def _packet_handler(self, pkt):
        if not self.is_running:
            return
        
        self.packet_count += 1
        self.byte_count += len(pkt)

        if SCAPY_AVAILABLE and pkt.haslayer(IP):
            src_ip = pkt[IP].src
            dst_ip = pkt[IP].dst
            proto = "TCP" if pkt.haslayer(TCP) else ("UDP" if pkt.haslayer(UDP) else ("ICMP" if pkt.haslayer(ICMP) else "OTHER"))
            
            src_port = 0
            dst_port = 0
            flags_dict = None

            if pkt.haslayer(TCP):
                src_port = pkt[TCP].sport
                dst_port = pkt[TCP].dport
                flags = pkt[TCP].flags
                flags_dict = {
                    "FIN": 1 if "F" in str(flags) else 0,
                    "SYN": 1 if "S" in str(flags) else 0,
                    "RST": 1 if "R" in str(flags) else 0,
                    "PSH": 1 if "P" in str(flags) else 0,
                    "ACK": 1 if "A" in str(flags) else 0,
                    "URG": 1 if "U" in str(flags) else 0
                }
            elif pkt.haslayer(UDP):
                src_port = pkt[UDP].sport
                dst_port = pkt[UDP].dport

            self.flow_extractor.process_packet_raw(
                src_ip=src_ip,
                src_port=src_port,
                dst_ip=dst_ip,
                dst_port=dst_port,
                proto=proto,
                packet_len=len(pkt),
                flags_dict=flags_dict,
                timestamp=time.time()
            )

    def _run_sniff(self):
        if not SCAPY_AVAILABLE:
            print("[Sniffer] Scapy not available. Running in simulated / mock mode.")
            return

        try:
            print(f"[Sniffer] Started live packet capture on interface: {self.iface or 'DEFAULT'}...")
            sniff(
                iface=self.iface,
                prn=self._packet_handler,
                store=False,
                stop_filter=lambda _: not self.is_running
            )
        except Exception as e:
            print(f"[Sniffer] Warning / Error during live capture: {e}")

    def _run_flow_flusher(self):
        while self.is_running:
            time.sleep(0.5)
            self.flow_extractor.flush_expired_flows()

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.sniffer_thread = threading.Thread(target=self._run_sniff, daemon=True)
        self.flush_thread = threading.Thread(target=self._run_flow_flusher, daemon=True)
        self.sniffer_thread.start()
        self.flush_thread.start()
        print("[Sniffer] Live packet sniffer and flow tracker activated.")

    def stop(self):
        self.is_running = False
        print("[Sniffer] Live packet sniffer stopped.")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "is_running": self.is_running,
            "packets_captured": self.packet_count,
            "bytes_captured": self.byte_count,
            "active_flows_tracking": len(self.flow_extractor.active_flows)
        }

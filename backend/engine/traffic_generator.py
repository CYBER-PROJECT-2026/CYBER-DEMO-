import time
import random
import threading
from typing import Optional, Callable, Dict, Any
from .flow_extractor import RealTimeFlowExtractor

class IoTTrafficGenerator:
    def __init__(self, flow_extractor: RealTimeFlowExtractor):
        self.flow_extractor = flow_extractor
        self.is_running = False
        self.generator_thread: Optional[threading.Thread] = None
        
        # IoT Device profiles in simulated network
        self.iot_devices = [
            {"ip": "192.168.1.101", "name": "Smart_Thermostat_01", "type": "MQTT_SENSOR"},
            {"ip": "192.168.1.102", "name": "IP_Camera_LivingRoom", "type": "RTSP_STREAM"},
            {"ip": "192.168.1.103", "name": "Smart_Doorlock_Front", "type": "COAP_CLIENT"},
            {"ip": "192.168.1.104", "name": "Medical_Infusion_Pump", "type": "TELEMETRY"},
            {"ip": "192.168.1.105", "name": "Industrial_PLC_Gateway", "type": "MODBUS_TCP"}
        ]
        self.gateway_ip = "192.168.1.1"
        self.cloud_broker_ip = "45.33.32.156"

    def inject_benign_telemetry(self, count: int = 15):
        """Simulates periodic healthy IoT sensor communication"""
        device = random.choice(self.iot_devices)
        src_ip = device["ip"]
        dst_ip = self.cloud_broker_ip if random.random() > 0.5 else self.gateway_ip
        src_port = random.randint(1024, 65535)
        dst_port = 1883 if "MQTT" in device["type"] else 80

        # Normal TCP handshake + payload
        now = time.time()
        # SYN
        self.flow_extractor.process_packet_raw(src_ip, src_port, dst_ip, dst_port, "TCP", 64, {"SYN": 1}, now)
        # SYN-ACK
        self.flow_extractor.process_packet_raw(dst_ip, dst_port, src_ip, src_port, "TCP", 64, {"SYN": 1, "ACK": 1}, now + 0.005)
        # ACK
        self.flow_extractor.process_packet_raw(src_ip, src_port, dst_ip, dst_port, "TCP", 54, {"ACK": 1}, now + 0.010)

        # Telemetry packets
        for _ in range(count):
            self.flow_extractor.process_packet_raw(
                src_ip, src_port, dst_ip, dst_port, "TCP",
                random.randint(120, 500),
                {"PSH": 1, "ACK": 1},
                now + random.uniform(0.02, 0.1)
            )

    def inject_ddos_syn_flood(self, attacker_ip: str = "192.168.1.200", target_ip: str = "192.168.1.104", burst_count: int = 150):
        """Simulates high-velocity SYN Flood targeting an IoT medical device"""
        target_port = 80
        now = time.time()

        for i in range(burst_count):
            sport = random.randint(1024, 65535)
            # High Rate SYN packets with zero ACK/FIN returns
            self.flow_extractor.process_packet_raw(
                attacker_ip, sport, target_ip, target_port, "TCP",
                random.randint(40, 64),
                {"SYN": 1, "ACK": 0, "FIN": 0},
                now + (i * 0.0002)
            )

    def inject_dos_flood(self, attacker_ip: str = "192.168.1.188", target_ip: str = "192.168.1.102", burst_count: int = 80):
        """Simulates UDP/HTTP amplification DoS attack against IP Camera"""
        target_port = 554
        now = time.time()

        for i in range(burst_count):
            sport = random.randint(1024, 65535)
            self.flow_extractor.process_packet_raw(
                attacker_ip, sport, target_ip, target_port, "UDP",
                random.randint(512, 1400),
                None,
                now + (i * 0.0005)
            )

    def inject_spoofing_attack(self, attacker_ip: str = "192.168.1.240", spoofed_ip: str = "192.168.1.1", target_ip: str = "192.168.1.105", burst_count: int = 40):
        """Simulates ARP / IP Spoofing / RST injection against industrial PLC"""
        now = time.time()
        for i in range(burst_count):
            sport = random.randint(1024, 65535)
            self.flow_extractor.process_packet_raw(
                spoofed_ip, sport, target_ip, 502, "TCP",
                random.randint(64, 128),
                {"RST": 1, "SYN": 1},
                now + (i * 0.001)
            )

    def start_background_simulation(self, interval_sec: float = 2.0):
        """Runs continuous background benign traffic with periodic simulated attacks"""
        if self.is_running:
            return
        self.is_running = True

        def _loop():
            step = 0
            while self.is_running:
                step += 1
                # Inject regular benign sensor packets
                self.inject_benign_telemetry(random.randint(5, 12))
                
                # Every 5 steps, inject a targeted IoT attack for live demonstration
                if step % 8 == 0:
                    attack_choice = random.choice(["DDOS", "DOS", "SPOOF"])
                    if attack_choice == "DDOS":
                        self.inject_ddos_syn_flood()
                    elif attack_choice == "DOS":
                        self.inject_dos_flood()
                    else:
                        self.inject_spoofing_attack()

                time.sleep(interval_sec)

        self.generator_thread = threading.Thread(target=_loop, daemon=True)
        self.generator_thread.start()

    def stop_background_simulation(self):
        self.is_running = False

from typing import List, Dict, Any

CYBERSECURITY_KNOWLEDGE_BASE: List[Dict[str, Any]] = [
    # --- MITRE ATT&CK TECHNIQUES ---
    {
        "id": "MITRE-T1498",
        "category": "MITRE_ATTACK",
        "title": "T1498: Network Denial of Service",
        "tactic": "Impact",
        "description": "Adversaries may perform Network Denial of Service (DoS) attacks to degrade or block the availability of targeted IoT services, gateways, and networks. This includes Direct Network Floods (SYN Flood, UDP Flood, ICMP Flood) and Reflection/Amplification attacks.",
        "mitigation": "Deploy stateful firewall ingress filtering, TCP SYN cookies (syncookies=1), upstream cloud scrubbers, dynamic rate limiting per IP, and drop malformed packet bursts.",
        "detection_signatures": ["syn_flag_number == 1", "Drate == 0", "Rate > 1000", "syn_count > 50"]
    },
    {
        "id": "MITRE-T1499",
        "category": "MITRE_ATTACK",
        "title": "T1499: Endpoint Denial of Service",
        "tactic": "Impact",
        "description": "Adversaries may target specific application protocols (HTTP/RTSP/CoAP) on resource-constrained IoT devices to exhaust memory, CPU, or socket pools (e.g. Slowloris, HTTP POST Floods).",
        "mitigation": "Set aggressive connection keep-alive timeouts, limit concurrent open sockets per client IP, deploy reverse proxies with connection throttling.",
        "detection_signatures": ["psh_flag_number == 1", "flow_duration > 5.0", "Weight > 200"]
    },
    {
        "id": "MITRE-T1557",
        "category": "MITRE_ATTACK",
        "title": "T1557: Adversary-in-the-Middle (AiTM) / ARP & IP Spoofing",
        "tactic": "Credential Access, Collection",
        "description": "Adversaries attempt to position themselves between two IoT communicating entities using ARP poisoning, DHCP spoofing, or IP address forgery to intercept sensitive control telemetry or inject rogue commands.",
        "mitigation": "Enable Dynamic ARP Inspection (DAI), DHCP Snooping, 802.1X Port Authentication, and static ARP bindings on critical PLC/Medical IoT gateways.",
        "detection_signatures": ["rst_flag_number == 1", "Protocol Type == 6", "Spoofed MAC/IP"]
    },
    {
        "id": "MITRE-T1046",
        "category": "MITRE_ATTACK",
        "title": "T1046: Network Service Discovery & Port Scanning",
        "tactic": "Discovery",
        "description": "Adversaries scan IoT network segments to enumerate open ports (e.g., Telnet 23, SSH 22, RTSP 554, MQTT 1883, Modbus 502) prior to launching exploit payloads.",
        "mitigation": "Disable unnecessary services, enforce management VLAN isolation, configure automated temporary IP blacklisting for sequential port hit thresholds.",
        "detection_signatures": ["Rate > 500", "syn_flag_number == 1", "ack_flag_number == 0"]
    },

    # --- KNOWN CVEs IN IOT ECOSYSTEMS ---
    {
        "id": "CVE-2023-1389",
        "category": "CVE_NVD",
        "title": "CVE-2023-1389: TP-Link Archer AX21 Command Injection (Mirai Variant)",
        "cvss": 8.8,
        "description": "Unauthenticated command injection in web management interface allowing remote attackers to execute arbitrary shell commands via crafted locale parameters, heavily weaponized by Mirai/Mozi botnets for DDoS reflection.",
        "mitigation": "Apply vendor firmware patch >= v1.1.4, block inbound port 80/443 from external WAN, isolate router administration to dedicated management subnet.",
        "affected_devices": ["TP-Link Routers", "IoT Home Gateways"]
    },
    {
        "id": "CVE-2021-36260",
        "category": "CVE_NVD",
        "title": "CVE-2021-36260: Hikvision IP Camera Pre-Auth Remote Code Execution",
        "cvss": 9.8,
        "description": "A buffer overflow in the web server of multiple Hikvision IP camera models allows remote unauthenticated attackers to gain full root shell access by sending specially crafted XML packets.",
        "mitigation": "Isolate IP cameras on dedicated surveillance VLAN with zero WAN egress, update firmware, block external access to port 80/8000/554.",
        "affected_devices": ["Hikvision IP Cameras", "NVR/DVR Systems"]
    },
    {
        "id": "CVE-2020-11896",
        "category": "CVE_NVD",
        "title": "CVE-2020-11896: Ripple20 Treck TCP/IP Stack Vulnerabilities",
        "cvss": 10.0,
        "description": "Multiple zero-day vulnerabilities in Treck embedded TCP/IP stack affecting millions of medical infusion pumps, industrial PLCs, and IoT smart devices, permitting remote code execution via malformed IPv4/ICMP packets.",
        "mitigation": "Deploy deep packet inspection (DPI) to drop malformed IP fragments and unauthorized IP tunneling, segment clinical/industrial IoT onto air-gapped subnets.",
        "affected_devices": ["Medical Infusion Pumps", "Schneider/Rockwell PLCs", "Smart Meters"]
    },

    # --- NIST & CISA ADVISORIES ---
    {
        "id": "NIST-SP-800-82",
        "category": "STANDARDS",
        "title": "NIST SP 800-82 Rev 3: Guide to Operational Technology (OT) & IoT Security",
        "description": "Provides formal recommendations for securing ICS and IoT networks against availability loss, data forgery, and network disruption.",
        "guidelines": [
            "Implement network zoning (Purdue Model) separating IoT field devices from corporate IT.",
            "Deploy stateful edge inspection and automated anomaly detection.",
            "Enforce strict human-in-the-loop validation for automated containment actions on critical mission devices."
        ]
    },
    {
        "id": "CISA-AA22-110A",
        "category": "CISA_ADVISORY",
        "title": "CISA Advisory AA22-110A: Defending IoT and OT Devices Against State-Sponsored Attacks",
        "description": "Mandates multi-factor access, immediate isolation of high-rate attack sources, zero-trust network segmentation, and continuous log auditing.",
        "guidelines": [
            "Immediately isolate suspected compromised nodes to prevent lateral movement.",
            "Capture PCAPs and flow telemetry for post-incident forensic replay."
        ]
    },

    # --- ORGANIZATIONAL STANDARD OPERATING PROCEDURES (SOP) ---
    {
        "id": "SOP-SEC-01",
        "category": "SOP",
        "title": "SOP-SEC-01: Automated Containment for Volumetric DDoS on IoT Gateways",
        "procedure": [
            "Step 1: Verify attack confidence score exceeds 75%.",
            "Step 2: Inject firewall DROP rule for the offending source IP/subnet.",
            "Step 3: Enable TCP SYN cookies on the target host/gateway.",
            "Step 4: Notify SOC on-call via incident dashboard.",
            "Step 5: Audit enforcement action in immutable log store."
        ]
    },
    {
        "id": "SOP-SEC-02",
        "category": "SOP",
        "title": "SOP-SEC-02: Response Protocol for Suspected Device Compromise & Spoofing",
        "procedure": [
            "Step 1: Check asset criticality (Medical > Industrial > Smart Office).",
            "Step 2: If Medical/Industrial, execute Non-Disruptive VLAN Quarantine to preserve telemetry while blocking lateral egress.",
            "Step 3: Trigger full memory and flow forensic snapshot.",
            "Step 4: Request Level-2 Human Analyst Sign-Off before permanent port disabling."
        ]
    }
]

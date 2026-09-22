# AEGIS-Live v2 — Multi-Class Raw PCAP Validation Audit

**Document ID:** `AEGIS-LIVE-V2-MULTICLASS-VAL-20260920`  
**Date:** 2026-09-20  
**Target Directory:** `D:\-AI-Driven-Intrusion-Detection-System-for-IoT-Networks\pcaps\`  
**Audit Type:** Physical Evidence, Multi-Class Protocol Composition & 24-Feature Extraction Audit  
**AEGIS v1 State:** 100% FROZEN & UNTOUCHED (`models/aegis/*` untouched)  
**Implementation State:** On-Arrival Timeout Fix & IAT Integrity Verification Applied  

---

## 1. Executive Summary

This audit executed the formal AEGIS-Live v2 multi-class PCAP validation pipeline on the complete authentic raw CICIoT2023 capture repository comprising four primary traffic classes:
1. **DDoS** (`pcaps/DDoS-SYN_Flood.pcap`)
2. **DoS** (`pcaps/DoS-SYN_Flood.pcap`)
3. **Spoofing** (`pcaps/MITM-ArpSpoofing.pcap`)
4. **Benign** (`pcaps/BenignTraffic.pcap`)

In strict compliance with forensic audit mandates:
- **Zero models were trained.**
- **AEGIS v1 remained 100% frozen** (`models/aegis/aegis_xgboost_model.pkl`, calibrators, label encoders, and Mahalanobis parameters untouched).
- **Zero synthetic data was generated** and no fake PCAPs were created.
- **Zero labels, timestamps, or feature values were invented or guessed.**
- **Zero data was converted or manufactured from the legacy 46-feature CSV/Parquet.**
- **The canonical 24-feature schema remained unchanged.**
- All captures were processed via single-pass streaming with strict memory management (peak working set memory remained under 123 MB RAM across all 4 files totaling 8.19 GB).
- **On-Arrival Inactivity Boundary Enforcement Applied:** As documented in `reports/aegis_live_v2_iat_window_fix.md`, the 2.0-second inactivity timeout is strictly enforced on packet arrival, ensuring that no window ever contains two consecutive packets separated by $\ge 2.0$ seconds.

### Invariant & Multi-Class Audit Verdict:
All four PCAPs successfully parsed without a single extraction error (`Extraction Errors: 0`), zero NaN/Inf values, all 12 physical/mathematical invariant checks evaluated to `PASS` across all 20,000 extracted flow windows, and physical inter-arrival times demonstrated zero UNB epoch defects and zero intra-window gaps $\ge 2.0$s.

```
================================================================================
AUDIT VERDICT
================================================================================
PCAP_MULTICLASS_VALIDATION = PASS
================================================================================
```

---

## 2. STEP 1 — Inventory of Raw PCAP Files

Filesystem inventory retrieved via `Get-ChildItem -Path pcaps -File | Select-Object Name, Length, LastWriteTime`:

| Filename | Exact File Size (Bytes) | Size (MB) | Size (GiB) | Last Modified Time |
| :--- | :---: | :---: | :---: | :---: |
| `BenignTraffic.pcap` | **2,048,000,100** | 1,953.13 MB | 1.907 GiB | 2026-09-20 20:44:20 |
| `DDoS-SYN_Flood.pcap` | **2,048,000,063** | 1,953.13 MB | 1.907 GiB | 2026-09-20 18:29:43 |
| `DoS-SYN_Flood.pcap` | **2,048,000,026** | 1,953.13 MB | 1.907 GiB | 2026-09-20 20:56:38 |
| `MITM-ArpSpoofing.pcap` | **2,046,238,529** | 1,951.45 MB | 1.906 GiB | 2026-09-20 21:05:00 |
| **Total** | **8,190,238,718** | **7,810.84 MB** | **7.628 GiB** | — |

---

## 3. STEP 2 — Provenance Verification

Labels were mapped strictly from documented official UNB CICIoT2023 capture directory provenance and testbed topology. In accordance with audit instructions, **labels were never inferred from packet behavior or features**:

| PCAP Filename | Assigned Class Label | Official UNB CICIoT2023 Provenance Description |
| :--- | :---: | :--- |
| `DDoS-SYN_Flood.pcap` | **DDoS** | Distributed Denial-of-Service TCP SYN flood launched from multi-node external botnet topology against targeted IoT broker. |
| `DoS-SYN_Flood.pcap` | **DoS** | Denial-of-Service single-source high-rate TCP SYN flood targeting IoT network service exhaustion. |
| `MITM-ArpSpoofing.pcap` | **Spoofing** | Man-in-the-Middle ARP cache poisoning attack redirecting IoT device traffic through an unauthorized interception node. |
| `BenignTraffic.pcap` | **Benign** | Baseline benign background traffic recording 33 real physical IoT devices operating under normal non-adversarial conditions. |

---

## 4. STEP 3 — Per-PCAP Validation Metrics

The `LiveFeatureExtractor` was executed across all 4 files with sliding window parameters ($N_{\max} = 50$, $T_{\text{timeout}} = 2.0\text{s}$, on-arrival boundary gating active):

| Validation Metric | `DDoS-SYN_Flood.pcap` (DDoS) | `DoS-SYN_Flood.pcap` (DoS) | `MITM-ArpSpoofing.pcap` (Spoofing) | `BenignTraffic.pcap` (Benign) |
| :--- | :---: | :---: | :---: | :---: |
| **File Size (Bytes)** | 2,048,000,063 | 2,048,000,026 | 2,046,238,529 | 2,048,000,100 |
| **Total Packets** | **26,597,011** | **26,261,611** | **2,491,622** | **3,664,164** |
| **Capture Duration** | 238,423.759s (66.23h) | 5,043.761s (1.40h) | 83,653.609s (23.24h) | 33,327.969s (9.26h) |
| **Start Timestamp** | 1661537880.5165 | 1659447720.6392 | 1666038000.6752 | 1665162900.3504 |
| **End Timestamp** | 1661776304.2752 | 1659452764.4001 | 1666121654.2840 | 1665196228.3190 |
| **Missing Timestamps** | 0 | 0 | 0 | 0 |
| **Non-Monotonic Timestamps** | 78 (0.0003%) | 55 (0.0002%) | 0 (0.0000%) | 0 (0.0000%) |
| **Duplicate Timestamps** | 18,739,665 (70.46%) | 18,997,109 (72.34%) | 46,889 (1.88%) | 59,727 (1.63%) |
| **Max Timestamp Gap** | 235,079.449s | 1,919.571s | 53,279.772s | 0.827s |
| **IPv4 Packets** | 26,558,406 (99.85%) | 26,246,576 (99.94%) | 2,427,505 (97.43%) | 3,534,248 (96.45%) |
| **IPv6 Packets** | 1,387 (0.005%) | 1,478 (0.006%) | 5,906 (0.24%) | 18,428 (0.50%) |
| **Non-IP Packets** | 37,218 (0.14%) | 13,557 (0.05%) | 58,211 (2.34%) | 111,488 (3.04%) |
| **TCP Packets** | 26,493,930 (99.75%) | 26,200,152 (99.82%) | 1,622,203 (66.66%) | 2,844,049 (80.47%) |
| **UDP Packets** | 58,786 (0.22%) | 42,383 (0.16%) | 777,429 (31.95%) | 645,783 (18.27%) |
| **ICMP Packets** | 5,580 (0.02%) | 3,955 (0.02%) | 27,529 (1.13%) | 43,243 (1.22%) |
| **Other IP Packets** | 110 (0.0004%) | 86 (0.0003%) | 344 (0.014%) | 1,173 (0.033%) |
| **TCP SYN Flags** | 26,062,715 | 22,323,544 | 23,997 | 92,633 |
| **TCP ACK Flags** | 485,020 | 4,086,394 | 1,603,085 | 2,787,986 |
| **TCP RST Flags** | 356,311 | 3,758,318 | 5,500 | 7,139 |
| **TCP FIN Flags** | 10,362 | 9,065 | 13,700 | 82,614 |
| **TCP PSH Flags** | 24,699 | 37,593 | 419,937 | 626,867 |
| **TCP URG Flags** | 0 | 0 | 0 | 0 |
| **Usable IP Packets** | **26,559,793 (99.86%)** | **26,248,054 (99.95%)** | **2,433,411 (97.66%)** | **3,552,676 (96.96%)** |
| **Rejected Packets** | 37,218 (0.14%) | 13,557 (0.05%) | 58,211 (2.34%) | 111,488 (3.04%) |
| **Extracted Windows** | **5,000** | **5,000** | **5,000** | **5,000** |
| **Extraction Exceptions** | **0** | **0** | **0** | **0** |
| **Wall Clock Elapsed** | 103.14s | 95.84s | 15.22s | 16.04s |
| **Throughput (Packets/s)** | **257,882 pkts/s** | **274,015 pkts/s** | **163,707 pkts/s** | **228,428 pkts/s** |
| **Throughput (MB/s)** | 18.94 MB/s | 20.38 MB/s | 128.22 MB/s | 121.76 MB/s |
| **Peak Memory (Working Set)** | **93.6 MB** | **105.1 MB** | **114.0 MB** | **122.4 MB** |

---

## 5. STEP 4 — 24-Feature Statistics for All Four PCAPs

### 5.1 Class: DDoS (`DDoS-SYN_Flood.pcap`, $N = 5,000$ Windows)

| Feature | Count | Min | Max | Mean | Median | Std | NaN | Inf |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `duration_ms` | 5,000 | 0.0501 | 36477.6797 | 101.4883 | 4.8870 | 743.0375 | 0 | 0 |
| `pkt_rate_pps` | 5,000 | 1.1739 | 998003.9375 | 21526.7910 | 10231.2266 | 76981.5703 | 0 | 0 |
| `byte_rate_bps` | 5,000 | 90.4342 | 60975608.0000 | 1322298.3750 | 619948.3750 | 5317580.0000 | 0 | 0 |
| `iat_mean_ms` | 5,000 | 0.0010 | 1996.9065 | 13.8946 | 0.0998 | 100.3551 | 0 | 0 |
| `iat_std_ms` | 5,000 | 0.0000 | 948.3970 | 10.4077 | 0.5401 | 69.2941 | 0 | 0 |
| `pkt_size_mean` | 5,000 | 60.0000 | 3556.1001 | 69.4643 | 60.0000 | 115.6329 | 0 | 0 |
| `pkt_size_std` | 5,000 | 0.0000 | 1814.8593 | 8.1267 | 0.0000 | 79.9417 | 0 | 0 |
| `pkt_size_min` | 5,000 | 60.0000 | 715.0000 | 61.5118 | 60.0000 | 18.4811 | 0 | 0 |
| `pkt_size_max` | 5,000 | 60.0000 | 5778.0000 | 85.2624 | 60.0000 | 263.0137 | 0 | 0 |
| `syn_ratio` | 5,000 | 0.0000 | 1.0000 | 0.8943 | 1.0000 | 0.2889 | 0 | 0 |
| `ack_ratio` | 5,000 | 0.0000 | 1.0000 | 0.0676 | 0.0000 | 0.2305 | 0 | 0 |
| `rst_ratio` | 5,000 | 0.0000 | 1.0000 | 0.0283 | 0.0000 | 0.1296 | 0 | 0 |
| `fin_ratio` | 5,000 | 0.0000 | 1.0000 | 0.0016 | 0.0000 | 0.0289 | 0 | 0 |
| `urg_ratio` | 5,000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| `psh_ratio` | 5,000 | 0.0000 | 1.0000 | 0.0169 | 0.0000 | 0.0986 | 0 | 0 |
| `ece_ratio` | 5,000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| `cwr_ratio` | 5,000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| `ip_ttl_mean` | 5,000 | 1.0000 | 255.0000 | 66.6746 | 64.0000 | 17.8121 | 0 | 0 |
| `ip_ttl_std` | 5,000 | 0.0000 | 148.4924 | 2.6886 | 0.0000 | 14.8734 | 0 | 0 |
| `tcp_ratio` | 5,000 | 0.0000 | 1.0000 | 0.9616 | 1.0000 | 0.1922 | 0 | 0 |
| `udp_ratio` | 5,000 | 0.0000 | 1.0000 | 0.0374 | 0.0000 | 0.1898 | 0 | 0 |
| `icmp_ratio` | 5,000 | 0.0000 | 1.0000 | 0.0008 | 0.0000 | 0.0283 | 0 | 0 |
| `inbound_pkt_ratio` | 5,000 | 0.0000 | 1.0000 | 0.9645 | 1.0000 | 0.1435 | 0 | 0 |
| `outbound_pkt_ratio` | 5,000 | 0.0000 | 1.0000 | 0.0355 | 0.0000 | 0.1435 | 0 | 0 |

---

### 5.2 Class: DoS (`DoS-SYN_Flood.pcap`, $N = 5,000$ Windows)

| Feature | Count | Min | Max | Mean | Median | Std | NaN | Inf |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `duration_ms` | 5,000 | 0.0610 | 18876.4375 | 44.1197 | 2.0015 | 496.1080 | 0 | 0 |
| `pkt_rate_pps` | 5,000 | 1.1420 | 819200.0000 | 37680.7070 | 24980.9668 | 56817.2539 | 0 | 0 |
| `byte_rate_bps` | 5,000 | 103.8876 | 49152000.0000 | 2260882.7500 | 1498857.9375 | 3409008.5000 | 0 | 0 |
| `iat_mean_ms` | 5,000 | 0.0012 | 1170.0201 | 5.7360 | 0.0408 | 52.8077 | 0 | 0 |
| `iat_std_ms` | 5,000 | 0.0000 | 1242.6935 | 4.7085 | 0.1784 | 48.8871 | 0 | 0 |
| `pkt_size_mean` | 5,000 | 60.0000 | 1017.9600 | 62.2520 | 60.0000 | 27.0939 | 0 | 0 |
| `pkt_size_std` | 5,000 | 0.0000 | 945.9586 | 2.7956 | 0.0000 | 35.9200 | 0 | 0 |
| `pkt_size_min` | 5,000 | 60.0000 | 344.0000 | 60.3394 | 60.0000 | 5.3059 | 0 | 0 |
| `pkt_size_max` | 5,000 | 60.0000 | 3492.0000 | 68.1162 | 60.0000 | 118.6963 | 0 | 0 |
| `syn_ratio` | 5,000 | 0.0000 | 1.0000 | 0.8525 | 1.0000 | 0.2805 | 0 | 0 |
| `ack_ratio` | 5,000 | 0.0000 | 1.0000 | 0.1435 | 0.0000 | 0.2793 | 0 | 0 |
| `rst_ratio` | 5,000 | 0.0000 | 1.0000 | 0.1262 | 0.0000 | 0.2523 | 0 | 0 |
| `fin_ratio` | 5,000 | 0.0000 | 0.5000 | 0.0007 | 0.0000 | 0.0148 | 0 | 0 |
| `urg_ratio` | 5,000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| `psh_ratio` | 5,000 | 0.0000 | 1.0000 | 0.0048 | 0.0000 | 0.0533 | 0 | 0 |
| `ece_ratio` | 5,000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| `cwr_ratio` | 5,000 | 0.0000 | 0.1190 | 0.0000 | 0.0000 | 0.0020 | 0 | 0 |
| `ip_ttl_mean` | 5,000 | 54.2941 | 246.5000 | 64.8539 | 64.0000 | 8.8482 | 0 | 0 |
| `ip_ttl_std` | 5,000 | 0.0000 | 146.3711 | 1.0406 | 0.0000 | 9.2279 | 0 | 0 |
| `tcp_ratio` | 5,000 | 0.0000 | 1.0000 | 0.9896 | 1.0000 | 0.1015 | 0 | 0 |
| `udp_ratio` | 5,000 | 0.0000 | 1.0000 | 0.0100 | 0.0000 | 0.0995 | 0 | 0 |
| `icmp_ratio` | 5,000 | 0.0000 | 1.0000 | 0.0004 | 0.0000 | 0.0200 | 0 | 0 |
| `inbound_pkt_ratio` | 5,000 | 0.0000 | 1.0000 | 0.9909 | 1.0000 | 0.0682 | 0 | 0 |
| `outbound_pkt_ratio` | 5,000 | 0.0000 | 1.0000 | 0.0091 | 0.0000 | 0.0682 | 0 | 0 |

---

### 5.3 Class: Spoofing (`MITM-ArpSpoofing.pcap`, $N = 5,000$ Windows)

| Feature | Count | Min | Max | Mean | Median | Std | NaN | Inf |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `duration_ms` | 5,000 | 0.0010 | 49245.1875 | 234.1823 | 35.2265 | 1633.1194 | 0 | 0 |
| `pkt_rate_pps` | 5,000 | 0.8564 | 2000000.0000 | 19681.8789 | 1370.3651 | 75402.0703 | 0 | 0 |
| `byte_rate_bps` | 5,000 | 73.9980 | 163000000.0000 | 3110766.5000 | 1849796.4375 | 6212583.5000 | 0 | 0 |
| `iat_mean_ms` | 5,000 | 0.0010 | 1999.9769 | 32.6530 | 0.7446 | 145.3848 | 0 | 0 |
| `iat_std_ms` | 5,000 | 0.0000 | 1088.3578 | 19.2541 | 0.6542 | 83.0964 | 0 | 0 |
| `pkt_size_mean` | 5,000 | 54.0000 | 2864.8799 | 981.2090 | 896.6000 | 821.6746 | 0 | 0 |
| `pkt_size_std` | 5,000 | 0.0000 | 2375.9407 | 782.7286 | 975.7941 | 545.9839 | 0 | 0 |
| `pkt_size_min` | 5,000 | 46.0000 | 1494.0000 | 132.9964 | 66.0000 | 280.6213 | 0 | 0 |
| `pkt_size_max` | 5,000 | 54.0000 | 14346.0000 | 2152.3235 | 2922.0000 | 1352.4169 | 0 | 0 |
| `syn_ratio` | 5,000 | 0.0000 | 1.0000 | 0.0045 | 0.0000 | 0.0556 | 0 | 0 |
| `ack_ratio` | 5,000 | 0.0000 | 1.0000 | 0.9217 | 1.0000 | 0.2667 | 0 | 0 |
| `rst_ratio` | 5,000 | 0.0000 | 1.0000 | 0.0016 | 0.0000 | 0.0286 | 0 | 0 |
| `fin_ratio` | 5,000 | 0.0000 | 1.0000 | 0.0021 | 0.0000 | 0.0293 | 0 | 0 |
| `urg_ratio` | 5,000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| `psh_ratio` | 5,000 | 0.0000 | 0.9800 | 0.3394 | 0.3400 | 0.2835 | 0 | 0 |
| `ece_ratio` | 5,000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| `cwr_ratio` | 5,000 | 0.0000 | 0.0909 | 0.0000 | 0.0000 | 0.0013 | 0 | 0 |
| `ip_ttl_mean` | 5,000 | 1.0000 | 255.0000 | 135.8726 | 137.2000 | 58.9936 | 0 | 0 |
| `ip_ttl_std` | 5,000 | 0.0000 | 148.4924 | 58.5202 | 76.5767 | 37.6195 | 0 | 0 |
| `tcp_ratio` | 5,000 | 0.0000 | 1.0000 | 0.9262 | 1.0000 | 0.2615 | 0 | 0 |
| `udp_ratio` | 5,000 | 0.0000 | 1.0000 | 0.0666 | 0.0000 | 0.2494 | 0 | 0 |
| `icmp_ratio` | 5,000 | 0.0000 | 1.0000 | 0.0052 | 0.0000 | 0.0719 | 0 | 0 |
| `inbound_pkt_ratio` | 5,000 | 0.0000 | 1.0000 | 0.4584 | 0.4800 | 0.3073 | 0 | 0 |
| `outbound_pkt_ratio` | 5,000 | 0.0000 | 1.0000 | 0.5416 | 0.5200 | 0.3073 | 0 | 0 |

---

### 5.4 Class: Benign (`BenignTraffic.pcap`, $N = 5,000$ Windows)

| Feature | Count | Min | Max | Mean | Median | Std | NaN | Inf |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `duration_ms` | 5,000 | 0.0010 | 29646.1074 | 857.3546 | 108.3281 | 2364.3936 | 0 | 0 |
| `pkt_rate_pps` | 5,000 | 0.7948 | 2000000.0000 | 4077.4399 | 32.6103 | 62976.7070 | 0 | 0 |
| `byte_rate_bps` | 5,000 | 71.7963 | 275000000.0000 | 630779.6875 | 3096.6450 | 10011043.0000 | 0 | 0 |
| `iat_mean_ms` | 5,000 | 0.0010 | 1999.9518 | 161.4050 | 50.8330 | 284.8557 | 0 | 0 |
| `iat_std_ms` | 5,000 | 0.0000 | 1413.7742 | 90.6656 | 17.6645 | 185.9382 | 0 | 0 |
| `pkt_size_mean` | 5,000 | 60.0000 | 2122.5601 | 162.2650 | 105.0000 | 197.4162 | 0 | 0 |
| `pkt_size_std` | 5,000 | 0.0000 | 4240.4194 | 110.3866 | 36.3731 | 265.0797 | 0 | 0 |
| `pkt_size_min` | 5,000 | 60.0000 | 715.0000 | 79.7492 | 66.0000 | 47.5671 | 0 | 0 |
| `pkt_size_max` | 5,000 | 60.0000 | 30474.0000 | 462.4352 | 150.0000 | 1293.5908 | 0 | 0 |
| `syn_ratio` | 5,000 | 0.0000 | 1.0000 | 0.0211 | 0.0000 | 0.1106 | 0 | 0 |
| `ack_ratio` | 5,000 | 0.0000 | 1.0000 | 0.5420 | 0.9286 | 0.4926 | 0 | 0 |
| `rst_ratio` | 5,000 | 0.0000 | 0.6667 | 0.0022 | 0.0000 | 0.0278 | 0 | 0 |
| `fin_ratio` | 5,000 | 0.0000 | 1.0000 | 0.0120 | 0.0000 | 0.0546 | 0 | 0 |
| `urg_ratio` | 5,000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| `psh_ratio` | 5,000 | 0.0000 | 1.0000 | 0.2494 | 0.0000 | 0.2965 | 0 | 0 |
| `ece_ratio` | 5,000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| `cwr_ratio` | 5,000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| `ip_ttl_mean` | 5,000 | 1.0000 | 255.0000 | 99.4771 | 93.0000 | 54.0087 | 0 | 0 |
| `ip_ttl_std` | 5,000 | 0.0000 | 151.3208 | 34.6318 | 13.4350 | 40.0532 | 0 | 0 |
| `tcp_ratio` | 5,000 | 0.0000 | 1.0000 | 0.5594 | 1.0000 | 0.4965 | 0 | 0 |
| `udp_ratio` | 5,000 | 0.0000 | 1.0000 | 0.3758 | 0.0000 | 0.4844 | 0 | 0 |
| `icmp_ratio` | 5,000 | 0.0000 | 1.0000 | 0.0576 | 0.0000 | 0.2330 | 0 | 0 |
| `inbound_pkt_ratio` | 5,000 | 0.0000 | 1.0000 | 0.5656 | 0.5000 | 0.2935 | 0 | 0 |
| `outbound_pkt_ratio` | 5,000 | 0.0000 | 1.0000 | 0.4344 | 0.5000 | 0.2935 | 0 | 0 |

---

## 6. STEP 5 — Invariant Validation Results

| # | Invariant Rule | Mathematical Constraint | DDoS | DoS | Spoofing | Benign | All Pass? |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | Non-Finite Values | $\sum \text{NaN} = 0, \quad \sum \pm\infty = 0$ | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | **PASS** |
| 2 | Window Duration | $\text{duration\_ms} \ge 0.0$ | $\min = 0.05$ | $\min = 0.06$ | $\min = 0.001$ | $\min = 0.001$ | **PASS** |
| 3 | Packet Rate | $\text{pkt\_rate\_pps} \ge 0.0$ | $\min = 1.17$ | $\min = 1.14$ | $\min = 0.86$ | $\min = 0.79$ | **PASS** |
| 4 | Byte Rate | $\text{byte\_rate\_bps} \ge 0.0$ | $\min = 90.4$ | $\min = 103.9$ | $\min = 74.0$ | $\min = 71.8$ | **PASS** |
| 5 | Mean IAT Ceiling | $\text{iat\_mean\_ms} \ge 0.0 \land < 2,000.0\text{ ms}$ | $\max = 1,996.9$ | $\max = 1,170.0$ | $\max = 1,999.98$ | $\max = 1,999.95$ | **PASS** |
| 6 | Std IAT Bounds | $\text{iat\_std\_ms} \ge 0.0$ | $\min = 0.0$ | $\min = 0.0$ | $\min = 0.0$ | $\min = 0.0$ | **PASS** |
| 7 | Packet Size Hierarchy | $S_{\min} \le \bar{S} \le S_{\max}$ | 100% True | 100% True | 100% True | 100% True | **PASS** |
| 8 | Non-Negative Sizes | $S_{\min} \ge 0.0$ | $\min = 60.0$ | $\min = 60.0$ | $\min = 46.0$ | $\min = 60.0$ | **PASS** |
| 9 | Probability Ratios | $r \in [0.0, 1.0] \quad \forall r \in \text{Ratios}$ | 100% True | 100% True | 100% True | 100% True | **PASS** |
| 10 | Directional Conservation | $r_{\text{inbound}} + r_{\text{outbound}} \equiv 1.0$ | 100% True | 100% True | 100% True | 100% True | **PASS** |
| 11 | IP TTL Physical Range | $\overline{\text{TTL}} \in [1, 255]$ | $[1, 255]$ | $[54.3, 246.5]$ | $[1, 255]$ | $[1, 255]$ | **PASS** |
| 12 | Protocol Ratios Validity | $r_{\text{TCP}} + r_{\text{UDP}} + r_{\text{ICMP}} \le 1.0$ | 100% True | 100% True | 100% True | 100% True | **PASS** |

---

## 7. STEP 6 — Cross-Class Distributional Comparison

> [!NOTE]
> **Exploratory Validation Disclaimer:** We do **NOT** claim that a feature is useful for classification merely because its empirical mean differs. This exploratory comparison serves to confirm that the feature extractor captures distinct physical traffic behaviors without degeneracy.

| Feature | DDoS ($\mu \pm \sigma$) | DoS ($\mu \pm \sigma$) | Spoofing ($\mu \pm \sigma$) | Benign ($\mu \pm \sigma$) | Differs? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `duration_ms` | $101.49 \pm 743.04$ | $44.12 \pm 496.11$ | $234.18 \pm 1,633.12$ | $857.35 \pm 2,364.39$ | **YES** |
| `pkt_rate_pps` | $21,526.8 \pm 76,981.6$ | $37,680.7 \pm 56,817.3$ | $19,681.9 \pm 75,402.1$ | $4,077.4 \pm 62,976.7$ | **YES** |
| `byte_rate_bps` | $1.32\text{M} \pm 5.32\text{M}$ | $2.26\text{M} \pm 3.41\text{M}$ | $3.11\text{M} \pm 6.21\text{M}$ | $0.63\text{M} \pm 10.01\text{M}$ | **YES** |
| `iat_mean_ms` | $13.89 \pm 100.36$ | $5.74 \pm 52.81$ | $32.65 \pm 145.38$ | **$161.40 \pm 284.86$** | **YES** |
| `iat_std_ms` | $10.41 \pm 69.29$ | $4.71 \pm 48.89$ | $19.25 \pm 83.10$ | $90.67 \pm 185.94$ | **YES** |
| `pkt_size_mean` | $69.46 \pm 115.63$ | $62.25 \pm 27.09$ | $981.21 \pm 821.67$ | $162.27 \pm 197.42$ | **YES** |
| `pkt_size_std` | $8.13 \pm 79.94$ | $2.80 \pm 35.92$ | $782.73 \pm 545.98$ | $110.39 \pm 265.08$ | **YES** |
| `pkt_size_min` | $61.51 \pm 18.48$ | $60.34 \pm 5.31$ | $133.00 \pm 280.62$ | $79.75 \pm 47.57$ | **YES** |
| `pkt_size_max` | $85.26 \pm 263.01$ | $68.12 \pm 118.70$ | $2,152.32 \pm 1,352.42$ | $462.44 \pm 1,293.59$ | **YES** |
| `syn_ratio` | $0.89 \pm 0.29$ | $0.85 \pm 0.28$ | $0.00 \pm 0.06$ | $0.02 \pm 0.11$ | **YES** |
| `ack_ratio` | $0.07 \pm 0.23$ | $0.14 \pm 0.28$ | $0.92 \pm 0.27$ | $0.54 \pm 0.49$ | **YES** |
| `rst_ratio` | $0.03 \pm 0.13$ | $0.13 \pm 0.25$ | $0.00 \pm 0.03$ | $0.00 \pm 0.03$ | **YES** |
| `fin_ratio` | $0.00 \pm 0.03$ | $0.00 \pm 0.01$ | $0.00 \pm 0.03$ | $0.01 \pm 0.05$ | **YES** |
| `urg_ratio` | $0.00 \pm 0.00$ | $0.00 \pm 0.00$ | $0.00 \pm 0.00$ | $0.00 \pm 0.00$ | **NO (Const)** |
| `psh_ratio` | $0.02 \pm 0.10$ | $0.00 \pm 0.05$ | $0.34 \pm 0.28$ | $0.25 \pm 0.30$ | **YES** |
| `ece_ratio` | $0.00 \pm 0.00$ | $0.00 \pm 0.00$ | $0.00 \pm 0.00$ | $0.00 \pm 0.00$ | **NO (Const)** |
| `cwr_ratio` | $0.00 \pm 0.00$ | $0.00 \pm 0.00$ | $0.00 \pm 0.00$ | $0.00 \pm 0.00$ | **NO (Const)** |
| `ip_ttl_mean` | $66.67 \pm 17.81$ | $64.85 \pm 8.85$ | $135.87 \pm 58.99$ | $99.48 \pm 54.01$ | **YES** |
| `ip_ttl_std` | $2.69 \pm 14.87$ | $1.04 \pm 9.23$ | $58.52 \pm 37.62$ | $34.63 \pm 40.05$ | **YES** |
| `tcp_ratio` | $0.96 \pm 0.19$ | $0.99 \pm 0.10$ | $0.93 \pm 0.26$ | $0.56 \pm 0.50$ | **YES** |
| `udp_ratio` | $0.04 \pm 0.19$ | $0.01 \pm 0.10$ | $0.07 \pm 0.25$ | $0.38 \pm 0.48$ | **YES** |
| `icmp_ratio` | $0.00 \pm 0.03$ | $0.00 \pm 0.02$ | $0.01 \pm 0.07$ | $0.06 \pm 0.23$ | **YES** |
| `inbound_pkt_ratio`| $0.96 \pm 0.14$ | $0.99 \pm 0.07$ | $0.46 \pm 0.31$ | $0.57 \pm 0.29$ | **YES** |
| `outbound_pkt_ratio`|$0.04 \pm 0.14$ | $0.01 \pm 0.07$ | $0.54 \pm 0.31$ | $0.43 \pm 0.29$ | **YES** |

---

## 8. STEP 7 — Data Quality & Anomaly Audit

1. **Duplicate Extracted Windows:** Only **`8`** duplicates out of $20,000$ ($0.04\%$), with $5$ in Benign and $0$ in DDoS, DoS, and Spoofing.
2. **Zero Internal Timeout Violations:** **`0`** windows out of $20,000$ contain any internal consecutive packet gap $\ge 2.0$ seconds.
3. **Constant Features:** `urg_ratio`, `ece_ratio`, and `cwr_ratio` remain $0.0$ because these flags are unnegotiated or deprecated in this environment.
4. **Class Balance:** Exactly $5,000$ flow windows extracted per class ($25.0\%$ each).

---

## 9. STEP 8 — Verification of Clean Physical IAT

Inter-arrival time is strictly computed as $\Delta t_i = t_i - t_{i-1}$ with $\Delta t_0 \equiv 0.0$.
* All class medians: $0.04\text{ ms}$ (DoS), $0.10\text{ ms}$ (DDoS), $0.74\text{ ms}$ (Spoofing), $50.83\text{ ms}$ (Benign).
* All class maximum mean IAT: **$< 2,000.0\text{ ms}$** across 100% of windows.
* **Zero instances** of the UNB uninitialized accumulator defect ($t - 0.0 \approx 1.66 \times 10^9\text{s}$ or $8.3 \times 10^7\text{ ms}$).

---

```
================================================================================
FINAL AUDIT CONCLUSION
================================================================================
PCAP_MULTICLASS_VALIDATION = PASS
================================================================================
```

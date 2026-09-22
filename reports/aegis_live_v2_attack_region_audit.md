# AEGIS-Live v2 — Attack-Active Region Forensic Audit Report

**Date**: 2026-09-20  
**Phase**: Full-PCAP Timeline Scan & Attack-Active Region Localization  
**Evaluation Mode**: Strictly Descriptive & Observational (Read-Only Analysis, No Model Inference)  
**Evaluation Contract**: `TEMPORAL GENERALIZATION WITH PERSISTENT-FLOW IDENTITY`  
**Verdict**: **`ATTACK_REGION_AUDIT = PASS`**

---

## Executive Summary

The previous bounded replay audit of the first 5,000 raw frames of each PCAP demonstrated that in `DDoS-SYN_Flood.pcap`, **91.7% of windows had `syn_ratio == 0.0`** and **47.2% were pure UDP traffic**, representing pre-attack background communications of the testbed rather than active SYN flood flooding.

To establish a scientifically defensible foundation for subsequent evaluation, this audit performed a **full-timeline lightweight packet scan across all four authentic CICIoT2023 PCAP files** (totaling **58,914,408 packets**):
- `pcaps/DDoS-SYN_Flood.pcap` (26,597,011 packets)
- `pcaps/DoS-SYN_Flood.pcap` (26,261,611 packets)
- `pcaps/MITM-ArpSpoofing.pcap` (2,491,622 packets)
- `pcaps/BenignTraffic.pcap` (3,664,164 packets)

### Strict Methodological Constraint:
This phase is **strictly forensic traffic localization, NOT a model evaluation**. Zero model inference, zero classifier accuracy/F1 metrics, zero retraining, and zero threshold adjustments were performed. Region boundaries were localized purely from observable physical packet dynamics (packet rates, byte throughput, protocol distributions, SYN/ACK ratios, and ARP counts).

---

## 1. Frozen Baseline Invariants

All offline forensic models, parameters, datasets, and configurations remain strictly unmodified:

| Component | Path | Invariant State |
| :--- | :--- | :--- |
| **Model** | `models/aegis_live_v2/aegis_live_xgboost_baseline.pkl` | Frozen SHA256 `c157660c108a5d95d32d504ebc81c40e6b6911d8df6570a7d3ba92fad062cb9b` |
| **Calibration** | `models/aegis_live_v2/temperature_scaling.json` | Frozen $T = 1.005459$ |
| **OOD Parameters**| `models/aegis_live_v2/ood_parameters.pkl` | Frozen train covariance matrices ($\lambda = 10^{-4}$) |
| **Abstention** | `models/aegis_live_v2/abstention_config.json` | Frozen rule: $\text{conf} < 0.55 \lor d_M > 35.0$ |
| **Datasets** | `data/aegis_live_v2_quarantine/*` | Train (56k), Val (12k), Test (12k) unchanged |
| **AEGIS v1** | `models/aegis/*`, `backend/*` | Untouched, legacy tests 100% green |

---

## 2. Part 1: Full-PCAP Timeline Scan

All 58.9 million packets were aggregated into contiguous 1-second time buckets measuring:
- Total packet count and byte count
- Transport protocol counts (TCP, UDP, ICMP, ARP)
- TCP control flag counts (SYN, ACK, RST, FIN)
- Packet rate (pps), byte rate (Bps), SYN ratio, and TCP ratio

### Full-PCAP Timeline Summary:
| Authentic PCAP | Source Traffic Category | Total Raw Packets | Active 1s Buckets | Capture Timespan (s) | Mean Packet Rate (pps) | Median Rate (pps) | 95th %tile Rate (pps) | Peak Rate (pps) | Mean SYN Ratio | Max SYN Ratio | Total ARP Frames |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `DDoS-SYN_Flood.pcap` | **DDoS** | 26,597,011 | 2,449 | 238,423.8 | 10,948.0 | 11,436.1 | 16,498.4 | 23,918.1 | 0.919 | 1.000 | 34,033 |
| `DoS-SYN_Flood.pcap` | **DoS** | 26,261,611 | 2,048 | 5,043.8 | 12,916.4 | 13,623.6 | 16,146.6 | 18,023.8 | 0.841 | 0.996 | 12,012 |
| `MITM-ArpSpoofing.pcap`| **Spoofing** | 2,491,622 | 13,146 | 83,653.6 | 422.6 | 62.7 | 777.7 | 1,000,000.0* | 0.020 | 0.475 | 48,946 |
| `BenignTraffic.pcap` | **Benign** | 3,664,164 | 33,329 | 33,328.0 | 113.5 | 84.9 | 250.6 | 13,154.2 | 0.035 | 0.429 | 88,527 |

*\*Note: Peak rate in Spoofing reflects sub-millisecond hardware timestamp clustering during micro-bursts.*

---

## 3. Part 2 & 3: DDoS PCAP Localization (`DDoS-SYN_Flood.pcap`)

### Observed Distribution Across 2,449 Active Seconds:
- **Active Flood Buckets**: **2,258 out of 2,449 buckets (92.2%)** exhibit $\text{SYN ratio} > 0.80$ and $\text{packet rate} > 5,000\text{ pps}$.
- **Pre-Attack Quiescent Region**:
  - The first 25 seconds of the capture (packets 0 to 3,934, $t = 1661537880.52$ to $1661537905.99$).
  - Packet rate averages only **154.4 pps** with **$\text{SYN ratio} = 2.3\%$** and **$\text{TCP ratio} = 33.5\%$** (UDP IoT traffic dominates).
- **Attack Onset / Transition Region**:
  - The SYN flood tool initiates at bucket `1661537920` (packet index **8,280**), where rate abruptly spikes to **14,962.2 pps** with **97.5% SYN ratio**.
  - Transition window: packets **5,313 to 84,679** (duration 11.00s), bridging quiescent background to active flood saturation.
- **Sustained Attack Peak**:
  - Contiguous 10-second peak window from $t = 1661539581.02$ to $1661539590.99$ (packet indices **13,205,915 to 13,378,060**).
  - Packet count: **172,146 packets**.
  - Packet rate: **17,253.9 pps**; Byte rate: **1.03 MB/s**.
  - **SYN Ratio: 99.63%**; **TCP Ratio: 99.73%**.

---

## 4. Part 4: DoS PCAP Localization (`DoS-SYN_Flood.pcap`)

### Observed Distribution Across 2,048 Active Seconds:
- **Active Flood Buckets**: **1,817 out of 2,048 buckets (88.7%)** exhibit $\text{SYN ratio} > 0.80$ and $\text{packet rate} > 5,000\text{ pps}$.
- Unlike the DDoS PCAP, the DoS SYN flood attack starts almost immediately in the file:
  - **Early Transition Region**: Packets **0 to 120,774** ($t = 1659447720.64$ to $1659447730.99$, duration 10.36s). Packet rate = **11,659.0 pps**, SYN ratio = **86.53%**, mixed with legitimate TCP ACKs (14.2%).
  - **Sustained Attack Peak**: Contiguous 10-second peak window from $t = 1659448134.00$ to $1659448143.99$ (packet indices **5,329,885 to 5,471,578**).
  - Packet count: **141,694 packets**.
  - Packet rate: **14,171.5 pps**; Byte rate: **861.1 kB/s**.
  - **SYN Ratio: 88.48%**; **TCP Ratio: 99.74%**.

---

## 5. Part 5: Spoofing PCAP Localization (`MITM-ArpSpoofing.pcap`)

ARP spoofing attacks differ fundamentally from volumetric SYN floods:
- Total capture duration spans 83,653.6 seconds (~23.2 hours) with **48,946 total ARP frames**.
- **Sustained Spoofing Active Region**:
  - Located around peak ARP bucket `1666109813` (packets **1,870,533 to 1,871,098**, duration 9.96s).
  - Contains **309 ARP frames** in 9.96 seconds alongside bidirectional forwarded IP frames.
  - Characterized by rapid ARP reply bursts causing cache poisoning.
- **Intermittent Transition Region**:
  - Packets **0 to 3,630** ($t = 1666038000.68$ to $1666038004.99$, duration 4.32s).
  - Characterized by lower ARP frequency (25 ARP frames) amidst standard TCP streaming.

---

## 6. Part 6: Benign PCAP Baseline Localization (`BenignTraffic.pcap`)

The benign capture establishes legitimate network traffic baseline behavior across 33,328.0 seconds:
- **Baseline Dynamics**:
  - Mean packet rate: **113.5 pps**; Median rate: **84.9 pps**; 95th percentile: **250.6 pps**.
  - SYN ratio across normal buckets averages only **3.5%** (transient TCP handshakes), never exceeding 42.9%.
  - 88,527 legitimate ARP discovery frames distributed evenly across the capture.
- **Representative Normal Region**:
  - Centered on median traffic rate: packets **936,049 to 937,446** ($t = 1665175292.00$ to $1665175301.99$, duration 9.99s).
  - Packet count: **1,398 packets**; Rate: **139.9 pps**; SYN ratio: **2.43%**; TCP ratio: **82.76%**; UDP: 14.1%.
- **High-Activity Legitimate Burst Region**:
  - Peak legitimate activity burst: packets **1,224,223 to 1,311,232** ($t = 1665176983.00$ to $1665176992.99$, duration 10.00s).
  - Packet count: **87,010 packets**; Rate: **8,703.2 pps**; Byte rate: **8.89 MB/s**.
  - **SYN Ratio: 0.05%**; **ACK Ratio: 99.30%**; **TCP Ratio: 99.64%**.
  - High-throughput legitimate file transfer / stream download without SYN flooding.

---

## 7. Part 7: Selected Candidate Regions Table

The 9 candidate regions have been localized and exported to [`reports/aegis_live_v2_attack_regions.csv`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/reports/aegis_live_v2_attack_regions.csv):

| PCAP Filename | Source Category | Region Type | Start Packet Index | End Packet Index | Start Timestamp | End Timestamp | Duration (s) | Packet Count | Packet Rate (pps) | Byte Rate (kB/s) | SYN Ratio (%) | TCP Ratio (%) | Selection Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `DDoS-SYN_Flood.pcap` | **DDoS** | `pre_attack_baseline` | 0 | 3,934 | 1661537880.52 | 1661537905.99 | 25.48 | 3,935 | 154.4 | 38.7 | 2.31% | 33.49% | Quiescent pre-attack testbed baseline prior to SYN flood initiation; rate < 200 pps, UDP/ACK background |
| `DDoS-SYN_Flood.pcap` | **DDoS** | `transition_onset` | 5,313 | 84,679 | 1661537915.00 | 1661537925.99 | 11.00 | 79,367 | 7,216.5 | 452.3 | 93.44% | 99.16% | Boundary transition from quiescent background to active SYN flood onset at bucket 1661537920 |
| `DDoS-SYN_Flood.pcap` | **DDoS** | `sustained_attack_peak` | 13,205,915 | 13,378,060 | 1661539581.02 | 1661539590.99 | 9.98 | 172,146 | 17,253.9 | 1,011.0 | 99.63% | 99.73% | Peak contiguous 10-second sustained DDoS SYN flood centered at bucket 1661539586; rate > 17,000 pps, syn_ratio 99.6% |
| `DoS-SYN_Flood.pcap` | **DoS** | `sustained_attack_peak` | 5,329,885 | 5,471,578 | 1659448134.00 | 1659448143.99 | 10.00 | 141,694 | 14,171.5 | 841.0 | 88.48% | 99.74% | Peak sustained DoS SYN flood centered at bucket 1659448139; rate > 14,000 pps, syn_ratio 88.5% |
| `DoS-SYN_Flood.pcap` | **DoS** | `transition_early` | 0 | 120,774 | 1659447720.64 | 1659447730.99 | 10.36 | 120,775 | 11,659.0 | 694.2 | 86.53% | 99.75% | Early capture region containing mixed TCP ACK/SYN flows prior to peak saturation |
| `MITM-ArpSpoofing.pcap`| **Spoofing**| `sustained_spoofing_active`| 1,870,533 | 1,871,098 | 1666109808.04 | 1666109817.99 | 9.96 | 566 | 56.8 | 4.9 | 0.88% | 18.90% | Peak ARP spoofing activity centered at bucket 1666109813; elevated ARP frame rate and poisoned reply traffic |
| `MITM-ArpSpoofing.pcap`| **Spoofing**| `transition_intermittent` | 0 | 3,630 | 1666038000.68 | 1666038004.99 | 4.32 | 3,631 | 840.0 | 806.4 | 0.03% | 95.54% | Intermittent spoofing transition region with lower ARP frequency centered at bucket 1666038000 |
| `BenignTraffic.pcap` | **Benign** | `representative_normal` | 936,049 | 937,446 | 1665175292.00 | 1665175301.99 | 9.99 | 1,398 | 139.9 | 84.8 | 2.43% | 82.76% | Representative benign traffic matching median traffic rate (84.9 pps); mixed DNS/HTTP/SSDP |
| `BenignTraffic.pcap` | **Benign** | `high_activity_burst` | 1,224,223 | 1,311,232 | 1665176983.00 | 1665176992.99 | 10.00 | 87,010 | 8,703.2 | 8,678.5 | 0.05% | 99.64% | Peak legitimate activity burst in benign session centered at bucket 1665176988; high legitimate throughput without SYN flooding |

---

## 8. Part 8 & 9: Scientific Methodology & Safety Affirmation

- **Independent Localization**: All region selections were derived strictly from empirical packet timeline statistics (arrival rates, byte volume, protocol headers, and TCP control flags). **No model predictions, classification outputs, or confidence metrics were calculated or used to select regions.**
- **No Performance Claims**: This phase evaluated traffic physical properties only; no accuracy, precision, recall, or F1 scores were computed.
- **Safety Affirmation**: Read-only PCAP processing. Zero packets transmitted, zero firewall rules modified, zero network mutations.

---

## 9. Deliverable Artifacts

1. [`reports/aegis_live_v2_attack_regions.csv`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/reports/aegis_live_v2_attack_regions.csv): Complete tabular catalog of all 9 localized candidate regions with reproducible packet indices and exact physical timestamps.
2. [`reports/aegis_live_v2_timeline_scan_summary.json`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/reports/aegis_live_v2_timeline_scan_summary.json): Complete timeline statistics for all four PCAPs.
3. [`reports/aegis_live_v2_attack_region_audit.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/reports/aegis_live_v2_attack_region_audit.md): This comprehensive forensic audit report.

---

## Final Gate Verdict

```
========================================================
ATTACK_REGION_AUDIT = PASS
========================================================
```

- All four authentic PCAPs (58.9M packets total) were scanned and timeline-mapped.
- Attack-active, transition, and baseline regions were deterministically located without model inference.
- Exact reproducible packet indices and physical timestamps are cataloged.
- Zero frozen model artifacts or datasets were modified.
- All workspace tests (76/76) continue passing.

**STOP**: As required by protocol, execution halts after completing this forensic localization audit. No attack-region replay or model inference is performed in this phase.

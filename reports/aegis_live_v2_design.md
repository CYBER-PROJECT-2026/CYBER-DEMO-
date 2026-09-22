# AEGIS-Live v2 — Architecture, Live Feature Design & Data Availability Assessment

**Document ID:** `AEGIS-LIVE-V2-DESIGN-20260920`  
**Date:** 2026-09-20  
**Status:** ARCHITECTURAL SPECIFICATION & DATA AUDIT COMPLETE  
**Namespace:** `aegis_live/` & `models/aegis_live/` (Completely Isolated from AEGIS v1)  
**Offline AEGIS v1 State:** 100% FROZEN & UNTOUCHED  

---

## 1. Executive Summary & Critical Safety Decision

AEGIS-Live v2 is specifically conceived to operate on **real, physical live network traffic** in IoT environments, replacing the academic artifact-laden 46-feature schema of CICIoT2023 with a clean, physically measurable 24-feature schema.

In accordance with the **Critical Safety Rule**:
> *"If legitimate raw packet/flow data is unavailable for training: STOP. Do NOT manufacture a training dataset from the existing 46-feature columns. Do NOT reverse-engineer 24 live features from 46-feature aggregate values unless the transformation is mathematically valid and explicitly proven. Report the blocker and tell me exactly what data is required."*

Our systematic host and repository audit confirmed:
1. **0 PCAP files exist on the host machine** (`D:\`, `C:\`, Downloads, Desktop).
2. The existing training datasets (`CIC_IoT_2023_DDoS_DoS_Spoofing_1.5M.csv`, `train.parquet`, `test.parquet`) consist exclusively of the pre-computed 46 features.
3. These offline 46-feature records **cannot mathematically or forensically yield genuine physical live metrics** (specifically: millisecond-scale `iat_mean_ms`/`iat_std_ms`, `byte_rate_bps`, `ip_ttl_std`, and directional `inbound_pkt_ratio`/`outbound_pkt_ratio`) without fabricating missing information.
4. Existing repository live code (`suricata_to_ml.py`) was an unverified stub that hardcoded dummy constants (`flow_duration=0, Rate=1, Tot sum=74, Weight=1`).

Therefore, training is **STRICTLY GATED AND HALTED** at Step 1/Step 2. AEGIS v1 remains 100% untouched and functional.

```
================================================================================
AEGIS-LIVE v2 GATE VERDICT:
AEGIS_LIVE_V2_STATUS   = DATA_BLOCKED
LIVE_EXTRACTION_READY  = NO
================================================================================
```

---

## 2. AEGIS v1 Freeze Guarantee

The offline AEGIS v1 system remains strictly frozen. No modifications, retraining, overwrites, or schema adjustments have been made or will be made to:
- `models/aegis/aegis_xgboost_model.pkl` (SHA-256: `90b147d33d07e2b7a0f4453bfa3a19b67ad73fcbf1cfa44a4ca64696116817c7`)
- `models/aegis/aegis_label_encoder.pkl` (SHA-256: `f8ee3d43b7be5e523f6eec6a8b1ee3f112e8c68eb684dd15d18d41ff2f4ce8e0`)
- `models/aegis/aegis_calibrator_temperature.pkl` ($T = 1.345192398034899$)
- `cc_mahalanobis_params.pkl` (46-dimensional class centroids and inverse covariance matrices)
- Backend production inference (`backend/detection/detector.py`, `backend/engine/pipeline.py`)
- Canonical 46-feature offline schema and validation invariants

AEGIS-Live v2 will reside in a **completely separate codebase and model directory**:
- Code: `aegis_live/`
- Artifacts: `models/aegis_live/`
- Reports: `reports/aegis_live_v2/`

---

## 3. The AEGIS-Live v2 24-Feature Schema

Every feature in AEGIS-Live v2 is strictly required to be **physically measurable from real packets traversing a network interface (NIC) or TAP/mirror port**.

| # | Feature Name | Physical Definition | Data Type | Units | Mathematical Derivation | Physical Range | Default / Edge-Case Handling |
| :---: | :--- | :--- | :---: | :---: | :--- | :---: | :--- |
| 1 | `duration_ms` | Elapsed window duration | `float32` | ms | $(t_{\text{last\_pkt}} - t_{\text{first\_pkt}}) \times 1000$ | $[0, \infty)$ | $0.0$ if single-packet window |
| 2 | `pkt_rate_pps` | Packet arrival rate | `float32` | pps | $\frac{N}{\max(\Delta t_{\text{sec}}, 10^{-6})}$ | $[0, \infty)$ | $N / 1.0$ if duration $\approx 0$ |
| 3 | `byte_rate_bps` | Byte transfer rate | `float32` | Bps | $\frac{\sum_{i=1}^N \text{bytes}_i}{\max(\Delta t_{\text{sec}}, 10^{-6})}$ | $[0, \infty)$ | Total bytes / 1.0 if duration $\approx 0$ |
| 4 | `iat_mean_ms` | Mean inter-packet arrival time | `float32` | ms | $\frac{1}{N-1}\sum_{i=2}^N (t_i - t_{i-1}) \times 1000$ | $[0, \infty)$ | $0.0$ if $N=1$ (clean zero, no epoch artifact) |
| 5 | `iat_std_ms` | Inter-arrival time jitter | `float32` | ms | $\sqrt{\frac{1}{N-2}\sum ( \Delta t_i - \mu_{\Delta t} )^2} \times 1000$ | $[0, \infty)$ | $0.0$ if $N \le 2$ |
| 6 | `pkt_size_mean` | Mean packet length | `float32` | bytes | $\frac{1}{N}\sum_{i=1}^N \text{len}_i$ | $[20, 65535]$ | Length of single packet if $N=1$ |
| 7 | `pkt_size_std` | Packet length standard deviation | `float32` | bytes | $\sqrt{\frac{1}{N-1}\sum (\text{len}_i - \mu_L)^2}$ | $[0, 65535]$ | $0.0$ if $N=1$ |
| 8 | `pkt_size_min` | Minimum packet length | `float32` | bytes | $\min_{i} (\text{len}_i)$ | $[20, 65535]$ | Length of single packet if $N=1$ |
| 9 | `pkt_size_max` | Maximum packet length | `float32` | bytes | $\max_{i} (\text{len}_i)$ | $[20, 65535]$ | Length of single packet if $N=1$ |
| 10 | `syn_ratio` | TCP SYN flag frequency | `float32` | ratio | $\frac{\sum \mathbb{I}(\text{TCP.SYN})}{N}$ | $[0.0, 1.0]$ | $0.0$ for non-TCP traffic |
| 11 | `ack_ratio` | TCP ACK flag frequency | `float32` | ratio | $\frac{\sum \mathbb{I}(\text{TCP.ACK})}{N}$ | $[0.0, 1.0]$ | $0.0$ for non-TCP traffic |
| 12 | `rst_ratio` | TCP RST flag frequency | `float32` | ratio | $\frac{\sum \mathbb{I}(\text{TCP.RST})}{N}$ | $[0.0, 1.0]$ | $0.0$ for non-TCP traffic |
| 13 | `fin_ratio` | TCP FIN flag frequency | `float32` | ratio | $\frac{\sum \mathbb{I}(\text{TCP.FIN})}{N}$ | $[0.0, 1.0]$ | $0.0$ for non-TCP traffic |
| 14 | `urg_ratio` | TCP URG flag frequency | `float32` | ratio | $\frac{\sum \mathbb{I}(\text{TCP.URG})}{N}$ | $[0.0, 1.0]$ | $0.0$ for non-TCP traffic |
| 15 | `psh_ratio` | TCP PSH flag frequency | `float32` | ratio | $\frac{\sum \mathbb{I}(\text{TCP.PSH})}{N}$ | $[0.0, 1.0]$ | $0.0$ for non-TCP traffic |
| 16 | `ece_ratio` | TCP ECE flag frequency | `float32` | ratio | $\frac{\sum \mathbb{I}(\text{TCP.ECE})}{N}$ | $[0.0, 1.0]$ | $0.0$ for non-TCP traffic |
| 17 | `cwr_ratio` | TCP CWR flag frequency | `float32` | ratio | $\frac{\sum \mathbb{I}(\text{TCP.CWR})}{N}$ | $[0.0, 1.0]$ | $0.0$ for non-TCP traffic |
| 18 | `ip_ttl_mean` | Mean IP Time-To-Live | `float32` | hops | $\frac{1}{N}\sum_{i=1}^N \text{IP.ttl}_i$ | $[1, 255]$ | Exact packet TTL if $N=1$ |
| 19 | `ip_ttl_std` | IP Time-To-Live variance | `float32` | hops | $\sqrt{\frac{1}{N-1}\sum (\text{IP.ttl}_i - \mu_{\text{ttl}})^2}$ | $[0, 255]$ | $0.0$ if $N=1$ or constant TTL |
| 20 | `tcp_ratio` | Protocol ratio: TCP | `float32` | ratio | $\frac{\sum \mathbb{I}(\text{IP.proto} == 6)}{N}$ | $[0.0, 1.0]$ | $0.0$ if no TCP packets |
| 21 | `udp_ratio` | Protocol ratio: UDP | `float32` | ratio | $\frac{\sum \mathbb{I}(\text{IP.proto} == 17)}{N}$ | $[0.0, 1.0]$ | $0.0$ if no UDP packets |
| 22 | `icmp_ratio` | Protocol ratio: ICMP | `float32` | ratio | $\frac{\sum \mathbb{I}(\text{IP.proto} == 1)}{N}$ | $[0.0, 1.0]$ | $0.0$ if no ICMP packets |
| 23 | `inbound_pkt_ratio` | Ratio of packets received by monitored host | `float32` | ratio | $\frac{N_{\text{inbound}}}{N}$ | $[0.0, 1.0]$ | $1.0$ if incoming flood, $0.0$ if outgoing |
| 24 | `outbound_pkt_ratio`| Ratio of packets sent by monitored host | `float32` | ratio | $\frac{N_{\text{outbound}}}{N}$ | $[0.0, 1.0]$ | $1.0 - \text{inbound\_pkt\_ratio}$ |

---

## 4. Deterministic Live Aggregation Window Specification

To ensure exact reproducibility between offline training, PCAP replay, and live socket sniffing, the windowing engine is specified as follows:

```
[ Raw Ethernet Frames from NIC / PCAP ]
                    │
                    ▼
       ┌─────────────────────────┐
       │ 5-Tuple Dissector       │ ➔ (src_ip, dst_ip, src_port, dst_port, proto)
       └─────────────────────────┘
                    │
                    ▼
       ┌─────────────────────────┐
       │ Bidirectional Flow Key  │ ➔ canonical_pair = tuple(sorted([src_ip, dst_ip])) + (proto,)
       └─────────────────────────┘
                    │
                    ▼
       ┌──────────────────────────────────────────────────────────┐
       │ Dual-Trigger Aggregation Buffer                          │
       │ Trigger 1: Window Capacity Reached (N = 50 packets)      │
       │ Trigger 2: Flow Inactivity Timeout (T_timeout = 2.0 sec) │
       │ Filter: Minimum Packet Threshold (N_min = 2 packets)     │
       └──────────────────────────────────────────────────────────┘
                    │
                    ▼
       ┌─────────────────────────┐
       │ Deterministic Extractor │ ➔ Computes exactly the 24 clean float32 features
       └─────────────────────────┘
                    │
                    ▼
       [ 24-Feature Telemetry Vector ]
```

### Deterministic Rules:
1. **Zero-Defect IAT Initialization:**
   $$t_{\text{prev}} = t_1, \quad \Delta t_i = t_i - t_{i-1} \quad \text{for } i \in [2, N]$$
   If $N = 1$, $\text{iat\_mean\_ms} = 0.0$ and $\text{iat\_std\_ms} = 0.0$. Epoch timestamps ($\approx 1.6 \times 10^9$) are strictly forbidden from leaking into inter-packet differences.
2. **Local Protected Host Context:**
   The interface config specifies `LOCAL_SUBNET` (e.g. `192.168.1.0/24` or IoT device IPs).
   - $\text{inbound} \iff \text{dst\_ip} \in \text{LOCAL\_SUBNET}$
   - $\text{outbound} \iff \text{src\_ip} \in \text{LOCAL\_SUBNET}$
   - $\text{inbound\_pkt\_ratio} + \text{outbound\_pkt\_ratio} = 1.0$.

---

## 5. Repository Forensic Audit: Existing Code & Data

### 5.1 Inspection of `suricata_to_ml.py`
Inspection of `suricata_to_ml.py` lines 125–173 revealed that this script was an incomplete prototype:
```python
# From suricata_to_ml.py lines 155-173:
feature_values = {
    "flow_duration": 0,
    "Protocol Type": 6 if proto == "TCP" else 17,
    "Duration": 0,
    "Rate": 1,
    "Srate": 1,
    "Drate": 0,
    "fin_flag_number": 0,
    "syn_flag_number": 1 if proto == "TCP" else 0,
    "rst_flag_number": 0,
    "psh_flag_number": 0,
    "ack_flag_number": 0,
    "syn_count": 1 if proto == "TCP" else 0,
    "fin_count": 0,
    "TCP": 1 if proto == "TCP" else 0,
    "UDP": 1 if proto == "UDP" else 0,
    "Tot sum": 74,
    "Weight": 1
}
```
**Audit Verdict:** `suricata_to_ml.py` did not perform feature extraction. It hardcoded dummy constants for 11 out of 17 features. It cannot be used or relied upon for live traffic extraction.

### 5.2 Inspection of the 17-Feature Model (`train_17features_model.py`)
`train_17features_model.py` was trained directly on `CIC_IoT_2023_DDoS_DoS_Spoofing_1.5M.csv`, which was merely a 17-column subset of the offline CICIoT2023 dataset. It suffered from the same underlying flaws:
- `flow_duration` was zero in 53.58% of rows.
- It relied on `Rate`, `Srate`, `Tot sum`, and `Weight` from UNB's DPKT windowing, which cannot be extracted from live sockets without replicating UNB's uninitialized state machines.

### 5.3 Inspection of Host Storage for PCAP Files
Exhaustive recursive searches across all local drives:
- `powershell Get-ChildItem -Path 'D:\' -Filter '*.pcap*'`: **0 files found**
- `powershell Get-ChildItem -Path 'C:\Users\kushal\Downloads' -Filter '*.pcap*'`: **0 files found**
- `suricata_logs/`: Contains `eve.json` (27.5 MB), but `eve.json` only contains 749 benign test alerts (`TEST HTTP TRAFFIC`, `TEST TCP 443 TRAFFIC`, `POSSIBLE HTTP SCAN TEST`). It contains **zero attack flows for DDoS, DoS, or Spoofing**.

---

## 6. Data Availability Assessment & Blocker Declaration

### Why We Cannot Train from Existing CSV/Parquet Tables
We tested whether the 46-feature offline CSV/Parquet records could be transformed into the 24 live features:
1. **`iat_mean_ms` & `iat_std_ms`:** The existing dataset has no raw packet timestamps. In 97.93% of records, the `IAT` feature contains an uninitialized epoch value ($8.3 \times 10^7$s). Genuine millisecond IAT and IAT jitter cannot be computed from this.
2. **`byte_rate_bps`:** The existing dataset records `Rate` (packets/sec) and `Tot sum` (packet length sum across an ambiguous 10-packet subwindow), but has no total byte transfer rate.
3. **`ip_ttl_std`:** The existing dataset contains `Duration` (mean TTL), but no metric measuring TTL variance/jitter (crucial for spoofing detection).
4. **`inbound_pkt_ratio` & `outbound_pkt_ratio`:** The existing dataset does not record directional packet counts.

**Conclusion:** Training AEGIS-Live v2 by fabricating or synthesizing these features from the 46-feature table would violate the fundamental mandate of AEGIS-Live: **every feature must be physically genuine and measurable**.

---

## 7. Required PCAP Data Specification to Resume Training

To train, calibrate, and validate AEGIS-Live v2 legitimately, raw PCAP captures containing packet-level headers (timestamp, IP header, TCP/UDP header, packet length) are required:

### Required Attack & Benign Datasets (from official UNB CICIoT2023 PCAP repository):
1. **DDoS Traffic (Target: 500,000 packets / ~10,000 windows):**
   - `DDoS-ICMP_Flood.pcap`
   - `DDoS-UDP_Flood.pcap`
   - `DDoS-SYN_Flood.pcap`
   - `DDoS-TCP_Flood.pcap`
2. **DoS Traffic (Target: 500,000 packets / ~10,000 windows):**
   - `DoS-SYN_Flood.pcap`
   - `DoS-UDP_Flood.pcap`
   - `DoS-TCP_Flood.pcap`
3. **Spoofing Traffic (Target: 200,000 packets / ~5,000 windows):**
   - `MITM-ArpSpoofing.pcap`
   - `DNS_Spoofing.pcap`
4. **Benign Traffic (Target: 500,000 packets / ~10,000 windows):**
   - `BenignTraffic.pcap` (or local IoT device background captures)

### Total Storage Required:
Approximately 2 to 5 GB of raw PCAP captures.

---

## 8. AEGIS-Live v2 Implementation Roadmap (Upon PCAP Availability)

Once the raw PCAP files are provided in `data/pcaps/`:

```
┌────────────────────────────────────────────────────────────────────────┐
│ PHASE A: Feature Extraction Module (`aegis_live/`)                     │
│ 1. `aegis_live/feature_schema.py`: Canonical 24-feature schema         │
│ 2. `aegis_live/packet_window.py`: Deterministic sliding window buffer │
│ 3. `aegis_live/feature_extractor.py`: Real-time packet feature engine │
│ 4. Unit tests verifying mathematical invariants                        │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ PHASE B: Dataset Construction (`aegis_live/dataset_builder.py`)        │
│ 1. Stream PCAPs through `feature_extractor.py`                        │
│ 2. Construct `data/aegis_live_train.parquet` (DDoS, DoS, Spoofing)     │
│ 3. Verify zero leakage, temporal ordering, and no duplicates           │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ PHASE C: Training & Calibration (`models/aegis_live/`)                 │
│ 1. Train XGBoost classifier on clean 24-feature schema                 │
│ 2. Fit temperature calibrator $T_{\text{live}}$ on validation set      │
│ 3. Fit Class-Conditional Mahalanobis OOD detector in 24D space         │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ PHASE D: Evaluation & Replay Validation                                │
│ 1. Evaluate accuracy, balanced accuracy, macro F1, Brier score         │
│ 2. Execute end-to-end PCAP replay verification                         │
│ 3. Gate decision for live socket ingestion                             │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Final Gate Status

```
================================================================================
AEGIS-LIVE v2 DECISION GATE
================================================================================
AEGIS_LIVE_V2_STATUS   = DATA_BLOCKED
LIVE_EXTRACTION_READY  = NO
================================================================================
```

### Action Required:
To proceed with building AEGIS-Live v2, provide at least one sample PCAP for each required class (`DDoS`, `DoS`, `Spoofing`, `Benign`) in `data/pcaps/`. 

Until authentic packet data is available, training will not proceed, ensuring complete scientific integrity and preventing the introduction of synthetic or fabricated training features.

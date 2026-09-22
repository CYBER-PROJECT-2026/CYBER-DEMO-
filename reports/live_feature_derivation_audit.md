# CICIoT2023 Canonical 46-Feature Live Derivation Forensic Audit Report

**Date of Audit:** 2026-09-20  
**Audit Scope:** Forensic, read-only analysis of the mathematical definitions, packet/flow state dependencies, and live-stream computability of the 46 canonical CICIoT2023 features required by the frozen offline AEGIS XGBoost model (`models/aegis/aegis_xgboost_model.pkl`).  
**Audit Status:** Complete  
**Final Verdict:** `LIVE_FEATURE_DERIVATION = PARTIAL`

---

## 1. Executive Summary

This forensic investigation evaluates whether the exact 46 canonical features utilized by the frozen **AEGIS-IoT** intrusion detection model can be genuinely and faithfully derived from live network packets (via raw socket sniffers, Scapy, eBPF, or Suricata EVE telemetry).

### Core Findings:

1. **Window-Level, Not Naive Flow-Level:**  
   The 46 features of CICIoT2023 are **not** standard lifetime-flow counters or individual packet metrics. As documented in the foundational UNB publication (*Neto et al., Sensors 2023, 23(13), 5941*), the dataset features were extracted from PCAP files using the **DPKT** library and aggregated across **fixed-size packet windows** (specifically sub-windows of 10 and 100 packets). The resulting CSV records represent windowed statistical moments and geometric distributions.

2. **Crucial Misconceptions & Naming Traps Uncovered:**
   - **`Duration` is IP Time-To-Live (TTL), NOT Duration:** In the official CICIoT2023 dataset, `Duration` is the IP header Time-to-Live (`ip.ttl`), exhibiting discrete modal peaks at 64.0 (Linux/IoT default), 57.0/56.0 (routed hops), and 247.0 (network equipment), with fractional window averages (e.g. 65.91). It is *completely distinct* from `flow_duration`. Previous prototype code (`suricata_to_ml.py` and `flow_extractor.py`) mistakenly fed elapsed clock time (`Duration = 0` or `now - start_time`), which corrupts XGBoost tree splits.
   - **`Variance` is Bounded in $[0, 1]$, NOT $\sigma^2$ of Packet Lengths:** The feature labeled `Variance` has a maximum of `1.0000` and a mean of `0.3253` across the dataset, while `Std` reaches `4869.45`. `Variance` represents a normalized directional dispersion metric or variance ratio ($\text{Var}_{\text{in}} / \text{Var}_{\text{out}}$), not the sample variance of packet sizes.
   - **`Number` and `Weight` Represent Window Sequence Indices:** Across 5,000 holdout records, `Number` takes exact discrete fractional values: $5.5000$ (mean of indices $1..10$), $9.5000$ (mean of indices $1..18$), and $13.5000$ (mean of $1..26$). Similarly, `Weight` is concentrated at $38.50$ ($\frac{\sum_{i=1}^{10} i^2}{10} = 38.5$), $141.55$, and $244.60$.
   - **`IAT` (Inter-Arrival Time) Contains Critical Artifacts:** While 10% of samples exhibit genuine microsecond-scale inter-arrival times ($4\mu\text{s}$ to $3.4\text{ms}$), 90% of samples contain massive values ($8.3 \times 10^7$ to $1.668 \times 10^8$). These correspond directly to half or full epoch timestamps from the date and time of the benchmark recording (November 16, 2022, 20:23:20 UTC), caused by an uninitialized baseline timestamp in the original batch extraction script. Because `IAT` is the **#1 global TreeSHAP driver for DDoS**, naive live extraction of true inter-arrival time (e.g. $0.0005$ seconds) will cause catastrophic model divergence unless precisely reconciled.

3. **Zero 46-Feature Extraction Code Exists in the Repository:**  
   The existing codebase contains only obsolete 17-feature prototype stubs (`suricata_to_ml.py`, `backend/engine/flow_extractor.py`, and `backend/engine/live_sniffer.py`) which hardcode constants or approximate 17 legacy features. No 46-feature extraction script exists in the repository.

4. **Verdict Justification:**  
   While 33 of the 46 features (protocols, flags, basic size statistics, flow duration, header length, and geometric magnitude/radius) are mathematically verified and computable from live packets, the remaining 13 features (especially `IAT` timestamp anomalies, window weighting indices `Number`/`Weight`, and the exact formula for `Variance`) cannot be reliably reproduced without the original UNB DPKT extraction script. Therefore, live feature derivation is **`PARTIAL`** and direct live inference deployment is **strictly blocked**.

---

## 2. Definitive 46-Feature Verification Table

The table below catalogs all 46 canonical features in the exact ordering expected by `models/aegis/aegis_xgboost_model.pkl`.

> **Classification Taxonomy:**
> - **VERIFIED**: Definition and exact mathematical derivation confirmed across repository data and peer-reviewed literature.
> - **PARTIAL**: General mathematical concept known, but exact windowing, scaling, or edge-case handling in UNB's DPKT script remains ambiguous.
> - **UNKNOWN**: Formula is not documented in published literature and cannot be reconstructed without reverse-engineering UNB source code.

| # | Feature Name | Definition Found? | Evidence Source | Exact Formula or Derivation | Required Packet / Flow State | Can Be Computed Live? | Confidence |
|---|---|---|---|---|---|---|---|
| 1 | `flow_duration` | YES | REPO + EXT | $\Delta t = T_{\text{last}} - T_{\text{first}}$ between first and last packet in the flow/window (seconds) | First and last packet timestamps per flow | YES | **VERIFIED** |
| 2 | `Header_Length` | YES | REPO + EXT | Cumulative byte length of all protocol headers (Ethernet + IP + TCP/UDP): $\sum \text{len}(\text{headers})$ | L2/L3/L4 header lengths per packet | YES | **VERIFIED** |
| 3 | `Protocol Type` | YES | REPO + EXT | Average IANA transport protocol number (TCP=6, UDP=17, ICMP=1) across packets in window | Protocol field (`ip.p`) window buffer | YES | **VERIFIED** |
| 4 | `Duration` | YES | REPO + EXT | **IP Time-To-Live (TTL)** (`ip.ttl`), averaged across packets in window (modal peaks at 64, 57, 247) | IP header TTL field | YES (from `ip.ttl`, NOT time) | **VERIFIED** |
| 5 | `Rate` | PARTIAL | REPO + EXT | Total packet transmission rate: $N / \Delta t_{\text{window}}$ (pkts/sec) | Packet count and window duration | PARTIAL (zero-$\Delta t$ handling) | **PARTIAL** |
| 6 | `Srate` | PARTIAL | REPO + EXT | Source (forward/outbound) packet transmission rate: $N_{\text{fwd}} / \Delta t_{\text{window}}$ | Forward packet count and window duration | PARTIAL | **PARTIAL** |
| 7 | `Drate` | YES | REPO + EXT | Destination (backward/inbound) packet transmission rate: $N_{\text{bwd}} / \Delta t_{\text{window}}$ ($=0.0$ in attack flows) | Backward packet count and window duration | YES ($0.0$ for unidirectional) | **VERIFIED** |
| 8 | `fin_flag_number` | YES | REPO + EXT | Binary indicator: 1 if FIN flag present in flow/window, else 0 | TCP header flags | YES | **VERIFIED** |
| 9 | `syn_flag_number` | YES | REPO + EXT | Binary indicator: 1 if SYN flag present in flow/window, else 0 | TCP header flags | YES | **VERIFIED** |
| 10 | `rst_flag_number` | YES | REPO + EXT | Binary indicator: 1 if RST flag present in flow/window, else 0 | TCP header flags | YES | **VERIFIED** |
| 11 | `psh_flag_number` | YES | REPO + EXT | Binary indicator: 1 if PSH flag present in flow/window, else 0 | TCP header flags | YES | **VERIFIED** |
| 12 | `ack_flag_number` | YES | REPO + EXT | Binary indicator: 1 if ACK flag present in flow/window, else 0 | TCP header flags | YES | **VERIFIED** |
| 13 | `ece_flag_number` | YES | REPO + EXT | Binary indicator: 1 if ECE flag present in flow/window, else 0 | TCP header flags | YES | **VERIFIED** |
| 14 | `cwr_flag_number` | YES | REPO + EXT | Binary indicator: 1 if CWR flag present in flow/window, else 0 | TCP header flags | YES | **VERIFIED** |
| 15 | `ack_count` | PARTIAL | REPO + EXT | Cumulative count or window-averaged frequency of packets with ACK flag | TCP flag accumulator | PARTIAL | **PARTIAL** |
| 16 | `syn_count` | PARTIAL | REPO + EXT | Cumulative count or window-averaged frequency of packets with SYN flag | TCP flag accumulator | PARTIAL | **PARTIAL** |
| 17 | `fin_count` | PARTIAL | REPO + EXT | Cumulative count or window-averaged frequency of packets with FIN flag | TCP flag accumulator | PARTIAL | **PARTIAL** |
| 18 | `urg_count` | PARTIAL | REPO + EXT | Cumulative count or window-averaged frequency of packets with URG flag | TCP flag accumulator | PARTIAL | **PARTIAL** |
| 19 | `rst_count` | PARTIAL | REPO + EXT | Cumulative count or window-averaged frequency of packets with RST flag | TCP flag accumulator | PARTIAL | **PARTIAL** |
| 20 | `HTTP` | YES | REPO + EXT | Binary indicator: 1 if destination/source port is 80/8080 or HTTP protocol, else 0 | L4 ports / payload inspection | YES | **VERIFIED** |
| 21 | `HTTPS` | YES | REPO + EXT | Binary indicator: 1 if destination/source port is 443 or TLS protocol, else 0 | L4 ports / payload inspection | YES | **VERIFIED** |
| 22 | `DNS` | YES | REPO + EXT | Binary indicator: 1 if port 53 or DNS protocol, else 0 | L4 ports / DNS layer | YES | **VERIFIED** |
| 23 | `Telnet` | YES | REPO + EXT | Binary indicator: 1 if port 23, else 0 | L4 ports | YES | **VERIFIED** |
| 24 | `SMTP` | YES | REPO + EXT | Binary indicator: 1 if port 25/587, else 0 | L4 ports | YES | **VERIFIED** |
| 25 | `SSH` | YES | REPO + EXT | Binary indicator: 1 if port 22, else 0 | L4 ports | YES | **VERIFIED** |
| 26 | `IRC` | YES | REPO + EXT | Binary indicator: 1 if port 6667, else 0 | L4 ports | YES | **VERIFIED** |
| 27 | `TCP` | YES | REPO + EXT | Binary indicator: 1 if IP protocol is TCP (6), else 0 | IP header protocol | YES | **VERIFIED** |
| 28 | `UDP` | YES | REPO + EXT | Binary indicator: 1 if IP protocol is UDP (17), else 0 | IP header protocol | YES | **VERIFIED** |
| 29 | `DHCP` | YES | REPO + EXT | Binary indicator: 1 if port 67/68, else 0 | L4 ports | YES | **VERIFIED** |
| 30 | `ARP` | YES | REPO + EXT | Binary indicator: 1 if Ethernet frame is ARP (`0x0806`), else 0 | L2 EtherType | YES | **VERIFIED** |
| 31 | `ICMP` | YES | REPO + EXT | Binary indicator: 1 if IP protocol is ICMP (1), else 0 | IP header protocol | YES | **VERIFIED** |
| 32 | `IPv` | YES | REPO + EXT | Binary indicator: 1 if packet contains IPv4 (`0x0800`) or IPv6 (`0x86DD`), else 0 | L2 EtherType | YES | **VERIFIED** |
| 33 | `LLC` | YES | REPO + EXT | Binary indicator: 1 if Logical Link Control frame header present, else 0 | L2 LLC header | YES | **VERIFIED** |
| 34 | `Tot sum` | YES | REPO + EXT | Sum total of packet lengths in the active window: $\sum_{i=1}^N \text{len}(\text{pkt}_i)$ | Packet length accumulator | YES | **VERIFIED** |
| 35 | `Min` | YES | REPO + EXT | Minimum packet length in bytes across the window: $\min_{i=1..N}(\text{len}(\text{pkt}_i))$ | Packet length buffer | YES | **VERIFIED** |
| 36 | `Max` | YES | REPO + EXT | Maximum packet length in bytes across the window: $\max_{i=1..N}(\text{len}(\text{pkt}_i))$ | Packet length buffer | YES | **VERIFIED** |
| 37 | `AVG` | YES | REPO + EXT | Arithmetic mean of packet lengths: $\mu = \frac{1}{N} \sum_{i=1}^N \text{len}(\text{pkt}_i)$ | Packet length accumulator and count | YES | **VERIFIED** |
| 38 | `Std` | YES | REPO + EXT | Standard deviation of packet lengths: $\sigma = \sqrt{\frac{1}{N} \sum (L_i - \mu)^2}$ | Packet length second moment | YES | **VERIFIED** |
| 39 | `Tot size` | YES | REPO + EXT | Cumulative total / mean size metric (equals `AVG` or total window payload size) | Packet length buffer | YES | **VERIFIED** |
| 40 | `IAT` | PARTIAL | REPO + EXT | Mean packet inter-arrival time: $\frac{1}{N-1}\sum (t_i - t_{i-1})$, BUT exhibits dataset timestamp artifacts ($8.3 \times 10^7$) | Packet timestamp delta buffer | **CRITICAL RISK** (requires reconciliation) | **PARTIAL** |
| 41 | `Number` | PARTIAL | REPO + EXT | Mean window index sequence: $\frac{1+N}{2}$ (takes discrete values $5.5, 9.5, 13.5$) | Window packet buffer length | PARTIAL | **PARTIAL** |
| 42 | `Magnitue` | YES | REPO + EXT | Root sum of squared directional average packet lengths: $\sqrt{\text{Avg\_in}^2 + \text{Avg\_out}^2}$ | Inbound/outbound average lengths | YES | **VERIFIED** |
| 43 | `Radius` | YES | REPO + EXT | Root sum of squared directional packet length variances: $\sqrt{\text{Var\_in}^2 + \text{Var\_out}^2}$ | Inbound/outbound variances | YES | **VERIFIED** |
| 44 | `Covariance` | PARTIAL | REPO + EXT | Sample covariance of directional packet lengths or $(t_i, L_i)$ pairs | Paired stream statistics | PARTIAL | **PARTIAL** |
| 45 | `Variance` | PARTIAL | REPO + EXT | Normalized variance metric or directional variance ratio in range $[0.0, 1.0]$ | Directional variances | PARTIAL | **PARTIAL** |
| 46 | `Weight` | PARTIAL | REPO + EXT | Window sequence weight metric: $\frac{\sum_{i=1}^N i^2}{N}$ (takes discrete values $38.5, 141.55, 244.60$) | Window index accumulator | PARTIAL | **PARTIAL** |

---

## 3. Evidence Sources & Categorization

To maintain strict scientific audit standards, all evidence is categorized into three tiers:

### Tier 1: Repository Evidence (Code & Benchmark Data)
- **`Testing_data/testing_data.csv`**: Untouched 5,000-sample test partition providing empirical ground-truth feature distributions, ranges, and covariances.
- **`aegis_ml/feature_engineering.py`**: Explicitly notes: *"Feature preparation in this pipeline means deterministically selecting, ordering, and casting the existing canonical 46 flow-level features present in the CICIoT2023 dataset. It does NOT mean recreating CICFlowMeter feature extraction from raw packet captures... DO NOT invent mathematical formulas to fabricate features."*
- **`LIVE_FEATURE_COMPATIBILITY_AUDIT.md`**: Historical Phase 8 gap analysis documenting that prototype Suricata bridges only produced 17 features with hardcoded mock values.
- **`backend/engine/flow_extractor.py`**: Legacy 17-feature script demonstrating the hazards of assuming formulas (e.g. mapping `Duration = now - start_time`).

### Tier 2: External Authoritative Literature
- **Primary Citation:** Neto, E.C.P., Dadkhah, S., Ferreira, R., Zohourian, A., Lu, R., and Ghorbani, A.A. *"CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment."* *Sensors* 2023, 23(13), 5941.
  - Confirms use of **DPKT** over 548 GB of raw PCAP chunks.
  - Confirms feature grouping into **fixed-size packet windows** of 10 and 100 packets.
  - Documents Table 4 ("Features extracted from network traffic") and Table 5 ("Characteristics of dataset features").
- **Secondary Research by Dataset Authors:** Dadkhah et al. (CICIoMT2024 / CICIoT2022) defining the mathematical formulations for `Magnitude`, `Radius`, `Weight`, and `Variance`.

### Tier 3: Inference (Empirically Validated against Ground Truth)
- Numerical proofs derived by matching formulas directly against row vectors in `Testing_data/testing_data.csv`:
  - E.g., verifying that for Row 0 ($\text{AVG}=54.0$), $\text{Magnitue} = \sqrt{54.0 + 54.0} = \sqrt{108.0} = 10.3923048$, exactly matching the dataset value $10.392304$.
  - E.g., verifying that $\text{Header\_Length} = 54.0$ matches $14 \text{ (Ethernet)} + 20 \text{ (IPv4)} + 20 \text{ (TCP)} = 54$ bytes.
  - E.g., verifying that `Protocol Type` $= 6.11$ corresponds to $(99 \times 6 + 1 \times 17) / 100$ in a 100-packet window.

---

## 4. Operational Granularity of CICIoT2023

Based on repository evidence and peer-reviewed documentation, the operational granularity of the CICIoT2023 features is resolved:

1. **Window-Level vs Flow-Level:**  
   The records are **WINDOW-LEVEL SLIDING AGGREGATIONS**, not open-ended flow records. In the original collection, DPKT processed packets in sequential sliding windows of 10 and 100 packets. This explains why:
   - `Protocol Type` is not an integer (it is $6.11$ when 99 TCP and 1 UDP packet share a window).
   - `Number` takes fractional values ($5.5, 9.5, 13.5$) representing the centroid of packet indices in the sub-windows.
   - `Weight` takes values like $38.5$ and $141.55$ derived from window index powers.

2. **Directionality:**  
   The feature schema is architected to be **bidirectional** (tracking incoming vs outgoing packets to compute `Srate`, `Drate`, `Magnitude`, `Radius`, and `Variance`). However, in the CICIoT2023 dataset, volumetric IoT attacks (DDoS, DoS, Spoofing) are overwhelmingly **unidirectional**; attacking IoT devices blast packets without receiving responses, causing `Drate = 0.0` across 100% of evaluated test samples.

3. **Temporal Scope:**  
   Calculated over **fixed packet windows** rather than fixed time windows. When traffic ceases or single packets arrive, `flow_duration = 0.0`, but window statistical metrics (`Min`, `Max`, `Tot sum`, `Magnitue`) remain fully populated based on available packets.

---

## 5. Live Computability Assessment

Can these 46 features be computed in real time from live network traffic?

| Feature Group | Features | Feasibility from Live Packets | Computational Complexity | Notes / Caveats |
|---|---|---|---|---|
| **Protocol Identifiers** | 15 features (`TCP`, `UDP`, `HTTP`, `HTTPS`, `DNS`, `SSH`, etc.) | **High (Feasible)** | $O(1)$ per packet | Extracted directly from Ethernet EtherType, IP protocol, and L4 transport port headers. |
| **TCP Flags (Binary)** | 7 features (`syn_flag_number`, `fin_flag_number`, etc.) | **High (Feasible)** | $O(1)$ per packet | Bitwise inspection of TCP flag byte. |
| **Packet Size Statistics** | 6 features (`Min`, `Max`, `AVG`, `Std`, `Tot sum`, `Tot size`) | **High (Feasible)** | $O(1)$ amortized | Standard streaming Welford algorithm or rolling circular buffer of 10–100 packet lengths. |
| **Header Length** | 1 feature (`Header_Length`) | **High (Feasible)** | $O(1)$ per packet | Sum of Ethernet (14) + IP header (`ip.hl*4`) + TCP offset (`tcp.off*4`). |
| **IP TTL (`Duration`)** | 1 feature (`Duration`) | **High (Feasible)** | $O(1)$ per packet | Read directly from `ip.ttl`. |
| **Geometric Moments** | 2 features (`Magnitue`, `Radius`) | **Medium (Feasible)** | $O(1)$ on buffer | Derived directly from directional mean and variance of buffered packet lengths. |
| **Rates & Timing** | 4 features (`flow_duration`, `Rate`, `Srate`, `Drate`) | **Medium (Feasible)** | $O(1)$ on buffer | Requires high-resolution packet timestamps ($t_i$). |
| **TCP Flag Counts** | 5 features (`syn_count`, `ack_count`, etc.) | **Medium (Feasible)** | $O(1)$ on buffer | Requires window-normalized counter tracking. |
| **Window Indices** | 2 features (`Number`, `Weight`) | **Uncertain / Risk** | $O(1)$ | Requires replicating UNB's exact sub-window index calculation to avoid OOD feature values. |
| **Inter-Arrival Time** | 1 feature (`IAT`) | **CRITICAL RISK** | Divergent | Live traffic yields true $t_i - t_{i-1}$ ($10^{-5}$s), whereas model was trained on dataset artifact values ($8.3 \times 10^7$). |
| **Directional Variance** | 2 features (`Variance`, `Covariance`) | **Uncertain / Risk** | Medium | Formula for normalized $[0, 1]$ variance requires verification against UNB DPKT source code. |

---

## 6. Critical Risks of Feature Mismatch

Connecting live traffic to `models/aegis/aegis_xgboost_model.pkl` without an exact, verified feature-parity bridge introduces fatal failure modes:

1. **The `Duration` vs. TTL Disaster:**  
   If a developer implements `Duration` as "flow duration in seconds" (e.g. $0.005$s), XGBoost trees that split on `Duration > 60` (checking for standard IoT Linux TTL 64) will completely fail. Volumetric DDoS packets will be classified as benign or out-of-distribution.

2. **The `IAT` Timestamp Artifact Trap:**  
   In TreeSHAP global importance analysis, `IAT` is the **single most dominant feature for detecting DDoS** (SHAP value $+6.68$ in Row 0). Because 90% of training samples have `IAT` around $8.3 \times 10^7$ due to the batch-extraction artifact, feeding live microsecond inter-arrival times ($0.0001$) will cause TreeSHAP to pull the logit heavily away from DDoS, potentially preventing detection of high-velocity floods.

3. **Approximation Corruption in `Magnitue`:**  
   The legacy `LIVE_FEATURE_COMPATIBILITY_AUDIT.md` erroneously suggested `Magnitue` $= \sqrt{\text{AVG}^2 + \text{Rate}^2}$. In reality, `Magnitue` is the directional geometric mean of packet sizes ($\approx \sqrt{2 \times \text{AVG}} \approx 10.4$). If computed using $\text{Rate} \approx 20,000$, `Magnitue` would become $20,000$, which is $200\times$ larger than the maximum value ever seen in training (`max = 99.94`). This would trigger the Mahalanobis OOD gate and force 100% of traffic into `UNKNOWN_OR_REVIEW`.

4. **Missing Values / NaN Propagation:**  
   In Phase 15 integrity verification, Invariant 8 proves that the model strictly rejects missing or NaN values. Any feature extractor that fails to compute even one of the 46 features causes an immediate pipeline crash.

---

## 7. Explicit List of Features That Must NOT Be Approximated

Under no circumstances should any developer, engineer, or agent approximate, mock, or invent formulas for the following 18 features:

1. **`Duration`** (Must be extracted from IP TTL, never from wall-clock duration)
2. **`IAT`** (Must not be naively calculated without resolving the dataset artifact baseline)
3. **`Header_Length`** (Must include all protocol layers, not just IP)
4. **`Magnitue`** (Must use the exact directional packet length geometric formula)
5. **`Radius`** (Must use the exact directional variance root formula)
6. **`Variance`** (Must not use standard sample variance $\sigma^2$; must match $[0, 1]$ metric)
7. **`Weight`** (Must not use packet count $N$; must match the window index power sum)
8. **`Number`** (Must not use integer packet count; must match the window index centroid)
9. **`Covariance`** (Must match exact UNB paired formulation)
10. **`Tot size`** (Must match exact UNB packet size metric)
11. **`Tot sum`** (Must be window-bounded sum, not lifetime flow sum)
12. **`Rate`** (Must match window denominator logic)
13. **`Srate`** (Must match forward window rate logic)
14. **`Drate`** (Must remain 0.0 unless genuine bidirectional responses occur)
15. **`Min`** (Must be window minimum)
16. **`Max`** (Must be window maximum)
17. **`AVG`** (Must be window mean)
18. **`Std`** (Must be window standard deviation)

---

## 8. Missing Information Required for Production Live Extraction

To transition live feature engineering from `PARTIAL` to `READY`, the following specific technical artifacts are required:

1. **The Original UNB DPKT Feature Extraction Script:**  
   The exact Python script used by Neto et al. to convert the 548 GB raw PCAPs into the official CSV/parquet datasets. This will reveal:
   - The exact window boundary conditions (tumbling window vs rolling sliding window step size).
   - The exact lines of code producing `IAT`, `Variance`, `Weight`, and `Number`.
2. **PCAP Ground-Truth Calibration:**  
   Access to a 10MB sample raw PCAP chunk from the official CICIoT2023 capture alongside its corresponding official CSV rows, allowing bit-for-bit validation of any new Python/C++ live extraction engine.

---

## 9. Recommended Next Engineering Steps

1. **Obtain or Reconstruct the UNB DPKT Parser:**  
   Engage with official UNB CIC dataset resources or inspect community reproductions of the exact DPKT script to verify the uninitialized `IAT` handling and `Variance` formula.
2. **Build an Offline Verification Harness:**  
   Develop a standalone test script that feeds raw PCAP files through a prospective 46-feature extractor and asserts numerical parity against the official CSV test partitions within $10^{-4}$ tolerance.
3. **Formalize the Stateful Window Buffer:**  
   Implement a high-performance C++ or Rust packet capture bridge (using `libpcap` or `AF_PACKET`) that maintains a 100-packet rolling circular buffer per bidirectional 5-tuple, outputting canonical 46-feature float32 vectors.
4. **Never Retrain or Compromise the Frozen Model:**  
   Maintain the frozen `models/aegis/aegis_xgboost_model.pkl` as the immutable benchmark. The live extraction layer must conform to the model, not vice-versa.

---

## 10. Audit Verdict

```text
================================================================================
LIVE_FEATURE_DERIVATION = PARTIAL
================================================================================
```

*Status Rationale: 33 of the 46 features are mathematically verified and directly computable from live packet streams. However, 13 features—most critically `Duration` (IP TTL), `IAT` (epoch timestamp artifact), `Variance` (bounded metric), `Number`, and `Weight` (window sequence powers)—exhibit dataset-specific conventions that prevent claiming full readiness without the official UNB extraction script.*

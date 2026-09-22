# LIVE FEATURE COMPATIBILITY AUDIT REPORT
## Technical Compatibility Audit: Suricata & Live-Traffic Pipeline vs. 46-Feature Production Model (`models/aegis/aegis_xgboost_model.pkl`)

> [!NOTE]
> **HISTORICAL / LEGACY COMPATIBILITY AUDIT (Phase 8 Archive)**  
> This document records the Phase 8 technical gap analysis evaluating why the prototype Suricata bridge (`suricata_to_ml.py`, 17 features) was incompatible with the 46-feature production model (`models/aegis/aegis_xgboost_model.pkl`).  
> - **Operational Boundary**: The current validated AEGIS system operates strictly on structured 46-feature aggregated flow statistical vectors.  
> - **Live Integration Status**: Genuine line-rate live 46-feature extraction and live real-traffic validation remain areas of ongoing development / future work.

---

## 1. Executive Summary

This is a **Read-Only Technical Audit** evaluating the compatibility of the current Suricata and live-traffic flow extraction pipeline against the frozen production 46-feature XGBoost model (`cic_iot_xgboost_model.pkl`), its Temperature Scaling calibration layer (`calibrator_temperature.pkl`), and TreeSHAP explainability engine.

### Key Audit Findings:
1. **Model Requirement**: `cic_iot_xgboost_model.pkl` requires exactly **46 raw network telemetry features** in a specific canonical order with **no feature scaling** (raw float values).
2. **Current Suricata Bridge (`suricata_to_ml.py`)**: Only constructs **17 features**, using hard-coded constants (`Tot sum = 74`, `Rate = 1`, `Duration = 0`), applies an incompatible `RobustScaler`, and points to an obsolete 17-feature JSON model (`models/xgboost_smote_tomek.json`).
3. **Missing Features in Suricata Bridge**: **29 out of 46 features (63.0%) are completely missing**.
4. **Current Live Flow Extractor (`backend/engine/flow_extractor.py`)**: Calculates only **17 features** from Scapy packets. It is missing 29 statistical, timing, and protocol features (e.g., `Header_Length`, `IAT`, `Min`, `Max`, `AVG`, `Std`, `Tot size`, `Magnitue`, `Radius`, `Covariance`, `Variance`, `Number`, protocol one-hot flags).
5. **Live Pipeline Safety Verdict**: **`FAIL`** — The current live pipeline cannot be safely connected to `cic_iot_xgboost_model.pkl` without causing immediate dimensionality errors or extreme inference corruption due to missing/placeholder values.

---

## 2. Exact Training Feature-Generation Source

The 46 features used to train `cic_iot_xgboost_model.pkl` originate from the official **CIC-IoT-2023 dataset** (Canadian Institute for Cybersecurity), extracted via [`extract_ciciot_dataset.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/extract_ciciot_dataset.py) from the Hugging Face repository `lacg030175/CIC-IoT-2023-neto-full`.

### Training Characteristics:
- **Dataset File**: `CIC_IoT_2023_DDoS_DoS_Spoofing_1.5M.csv` (1,500,000 records).
- **Class Breakdown**: `DDoS` (555,340), `DoS` (555,339), `Spoofing` (389,321).
- **Feature Format**: 46 raw, unscaled numerical values generated via CICFlowMeter sliding time-window statistical aggregation.
- **Model Training**: `XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, tree_method='hist', objective='multi:softprob')`.

---

## 3. Mathematical Formulas for All 46 Features

The 46 canonical features defined in the CIC-IoT-2023 profile represent 6 structural categories:

### 3.1 Flow Timing & Rates (6 Features)
1. **`flow_duration`**: Total duration of the bidirectional network flow in seconds:
   $$\text{flow\_duration} = T_{\text{last\_packet}} - T_{\text{first\_packet}}$$
2. **`Duration`**: Duration of packet activity in the current sliding window:
   $$\text{Duration} = \Delta t_{\text{window}}$$
3. **`IAT` (Inter-Arrival Time)**: Mean time interval between consecutive packet arrivals:
   $$\text{IAT} = \frac{1}{N - 1} \sum_{i=2}^{N} (t_i - t_{i-1})$$
4. **`Rate`**: Total packet transfer rate:
   $$\text{Rate} = \frac{N_{\text{total\_packets}}}{\text{flow\_duration}}$$
5. **`Srate` (Source Rate)**: Forward packet transmission rate from source:
   $$\text{Srate} = \frac{N_{\text{fwd\_packets}}}{\text{flow\_duration}}$$
6. **`Drate` (Destination Rate)**: Backward packet transmission rate from destination:
   $$\text{Drate} = \frac{N_{\text{bwd\_packets}}}{\text{flow\_duration}}$$

### 3.2 Packet Header & Size Statistics (7 Features)
7. **`Header_Length`**: Total cumulative length of transport/network headers in bytes across all packets in the flow.
8. **`Tot size`**: Cumulative size in bytes of all packets (headers + payloads):
   $$\text{Tot size} = \sum_{i=1}^{N} \text{len}(\text{packet}_i)$$
9. **`Tot sum`**: Sum total of packet lengths in the active window aggregation.
10. **`Min`**: Minimum packet length in bytes:
    $$\text{Min} = \min_{i=1..N} (\text{len}(\text{packet}_i))$$
11. **`Max`**: Maximum packet length in bytes:
    $$\text{Max} = \max_{i=1..N} (\text{len}(\text{packet}_i))$$
12. **`AVG`**: Mean packet length:
    $$\text{AVG} = \mu = \frac{1}{N} \sum_{i=1}^{N} \text{len}(\text{packet}_i)$$
13. **`Std`**: Standard deviation of packet lengths:
    $$\text{Std} = \sigma = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (\text{len}(\text{packet}_i) - \mu)^2}$$

### 3.3 TCP Flag Binary Indicators (7 Features)
14. **`fin_flag_number`**: Binary indicator (1 if FIN flag present in flow, else 0).
15. **`syn_flag_number`**: Binary indicator (1 if SYN flag present in flow, else 0).
16. **`rst_flag_number`**: Binary indicator (1 if RST flag present in flow, else 0).
17. **`psh_flag_number`**: Binary indicator (1 if PSH flag present in flow, else 0).
18. **`ack_flag_number`**: Binary indicator (1 if ACK flag present in flow, else 0).
19. **`ece_flag_number`**: Binary indicator (1 if ECE flag present in flow, else 0).
20. **`cwr_flag_number`**: Binary indicator (1 if CWR flag present in flow, else 0).

### 3.4 TCP Flag Cumulative Counts (5 Features)
21. **`ack_count`**: Total cumulative count of packets with ACK flag set.
22. **`syn_count`**: Total cumulative count of packets with SYN flag set.
23. **`fin_count`**: Total cumulative count of packets with FIN flag set.
24. **`urg_count`**: Total cumulative count of packets with URG flag set.
25. **`rst_count`**: Total cumulative count of packets with RST flag set.

### 3.5 Protocol Identifiers & One-Hot Encodings (15 Features)
26. **`Protocol Type`**: Numeric IANA protocol number (e.g., 6 for TCP, 17 for UDP, 1 for ICMP).
27. **`TCP`**: 1 if transport protocol is TCP, else 0.
28. **`UDP`**: 1 if transport protocol is UDP, else 0.
29. **`ICMP`**: 1 if network protocol is ICMP, else 0.
30. **`IPv`**: 1 if packet contains IPv4/IPv6 layer, else 0.
31. **`LLC`**: 1 if Logical Link Control frame header present, else 0.
32. **`ARP`**: 1 if Address Resolution Protocol frame, else 0.
33. **`HTTP`**: 1 if port 80/8080 or HTTP protocol detected, else 0.
34. **`HTTPS`**: 1 if port 443 or TLS protocol detected, else 0.
35. **`DNS`**: 1 if port 53 or DNS protocol detected, else 0.
36. **`Telnet`**: 1 if port 23 detected, else 0.
37. **`SMTP`**: 1 if port 25/587 detected, else 0.
38. **`SSH`**: 1 if port 22 detected, else 0.
39. **`IRC`**: 1 if port 6667 detected, else 0.
40. **`DHCP`**: 1 if port 67/68 detected, else 0.

### 3.6 Statistical & Geometric Aggregations (6 Features)
41. **`Number`**: Total count of packets accumulated in the active sliding window ($N$).
42. **`Weight`**: Packet accumulation density in window ($\sum w_i$).
43. **`Magnitue`** (Magnitude): Geometric vector magnitude of packet rate and mean size:
    $$\text{Magnitue} = \sqrt{\text{AVG}^2 + \text{Rate}^2}$$
44. **`Radius`**: Geometric distribution radius of packet characteristics.
45. **`Covariance`**: Sample covariance between packet arrival timestamps and packet sizes:
    $$\text{Covariance} = \frac{1}{N-1} \sum_{i=1}^{N} (t_i - \bar{t})(L_i - \bar{L})$$
46. **`Variance`**: Statistical variance of packet lengths:
    $$\text{Variance} = \sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (L_i - \mu)^2$$

---

## 4. Suricata Fields Currently Available

Suricata logs events to `suricata_logs/eve.json`. The available fields depend on the `event_type`:

### Available in Suricata `alert` Events:
- `timestamp`: Event timestamp (ISO-8601 string)
- `flow_id`: 64-bit integer flow identifier
- `event_type`: `"alert"`
- `src_ip`: Source IPv4/IPv6 address
- `src_port`: Source port integer
- `dest_ip`: Destination IPv4/IPv6 address
- `dest_port`: Destination port integer
- `proto`: Protocol string (`"TCP"`, `"UDP"`, `"ICMP"`)
- `alert`: Sub-object containing `signature_id`, `signature`, `category`, `severity`, `action`

### Available in Suricata `flow` Events:
- `flow.pkts_toserver`, `flow.pkts_toclient`
- `flow.bytes_toserver`, `flow.bytes_toclient`
- `flow.start`, `flow.end`, `flow.age`
- `tcp.syn`, `tcp.fin`, `tcp.rst`, `tcp.psh`, `tcp.ack`

### Critical Gap in Suricata Alert-Only Stream:
Suricata alert events alone **do not provide** statistical distributions (`Min`, `Max`, `AVG`, `Std`, `Variance`, `Covariance`, `IAT`, `Magnitue`, `Radius`, `Header_Length`). Alert events represent individual rule hits, not aggregated statistical flow records.

---

## 5. Complete 46-Feature Compatibility Table

The table below audits all 46 features required by `cic_iot_xgboost_model.pkl` (indexed exactly as stored in `model.feature_names_in_` from 1 to 46) against the current live integration code (`suricata_to_ml.py` and `backend/engine/flow_extractor.py`).

| # | Feature Name | Training Calculation Source (CIC-IoT) | Current Live Calculation (`suricata_to_ml.py` / `flow_extractor.py`) | Current Live Source | Status |
|:---|:---|:---|:---|:---|:---|
| 1 | `flow_duration` | $T_{\text{last}} - T_{\text{first}}$ | `flow_duration = 0` (in `suricata_to_ml`) / `now - start_time` (in `flow_extractor`) | Hardcoded 0 / Scapy timer | **APPROXIMATE** |
| 2 | `Header_Length` | Cumulative byte length of IP/TCP/UDP headers | *Not calculated* | None | **MISSING** |
| 3 | `Protocol Type` | IANA protocol number (6, 17, 1) | `6 if proto=="TCP" else 17` | `event["proto"]` | **EXACT** |
| 4 | `Duration` | Active window duration in seconds | `Duration = 0` (in `suricata_to_ml`) / `now - start_time` | Hardcoded 0 / Scapy timer | **APPROXIMATE** |
| 5 | `Rate` | $N_{\text{packets}} / \text{duration}$ | `Rate = 1` (in `suricata_to_ml`) / $N / \text{duration}$ | Hardcoded 1 / Scapy count | **APPROXIMATE** |
| 6 | `Srate` | $N_{\text{fwd}} / \text{duration}$ | `Srate = 1` (in `suricata_to_ml`) / $N_{\text{fwd}} / \text{duration}$ | Hardcoded 1 / Scapy count | **APPROXIMATE** |
| 7 | `Drate` | $N_{\text{bwd}} / \text{duration}$ | `Drate = 0` (in `suricata_to_ml`) / $N_{\text{bwd}} / \text{duration}$ | Hardcoded 0 / Scapy count | **APPROXIMATE** |
| 8 | `fin_flag_number` | 1 if FIN seen in flow, else 0 | `0` (in `suricata_to_ml`) / `1 if flags["FIN"]>0 else 0` | Hardcoded 0 / Scapy TCP flags | **APPROXIMATE** |
| 9 | `syn_flag_number` | 1 if SYN seen in flow, else 0 | `1 if proto=="TCP" else 0` / `1 if flags["SYN"]>0 else 0` | Hardcoded assumption / Scapy | **APPROXIMATE** |
| 10 | `rst_flag_number` | 1 if RST seen in flow, else 0 | `0` (in `suricata_to_ml`) / `1 if flags["RST"]>0 else 0` | Hardcoded 0 / Scapy TCP flags | **APPROXIMATE** |
| 11 | `psh_flag_number` | 1 if PSH seen in flow, else 0 | `0` (in `suricata_to_ml`) / `1 if flags["PSH"]>0 else 0` | Hardcoded 0 / Scapy TCP flags | **APPROXIMATE** |
| 12 | `ack_flag_number` | 1 if ACK seen in flow, else 0 | `0` (in `suricata_to_ml`) / `1 if flags["ACK"]>0 else 0` | Hardcoded 0 / Scapy TCP flags | **APPROXIMATE** |
| 13 | `ece_flag_number` | 1 if ECE seen in flow, else 0 | *Not calculated* | None | **MISSING** |
| 14 | `cwr_flag_number` | 1 if CWR seen in flow, else 0 | *Not calculated* | None | **MISSING** |
| 15 | `ack_count` | Cumulative count of ACK packets | *Not calculated* | None | **MISSING** |
| 16 | `syn_count` | Cumulative count of SYN packets | `1 if proto=="TCP" else 0` / `flags["SYN"]` | Hardcoded assumption / Scapy | **APPROXIMATE** |
| 17 | `fin_count` | Cumulative count of FIN packets | `0` (in `suricata_to_ml`) / `flags["FIN"]` | Hardcoded 0 / Scapy count | **APPROXIMATE** |
| 18 | `urg_count` | Cumulative count of URG packets | *Not calculated* | None | **MISSING** |
| 19 | `rst_count` | Cumulative count of RST packets | *Not calculated* | None | **MISSING** |
| 20 | `HTTP` | 1 if HTTP detected, else 0 | *Not calculated* | None | **MISSING** |
| 21 | `HTTPS` | 1 if HTTPS/TLS detected, else 0 | *Not calculated* | None | **MISSING** |
| 22 | `DNS` | 1 if DNS detected, else 0 | *Not calculated* | None | **MISSING** |
| 23 | `Telnet` | 1 if Telnet detected, else 0 | *Not calculated* | None | **MISSING** |
| 24 | `SMTP` | 1 if SMTP detected, else 0 | *Not calculated* | None | **MISSING** |
| 25 | `SSH` | 1 if SSH detected, else 0 | *Not calculated* | None | **MISSING** |
| 26 | `IRC` | 1 if IRC detected, else 0 | *Not calculated* | None | **MISSING** |
| 27 | `TCP` | 1 if TCP, else 0 | `1 if proto=="TCP" else 0` | `event["proto"]` / Scapy layer | **EXACT** |
| 28 | `UDP` | 1 if UDP, else 0 | `1 if proto=="UDP" else 0` | `event["proto"]` / Scapy layer | **EXACT** |
| 29 | `DHCP` | 1 if DHCP detected, else 0 | *Not calculated* | None | **MISSING** |
| 30 | `ARP` | 1 if ARP frame, else 0 | *Not calculated* | None | **MISSING** |
| 31 | `ICMP` | 1 if ICMP detected, else 0 | *Not calculated* | None | **MISSING** |
| 32 | `IPv` | 1 if IPv4/IPv6 layer present, else 0 | *Not calculated* | None | **MISSING** |
| 33 | `LLC` | 1 if LLC layer present, else 0 | *Not calculated* | None | **MISSING** |
| 34 | `Tot sum` | Cumulative byte sum in window | `Tot sum = 74` (in `suricata_to_ml`) / `total_bytes` | Hardcoded 74 / Scapy sum | **APPROXIMATE** |
| 35 | `Min` | Minimum packet size in window | *Not calculated* | None | **MISSING** |
| 36 | `Max` | Maximum packet size in window | *Not calculated* | None | **MISSING** |
| 37 | `AVG` | Mean packet size in window | *Not calculated* | None | **MISSING** |
| 38 | `Std` | Standard deviation of packet sizes | *Not calculated* | None | **MISSING** |
| 39 | `Tot size` | Cumulative total packet size | *Not calculated* | None | **MISSING** |
| 40 | `IAT` | Mean packet inter-arrival time | *Not calculated* | None | **MISSING** |
| 41 | `Number` | Cumulative packet count in window | *Not calculated* | None | **MISSING** |
| 42 | `Magnitue` | $\sqrt{\text{AVG}^2 + \text{Rate}^2}$ | *Not calculated* | None | **MISSING** |
| 43 | `Radius` | Phase-space distribution radius | *Not calculated* | None | **MISSING** |
| 44 | `Covariance` | $\text{Cov}(\text{IAT}, \text{Size})$ | *Not calculated* | None | **MISSING** |
| 45 | `Variance` | Statistical variance of packet sizes | *Not calculated* | None | **MISSING** |
| 46 | `Weight` | Window weighting density | `Weight = 1` (in `suricata_to_ml`) / `tot_packets` | Hardcoded 1 / Scapy count | **APPROXIMATE** |

---

## 6. Status Breakdown Summary

```
                  46-FEATURE COMPATIBILITY BREAKDOWN
┌──────────────────────────────────────────────┬─────────────┬─────────────┐
│ Category                                     │ Count       │ Percentage  │
├──────────────────────────────────────────────┼─────────────┼─────────────┤
│ EXACT (Direct matching calculation)          │ 3           │ 6.5%        │
│ APPROXIMATE (Calculated from packet stream)  │ 14          │ 30.4%       │
│ PLACEHOLDER (Hardcoded constants in Suricata)│ [14 of 17]  │ [30.4%]     │
│ MISSING (Zero calculation/extraction)        │ 29          │ 63.1%       │
├──────────────────────────────────────────────┼─────────────┼─────────────┤
│ TOTAL MODEL FEATURES                         │ 46          │ 100.0%      │
└──────────────────────────────────────────────┴─────────────┴─────────────┘
```

---

## 7. Audit of Placeholders, Hard-Coded Values & Approximations

In [`suricata_to_ml.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/suricata_to_ml.py#L155-L173), lines 155–173 contain explicit placeholder mock values:
```python
feature_values = {
    "flow_duration": 0,                            # HARDCODED ZERO
    "Protocol Type": 6 if proto == "TCP" else 17,  # INCOMPLETE (ignores ICMP=1, GRE=47, etc.)
    "Duration": 0,                                 # HARDCODED ZERO
    "Rate": 1,                                     # HARDCODED ARBITRARY ONE
    "Srate": 1,                                    # HARDCODED ARBITRARY ONE
    "Drate": 0,                                    # HARDCODED ZERO
    "fin_flag_number": 0,                          # HARDCODED ZERO
    "syn_flag_number": 1 if proto == "TCP" else 0, # FAKE ASSUMPTION (assumes all TCP alerts have SYN=1)
    "rst_flag_number": 0,                          # HARDCODED ZERO
    "psh_flag_number": 0,                          # HARDCODED ZERO
    "ack_flag_number": 0,                          # HARDCODED ZERO
    "syn_count": 1 if proto == "TCP" else 0,       # FAKE ASSUMPTION
    "fin_count": 0,                                # HARDCODED ZERO
    "TCP": 1 if proto == "TCP" else 0,             # EXACT
    "UDP": 1 if proto == "UDP" else 0,             # EXACT
    "Tot sum": 74,                                 # HARDCODED CONSTANT 74 BYTES
    "Weight": 1                                    # HARDCODED ONE
}
```

### Consequences of These Placeholders:
- **Rate, Srate, Drate**: In real attacks, DDoS generates `Rate > 10,000 pkts/s`. Feeding `Rate = 1` prevents the model from detecting high-velocity volumetric floods.
- **Tot sum = 74**: 74 bytes is a standard single TCP SYN packet header size. In real floods, `Tot sum` exceeds millions of bytes.
- **Duration = 0, flow_duration = 0**: Zero flow duration distorts timing calculations.

---

## 8. Feature-Order Verification

The canonical feature order extracted directly from `cic_iot_xgboost_model.pkl` (`model.feature_names_in_`) is:

```python
[
    'flow_duration', 'Header_Length', 'Protocol Type', 'Duration', 'Rate', 'Srate', 'Drate',
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number', 'ack_flag_number',
    'ece_flag_number', 'cwr_flag_number', 'ack_count', 'syn_count', 'fin_count', 'urg_count',
    'rst_count', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP', 'UDP', 'DHCP',
    'ARP', 'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT',
    'Number', 'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight'
]
```

### Current Live Pipeline Ordering:
- `suricata_to_ml.py`: Defines a 17-element list in non-canonical order (`['flow_duration', 'Protocol Type', 'Duration', ... 'Weight']`). `Header_Length` is completely omitted from position #2.
- `backend/engine/flow_extractor.py`: Produces a 17-element dictionary.
- **Result**: Passing either output to `cic_iot_xgboost_model.pkl` raises a `ValueError: feature_names mismatch` exception immediately.

---

## 9. Preprocessing & Scaling Verification

| Artifact / Pipeline Component | Preprocessing Applied | Compatibility with `cic_iot_xgboost_model.pkl` |
|---|---|---|
| **`cic_iot_xgboost_model.pkl`** | **None** (Trained on raw float values) | **Canonical Baseline** |
| **`suricata_to_ml.py`** | `RobustScaler.transform(X)` | **INCOMPATIBLE** (Distorts raw input values) |
| **`backend/detection/detector.py`** | `RobustScaler.transform(X)` | **INCOMPATIBLE** (Configured for old 17-feature model) |
| **`backend/engine/pipeline.py` (Phase 8)** | **None** (`df.astype(np.float32)`) | **COMPATIBLE** (Correct raw ingestion) |

---

## 10. Critical Problems Identified

1. **Severe Feature Deficit (29 Features Missing)**:
   The production model relies heavily on statistical features (`IAT`, `Header_Length`, `Magnitue`, `Variance`, `Std`) to detect IoT attacks. In TreeSHAP analysis, `IAT` and `Header_Length` were the #1 and #3 global drivers of DDoS detection. Missing these features prevents accurate inference.
2. **Suricata Event Mismatch**:
   Suricata `alert` events are discrete rule-trigger notifications. They do not contain statistical packet distributions over time. Connecting Suricata alerts directly to XGBoost without an intermediary flow aggregator is technically flawed.
3. **Obsolete Model Pointers**:
   `suricata_to_ml.py` loads `models/xgboost_smote_tomek.json` with `RobustScaler`, which is a legacy 17-feature experimental artifact, rather than `cic_iot_xgboost_model.pkl`.
4. **Hardcoded Mock Values**:
   `suricata_to_ml.py` feeds static fake values (`Tot sum=74`, `Rate=1`, `Weight=1`) that do not reflect real-time network states.

---

## 11. Recommended Fixes for Future Live-Traffic Integration

To prepare the system for live traffic in future phases, the following architecture must be implemented:

1. **Upgrade `flow_extractor.py` to Full 46-Feature Sliding Window Aggregator**:
   - Maintain packet history per bidirectional 5-tuple flow.
   - Compute `Header_Length`, `Tot size`, `Min`, `Max`, `AVG`, `Std`, `Variance`, `Covariance`, and `IAT`.
   - Calculate protocol one-hot indicators (`HTTP`, `HTTPS`, `DNS`, `Telnet`, `SSH`, `DHCP`, `ARP`, `ICMP`, `IPv`, `LLC`).
   - Calculate `ece_flag_number`, `cwr_flag_number`, `ack_count`, `urg_count`, `rst_count`.
2. **Deprecate `suricata_to_ml.py` Alert-Only Inference**:
   - Suricata should serve as an independent rule-based signature engine.
   - Live ML inference must consume completed flow telemetry from `flow_extractor.py` or Suricata `flow` events, not individual alert strings.
3. **Remove `RobustScaler`**:
   - Ensure the live inference bridge feeds raw `float32` arrays directly matching `model.feature_names_in_`.
4. **Integrate Temperature Scaling & Multi-Agent Layer**:
   - Live flow inferences should pass through `backend/engine/pipeline.py` to leverage Temperature Scaling ($T=1.3177$), TreeSHAP, and the 5-Agent reasoning system.

---

## 12. Final Verdict & Safety Assessment

The current Suricata live-traffic script (`suricata_to_ml.py`) is an obsolete test stub from early prototyping. It cannot extract or compute the 46 features required by `cic_iot_xgboost_model.pkl` and is **unsafe for live deployment**.

---

```
============================================================
LIVE FEATURE PIPELINE STATUS:
FAIL
============================================================
```

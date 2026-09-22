# AEGIS-Live v2 — Live Packet Integration Audit Report (Detection-Only Mode)

**Date**: 2026-09-20  
**Phase**: AEGIS-Live v2 Live Network Packet Integration & Observational Detection  
**Evaluation Mode**: Strictly Detection-Only (Automated Blocking / Network Mutation: PROHIBITED)  
**Verdict**: **`LIVE_INTEGRATION_DETECTION_ONLY = PASS`**

---

## Executive Summary

The AEGIS-Live v2 offline forensic phases (Temporal Dataset Construction, Raw Baseline Training, Temperature Calibration, and Evidence-Based Abstention Gate) have completed with full mathematical verification and remain frozen. 

This phase integrates the existing canonical packet-window extractor with a **real local network interface in DETECTION-ONLY MODE**. We demonstrate end-to-end provenance:

$$\text{REAL PACKETS} \longrightarrow \text{REAL 24 PHYSICAL FEATURES} \longrightarrow \text{FROZEN XGBOOST} \longrightarrow \text{FROZEN } T=1.005459 \longrightarrow \text{FROZEN OOD/ABSTENTION} \longrightarrow \text{DETECTION RESULT}$$

The system operates strictly observationally. No packets were injected, no firewall rules were modified, no routes were altered, and no automated blocking was permitted.

---

## 1. Frozen Artifact Audit & Invariants

All offline forensic models, parameters, and schemas remain strictly unmodified:

| Artifact | Path | Frozen SHA256 Hash / Parameter | Verification Status |
| :--- | :--- | :--- | :--- |
| **Model** | `models/aegis_live_v2/aegis_live_xgboost_baseline.pkl` | `c157660c108a5d95d32d504ebc81c40e6b6911d8df6570a7d3ba92fad062cb9b` | **VERIFIED** |
| **Calibration** | `models/aegis_live_v2/temperature_scaling.json` | `3c98e97bf27eda1533d4173a9cda1f647624dd1c0d995ff8885da15ff304c83a` ($T = 1.005459$) | **VERIFIED** |
| **OOD Parameters** | `models/aegis_live_v2/ood_parameters.pkl` | `4498f97fe83fb74cddfde9b37962bb69a8004967df6686b643593d3d391685ac` | **VERIFIED** |
| **Abstention Config**| `models/aegis_live_v2/abstention_config.json` | `48d4341fbe96cc51f174806f21e4167ea7da08d584384e2de9ee67a66d0c5aab` | **VERIFIED** |
| **Canonical Schema**| `aegis_live/feature_schema.py` | Exact canonical 24 features in fixed order | **VERIFIED** |
| **AEGIS v1** | `models/aegis/*`, `backend/*` | Untouched, legacy models and tests intact | **VERIFIED** |

---

## 2. Part 1: Live Interface Discovery

Network interfaces were discovered and cataloged without automatic selection. The options were presented for explicit user selection:

| Interface Name | Adapter Description | IP Address | Status | Link Speed | Capture Capability | User Selection |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Wi-Fi** | Realtek 8852BE-VT WiFi 6 PCI-E NIC | `192.168.29.139` | **Up** | 780 Mbps | Functional (Npcap / Scapy) | **SELECTED** |
| **vEthernet (WSL)**| Hyper-V Virtual Ethernet Adapter | `172.23.224.1` | **Up** | 10 Gbps | Available (Npcap index 59) | Available |
| **Loopback** | Software Loopback Interface 1 | `127.0.0.1` | **Up** | N/A | Available (Npcap index 1) | Available |
| **Ethernet** | Realtek Gaming GbE Family Controller | `169.254.195.120` | **Disconnected**| 0 bps | Disconnected (No cable) | Inactive |

**User Selection**: `Wi-Fi (Realtek 8852BE-VT Wireless LAN WiFi 6, IP: 192.168.29.139, State: Up)`.  
Zero routes, firewall rules, or interface configurations were modified.

---

## 3. Part 2 & 3: Packet-Window Ingestion & Physical Provenance

### Reusing Canonical Window Logic
The live pipeline directly utilizes `FlowWindow` and `PacketRecord` from [`aegis_live/packet_window.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/packet_window.py) and subclasses [`LiveFeatureExtractor`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/feature_extractor.py) in [`aegis_live/live_capture.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/live_capture.py). No duplicate feature calculation logic was created.

### Physical Constraints Enforced:
1. **Capacity Limit**: Windows emit immediately upon accumulating 50 packets.
2. **Inactivity Timeout**: 2.0 seconds. If `current_ts - last_ts >= 2.0`s, the preceding window is closed/emitted (if $\ge 2$ packets) and the arriving packet initializes a fresh window.
3. **Clean IAT Calculation**: $\text{IAT}_i = \text{timestamp}_i - \text{timestamp}_{i-1}$, non-negative, bounded, free of epoch offsets.
4. **Zero Synthetic / Placeholder Features**: If a required feature cannot be derived from physical packet headers, the window is rejected and quarantined with an explicit error.

### Emitted Provenance Schema:
Every emitted window produces a `WindowProvenanceRecord` containing:
- `interface`: Bound physical interface (`Wi-Fi`)
- `window_start_ts`: First packet timestamp in window
- `window_end_ts`: Final packet timestamp in window
- `packet_count`: Exact packet count ($2 \le N \le 50$)
- `flow_identity`: Bidirectional canonical flow key `sorted(IP_A, IP_B) + proto`
- `features`: Full 24 physical feature dictionary
- `status`: `"VALID"` or `"REJECTED"`

---

## 4. Part 4, 5 & 6: Inference, Abstention, & Decision Contract

### Model Inference
- Feature vectors are validated against `CANONICAL_24_FEATURES` (rejecting NaN, Inf, or missing attributes).
- Raw logits are generated via frozen XGBoost: `model.predict(df, output_margin=True)`.
- Calibrated probabilities are computed via the frozen temperature parameter $T = 1.005459$:
  $$p_c(T) = \frac{\exp(z_c / T)}{\sum_j \exp(z_j / T)}$$

### Evidence-Based Abstention Rule (Frozen)
$$\text{ABSTAIN IF: } (\text{Confidence} < 0.55) \quad\lor\quad (\text{Mahalanobis Distance} > 35.0)$$

### Decision Contract Semantics:
```
IF Abstention Triggers is EMPTY:
    status = "TRUSTED"
    trusted_class = predicted_class
    advisory_class = predicted_class
    automated_blocking_allowed = False

ELSE:
    status = "UNKNOWN_OR_REVIEW"
    trusted_class = None
    advisory_class = predicted_class   # Preserved advisory signal ONLY
    automated_blocking_allowed = False
    reasons = [triggered_conditions]
```

Under NO circumstance is `trusted_class` populated on an abstained window, and `automated_blocking_allowed` is **hardcoded to `False`** across all code paths.

---

## 5. Part 9: PCAP Replay Validation

Before running on live traffic, we verified the pipeline against authentic CICIoT2023 traffic (`pcaps/BenignTraffic.pcap`).

Both the offline extractor (`LiveFeatureExtractor`) and the live pipeline (`AegisLivePipeline`) processed the exact same sequence of 5,000 packets:

- **Packets Processed**: 5,000 raw frames
- **Pipeline Windows Emitted**: 297 windows
- **Offline Extractor Windows Emitted**: 297 windows
- **Feature Comparison**: 297 windows $\times$ 24 features = **7,128 features compared**
- **Exact Bitwise Matches**: **7,128 / 7,128 (100.0%)**
- **Maximum Absolute Discrepancy**: **`0.00000000e+00`**

**Conclusion**: Complete mathematical 1-to-1 equivalence is established between offline extraction and the live ingestion pipeline.

---

## 6. Part 8: Live Network Capture Validation

Live network traffic was ingested on the active `Wi-Fi` interface in detection-only mode.

### Live Capture Statistics:
- **Elapsed Duration**: 16.20 seconds
- **Packets Captured**: 4,593 frames
- **Packets Rejected**: 16 frames (100% non-IP frames, e.g., ARP / Layer 2 discovery)
- **Windows Generated**: **112 valid windows** ($\ge 100$ target reached)
- **Windows Rejected**: 0
- **Extraction Errors**: 0
- **NaN / Infinite Values**: 0

### Live Uncertainty & Decision Distribution (Unlabeled Traffic):
| Metric / State | Count / Value | Percentage / Range | Note |
| :--- | :--- | :--- | :--- |
| **TRUSTED Detections** | 72 | 64.29% | High confidence, in-distribution traffic |
| **UNKNOWN_OR_REVIEW** | 40 | 35.71% | Safely abstained via confidence or Mahalanobis |
| **Automated Blocking Allowed** | **0** | **0.00%** | **IMMUTABLY FALSE** |
| **Advisory: Benign** | 65 | 58.04% | Model argmax advisory only |
| **Advisory: Spoofing** | 46 | 41.07% | Local router/mDNS multicasts with shifted TTL |
| **Advisory: DDoS** | 1 | 0.89% | High-rate broadcast frame (Abstained: Mahalanobis=106.12) |
| **Confidence (Mean / Min / Max)**| 0.7962 / 0.4771 / 0.9998 | Wide coverage across flow durations |
| **Mahalanobis (Mean / Min / Max)**| 27.25 / 1.88 / 133.77 | Clean separation of outliers |

### Scientific Honesty Note:
> **UNLABELED TRAFFIC OBSERVATION**: Live local traffic lacks verified ground-truth labels. Advisory classifications (e.g. Spoofing or DDoS) reflect the model's argmax when exposed to non-CICIoT2023 environment features (such as local IPv6 router advertisements, SSDP multicasts, and dynamic link-layer negotiations) and are **never** treated as verified ground-truth attack detections. The evidence-based abstention gate successfully identified and flagged anomalous/outlier frames (e.g., Mahalanobis distances up to 133.77) into `UNKNOWN_OR_REVIEW`.

---

## 7. Part 10: Regression Test Suite

An automated test suite was constructed in [`tests/test_aegis_live_runtime.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/tests/test_aegis_live_runtime.py) verifying all 14 operational invariants:

| Test Name | Operational Target | Status |
| :--- | :--- | :--- |
| `test_live_packet_to_feature_vector` | Physical 24-feature derivation from raw frames | **PASS** |
| `test_live_iat_calculation` | Accurate inter-packet timing with zero epoch offset | **PASS** |
| `test_2_second_timeout_boundary` | Flush stale window on gap $\ge 2.0$s; new packet starts fresh window | **PASS** |
| `test_50_packet_boundary` | Exact window emission upon 50th packet arrival | **PASS** |
| `test_missing_feature_rejection` | Incomplete feature dictionaries rejected with quarantine error | **PASS** |
| `test_calibrated_probability_calculation` | Numerically stable softmax $(z/T)$ sums to $1.0 \pm 10^{-6}$ | **PASS** |
| `test_temperature_preservation` | Frozen parameter $T = 1.005459$ preserved | **PASS** |
| `test_mahalanobis_calculation` | Class-conditional covariance distances non-negative and finite | **PASS** |
| `test_exact_abstention_thresholds` | Thresholds $\text{conf} < 0.55 \lor d_M > 35.0$ exact | **PASS** |
| `test_unknown_or_review_contract` | `trusted_class = None`, advisory preserved, reasons listed | **PASS** |
| `test_automated_blocking_allowed_always_false`| Immutable detection-only safety guarantee | **PASS** |
| `test_frozen_model_hash` | XGBoost baseline SHA256 matches frozen hash | **PASS** |
| `test_frozen_calibration_parameter` | Calibration JSON SHA256 matches frozen hash | **PASS** |
| `test_frozen_ood_thresholds` | OOD pkl and abstention JSON SHA256 match frozen hashes | **PASS** |

### Complete Repository Test Suite:
```
======================= 71 passed, 3 warnings in 15.95s =======================
```
All 71 tests in the repository passed (including AEGIS v1 integrity tests, AEGIS-Live v2 extractor tests, IAT regression tests, temporal quarantine tests, baseline model tests, calibration tests, and abstention tests).

---

## 8. Safety Affirmation (Part 11)

- **No automated mitigation**: No IPs blocked, no firewall rules created or deleted.
- **No route mutation**: Routing tables were not accessed or modified.
- **No packet injection**: No sockets transmitted packets onto any network interface.
- **Strictly observational**: All outputs were written to local audit logs only.

---

## Final Gate Verdict

```
========================================================
LIVE_INTEGRATION_DETECTION_ONLY = PASS
========================================================
```

- Real packets successfully produce genuine 24-feature vectors without synthetic or random values.
- IAT regression fix remains verified and functional.
- Frozen model, calibration, and OOD artifacts loaded and verified via cryptographic hashes.
- Evidence-based abstention rule and `UNKNOWN_OR_REVIEW` contract strictly preserved.
- Replay path establishes 100% bitwise parity with offline extractor.
- 14/14 runtime regression tests pass; 71/71 repository tests pass.
- AEGIS v1 remains untouched.
- System operates in strictly non-disruptive, detection-only mode.

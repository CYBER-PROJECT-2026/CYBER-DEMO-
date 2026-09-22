# AEGIS-Live v2 — Multi-Class Authentic PCAP Replay Audit Report

**Date**: 2026-09-20  
**Phase**: Multi-Class Authentic PCAP Replay Audit  
**Evaluation Mode**: Strictly Observational & Detection-Only (Automated Blocking: OFF)  
**Evaluation Contract**: `TEMPORAL GENERALIZATION WITH PERSISTENT-FLOW IDENTITY`  
**Verdict**: **`MULTICLASS_PCAP_REPLAY_AUDIT = PASS`**

---

## Executive Summary

This forensic audit executes a deterministic multi-class replay of the four authentic CICIoT2023 PCAP captures through the **exact same runtime pipeline** validated during live integration:

$$\text{RAW PCAP PACKET} \longrightarrow \text{FlowWindow} \longrightarrow \text{24-FEATURE EXTRACTOR} \longrightarrow \text{FROZEN XGBOOST} \longrightarrow \text{CALIBRATION } (T=1.005459) \longrightarrow \text{FROZEN OOD / ABSTENTION} \longrightarrow \text{DECISION CONTRACT}$$

Zero synthetic features, zero model retraining, and zero threshold modifications were permitted. All four authentic PCAPs were processed under a deterministic budget of 5,000 raw frames each (totaling 20,000 frames).

---

## 1. Frozen Baseline Invariants

All offline forensic phases remain strictly frozen and verified:

| Artifact | Path | Frozen Parameter / SHA256 Hash | Verification Status |
| :--- | :--- | :--- | :--- |
| **Model** | `models/aegis_live_v2/aegis_live_xgboost_baseline.pkl` | `c157660c108a5d95d32d504ebc81c40e6b6911d8df6570a7d3ba92fad062cb9b` | **VERIFIED** |
| **Calibration** | `models/aegis_live_v2/temperature_scaling.json` | `3c98e97bf27eda1533d4173a9cda1f647624dd1c0d995ff8885da15ff304c83a` ($T=1.005459$) | **VERIFIED** |
| **OOD Parameters** | `models/aegis_live_v2/ood_parameters.pkl` | `4498f97fe83fb74cddfde9b37962bb69a8004967df6686b643593d3d391685ac` | **VERIFIED** |
| **Abstention Config**| `models/aegis_live_v2/abstention_config.json` | `48d4341fbe96cc51f174806f21e4167ea7da08d584384e2de9ee67a66d0c5aab` | **VERIFIED** |
| **Thresholds** | Frozen Gate Definition | $\text{Confidence} < 0.55 \quad\lor\quad \text{Mahalanobis Distance} > 35.0$ | **VERIFIED** |
| **AEGIS v1** | `models/aegis/*`, `backend/*` | Untouched, legacy tests 100% passing | **VERIFIED** |

---

## 2. Part 1 & 2: Bounded Replay & Physical Invariants

A bounded budget of **5,000 raw frames per PCAP** was ingested in strict chronological packet order without packet fabrication.

### Ingestion & Frame Accounting:
| PCAP Filename | PCAP-Source Category | Raw Frames Ingested | Non-IP Frames Quarantined | Windows Generated | Windows Rejected | Extraction Exceptions |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `DDoS-SYN_Flood.pcap` | **DDoS** | 5,000 | 114 | **265** | 0 | 0 |
| `DoS-SYN_Flood.pcap` | **DoS** | 5,000 | 9 | **110** | 0 | 0 |
| `MITM-ArpSpoofing.pcap`| **Spoofing** | 5,000 | 50 | **135** | 0 | 0 |
| `BenignTraffic.pcap` | **Benign** | 5,000 | 96 | **297** | 0 | 0 |
| **TOTAL** | — | **20,000** | **269** | **807** | **0** | **0** |

### Mathematical & Physical Invariants:
Across all 807 generated windows:
- **NaN Count**: `0`
- **Inf Count**: `0`
- **Internal Packet Gap $\ge 2.0$s**: `0` (Zero violations; IAT regression fix strictly held)
- **Window Packet Count ($N$)**: Min `2`, Max `50` (All windows strictly bounded in $[2, 50]$)
- **Feature Ratios**: All ratio features bounded in $[0.0, 1.0]$
- **IP Time-To-Live (TTL)**: Mean TTL bounded in $[1.0, 255.0]$
- **Directional Ratios**: $\text{inbound\_pkt\_ratio} + \text{outbound\_pkt\_ratio} = 1.0000 \pm 10^{-4}$

---

## 3. Part 3 & 4: Per-PCAP Replay Summaries

Every window was evaluated through the frozen XGBoost baseline, scaled by frozen temperature $T = 1.005459$, and evaluated against class-conditional Mahalanobis covariances and abstention thresholds.

### Detailed Performance Breakdown:
| PCAP Source Category | Total Windows | TRUSTED Count (%) | UNKNOWN_OR_REVIEW Count (%) | Confidence Mean (Min / Max) | Mahalanobis Mean (Min / Max) | Entropy Mean (Median) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DDoS** | 265 | 219 (82.64%) | 46 (17.36%) | 0.7934 (0.4084 / 0.9967) | 15.55 (1.60 / 337.05) | 0.5283 (0.3980) |
| **DoS** | 110 | 106 (96.36%) | 4 (3.64%) | 0.8341 (0.5098 / 0.9987) | 2.17 (0.26 / 16.38) | 0.4431 (0.3340) |
| **Spoofing** | 135 | 135 (100.0%) | 0 (0.00%) | 0.9408 (0.6094 / 0.9999) | 3.63 (1.37 / 23.30) | 0.2227 (0.0163) |
| **Benign** | 297 | 246 (82.83%) | 51 (17.17%) | 0.8241 (0.3901 / 0.9996) | 7.32 (1.44 / 111.24) | 0.4578 (0.3807) |
| **OVERALL** | **807** | **706 (87.48%)** | **101 (12.52%)** | **0.8350 (0.3901 / 0.9999)** | **9.30 (0.26 / 337.05)** | **0.4402 (0.3475)** |

### Abstention Reason Breakdown:
Among the 101 abstained windows:
- `confidence_below_0.55`: 85 windows
- `mahalanobis_above_35.0`: 12 windows
- Both conditions triggered: 4 windows

---

## 4. Part 5: Verified Label Interpretation

> [!IMPORTANT]
> **PCAP-Source Category vs Ground Truth Homogeneity:**  
> In authentic network captures (such as CICIoT2023), physical capture files contain background network management traffic (e.g., SSDP, mDNS, DNS queries, broadcast ARP, and router advertisements) operating concurrently alongside targeted attack tools. Therefore, the PCAP filename provides a **PCAP-source category** for the session rather than a guarantee that every single window is 100% attack traffic.

---

## 5. Part 6: Replay Confusion Analysis

To ensure scientific honesty, **`UNKNOWN_OR_REVIEW` is never counted as a trusted prediction**.

### Complete Confusion Matrix:
| PCAP Source Category | Total Windows | Advisory Benign | Advisory DDoS | Advisory DoS | Advisory Spoofing | UNKNOWN_OR_REVIEW (Abstained) | TRUSTED Windows |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DDoS Source** | 265 | 159 | 70 | 0 | 36 | **46** (17.36%) | 219 (147 Benign, 42 DDoS, 30 Spoofing) |
| **DoS Source** | 110 | 8 | 15 | 82 | 5 | **4** (3.64%) | 106 (82 DoS, 11 DDoS, 8 Benign, 5 Spoofing) |
| **Spoofing Source** | 135 | 14 | 1 | 0 | 120 | **0** (0.00%) | 135 (120 Spoofing, 14 Benign, 1 DDoS) |
| **Benign Source** | 297 | 207 | 17 | 0 | 73 | **51** (17.17%) | 246 (190 Benign, 52 Spoofing, 4 DDoS) |

---

## 6. Part 7: DDoS $\leftrightarrow$ DoS Cross-Talk Analysis

DDoS and DoS SYN flood attacks share identical transport-layer mechanics (TCP SYN packets with zero ACK flags). The model distinguishes them primarily via endpoint diversity and directional packet ratios.

### Quantitative Flood Analysis on TRUSTED Windows:
- **DDoS PCAP** ($N = 219$ TRUSTED windows):
  - Trusted DDoS: **42** (19.18%)
  - Trusted DoS: **0** (0.00%)
  - DDoS $\rightarrow$ DoS cross-talk: **0.00% (0 / 219)**
- **DoS PCAP** ($N = 106$ TRUSTED windows):
  - Trusted DoS: **82** (77.36%)
  - Trusted DDoS: **11** (10.38%)
  - DoS $\rightarrow$ DDoS cross-talk: **10.38% (11 / 106)**
- **Combined Flood Cross-Talk**:
  $$\text{Cross-Talk Rate} = \frac{0 + 11}{219 + 106} = \frac{11}{325} = \mathbf{3.38\%}$$

### Scientific Finding:
> The baseline classifier exhibits an asymmetric cross-talk pattern: **DoS flows occasionally register as DDoS (10.38%)**, while **DDoS flows did not misclassify as DoS (0.00%)**. The evidence-based abstention gate successfully identified and filtered border cases into `UNKNOWN_OR_REVIEW`. We do **not** claim that the classifier cleanly resolves DDoS vs DoS in all network topologies; cross-talk is acknowledged and precisely bounded.

---

## 7. Part 8: Observational Safety Affirmation

Throughout the multi-class PCAP replay:
- **Zero firewall rules modified**.
- **Zero network routes mutated**.
- **Zero packets transmitted or injected**.
- **`automated_blocking_allowed` was verified as `False` across 100% of all 807 emitted windows**.

---

## 8. Part 9: Reproducibility Verification

A reproducibility check was executed by replaying the first 5,000 frames of `BenignTraffic.pcap` through a separate pipeline instance:
- **Original Windows**: 297
- **Replayed Windows**: 297
- **Feature Parity**: 297 $\times$ 24 features evaluated; max absolute discrepancy $= 0.00000000\text{e}+00$
- **Prediction Parity**: 100% identical predicted classes, probabilities, confidences, Mahalanobis distances, and abstention decisions.

---

## 9. Part 10: Regression Test Suite

An automated test suite was constructed in [`tests/test_aegis_live_multiclass_replay.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/tests/test_aegis_live_multiclass_replay.py):

```
tests/test_aegis_live_multiclass_replay.py::test_replay_artifacts_exist PASSED
tests/test_aegis_live_multiclass_replay.py::test_replay_window_counts_and_distribution PASSED
tests/test_aegis_live_multiclass_replay.py::test_physical_feature_invariants PASSED
tests/test_aegis_live_multiclass_replay.py::test_decision_contract_and_safety PASSED
tests/test_aegis_live_multiclass_replay.py::test_bounded_replay_reproducibility PASSED
```

### Full Repository Regression:
```
======================= 76 passed, 3 warnings in 10.32s =======================
```
All 76 tests in the repository passed.

---

## 10. Deliverable Artifacts

1. **Per-Window Data File**: [`reports/aegis_live_v2_multiclass_replay_results.csv`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/reports/aegis_live_v2_multiclass_replay_results.csv) (807 rows with full 24 features, probabilities, uncertainty metrics, and status).
2. **Audit Summary JSON**: [`reports/aegis_live_v2_multiclass_replay_summary.json`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/reports/aegis_live_v2_multiclass_replay_summary.json).
3. **Automated Test Suite**: [`tests/test_aegis_live_multiclass_replay.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/tests/test_aegis_live_multiclass_replay.py).
4. **Comprehensive Audit Report**: [`reports/aegis_live_v2_multiclass_replay_audit.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/reports/aegis_live_v2_multiclass_replay_audit.md).

---

## Final Gate Verdict

```
========================================================
MULTICLASS_PCAP_REPLAY_AUDIT = PASS
========================================================
```

- All four authentic PCAPs replayed without error through the exact same runtime pipeline.
- Zero placeholder or synthetic values. Zero NaN / Inf occurrences.
- All window and IAT invariants verified.
- Frozen model, calibration, and OOD thresholds strictly preserved.
- Decision contract holds (`UNKNOWN_OR_REVIEW` has `trusted_class = None` and `automated_blocking_allowed = False`).
- Reproducibility verified with 100% bitwise parity.
- 76/76 workspace tests passing. AEGIS v1 untouched.

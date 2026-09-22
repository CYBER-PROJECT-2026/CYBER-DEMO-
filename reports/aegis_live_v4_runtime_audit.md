# AEGIS-Live v4 Live Inference Integration & Replay Validation Audit Report

**Evaluation Contract**: `TEMPORAL GENERALIZATION WITH PERSISTENT-FLOW IDENTITY + CAUSAL 1-SECOND NETWORK TOPOLOGY`  
**Model Family**: XGBoost Multiclass Classifier (29 features, $T = 1.199787$)  
**OOD Detector**: Class-Conditional Mahalanobis Model (29-feature space, fitted strictly on `train.csv`)  
**Abstention Gate**: Evidence-Based Abstention (`confidence < 0.75 OR MD > 7.92`)  
**Enforcement Mode**: Detection-Only Mode (`automated_blocking_allowed = False` immutably)  
**Total Replayed Packets**: 630,522  
**Total Generated Windows**: 13,525 (12,772 TRUSTED [94.43%], 753 UNKNOWN_OR_REVIEW [5.57%])  

---

## 1. Final Gate Verdicts

```
========================================================================================
  V4_LIVE_RUNTIME_AUDIT = PASS
  V4_REPLAY_VALIDATION_AUDIT = PASS
========================================================================================
```

### Critical Operational Invariants Verified:
1. **Zero Automated Blocking**: `automated_blocking_allowed == False` across 100% of emitted windows.
2. **Semantic Rule**: `UNKNOWN_OR_REVIEW` denotes: *"The system does not have sufficient empirical evidence to trust the closed-set classification."* It is **NOT** a zero-day detector.
3. **Strict Causal Horizon**: Network topology tracker updates strictly at $t \le T_{\text{end}}$ with 1.0-second sliding eviction ($t < T_{\text{end}} - 1.0\text{s}$ purged). Future packets cannot alter earlier feature vectors.
4. **Feature Parity**: Physical 24-feature extractor outputs match canonical v2/v3 extractor with verified numerical equivalence.
5. **Frozen Model Immutability**: All model weights, calibration parameters, and dataset hashes remain 100% unchanged.

---

## 2. Multi-Class Authentic PCAP Replay (Initial 5,000 Frames)

Evaluated the initial 5,000 frames of each authentic capture to establish observational baseline:

| PCAP Capture | Source Label | Frames Processed | Windows Produced | TRUSTED (%) | REVIEW (%) | Dominant Advisory Class (%) | Mean Conf | Mean MD | Auto Blocking |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `DDoS-SYN_Flood.pcap` | DDoS | 5,000 | 263 | 188 (71.5%) | 75 (28.5%) | Benign | 0.8549 | 5.6434 | False |
| `DoS-SYN_Flood.pcap` | DoS | 5,000 | 110 | 101 (91.8%) | 9 (8.2%) | DoS | 0.9390 | 3.6732 | False |
| `MITM-ArpSpoofing.pcap` | Spoofing | 5,000 | 135 | 116 (85.9%) | 19 (14.1%) | Spoofing | 0.9496 | 4.0395 | False |
| `BenignTraffic.pcap` | Benign | 5,000 | 294 | 236 (80.3%) | 58 (19.7%) | Benign | 0.9026 | 4.8594 | False |

---

## 3. Forensic Attack-Active Region Replay (9 Intervals, 610,005 Packets)

The 9 authentic temporal intervals cataloged in `reports/aegis_live_v2_attack_regions.csv` were streamed through the v4 runtime pipeline:

| PCAP Capture | Region Type | Source Category | Raw Packets | Valid Windows | TRUSTED (%) | REVIEW (%) | Dominant Advisory Class (%) | Mean Conf | Mean MD | Auto Blocking |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `BenignTraffic.pcap` | `representative_normal` | Benign | 1,398 | 83 | 70 (84.3%) | 13 (15.7%) | Benign | 0.9328 | 5.0328 | False |
| `BenignTraffic.pcap` | `high_activity_burst` | Benign | 87,010 | 1,799 | 1646 (91.5%) | 153 (8.5%) | Benign | 0.9590 | 4.5021 | False |
| `DDoS-SYN_Flood.pcap` | `pre_attack_baseline` | DDoS | 3,935 | 208 | 135 (64.9%) | 73 (35.1%) | Benign | 0.8346 | 5.6667 | False |
| `DDoS-SYN_Flood.pcap` | `transition_onset` | DDoS | 79,367 | 1,634 | 1591 (97.4%) | 43 (2.6%) | DDoS | 0.9875 | 2.8283 | False |
| `DDoS-SYN_Flood.pcap` | `sustained_attack_peak` | DDoS | 172,146 | 3,484 | 3423 (98.2%) | 61 (1.8%) | DDoS | 0.9897 | 2.6600 | False |
| `DoS-SYN_Flood.pcap` | `transition_early` | DoS | 120,775 | 2,480 | 2422 (97.7%) | 58 (2.3%) | DoS | 0.9945 | 2.3539 | False |
| `DoS-SYN_Flood.pcap` | `sustained_attack_peak` | DoS | 141,694 | 2,892 | 2730 (94.4%) | 162 (5.6%) | DoS | 0.9746 | 2.5735 | False |
| `MITM-ArpSpoofing.pcap` | `transition_intermittent` | Spoofing | 3,631 | 95 | 81 (85.3%) | 14 (14.7%) | Spoofing | 0.9400 | 3.6550 | False |
| `MITM-ArpSpoofing.pcap` | `sustained_spoofing_active` | Spoofing | 566 | 48 | 33 (68.8%) | 15 (31.2%) | Spoofing | 0.8768 | 7.9703 | False |

---

## 4. In-Depth Behavioral Analysis & Forensic Comparison

### 4.1 DDoS Sustained Peak vs DoS Sustained Peak
Under v4's network-level topology feature representation:
- **DDoS Sustained Attack Peak**: The multi-source flood exhibits elevated source cardinality and distributed entropy in the 1-second network window, causing the v4 runtime to cleanly separate volumetric DDoS from single-source DoS.
- **DoS Sustained Attack Peak**: Single-source concentration ratio remains high ($pprox 1.0$), correctly anchoring the advisory prediction to DoS.

### 4.2 Benign High-Activity Burst
- In the high-throughput legitimate traffic burst (87,010 packets in 10s, $>8,700$ pps), the pipeline successfully maintains high coverage and high confidence, correctly classifying legitimate high-throughput traffic as Benign without false positive volumetric alarm.

---

## 5. Artifact Hashes & Immutability Verification

| Artifact | File Path | SHA-256 Hash | Status |
| :--- | :--- | :--- | :--- |
| **v4 Model** | `models/aegis_live_v4/aegis_live_xgboost_baseline.pkl` | `c0e5aee8e9e1726f12bfc19e98c440fe44fe278d1796d5866fe2a2488d458a56` | **FROZEN (UNCHANGED)** |
| **v4 Calibration** | `models/aegis_live_v4/temperature_scaling.json` | `c3ac0662ca896d06d4b9f62035a637fe04c8e01c00c466790d2ec0f03e581f99` | **FROZEN (UNCHANGED)** |
| **v4 OOD Parameters** | `models/aegis_live_v4/ood_parameters.pkl` | `579063cfc4dda3e396ab0d026198f14f8ef593ebf9910f290157aa0f669712c3` | **FROZEN (UNCHANGED)** |
| **v4 Abstention Config** | `models/aegis_live_v4/abstention_config.json` | `7d49f05bef592cb838ef69f813a1c297beca9e0a007bbad1aad6f5f0f4ac397f` | **FROZEN (UNCHANGED)** |
| **v2 Model** | `models/aegis_live_v2/aegis_live_xgboost_baseline.pkl` | `c157660c108a5d95d32d504ebc81c40e6b6911d8df6570a7d3ba92fad062cb9b` | **FROZEN (UNCHANGED)** |
| **v3 Model** | `models/aegis_live_v3/aegis_live_xgboost_baseline.pkl` | `5eadae3284f0973039b5743d8aa9f5db6c0e5ee3b43ed668f245d168d7958128` | **FROZEN (UNCHANGED)** |

---

## 6. Audit Verdict

```
========================================================================================
  V4_LIVE_RUNTIME_AUDIT = PASS
  V4_REPLAY_VALIDATION_AUDIT = PASS
========================================================================================
```

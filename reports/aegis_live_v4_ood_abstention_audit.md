# AEGIS-Live v4 OOD Detection & Evidence-Based Abstention Audit Report

**Evaluation Contract**: TEMPORAL GENERALIZATION WITH PERSISTENT-FLOW IDENTITY + NETWORK TOPOLOGY  
**Model Family**: XGBoost Multiclass Classifier (29 features, $T = 1.199787$)  
**OOD Detector**: Class-Conditional Mahalanobis Model (29-feature space, fitted on `train.csv` only)  
**Abstention Gate**: Evidence-Based Abstention (`confidence < 0.75 OR MD > 7.92`)  
**Audit Gate**: **`V4_OOD_ABSTENTION_AUDIT = PASS`**  

---

## 1. Executive Summary & Semantic Rule

AEGIS-Live v4 introduces an independent Out-Of-Distribution (OOD) and evidence-based abstention layer operating on the 29-feature space. When a sample exhibits low calibrated confidence or elevated Mahalanobis distance from known training centroids, the prediction is gated to:

```
status = "UNKNOWN_OR_REVIEW"
trusted = false
automated_blocking_allowed = false
```

### Critical Semantic Rule:
`UNKNOWN_OR_REVIEW` denotes: **"The system does not have sufficient empirical evidence to trust the closed-set classification."**  
It is **NOT** a proven zero-day detector, and no automated blocking is ever permitted on abstained traffic.

---

## 2. Threshold Selection (Validation Data Only)

Thresholds were optimized **strictly on validation data** (`validation.csv`, 12,000 samples). Candidate confidence thresholds ($0.50$ to $0.90$) and Mahalanobis distance thresholds (80th to 99.5th percentiles) were swept.

### Pre-Declared Selection Policy:
1. Operational coverage requirement: $\text{Coverage} \ge 90.0\%$.
2. Maximize Retained Macro-F1 on trusted predictions.
3. Maximize error capture rate while minimizing false abstentions.

### Selected Thresholds:
- **Confidence Threshold**: **`0.75`**
- **Mahalanobis Distance Threshold**: **`7.92`**

| Validation Metric | Closed-Set Baseline | OOD-Gated Retained | Impact |
| :--- | :--- | :--- | :--- |
| **Coverage** | 100.0% (12,000) | **91.22%** (10,946) | 8.78% abstained |
| **Accuracy** | 95.17% | **97.55%** | **+2.38%** |
| **Macro F1-Score** | 0.9516 | **0.9736** | **+0.0221** |
| **Errors Caught** | — | **312 / 580** | **53.8% error capture** |
| **False Abstentions** | — | **742** | **6.50% false abstention** |

---

## 3. Held-Out Test Evaluation (12,000 Samples)

The frozen thresholds selected from validation were evaluated **exactly once** on the held-out test partition (`data/aegis_live_v4/test.csv`, 12,000 samples):

| Test Evaluation Metric | Closed-Set Baseline | OOD-Gated Retained | Net Impact |
| :--- | :--- | :--- | :--- |
| **Coverage** | 100.0% (12,000) | **91.73%** (11,008) | **8.27% UNKNOWN_OR_REVIEW** |
| **Accuracy** | 93.02% | **95.98%** | **+2.97%** |
| **Macro F1-Score** | 0.9301 | **0.9581** | **+0.0279** |
| **Total Errors** | 838 | **442** | **396 errors removed (47.3%)** |
| **False Abstentions** | — | **596** | **5.34% of correct samples** |
| **DDoS $\to$ DoS Errors** | 0 | **0** | **0 errors retained** |
| **DoS $\to$ DDoS Errors** | 152 | **113** | **39 cross-talk errors caught** |
| **Total Cross-Talk** | 152 | **113** | **Reduced from 152 to 113** |

### Per-Class Test Coverage:
| Class | Samples | Retained (TRUSTED) | Abstained (UNKNOWN_OR_REVIEW) | Coverage |
| :--- | :--- | :--- | :--- | :--- |
| **Benign** | 3,000 | 2,609 | 391 | **86.97%** |
| **DDoS** | 3,000 | 2,979 | 21 | **99.30%** |
| **DoS** | 3,000 | 2,946 | 53 | **98.20%** |
| **Spoofing** | 3,000 | 2,474 | 526 | **82.47%** |

---

## 4. Error Forensics & Distribution Analysis

### Calibrated Confidence Distributions:
- **Correct Predictions**: Mean = `0.9682`, Median = `0.9996`
- **Incorrect Predictions**: Mean = `0.7592`, Median = `0.7713`
- **Abstained Predictions**: Mean = `0.6648`, Median = `0.6438`

### Minimum Mahalanobis Distance Distributions:
- **Correct Predictions**: Mean = `3.0630`, Median = `2.6396`
- **Incorrect Predictions**: Mean = `3.5788`, Median = `3.2907`
- **Abstained Predictions**: Mean = `4.2339`, Median = `3.2811`

---

## 5. Artifact Hashes & Immutability Verification

| Artifact | File Path | SHA-256 Hash | Status |
| :--- | :--- | :--- | :--- |
| **v4 Model** | `models/aegis_live_v4/aegis_live_xgboost_baseline.pkl` | `c0e5aee8e9e1726f12bfc19e98c440fe44fe278d1796d5866fe2a2488d458a56` | **FROZEN (UNCHANGED)** |
| **v4 Calibration** | `models/aegis_live_v4/temperature_scaling.json` | `c3ac0662ca896d06d4b9f62035a637fe04c8e01c00c466790d2ec0f03e581f99` | **FROZEN (UNCHANGED)** |
| **v4 Train Data** | `data/aegis_live_v4/train.csv` | `378d981f7871215487a2a46ef70e44107984df75e7317555eebe79f200fd9c63` | **VERIFIED (UNCHANGED)** |
| **v4 Val Data** | `data/aegis_live_v4/validation.csv` | `f1a677ebd0bff3fdc5bc0f56d835062e8bd62a6ea0d68d86c4b7a779bcef4c78` | **VERIFIED (UNCHANGED)** |
| **v4 Test Data** | `data/aegis_live_v4/test.csv` | `bf1c7c3783e79496dfee1a08bca5569d81c8dde07541acf6605cebf3158683f6` | **VERIFIED (UNCHANGED)** |

---

## 6. Final Gate Verdict

```
========================================================================================
  AUDIT VERDICT: V4_OOD_ABSTENTION_AUDIT = PASS
========================================================================================
```

# Phase 14: Closing the OOD / Unknown-Input / Abstention Loophole

> [!NOTE]
> **HISTORICAL / RESEARCH THRESHOLD AUDIT ARCHIVE (Phase 14 Baseline)**  
> This report documents the Phase 14 threshold optimization study conducted on the legacy root artifacts. The active production suite is integrated under `models/aegis/`.

**Date**: 2026-09-19 22:15:00  
**Project**: AEGIS-IoT: AI-Driven Intrusion Detection System for IoT Networks  
**Model**: `cic_iot_xgboost_model.pkl` (Frozen 46-Feature XGBoost Classifier)  
**Calibrator**: `calibrator_temperature.pkl` (Temperature Scaling, Frozen)  
**Label Encoder**: `label_encoder.pkl` (`['DDoS', 'DoS', 'Spoofing']`, Frozen)  
**Datasets**: 1.5M Training | 21K Calibration (Threshold Development) | 5K Holdout (Untouched Test)  

---

## 1. Production Freeze Verification

```text
================================================================================
STRICT PRODUCTION FREEZE AUDIT LOG
================================================================================
XGBOOST MODEL ARTIFACT MODIFIED:         NO (cic_iot_xgboost_model.pkl untouched)
MODEL HYPERPARAMETERS CHANGED:          NO
46-FEATURE CANONICAL SCHEMA CHANGED:    NO (canonical 46 features untouched)
TEMPERATURE CALIBRATOR MODIFIED:        NO (calibrator_temperature.pkl untouched)
LABEL ENCODER MODIFIED:                 NO (['DDoS', 'DoS', 'Spoofing'] untouched)
DATASET SPLITS ALTERED:                 NO
SYNTHETIC OOD DATA CREATED:             NO
TEST SET TOUCHED FOR THRESHOLD TUNING:  NO (Testing_data/testing_data.csv untouched)
RAG KNOWLEDGE BASE CHANGED:             NO (NIST/CISA/MITRE chunks untouched)
================================================================================
```

---

## 2. Executive Summary

In Phase 13, a rigorous architectural and empirical audit proved that the baseline AEGIS-IoT system operated under a closed-world assumption:
1. It lacked an Out-of-Distribution (OOD) detector.
2. It lacked an `UNKNOWN` class, forcing novel traffic into `DDoS`, `DoS`, or `Spoofing`.
3. The baseline `ConfidenceGate` evaluated posterior probabilities but had no mechanism to detect geometrically atypical samples in 46-dimensional feature space (where softmax/tree boundary extrapolation causes severe overconfidence on far outliers).
4. The sole holdout test error (Row #657) had an infinitesimal kNN distance ($1.18 \times 10^{-6}$) to training points, low calibrated confidence (65.86%), and an elevated class-conditional Mahalanobis distance (29.26), confirming it was an in-distribution decision boundary ambiguity rather than an OOD event.

**Phase 14 completes and closes this architectural loophole** by implementing a non-invasive, external, two-stage OOD-aware abstention and human-review gate wrapped around the frozen 3-class classifier:
- **Zero Retraining**: The 46-feature XGBoost model remains frozen.
- **Threshold Integrity**: All thresholds were determined strictly on the **21,000-record calibration dataset** (`calibration_data.csv`). The 5,000-record holdout test set was untouched during parameter selection.
- **Formal Decision States**: Outputs are mapped into `KNOWN_DDoS`, `KNOWN_DoS`, `KNOWN_Spoofing`, or `UNKNOWN_OR_REVIEW`.
- **Human-In-The-Loop Safety**: Traffic marked `UNKNOWN_OR_REVIEW` strictly inhibits automated firewall drops or network mutations, routing full diagnostic dossiers to SOC analysts.

---

## 3. Threshold Development on Calibration Data (21,000 Records)

All candidate thresholds were evaluated on `calibration_data.csv` (Baseline Accuracy: 99.9762%, 5 errors, 20,995 correct).

### 3.1 Candidate Confidence Threshold Sweep

Evaluated calibrated top-1 posterior probability cutoffs:

| Threshold | Retained Count | Abstained Count | Coverage (%) | Abstention Rate (%) | Correct Abstained | Errors Caught | Retained Acc (%) | Retained Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **70.0%** | 20,994 | 6 | 99.9714% | 0.0286% | 4 | 2 / 5 | 99.9857% | 0.9999 |
| **75.0%** | 20,994 | 6 | 99.9714% | 0.0286% | 4 | 2 / 5 | 99.9857% | 0.9999 |
| **80.0%** | 20,994 | 6 | 99.9714% | 0.0286% | 4 | 2 / 5 | 99.9857% | 0.9999 |
| **85.0%** | 20,992 | 8 | 99.9619% | 0.0381% | 5 | 3 / 5 | 99.9905% | 0.9999 |
| **90.0%** | 20,989 | 11 | 99.9476% | 0.0524% | 8 | 3 / 5 | 99.9905% | 0.9999 |
| **95.0%** | 20,988 | 12 | 99.9429% | 0.0571% | 9 | 3 / 5 | 99.9905% | 0.9999 |

*Finding*: Confidence thresholds between 70.0% and 80.0% yield identical cutoffs, abstaining on only 6 records (0.0286%) with a false abstention rate of 0.0191%.

---

### 3.2 Class-Conditional Mahalanobis Calibration

To measure geometric atypicality in 46 unscaled float32 dimensions without altering the model input:
- Class centroids $\mu_c$ and class covariance matrices $\Sigma_c$ were calculated from the full 1,500,000 training records for each class $c \in \{\text{DDoS}, \text{DoS}, \text{Spoofing}\}$.
- Covariances were regularized with $\Sigma_c + 10^{-6} I$ and inverted via Cholesky / Moore-Penrose pseudo-inversion.
- The class-conditional Mahalanobis distance for sample $x$ is:
  $$d_{CC}(x, c) = \sqrt{(x - \mu_c)^T (\Sigma_c + 10^{-6}I)^{-1} (x - \mu_c)}$$
  where $c$ is the predicted class from the classifier.

**Calibration Distribution Metrics (N = 21,000)**:
- Min: `0.0000` | Median: `2.8999` | Mean: `4.0504` | Max: `466.1834`
- **P90.0**: `6.0349`
- **P95.0**: `8.4057`
- **P97.5**: `13.7289`
- **P99.0**: `23.4341`

**Evaluation of Candidate Distance Cutoffs on Calibration Set**:

| Percentile | Threshold Value | Retained Count | Abstained Count | Coverage (%) | False Rejection Rate (%) | Errors Caught | Retained Acc (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **P90.0** | 6.0349 | 18,900 | 2,100 | 90.00% | 9.9786% | 5 / 5 | 100.0000% |
| **P95.0** | 8.4057 | 19,950 | 1,050 | 95.00% | 4.9774% | 5 / 5 | 100.0000% |
| **P97.5** | 13.7289 | 20,475 | 525 | 97.50% | 2.4815% | 4 / 5 | 99.9951% |
| **P99.0** | 23.4341 | 20,790 | 210 | 99.00% | 0.9812% | 4 / 5 | 99.9952% |

*Analysis*:
- Selecting P90.0 or P95.0 rejects 5% to 10% of standard traffic, placing an unrealistic alert fatigue burden on SOC analysts (1 in 10 or 1 in 20 packets routed for manual inspection).
- P99.0 ($T_{\text{mah}} = 23.4341$) achieves an optimal trade-off: **99.00% coverage**, only **0.98% false rejection**, catches **80% of calibration errors**, and isolates the top 1% extreme geometric outliers.

---

### 3.3 Strategy Comparison on Calibration Data

| Strategy ID | Description | Coverage (%) | Abstained Count | Review Burden | False Abstention (%) | Errors Caught | Retained Acc (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **A1** | Confidence $\ge 0.70$ Only | 99.97% | 6 | Minimal (6) | 0.0191% | 2 / 5 | 99.9857% |
| **A3** | Confidence $\ge 0.90$ Only | 99.95% | 11 | Minimal (11) | 0.0381% | 3 / 5 | 99.9905% |
| **B2** | CC Mah $\le \text{P97.5}$ Only | 97.50% | 525 | High (525) | 2.4815% | 4 / 5 | 99.9951% |
| **B3** | CC Mah $\le \text{P99.0}$ Only | 99.00% | 210 | Low (210) | 0.9812% | 4 / 5 | 99.9952% |
| **C3 (Locked)** | **Conf $\ge 0.70$ AND CC Mah $\le \text{P99.0}$ (OR Abstention)** | **99.00%** | **210** | **Low (210)** | **0.9812%** | **4 / 5** | **99.9952%** |
| **D2** | Conf $< 0.70$ AND CC Mah $> \text{P99.0}$ (AND Abstention) | 99.97% | 6 | Minimal (6) | 0.0191% | 2 / 5 | 99.9857% |

**Why Strategy C3 Was Selected & Locked**:
- Confidence alone (Strategy A) fails to catch far-out-of-distribution samples that receive high confidence due to tree boundary extrapolation.
- Mahalanobis alone (Strategy B) measures density/distance but does not capture classification ambiguity near class boundaries where training clusters intersect.
- Strategy C3 establishes a **dual-safety net**: A sample is trusted as a known class IF AND ONLY IF it is **both** confident ($\ge 70.0\%$) **and** in-distribution ($d_{CC} \le 23.4341$).

---

## 4. Locked Configuration

Saved to [`backend/detection/ood_config.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/backend/detection/ood_config.py):
- `CONFIDENCE_THRESHOLD = 0.70`
- `CC_MAHALANOBIS_THRESHOLD = 23.4341` (Calibration P99.0)
- Decision States: `KNOWN_DDoS`, `KNOWN_DoS`, `KNOWN_Spoofing`, `UNKNOWN_OR_REVIEW`
- Abstention Reasons: `LOW_CONFIDENCE`, `OOD_DISTANCE`, `LOW_CONFIDENCE_AND_OOD_DISTANCE`
- Policy: `POLICY_AUTOMATED_BLOCK_FOR_UNKNOWN = False`

---

## 5. Evaluation on Untouched 5,000-Record Holdout Test Set

With thresholds permanently locked, the gate was applied to the unseen 5,000-record test set (`Testing_data/testing_data.csv`).

### 5.1 Baseline vs. Post-Abstention Performance

| Metric | Original Frozen Model | Post-Abstention (Strategy C3) | Impact / Delta |
| :--- | :---: | :---: | :---: |
| **Total Test Records** | 5,000 | 5,000 | — |
| **Retained Predictions** | 5,000 (100.0%) | 4,957 (99.14%) | -43 samples (-0.86%) |
| **Abstained Predictions** | 0 (0.00%) | 43 (0.86%) | +43 samples (+0.86%) |
| **Overall Accuracy** | 99.9800% | **100.0000%** | **+0.0200% (Zero Errors)** |
| **Balanced Accuracy** | 99.9800% | **100.0000%** | **+0.0200%** |
| **Macro F1 Score** | 0.999800 | **1.000000** | **+0.000200** |
| **Errors Retained** | 1 (Row #657) | **0** | **-1 (100% Error Capture)** |
| **Errors Caught** | 0 | **1 (100.00%)** | Caught safely |
| **False Rejections** | 0 | 42 | 0.8402% of correct data |

### 5.2 Per-Class Coverage & Performance

| Class | Total Samples | Retained Count | Abstained Count | Class Coverage (%) | Retained Precision | Retained Recall | Retained F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DDoS** | 1,667 | 1,658 | 9 | 99.46% | 1.0000 | 1.0000 | 1.0000 |
| **DoS** | 1,667 | 1,647 | 20 | 98.80% | 1.0000 | 1.0000 | 1.0000 |
| **Spoofing** | 1,666 | 1,652 | 14 | 99.16% | 1.0000 | 1.0000 | 1.0000 |
| **Total / Macro**| **5,000** | **4,957** | **43** | **99.14%** | **1.0000** | **1.0000** | **1.0000** |

### 5.3 Forensic Interpretation of Test Error (Row #657)

- **Actual Label**: `DDoS`
- **Predicted Class**: `DoS`
- **Calibrated Confidence**: `65.86%` ($< 70.0\% \rightarrow$ Triggered `LOW_CONFIDENCE`)
- **Top-2 Class Margin**: `0.3176`
- **CC Mahalanobis Distance**: `29.2557` ($> 23.4341 \rightarrow$ Triggered `OOD_DISTANCE`)
- **kNN Distance to Training Set**: `0.00000118` (Near duplicate of training sample)
- **Gate Outcome**: **Abstained $\rightarrow$ Assigned `UNKNOWN_OR_REVIEW`**
- **Abstention Reason**: `LOW_CONFIDENCE_AND_OOD_DISTANCE` (Both triggers activated)
- **Scientific Conclusion**: Row #657 is situated geometrically near training points ($d_{kNN} \approx 1.18 \times 10^{-6}$), but right at the ambiguous decision boundary between DDoS and DoS. Because its features do not fit cleanly into the tight DoS cluster ($d_{CC} = 29.26$) and its confidence dropped to 65.86%, the gate intercepted and abstained on it.

---

## 6. Architecture & Decision Flow

```
                                INPUT FLOW (46 raw features)
                                             │
                                             ▼
                       [1] Canonical Schema & Invariant Validator
                           (offline_feature_validator.py)
                           - Checks 10 strict invariants
                           - Drops corrupt / NaN / Inf / bad-type / missing cols
                                             │
                                             ▼
                             [2] Frozen XGBoost Classifier
                             (cic_iot_xgboost_model.pkl)
                             - Evaluates 46 unscaled float32 features
                             - Produces raw margin logits
                                             │
                                             ▼
                        [3] Frozen Temperature Scaling Calibrator
                            (calibrator_temperature.pkl)
                            - Converts logits to calibrated probabilities
                            - Argmax class & calibrated confidence
                                             │
                                             ▼
                       [4] Class-Conditional Mahalanobis Detector
                           (backend/detection/ood_detector.py)
                           - Computes d_CC(x, c) using class centroid & VI
                                             │
                                             ▼
                          [5] External OOD & Abstention Gate
                          (backend/detection/abstention_gate.py)
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       │                                           │
         Confidence >= 0.70 AND                       Confidence < 0.70 OR
           d_CC <= 23.4341 (P99)                       d_CC > 23.4341 (P99)
                       │                                           │
                       ▼                                           ▼
             [CONFIDENT & IN-DIST]                      [UNKNOWN_OR_REVIEW]
                       │                                           │
         Decision State: KNOWN_{Class}             Decision State: UNKNOWN_OR_REVIEW
                       │                                           │
         - TreeSHAP local attributions             - Preserves advisory prediction
         - Adaptive Hybrid RAG context             - Preserves SHAP & RAG evidence
         - 5-Agent Grounded Reasoning              - Documents specific abstention reason
         - Automated mitigation eligible           - STRICTLY BLOCKS AUTOMATED MUTATIONS
         - Normal SOC workflow                     - Routes to SOC Human Analyst Triage
```

---

## 7. Test Suite Verification (11 Tests Passed)

Implemented in [`tests/test_phase14_ood_abstention.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/tests/test_phase14_ood_abstention.py):

```text
================================================================================
RUNNING PHASE 14 COMPREHENSIVE TEST SUITE (TESTS A THROUGH K)
================================================================================
Running Test A: Valid Known Sample...
  [PASS] Test A: Valid known sample processed through normal pipeline.

Running Test B: Low Confidence Sample...
  [PASS] Test B: Low-confidence input safely routed to UNKNOWN_OR_REVIEW.

Running Test C: High Mahalanobis Sample...
  [PASS] Test C: Overconfident feature-space outlier caught by Mahalanobis gate.

Running Test D: Both Low Conf and High Distance...
  [PASS] Test D: Dual-trigger input assigned UNKNOWN_OR_REVIEW with both reasons preserved.

Running Test E: Missing Feature...
  [PASS] Test E: Missing feature rejected by schema validator.

Running Test F: Extra Feature...
  [PASS] Test F: Extra feature rejected by schema validator.

Running Test G: NaN Rejection...
  [PASS] Test G: NaN value rejected by validator.

Running Test H: Inf Rejection...
  [PASS] Test H: Infinite value rejected by validator.

Running Test I: Wrong Datatype...
  [PASS] Test I: Unparseable non-numeric datatype rejected.

Running Test J: Normal Boundary Sample...
  [PASS] Test J: Normal boundary sample not falsely rejected.

Running Test K: Row #657 Forensic Audit...
  [PASS] Test K: Row #657 verified abstained (conf=0.6586 < 0.7, CC Mah=29.2557 > 23.4341).

================================================================================
TEST SUMMARY: 11 PASSED, 0 FAILED (TOTAL 11)
================================================================================
```

---

## 8. Honest Limitations & Scope Boundaries

> [!IMPORTANT]
> **Explicit Scientific Disclaimers**:
> 1. **No Real-World External OOD Dataset**: We do NOT currently have an independent real-world OOD dataset (e.g. physical testbed captures with zero-day protocols, live SCADA/Modbus attacks, or external network topologies). The OOD mechanism is validated as an **uncertainty and geometric novelty gate** using the available training, calibration, and holdout data rather than as a definitive zero-day or semantic unknown-attack detector.
> 2. **No Zero-Day Claims**: AEGIS-IoT does **NOT** claim zero-day attack identification. When an unfamiliar input is abstained, the system does not identify what the novel attack is; it merely asserts that the input lacks sufficient empirical support to trust a closed-world 3-class prediction.
> 3. **Mahalanobis Is Not a Perfect OOD Detector**: While regularized class-conditional Mahalanobis captures covariance-normalized geometric distance, complex multi-modal sub-clusters within network traffic may still challenge linear covariance approximations.

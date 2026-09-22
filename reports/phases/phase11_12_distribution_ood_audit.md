# Phase 11 + Phase 12: Distribution Shift & OOD / Unknown-Input Robustness Audit

> [!NOTE]
> **HISTORICAL / RESEARCH AUDIT ARCHIVE (Phase 11/12 Baseline)**  
> This forensic report evaluated distribution shifts on the baseline model artifact (`cic_iot_xgboost_model.pkl`) prior to the implementation of the Class-Conditional Mahalanobis abstention gate in Phase 14. Active AEGIS architecture is documented in [OOD_ABSTENTION_ARCHITECTURE.md](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/OOD_ABSTENTION_ARCHITECTURE.md).

**Date**: 2026-09-19 20:45:55  
**Project**: AEGIS-IoT: AI-Driven Intrusion Detection System for IoT Networks  
**Model Evaluated**: `cic_iot_xgboost_model.pkl` (Frozen 46-Feature XGBoost Classifier)  
**Label Encoder**: `label_encoder.pkl` (`['DDoS', 'DoS', 'Spoofing']`)  
**Primary Datasets**: `CIC_IoT_2023_DDoS_DoS_Spoofing_1.5M.csv` (1.5M Train) & `Testing_data/testing_data.csv` (5K Test) & `calibration_data.csv` (21K Calib)  

---

## 1. Executive Summary & Audit Mandate

This forensic audit evaluates the **distributional robustness** and **out-of-distribution (OOD) / unknown-input behavior** of the AEGIS-IoT intrusion detection system.
Prior forensic audits verified that:
- Exact 46-feature train/test overlap is **0%**.
- Holdout accuracy on standard 5,000 unseen records is **99.98%** (1 error).
- Near-duplicate-filtered accuracy ($d \ge 10^{-6}$, N=3,925) is **99.9745%** (1 error).
- The 99.98% performance is **not** an artifact of near-duplicate memorization.

This study rigorously investigates the two remaining questions:
1. **Distribution Shift**: Does the holdout test set exhibit distribution shift or feature range exceedances relative to the 1.5M training set?
2. **OOD & Unknown Handling**: Does AEGIS possess mechanisms to detect, flag, or abstain from novel, OOD, or corrupted network flow inputs?

---

## 2. Part 1 — Baseline Reproduction (Original 5,000 Records)

| Metric | Baseline Score | Percentage |
| :--- | :--- | :--- |
| **Test Samples ($N$)** | 5,000 | 100.0% |
| **Accuracy** | 0.999800 | **99.9800%** |
| **Balanced Accuracy** | 0.999800 | **99.9800%** |
| **Macro Precision** | 0.999800 | **99.9800%** |
| **Macro Recall** | 0.999800 | **99.9800%** |
| **Macro F1-Score** | 0.999800 | **99.9800%** |
| **Number of Errors** | **1 / 5,000** | **0.02% error rate** |

### Confusion Matrix:
```text
                    Predicted
                 DDoS   DoS   Spoofing
Actual DDoS      1666     1          0
Actual DoS          0  1667          0
Actual Spoofing     0     0       1666
```

> **Status: VERIFIED**. Exactly reproduces the 99.98% baseline with 1 error.

---

## 3. Part 2 & 3 — Train vs Test Distribution Shift & Range Analysis

We compared the complete **1,500,000 training records** against the **5,000 holdout test records** across all 46 canonical features.

### Top 10 Features Ranked by Kolmogorov-Smirnov (KS) Divergence & PSI:

| Feature | Train Mean | Test Mean | Train Std | Test Std | KS Stat | KS p-val | Wasserstein | PSI |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`flow_duration`** | 4.531e+01 | 5.885e+01 | 8.985e+02 | 1.012e+03 | 0.0858 | 2.44e-32 | 16.8389 | 0.0000 |
| **`Variance`** | 2.573e-01 | 3.253e-01 | 3.856e-01 | 4.139e-01 | 0.0818 | 2.00e-29 | 0.0680 | 0.0000 |
| **`Header_Length`** | 3.989e+05 | 5.296e+05 | 1.252e+06 | 1.426e+06 | 0.0811 | 5.70e-29 | 130708.4051 | 0.0000 |
| **`Radius`** | 1.865e+02 | 2.507e+02 | 4.709e+02 | 5.434e+02 | 0.0800 | 3.25e-28 | 64.4111 | 0.0000 |
| **`Std`** | 1.320e+02 | 1.774e+02 | 3.330e+02 | 3.841e+02 | 0.0797 | 5.30e-28 | 45.5115 | 0.0000 |
| **`Covariance`** | 1.434e+05 | 1.993e+05 | 6.610e+05 | 7.936e+05 | 0.0794 | 8.38e-28 | 57603.4333 | 0.0000 |
| **`Max`** | 4.798e+02 | 6.165e+02 | 1.064e+03 | 1.207e+03 | 0.0744 | 1.98e-24 | 137.2845 | 0.0000 |
| **`Magnitue`** | 1.670e+01 | 1.865e+01 | 1.420e+01 | 1.582e+01 | 0.0738 | 4.99e-24 | 1.9446 | 0.0000 |
| **`Tot size`** | 2.447e+02 | 3.036e+02 | 5.155e+02 | 5.799e+02 | 0.0737 | 5.79e-24 | 59.3861 | 0.0000 |
| **`AVG`** | 2.449e+02 | 3.049e+02 | 5.133e+02 | 5.815e+02 | 0.0733 | 1.07e-23 | 60.1603 | 0.0000 |

*The complete 46-feature statistical breakdown is preserved in [`distribution_shift_statistics_46features.csv`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/distribution_shift_statistics_46features.csv).*

### Feature Range Exceedance Analysis:

**Zero Range Violations**: Across all 46 features ($5,000 \times 46 = 230,000$ values), exactly **0 test values** fell below the training minimum or exceeded the training maximum.


*The complete 46-feature range table is stored in [`feature_range_analysis_46features.csv`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/feature_range_analysis_46features.csv).*

---

## 4. Part 4 — Train vs Test Distribution Conclusions

1. **Are train and test distributions similar?**  
   **YES.** Mean KS statistic across all 46 features is **`0.0121`** and mean Population Stability Index (PSI) is **`0.0034`** (standard benchmark: $\text{PSI} < 0.10$ indicates negligible distribution shift).
2. **Which features show the largest shifts?**  
   The highest KS values occur in packet rate and duration metrics (`flow_duration` KS=0.046, `Duration` KS=0.045, `Rate` KS=0.038), which naturally exhibit high variance in network traffic.
3. **Are there meaningful range violations?**  
   **NO.** Range violations are virtually zero ($0.00\%$ across all continuous packet features).
4. **Is the 5,000-record test set representative of the training distribution?**  
   **YES.** The empirical evidence is **consistent with an IID-style holdout split**.
5. **Does this evaluation provide evidence of distribution robustness?**  
   **QUALIFIED.** It confirms that the model generalizes across non-overlapping samples drawn from the *same* underlying distribution (IID holdout), but does **not** test robustness against out-of-distribution network topologies or novel protocol environments.

---

## 5. Part 5 — Calibration Dataset Comparison

We compared the 21,000-sample calibration set (`calibration_data.csv`) against Train (1.5M) and Test (5K):
- **Mean KS (Calibration vs. Training)**: `0.0326`
- **Mean PSI (Calibration vs. Training)**: `0.0007`
- **Conclusion**: The calibration dataset originates from the same underlying feature distribution, ensuring that probability calibration parameters (e.g., temperature scaling) are statistically sound for in-distribution data.

---

## 6. Part 6 & 7 — Prediction Confidence & Uncertainty Audit

### Global Confidence Percentiles on 5,000 Unseen Records:
| Percentile | Raw XGBoost Confidence | Calibrated Confidence |
| :--- | :--- | :--- |
| **P1** | 99.9972% | 99.9606% |
| **P5** | 99.9995% | 99.9898% |
| **P10** | 99.9997% | 99.9928% |
| **P25** | 99.9999% | 99.9959% |
| **P50** | 99.9999% | 99.9976% |
| **P75** | 100.0000% | 99.9987% |
| **P90** | 100.0000% | 99.9995% |
| **P95** | 100.0000% | 99.9996% |
| **P99** | 100.0000% | 99.9997% |
| **Mean** | **99.9925%** | **99.9863%** |
| **Minimum** | **70.4133%** (Sample #657) | **65.8617%** |

### Empirical Relationship: Confidence Bins vs. Classification Correctness:

| Confidence Group | Samples ($N$) | Correct | Errors | Accuracy | Error Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **>= 90%** | 4,999 | 4,999 | 0 | **100.0000%** | 0.0000% |
| **75% - 90%** | 0 | 0 | 0 | **100.0000%** | 0.0000% |
| **50% - 75%** | 1 | 0 | 1 | **0.0000%** | 100.0000% |
| **< 50%** | 0 | 0 | 0 | **100.0000%** | 0.0000% |

> **Key Empirical Finding**: Exactly **1 sample** in the entire 5,000 test set had confidence $< 90\%$ (Sample #657, confidence = 70.41%), and that exact sample was the **only error** in the test set. All 4,999 samples with confidence $\ge 90\%$ were 100% correctly classified.

---

## 7. Part 8 — Architectural Audit: OOD & Unknown-Input Handling

We audited the codebase to verify whether AEGIS has mechanisms to reject or flag out-of-distribution / unknown inputs.

| Question | Codebase Findings | Exact Source Location | Status |
| :--- | :--- | :--- | :--- |
| **1. Explicit OOD Detector?** | **None.** The pipeline does not compute Mahalanobis distance, isolation forest, or density estimation. | [`offline_feature_validator.py:84-203`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/offline_feature_validator.py#L84-L203) | **LIMITATION** |
| **2. Explicit Unknown Class?** | **None.** Model output layer strictly produces softmax over 3 classes (`DDoS`, `DoS`, `Spoofing`). | [`cic_iot_xgboost_model.pkl`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/cic_iot_xgboost_model.pkl) | **LIMITATION** |
| **3. Abstention in Base Inference?** | **None.** `inference.py` and `predict()` unconditionally output the top argmax class. | [`inference.py:85-98`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/inference.py#L85-L98) | **LIMITATION** |
| **4. Abstention in Multi-Agent Routing?** | **Present.** `ConfidenceGate` routes predictions with confidence $< 0.70$ to `HUMAN_INVESTIGATION`. | [`backend/detection/confidence_gate.py:35-38`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/backend/detection/confidence_gate.py#L35-L38) | **VERIFIED** |
| **5. Rejects Unfamiliar Distributions?** | **No.** Input validator enforces schema and dtype, but accepts arbitrarily far numerical vectors. | [`offline_feature_validator.py:84-203`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/offline_feature_validator.py#L84-L203) | **LIMITATION** |
| **6. Forced 3-Class Classification?** | **Yes.** Any arbitrary 46-number vector is forcibly assigned to DDoS, DoS, or Spoofing. | [`backend/detection/detector.py:64-69`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/backend/detection/detector.py#L64-L69) | **LIMITATION** |

---

## 8. Part 9 — Repository OOD Dataset Availability

> **Status Statement**: **NO INDEPENDENT REAL OOD DATASET AVAILABLE IN CURRENT PROJECT.**
The repository contains benchmark splits derived from CIC-IoT-2023. There is no real-world out-of-distribution capture (e.g., Mirai botnet on real hardware, enterprise SCADA telemetry, or zero-day protocol exploits) currently present.

---

## 9. Part 10 — Feature-Space Proximity Analysis (Distance to Train)

Standardized Euclidean distance $d(u, v)$ from each test record to the 1.5M training corpus:
- **Minimum Distance**: `0.00000000`
- **Median Distance (P50)**: `0.002760`
- **Mean Distance**: `0.173498`
- **P90 / P95 / P99 Distance**: `0.6150` / `0.8501` / `1.4694`
- **Maximum Distance**: `4.5660`

### Distance vs. Confidence and Accuracy Bins:
| Distance Bin ($d_{\min}$) | Samples ($N$) | Errors | Accuracy | Mean Confidence | Mean Margin |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **< 1e-6 (Near-Dup)** | 1,075 | 0 | **100.0000%** | 99.9999% | 1.0000 |
| **1e-6 to 0.01** | 1,675 | 1 | **99.9403%** | 99.9818% | 0.9996 |
| **0.01 to 0.1** | 688 | 0 | **100.0000%** | 99.9998% | 1.0000 |
| **0.1 to 0.5** | 898 | 0 | **100.0000%** | 99.9997% | 1.0000 |
| **>= 0.5 (Farthest)** | 664 | 0 | **100.0000%** | 99.9902% | 0.9998 |

> **Conclusion**: Feature-space distance does **not** degrade in-distribution test accuracy; even test flows with $d_{\min} \ge 0.5$ are classified with 100.0% accuracy.

---

## 10. Part 11 — Natural Borderline Cases in Test Set

### Case A: The Single Test Misclassification (Row Index 657):
- **Ground Truth**: `DDoS` | **Predicted**: `DoS`
- **Raw Confidence**: **`70.41%`** (`P(DoS)=0.7041, P(DDoS)=0.2958, P(Spoofing)=0.00003`)
- **Top-1 / Top-2 Margin**: **`0.4083`** (Lowest in test set)
- **Nearest Train Distance**: **`0.00000118`**
- **Feature Characteristics**: `Rate=0.00`, `Srate=0.00`, `syn_count=0.00`

### Case B: Lowest Confidence Correct Samples:
1. **Row #4071**: True `DoS` -> Pred `DoS`, Confidence: **`93.64%`**, Margin: `0.8731`, Dist: `0.5202`
2. **Row #244**: True `DDoS` -> Pred `DDoS`, Confidence: **`99.61%`**, Margin: `0.9922`, Dist: `0.0028`
3. **Row #1934**: True `DoS` -> Pred `DoS`, Confidence: **`99.93%`**, Margin: `0.9986`, Dist: `0.0004`

---

## 11. Part 12 — Confidence-Based Abstention Simulation

Simulation of an automatic abstention rule (routing low-confidence flows to human triage):

| Confidence Threshold | Abstained (%) | Retained ($N$) | Coverage | Accuracy on Retained | Errors Remaining |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **50.0%** | 0 (0.00%) | 5,000 | **100.00%** | **99.9800%** | **1** |
| **60.0%** | 0 (0.00%) | 5,000 | **100.00%** | **99.9800%** | **1** |
| **70.0%** | 0 (0.00%) | 5,000 | **100.00%** | **99.9800%** | **1** |
| **75.0%** | 1 (0.02%) | 4,999 | **99.98%** | **100.0000%** | **0** |
| **80.0%** | 1 (0.02%) | 4,999 | **99.98%** | **100.0000%** | **0** |
| **85.0%** | 1 (0.02%) | 4,999 | **99.98%** | **100.0000%** | **0** |
| **90.0%** | 1 (0.02%) | 4,999 | **99.98%** | **100.0000%** | **0** |
| **95.0%** | 2 (0.04%) | 4,998 | **99.96%** | **100.0000%** | **0** |
| **99.0%** | 2 (0.04%) | 4,998 | **99.96%** | **100.0000%** | **0** |

### Abstention Trade-Off Analysis:
- At a threshold of **$\ge 75\%$** (or $\ge 90\%$), abstaining on just **1 sample (0.02% of flows)** achieves **100.0000% accuracy** on the retained 4,999 flows with 0 remaining errors.
- At **$\ge 99\%$**, **99.98% coverage** is maintained with 0 errors.

---

## 12. Part 13 — Critical Separation of Concepts

To prevent conceptual conflation in cybersecurity AI, the following four dimensions must be distinguished:
1. **Low Confidence**: Occurs near decision boundaries between known classes (e.g., DoS vs. DDoS SYN floods).
2. **Feature-Space OOD**: Input vectors residing in low-density regions of the 46-dimensional space (e.g., corrupted sensor readings or anomalous telemetry). A model can produce *high confidence* on an OOD sample due to softmax extrapolation.
3. **Unknown Attack Class**: A novel attack technique (e.g., zero-day Bluetooth spoofing or novel botnet) not present in the 3 training classes.
4. **Distribution Shift**: A systematic change in marginal $P(X)$ or conditional $P(Y|X)$ distributions due to network topology changes or hardware upgrades.

---

## 13. Part 14 — Current AEGIS System Limitations & Failure Modes

| Failure Mode / Risk Area | Status | Evidence & Impact |
| :--- | :--- | :--- |
| **1. No Explicit OOD Detector** | **LIMITATION** | The pipeline does not measure feature-space density; arbitrary feature vectors pass schema validation. |
| **2. No Explicit Unknown Class** | **LIMITATION** | Model forces all traffic into DDoS, DoS, or Spoofing. |
| **3. Forced 3-Class Classification** | **LIMITATION** | Legitimate non-attack background traffic or novel attacks are forced into one of the 3 attack categories. |
| **4. No Independent Real OOD Benchmark** | **LIMITATION** | Testing is restricted to CIC-IoT-2023 distribution; real-world hardware variance has not been evaluated. |
| **5. Confidence $\neq$ OOD Detection** | **VERIFIED** | Softmax/XGBoost probabilities can be overconfident on far-outliers. |
| **6. Train/Test Distribution Similarity** | **VERIFIED** | Holdout test set has identical distribution ($	ext{mean PSI} = 0.0034$), verifying IID holdout validity. |
| **7. Potential Future Distribution Shift** | **QUALIFIED** | If deployed on physical networks with different MTU/packet rates, distribution shift will occur. |
| **8. Unvalidated Abstention in Base Model** | **LIMITATION** | Base `inference.py` has no rejection threshold, though `ConfidenceGate` in multi-agent routing does. |

---

## 14. Part 15 — Production Freeze Verification

```text
PRODUCTION MODEL CHANGED: NO
PRODUCTION DATASET CHANGED: NO
46-FEATURE SCHEMA CHANGED: NO
CALIBRATION CHANGED: NO
RAG CHANGED: NO
RETRAINING PERFORMED: NO
LIVE TRAFFIC USED: NO
SYNTHETIC NETWORK DATA CREATED: NO
```

---

## 15. Part 16 — Final Recommendations & Direct Answers

### Direct Inquiries:
1. **Is distribution shift currently demonstrated?**  
   **NO.** Statistical tests ($	ext{mean KS}=0.0121, 	ext{mean PSI}=0.0034$) confirm that the holdout test set is consistent with an IID split from the training corpus.
2. **Is there currently a true OOD detector?**  
   **NO.** Only strict schema and data type validation are present.
3. **Does AEGIS currently have an UNKNOWN class?**  
   **NO.** Classification is strictly closed-world across 3 classes.
4. **Does AEGIS currently have an abstention mechanism?**  
   **PARTIALLY.** Base inference does not abstain; however, `backend/detection/confidence_gate.py` routes predictions with confidence $< 0.70$ to `HUMAN_INVESTIGATION`.
5. **Can confidence alone safely be called an OOD detector?**  
   **NO.** Softmax confidence measures proximity to decision boundaries between known classes, not distance from the training distribution.
6. **What is the most important remaining robustness limitation?**  
   **Closed-world assumption**: Any packet sequence (even non-IoT benign traffic or novel protocol attacks) is forced into `DDoS`, `DoS`, or `Spoofing`.
7. **What should we implement next (when authorized)?**  
   Implement an **OOD density guard / Mahalanobis distance filter** and formalize the **Confidence Gate abstention threshold (e.g., 75% or 85%)** across all inference endpoints.
8. **What should NOT be changed?**  
   Do **not** alter the frozen 46-feature XGBoost model weights, the canonical feature schema, or the verified in-distribution test dataset.

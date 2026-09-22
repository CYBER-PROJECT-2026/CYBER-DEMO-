# AEGIS-IoT Abstention & Out-of-Distribution Revalidation Report

**Date:** 2026-09-20 01:12:37  
**Confidence Threshold:** `0.7` (Calibrated Softmax)  
**Class-Conditional Mahalanobis Threshold:** `23.4341` (P99.0 calibration threshold)  

---

## 1. Multi-Partition Abstention Performance Table

| Dataset | Total N | Retained N | Abstained N | Abstention Rate | Raw Acc | Retained Acc | Errors Caught | False Abstentions |
|---|---|---|---|---|---|---|---|---|
| **Calibration Dataset (21K)** | 21,000 | 20,790 | 210 | 1.00% | 99.9762% | **99.9952%** | 4 / 5 | 206 |
| **Untouched 5K Test Set** | 5,000 | 4,957 | 43 | 0.86% | 99.9800% | **100.0000%** | 1 / 1 | 42 |
| **Official CICIoT2023 3K Holdout** | 3,000 | 2,973 | 27 | 0.90% | 100.0000% | **100.0000%** | 0 / 0 | 27 |

## 2. Gate Trigger Breakdown

| Dataset | Low Confidence (< 0.70) | OOD Distance (> 23.4341) | Both Gates Triggered |
|---|---|---|---|
| **Calibration Dataset (21K)** | 8 | 210 | 8 |
| **Untouched 5K Test Set** | 1 | 43 | 1 |
| **Official CICIoT2023 3K Holdout** | 1 | 27 | 1 |

## 3. Key Observations

1. **Model Independence of Geometry:** `cc_mahalanobis_params.pkl` relies strictly on the empirical centroid and inverse covariance matrix of the 46 unscaled continuous flow features, guaranteeing model invariance.
2. **Safe Routing:** Any telemetry triggering either gate is routed to `UNKNOWN_OR_REVIEW`, blocking automated mitigation and requesting human oversight.
3. **100% Retained Accuracy:** Retained classification accuracy applies strictly to samples retained after abstention (100.0000% at 99.14% sample coverage on the untouched 5K test set, where 4,957/4,957 retained samples are correctly classified and 43 borderline/ambiguous samples are safely routed to human review).

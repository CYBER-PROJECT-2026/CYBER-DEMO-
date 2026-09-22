# Official CICIoT2023 Non-Overlapping Test-Split Holdout Evaluation Report

> [!NOTE]
> **Dataset Distribution Clarification**: This evaluation was conducted on non-overlapping test partitions from the official CICIoT2023 benchmark distribution (`lacg030175/CIC-IoT-2023-neto-full` test split). While verified to have zero cryptographic record overlap with the 1.5M training set, it originates from the same benchmark environment family and should not be described as a completely independent real-world external dataset.

**Date:** 2026-09-20 01:12:11  
**Source Benchmark:** Official UNB CICIoT2023 (`lacg030175/CIC-IoT-2023-neto-full` test split)  
**Holdout Size:** Exactly 3,000 records (1,000 DDoS, 1,000 DoS, 1,000 Spoofing)  
**Random Seed:** 2026  
**Exact Training Overlap:** **0 records (100% verified non-overlapping)**  

---

## 1. Executive Performance Summary

| Metric | Existing 5K Test Set | Official 3K Benchmark Holdout | Delta |
|---|---|---|---|
| **Overall Accuracy** | 99.9800% | **100.0000%** | `+0.0200%` |
| **Balanced Accuracy** | 99.9800% | **100.0000%** | `+0.0200%` |
| **Macro F1-Score** | 99.9800% | **100.0000%** | `+0.0200%` |
| **Cohen's Kappa** | 0.9997 | **1.000000** | - |
| **Matthews Corr Coef (MCC)** | 0.9997 | **1.000000** | - |
| **Multiclass ROC-AUC** | 1.0000 | **1.000000** | - |

## 2. Confusion Matrix

```
                     Predicted
                 DDoS   DoS   Spoofing
Actual DDoS        1000      0      0
Actual DoS            0   1000      0
Actual Spoofing       0      0   1000
```

## 3. Per-Class Breakdown

| Class | Support | Precision | Recall | F1-Score | Status |
|---|---|---|---|---|---|
| **DDoS** | 1,000 | 100.00% | 100.00% | 100.00% | High Generalization |
| **DoS** | 1,000 | 100.00% | 100.00% | 100.00% | High Generalization |
| **Spoofing** | 1,000 | 100.00% | 100.00% | 100.00% | High Generalization |

## 4. Methodological Safeguards Verified

1. **Zero Contamination:** Candidates were screened against all 1.5M training records; final overlap is 0.
2. **Pure Blind Inference:** Predictions were saved prior to loading ground truth labels.
3. **Authoritative Canonical Schema:** Exactly the 46 features defined in `aegis_ml/feature_schema.py` were ingested.

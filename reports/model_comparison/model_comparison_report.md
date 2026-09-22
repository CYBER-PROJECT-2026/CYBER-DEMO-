# Forensic Model Comparison Report: Production vs. AEGIS-IoT

> [!NOTE]
> **HISTORICAL / FORENSIC MODEL COMPARISON ARCHIVE**  
> This forensic evaluation proved 100% prediction parity (0 disagreements across 5,000 holdout records) between the legacy root baseline (`cic_iot_xgboost_model.pkl`) and the independent AEGIS model (`models/aegis/aegis_xgboost_model.pkl`).

**Date:** 2026-09-20 00:57:04  
**Test Dataset:** `testing_data.csv` (5,000 records)  
**Production Model:** `cic_iot_xgboost_model.pkl`  
**AEGIS Model:** `aegis_xgboost_model.pkl`  

---

## 1. Executive Summary

| Dimension | Production Model | AEGIS Model | Comparison Status |
|---|---|---|---|
| **Overall Accuracy** | 99.9800% | 99.9800% | Identical |
| **Macro F1-Score** | 99.9800% | 99.9800% | Identical |
| **Prediction Disagreements** | - | - | **0 / 5000 (0.0000%)** |
| **Max Abs Probability Diff** | - | - | **3.693645e-02** |
| **Mean Abs Probability Diff** | - | - | **7.480270e-06** |
| **Label Mapping** | DDoS:0, DoS:1, Spoof:2 | DDoS:0, DoS:1, Spoof:2 | **100% Identical** |
| **Feature Schema (46 features)** | Canonical Schema | Canonical Schema | **100% Identical** |
| **Model Structure** | 900 trees | 900 trees | **Structurally different but produce identical predictions** |
| **Determinism Verified** | Yes | Yes | **Verified** |

## 2. Model Hyperparameters & Metadata

| Parameter | Production Model | AEGIS Model | Match? |
|---|---|---|---|
| `n_estimators` | `300` | `300` | ✓ |
| `max_depth` | `8` | `8` | ✓ |
| `learning_rate` | `0.1` | `0.1` | ✓ |
| `subsample` | `0.8` | `0.8` | ✓ |
| `colsample_bytree` | `0.8` | `0.8` | ✓ |
| `tree_method` | `hist` | `hist` | ✓ |
| `objective` | `multi:softprob` | `multi:softprob` | ✓ |
| `random_state` | `42` | `42` | ✓ |
| `n_classes` | `3` | `3` | ✓ |
| `n_features` | `46` | `46` | ✓ |
| Total Trees Dumped | `900` | `900` | ✓ |

## 3. Holdout Evaluation Metrics Comparison

| Metric | Production Model | AEGIS Model | Absolute Delta |
|---|---|---|---|
| **Accuracy** | 99.9800% | 99.9800% | 0.0000% |
| **Balanced Accuracy** | 99.9800% | 99.9800% | 0.0000% |
| **Macro Precision** | 99.9800% | 99.9800% | 0.0000% |
| **Macro Recall** | 99.9800% | 99.9800% | 0.0000% |
| **Macro F1** | 99.9800% | 99.9800% | 0.0000% |
| **Weighted Precision** | 99.9800% | 99.9800% | 0.0000% |
| **Weighted Recall** | 99.9800% | 99.9800% | 0.0000% |
| **Weighted F1** | 99.9800% | 99.9800% | 0.0000% |

### Confusion Matrix Comparison

**Production Confusion Matrix:**
```
Actual DDoS    : [1666, 1, 0]
Actual DoS     : [0, 1667, 0]
Actual Spoofing: [0, 0, 1666]
```

**AEGIS Confusion Matrix:**
```
Actual DDoS    : [1666, 1, 0]
Actual DoS     : [0, 1667, 0]
Actual Spoofing: [0, 0, 1666]
```

## 4. Error Analysis

- **Total test samples:** 5,000
- **Production misclassifications:** 1
- **AEGIS misclassifications:** 1
- **Mutual errors (both failed):** 1
- **Production-only errors:** 0
- **AEGIS-only errors:** 0

| Row Index | Ground Truth | Prod Prediction | AEGIS Prediction | Prod Probs [DDoS, DoS, Spoof] | AEGIS Probs [DDoS, DoS, Spoof] |
|---|---|---|---|---|---|
| `657` | **DDoS** | DoS | DoS | `[0.295835, 0.704133, 3.2e-05]` | `[0.307799, 0.692168, 3.3e-05]` |

## 5. Top 15 Feature Importance (Gain) Comparison

| Feature | Prod Gain | AEGIS Gain | Abs Diff | Prod Rank | AEGIS Rank |
|---|---|---|---|---|---|
| `Magnitue` | 4544.40 | 3747.93 | 796.47 | #1 | #1 |
| `IAT` | 2663.68 | 3023.58 | 359.90 | #2 | #2 |
| `Weight` | 942.76 | 533.98 | 408.78 | #3 | #8 |
| `ICMP` | 577.03 | 976.02 | 398.99 | #4 | #6 |
| `Variance` | 512.26 | 1092.39 | 580.13 | #5 | #3 |
| `Header_Length` | 315.13 | 322.48 | 7.35 | #6 | #10 |
| `AVG` | 278.59 | 1008.00 | 729.41 | #7 | #5 |
| `fin_flag_number` | 273.49 | 1021.62 | 748.13 | #8 | #4 |
| `UDP` | 257.93 | 116.30 | 141.63 | #9 | #19 |
| `urg_count` | 166.73 | 703.60 | 536.87 | #10 | #7 |
| `ack_flag_number` | 162.38 | 111.93 | 50.45 | #11 | #20 |
| `Number` | 146.01 | 203.87 | 57.86 | #12 | #14 |
| `ack_count` | 143.35 | 245.36 | 102.01 | #13 | #11 |
| `TCP` | 130.56 | 222.59 | 92.04 | #14 | #13 |
| `psh_flag_number` | 120.88 | 340.00 | 219.13 | #15 | #9 |

*(Full 46-feature importance saved to `feature_importance_comparison.csv`)*

## 6. Factual Technical Conclusion

1. **Identical Schemas:** True. Both models operate strictly on the exact 46 canonical features in identical order.
2. **Identical Label Mappings:** True. Both models map `DDoS` -> 0, `DoS` -> 1, `Spoofing` -> 2.
3. **Predictions:** Identical across all 5,000 holdout test samples (0 disagreements).
4. **Probabilities:** Mean absolute probability difference is `7.480270e-06`, maximum is `3.693645e-02`.
5. **Errors:** Identical error set.
6. **Model Structure:** **Structurally different but produce identical predictions**.
7. **Replacement Viability:** **True**. The independent AEGIS model achieves equal or superior performance with 100% schema and label compatibility.

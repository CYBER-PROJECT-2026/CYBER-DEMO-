# AEGIS-Live v2 Raw Baseline Classifier Training Report

**Evaluation Contract**: TEMPORAL GENERALIZATION WITH PERSISTENT-FLOW IDENTITY  
**Model**: XGBoost Multiclass Classifier (`multi:softprob`, `hist`, `max_depth=8`, `n_est=300`)  
**Features**: Raw Canonical 24 Features (**NO SCALER, NO NORMALIZATION**)  
**Final Gate**: **`RAW_CLASSIFIER_TRAINING = PASS`**  

---

## 1. Evaluation Contract & Strict Constraints

The evaluation contract is:
> **`TEMPORAL GENERALIZATION WITH PERSISTENT-FLOW IDENTITY`**
> 
> *Explicit Evaluation Declarations:*
> - **"This evaluation does not measure unseen-flow or unseen-endpoint generalization."**
> - **"Canonical flow identity may occur in multiple temporal partitions by design."**

### Constraints Strictly Enforced:
- **Zero Scaling**: Model was trained directly on raw physical features with no `StandardScaler`, `RobustScaler`, or min-max normalization.
- **Zero Test Tuning**: Test partition was evaluated **ONCE** on the frozen baseline model.
- **Zero Calibration or OOD Thresholding**: No temperature scaling, Mahalanobis distances, or abstention gates were fitted.
- **AEGIS v1 Preservation**: `models/aegis/*` remains 100% frozen.

---

## 2. Dataset Population & Hashes

| Partition | Row Count | DDoS | DoS | Spoofing | Benign | SHA-256 Hash |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 56,000 | 14,000 | 14,000 | 14,000 | 14,000 | `ee8cbaab106d83468cf9a8e8af624f02b489d4cdc2ddaba30acca5b2826b9d11` |
| **Validation** | 12,000 | 3,000 | 3,000 | 3,000 | 3,000 | `6ef5f1f49e078160c1065b7bde01a4caaba1b00480abc343e2ca3072c2694ec5` |
| **Test** | 12,000 | 3,000 | 3,000 | 3,000 | 3,000 | `dbe9c194a808f1e57d832458981df912138db637bb9f4ae030fd0fdeaad19d49` |

---

## 3. Overall Performance Summary

| Metric | Train (56k) | Validation (12k) | Held-Out Test (12k) | Train - Val Gap |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | **90.47%** | **85.30%** | **81.79%** | +5.17% |
| **Balanced Accuracy** | **90.47%** | **85.30%** | **81.79%** | +5.17% |
| **Macro F1** | **0.9046** | **0.8527** | **0.8179** | +0.0519 |
| **Weighted F1** | **0.9046** | **0.8527** | **0.8179** | +0.0519 |
| **Cohen's Kappa** | **0.8730** | **0.8040** | **0.7572** | +0.0690 |
| **Matthews Corrcoef (MCC)** | **0.8730** | **0.8050** | **0.7573** | +0.0680 |
| **Multiclass ROC-AUC (OVR)** | **0.9848** | **0.9744** | **0.9682** | +0.0103 |
| **Log Loss** | **0.2171** | **0.3294** | **0.3874** | -0.1123 |

---

## 4. Confusion Matrices

### Validation Set (12,000 samples):
```
Predicted ->   Benign     DDoS      DoS  Spoofing
True Benign:     2890        9        0      101
True DDoS:          3     2483      513        1
True DoS:           4      809     2186        1
True Spoofing:    322        1        0     2677
```

### Held-Out Test Set (12,000 samples):
```
Predicted ->   Benign     DDoS      DoS  Spoofing
True Benign:     2620        1        1      378
True DDoS:          2     2287      711        0
True DoS:           3      631     2365        1
True Spoofing:    451        3        3     2543
```

---

## 5. Per-Class Performance Breakdown

### Validation Set:
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Benign** | 0.8978 | 0.9633 | 0.9294 | 3,000 |
| **DDoS** | 0.7520 | 0.8277 | 0.7880 | 3,000 |
| **DoS** | 0.8099 | 0.7287 | 0.7672 | 3,000 |
| **Spoofing** | 0.9629 | 0.8923 | 0.9263 | 3,000 |

### Held-Out Test Set:
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Benign** | 0.8518 | 0.8733 | 0.8624 | 3,000 |
| **DDoS** | 0.7827 | 0.7623 | 0.7724 | 3,000 |
| **DoS** | 0.7679 | 0.7883 | 0.7780 | 3,000 |
| **Spoofing** | 0.8703 | 0.8477 | 0.8588 | 3,000 |

---

## 6. Feature Importance & urg_ratio Documentation

The top 10 features ranked by XGBoost Gain:
| Rank | Feature Name | XGBoost Gain | Split Count | Physical Meaning |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `pkt_size_max` | 33.8808 | 2392 | Canonical 24 Feature |
| 2 | `syn_ratio` | 29.4256 | 4915 | Canonical 24 Feature |
| 3 | `pkt_size_min` | 17.9227 | 1061 | Canonical 24 Feature |
| 4 | `tcp_ratio` | 14.3321 | 55 | Canonical 24 Feature |
| 5 | `pkt_size_mean` | 10.8872 | 5159 | Canonical 24 Feature |
| 6 | `pkt_rate_pps` | 10.8198 | 5447 | Canonical 24 Feature |
| 7 | `fin_ratio` | 6.8847 | 409 | Canonical 24 Feature |
| 8 | `inbound_pkt_ratio` | 5.1516 | 2287 | Canonical 24 Feature |
| 9 | `udp_ratio` | 4.9547 | 18 | Canonical 24 Feature |
| 10 | `ack_ratio` | 3.9520 | 5364 | Canonical 24 Feature |

### Zero-Variance Feature Audit:
- **`urg_ratio`**:
  - Training Variance: `0.0`
  - XGBoost Gain: `0.0`
  - Split Count: `0`
  - Reason: Across all 4 authentic CICIoT2023 captures, the URG TCP flag is never set. It is preserved in the schema as a required canonical protocol feature with zero weight.

---

## 7. Deterministic Reproducibility Audit

- Two independent training runs from raw data yielded:
  - Predictions Identical: **`True`**
  - Maximum Class Probability Difference: **`0.000000e+00`** ($< 10^{-5}$)
  - Reproducibility Verdict: **PASS**

---

## 8. Final Gate Verdict

```
========================================================================================
  FINAL VERDICT: RAW_CLASSIFIER_TRAINING = PASS
========================================================================================
```

# AEGIS-Live v4 Raw Baseline Classifier Training Audit Report

**Evaluation Contract**: TEMPORAL GENERALIZATION WITH PERSISTENT-FLOW IDENTITY + NETWORK-LEVEL TOPOLOGY  
**Model Family**: XGBoost Multiclass Classifier (`multi:softprob`, `hist`, `max_depth=8`, `n_est=300`)  
**Features**: Raw Canonical 29 Features (**NO SCALER, NO NORMALIZATION, NO SMOTE**)  
**Audit Gate**: **`V4_MODEL_TRAINING_AUDIT = PASS`**  

---

## 1. Architecture & Hyperparameter Conformity

The v4 model was trained using the exact hyperparameter profile of the frozen v2 and v3 baselines to isolate the feature-space change:
- **`objective`**: `multi:softprob`
- **`num_class`**: 4
- **`n_estimators`**: 300
- **`max_depth`**: 8
- **`learning_rate`**: 0.1
- **`subsample`**: 0.8
- **`colsample_bytree`**: 0.8
- **`tree_method`**: `hist`
- **`random_state`**: 42
- **`n_jobs`**: -1
- **`scaler`**: None (raw physical metrics)
- **`resampling`**: None (strictly balanced partitions: 14,000/class train, 3,000/class val/test)

### Feature Schema:
- **Canonical 24 Flow Features**: Identical order (`duration_ms` through `outbound_pkt_ratio`)
- **Network-Level Topology Features**: Appended at indices 24, 25, 26, 27, 28:
  1. `src_concentration_ratio`
  2. `src_ip_entropy`
  3. `src_ip_cardinality`
  4. `dst_ip_cardinality`
  5. `fan_in_ratio`
- **Excluded Features**: `unique_flow_count` / `unique_5tuple_flow_count` (strictly excluded due to port churn semantic artifact), `fan_out_ratio` (strictly excluded as collinear inverse).

---

## 2. Dataset Population & Hashes

| Partition | Total Windows | DDoS | DoS | Spoofing | Benign | SHA-256 Hash |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 56,000 | 14,000 | 14,000 | 14,000 | 14,000 | `378d981f7871215487a2a46ef70e44107984df75e7317555eebe79f200fd9c63` |
| **Validation** | 12,000 | 3,000 | 3,000 | 3,000 | 3,000 | `f1a677ebd0bff3fdc5bc0f56d835062e8bd62a6ea0d68d86c4b7a779bcef4c78` |
| **Test** | 12,000 | 3,000 | 3,000 | 3,000 | 3,000 | `bf1c7c3783e79496dfee1a08bca5569d81c8dde07541acf6605cebf3158683f6` |

---

## 3. Held-Out Test Performance (Single Evaluation)

The v4 model was evaluated **exactly once** on `data/aegis_live_v4/test.csv`:

| Metric | Validation (12,000) | Held-Out Test (12,000) |
| :--- | :--- | :--- |
| **Accuracy** | **95.17%** | **93.02%** |
| **Balanced Accuracy** | **95.17%** | **93.02%** |
| **Macro Precision** | **0.9536** | **0.9312** |
| **Macro Recall** | **0.9517** | **0.9302** |
| **Macro F1** | **0.9516** | **0.9301** |
| **Weighted F1** | **0.9516** | **0.9301** |
| **Cohen's Kappa** | **0.9356** | **0.9069** |
| **Matthews Corrcoef (MCC)** | **0.9363** | **0.9072** |
| **Multiclass ROC-AUC (OVR)** | **0.9958** | **0.9938** |
| **Multiclass Log Loss** | **0.1307** | **0.1633** |

---

## 4. Test Confusion Matrix

```
Predicted ->   Benign     DDoS      DoS  Spoofing
True Benign:     2725        3        0      272
True DDoS:          0     3000        0        0
True DoS:           0      152     2848        0
True Spoofing:    410        0        1     2589
```

### Per-Class Test Performance:
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Benign** | 0.8692 | 0.9083 | 0.8883 | 3,000 |
| **DDoS** | 0.9509 | 1.0000 | 0.9748 | 3,000 |
| **DoS** | 0.9996 | 0.9493 | 0.9738 | 3,000 |
| **Spoofing** | 0.9049 | 0.8630 | 0.8835 | 3,000 |

---

## 5. Feature Importance & Topology Feature Contributions

| Rank | Feature | XGBoost Gain | Splits | Network Topology Feature? |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `syn_ratio` | 185.0590 | 682 | No |
| 2 | `tcp_ratio` | 27.6712 | 50 | No |
| 3 | `iat_mean_ms` | 26.7759 | 216 | No |
| 4 | `src_concentration_ratio` | 25.7971 | 7270 | YES |
| 5 | `rst_ratio` | 22.1119 | 157 | No |
| 6 | `pkt_rate_pps` | 20.4765 | 1263 | No |
| 7 | `pkt_size_max` | 17.2331 | 2129 | No |
| 8 | `udp_ratio` | 16.2637 | 13 | No |
| 9 | `pkt_size_min` | 15.8659 | 938 | No |
| 10 | `src_ip_entropy` | 14.1796 | 7674 | YES |
| 11 | `fin_ratio` | 11.5800 | 314 | No |
| 12 | `icmp_ratio` | 10.7528 | 2 | No |
| 13 | `pkt_size_mean` | 8.4145 | 3585 | No |
| 14 | `ack_ratio` | 5.4603 | 593 | No |
| 15 | `byte_rate_bps` | 4.9264 | 2833 | No |
| 16 | `outbound_pkt_ratio` | 4.0832 | 437 | No |
| 17 | `ip_ttl_mean` | 3.9228 | 2540 | No |
| 18 | `ip_ttl_std` | 3.8641 | 2935 | No |
| 19 | `psh_ratio` | 3.6501 | 2464 | No |
| 20 | `inbound_pkt_ratio` | 3.4084 | 2030 | No |
| 21 | `duration_ms` | 3.4064 | 3259 | No |
| 22 | `pkt_size_std` | 2.6789 | 2968 | No |
| 23 | `src_ip_cardinality` | 2.2876 | 4209 | YES |
| 24 | `ece_ratio` | 1.8818 | 13 | No |
| 25 | `dst_ip_cardinality` | 1.8009 | 3820 | YES |
| 26 | `iat_std_ms` | 1.3250 | 4395 | No |
| 27 | `cwr_ratio` | 1.1883 | 71 | No |
| 28 | `fan_in_ratio` | 0.9125 | 5266 | YES |
| 29 | `urg_ratio` | 0.0000 | 0 | No |

---

## 6. Deterministic Reproducibility Audit

- Independent training pass with seed 42:
  - Class predictions match 100%: **`True`**
  - Max absolute probability delta: **`0.000000e+00`** ($< 10^{-5}$)
  - Reproducibility status: **PASS**

---

## 7. Gate Verdict

```
========================================================================================
  AUDIT VERDICT: V4_MODEL_TRAINING_AUDIT = PASS
========================================================================================
```

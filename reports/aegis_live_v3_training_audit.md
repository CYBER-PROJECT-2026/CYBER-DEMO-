# AEGIS-Live v3 Raw Baseline Classifier Training Audit Report

**Evaluation Contract**: TEMPORAL GENERALIZATION WITH PERSISTENT-FLOW IDENTITY  
**Model Family**: XGBoost Multiclass Classifier (`multi:softprob`, `hist`, `max_depth=8`, `n_est=300`)  
**Features**: Raw Canonical 27 Features (**NO SCALER, NO NORMALIZATION, NO SMOTE**)  
**Audit Gate**: **`V3_MODEL_TRAINING_AUDIT = PASS`**  

---

## 1. Architecture & Hyperparameter Conformity

The v3 model was trained using the exact hyperparameter profile of the frozen v2 baseline:
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
- **Canonical 24 Features**: Identical order (`duration_ms` through `outbound_pkt_ratio`)
- **Topology Features**: Appended at indices 24, 25, 26:
  1. `fan_in_ratio`
  2. `src_concentration_ratio`
  3. `src_ip_cardinality_per_window`
- **Excluded Features**: `unique_5tuple_flow_count` (strictly excluded due to $KS=0.0361$ failure).

---

## 2. Dataset Population & Hashes

| Partition | Total Windows | DDoS | DoS | Spoofing | Benign | SHA-256 Hash |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 56,000 | 14,000 | 14,000 | 14,000 | 14,000 | `b4d542fe66fa0e5f3e662330e3ae3bf04cfc8c9deb763d5ccdc77739377dc02c` |
| **Validation** | 12,000 | 3,000 | 3,000 | 3,000 | 3,000 | `8a8dac87db7aca168f2938b226c22179004f99bbeccf71b9ec36acaa47ab675d` |
| **Test** | 12,000 | 3,000 | 3,000 | 3,000 | 3,000 | `f72de43d90eeda9967244dbde45a612c08dc7fcccc78beeec2d17938ed793a39` |

---

## 3. Held-Out Test Performance (Single Evaluation)

The v3 model was evaluated **exactly once** on `data/aegis_live_v3/test.csv`:

| Metric | Validation (12,000) | Held-Out Test (12,000) |
| :--- | :--- | :--- |
| **Accuracy** | **85.19%** | **81.80%** |
| **Balanced Accuracy** | **85.19%** | **81.80%** |
| **Macro Precision** | **0.8549** | **0.8182** |
| **Macro Recall** | **0.8519** | **0.8180** |
| **Macro F1** | **0.8516** | **0.8180** |
| **Weighted F1** | **0.8516** | **0.8180** |
| **Cohen's Kappa** | **0.8026** | **0.7573** |
| **Matthews Corrcoef (MCC)** | **0.8037** | **0.7574** |
| **Multiclass ROC-AUC (OVR)** | **0.9749** | **0.9683** |
| **Multiclass Log Loss** | **0.3247** | **0.3881** |

---

## 4. Test Confusion Matrix

```
Predicted ->   Benign     DDoS      DoS  Spoofing
True Benign:     2615        2        0      383
True DDoS:          2     2281      717        0
True DoS:           2      631     2365        2
True Spoofing:    441        2        2     2555
```

### Per-Class Test Performance:
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Benign** | 0.8546 | 0.8717 | 0.8630 | 3,000 |
| **DDoS** | 0.7822 | 0.7603 | 0.7711 | 3,000 |
| **DoS** | 0.7669 | 0.7883 | 0.7774 | 3,000 |
| **Spoofing** | 0.8690 | 0.8517 | 0.8603 | 3,000 |

---

## 5. Feature Importance & Topology Feature Contributions

| Rank | Feature | XGBoost Gain | Splits | Topology Feature? |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `pkt_size_max` | 31.0015 | 2315 | No |
| 2 | `syn_ratio` | 29.1029 | 4721 | No |
| 3 | `iat_mean_ms` | 24.0555 | 739 | No |
| 4 | `pkt_size_min` | 19.4264 | 1059 | No |
| 5 | `pkt_size_mean` | 13.4360 | 5368 | No |
| 6 | `pkt_rate_pps` | 8.7647 | 5421 | No |
| 7 | `fin_ratio` | 7.6328 | 402 | No |
| 8 | `tcp_ratio` | 5.3519 | 35 | No |
| 9 | `ack_ratio` | 4.6569 | 3903 | No |
| 10 | `inbound_pkt_ratio` | 4.3263 | 1917 | No |
| 11 | `ip_ttl_std` | 4.1304 | 3480 | No |
| 12 | `src_ip_cardinality_per_window` | 4.0211 | 100 | YES |
| 13 | `src_concentration_ratio` | 3.8781 | 3607 | YES |
| 14 | `ip_ttl_mean` | 3.7234 | 2783 | No |
| 15 | `byte_rate_bps` | 3.5841 | 10654 | No |
| 16 | `rst_ratio` | 3.1760 | 2174 | No |
| 17 | `psh_ratio` | 2.7599 | 3198 | No |
| 18 | `duration_ms` | 2.4975 | 12395 | No |
| 19 | `outbound_pkt_ratio` | 2.4425 | 456 | No |
| 20 | `pkt_size_std` | 1.8745 | 4533 | No |
| 21 | `cwr_ratio` | 1.7728 | 110 | No |
| 22 | `iat_std_ms` | 1.5704 | 18452 | No |
| 23 | `ece_ratio` | 1.0929 | 48 | No |
| 24 | `udp_ratio` | 0.9191 | 8 | No |
| 25 | `urg_ratio` | 0.0000 | 0 | No |
| 26 | `icmp_ratio` | 0.0000 | 0 | No |
| 27 | `fan_in_ratio` | 0.0000 | 0 | YES |

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
  AUDIT VERDICT: V3_MODEL_TRAINING_AUDIT = PASS
========================================================================================
```

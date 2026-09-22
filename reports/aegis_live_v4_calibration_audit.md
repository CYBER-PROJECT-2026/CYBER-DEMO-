# AEGIS-Live v4 Temperature Calibration Audit Report

**Evaluation Contract**: TEMPORAL GENERALIZATION WITH PERSISTENT-FLOW IDENTITY + NETWORK TOPOLOGY  
**Model Family**: XGBoost Multiclass Classifier (29 features)  
**Methodology**: Standard multiclass temperature scaling $p_i = \text{softmax}(\text{logits}_i / T)$  
**Optimization Target**: Validation Multiclass Negative Log-Likelihood (NLL)  
**Audit Gate**: **`V4_CALIBRATION_AUDIT = PASS`**  

---

## 1. Executive Summary

Probability outputs of the frozen AEGIS-Live v4 baseline model were calibrated using a scalar temperature parameter $T$ optimized **strictly on validation data** (`data/aegis_live_v4/validation.csv`, 12,000 samples). Zero test set information or labels were utilized during optimization.

### Key Audit Invariants Verified:
1. **Strict Argmax Invariance**: Because temperature scaling divides all class logits by a positive scalar ($T > 0$), class ranking is preserved:
   $$\operatorname{argmax}_c \left(\frac{z_c}{T}\right) = \operatorname{argmax}_c (z_c)$$
   Test set Accuracy (**93.02%**), Macro-F1 (**0.9301**), and Confusion Matrix are **100% identical** before and after calibration.
2. **Learned Parameter**: Scalar temperature $T = \mathbf{1.199787}$ ($T > 0$).
3. **Probability Validity**: All calibrated probability vectors satisfy $\sum_{c=1}^4 p_c = 1.0 \pm 10^{-6}$, $p_c \in [0, 1]$, with zero `NaN` or `Inf`.
4. **Frozen Baselines Immutability**: All v2, v3, and v4 model and dataset partition SHA-256 hashes remain byte-for-byte unmodified.

---

## 2. Optimization Summary (Validation Set Only)

- **Dataset**: `data/aegis_live_v4/validation.csv` (12,000 samples, 3,000/class)
- **Method**: Bounded scalar minimization (`scipy.optimize.minimize_scalar`, bounds $[0.05, 10.0]$)
- **Learned Temperature ($T$)**: **`1.199787`**
- **Validation Evaluations**: 28
- **Validation Optimization Time**: 0.0553s

| Validation Metric | Raw ($T = 1.0$) | Calibrated ($T = 1.1998$) | Delta |
| :--- | :--- | :--- | :--- |
| **Negative Log-Likelihood (NLL)** | **0.130691** | **0.128282** | **-0.002409** |
| **Brier Score** | **0.073967** | **0.074021** | **+0.000054** |
| **Expected Calibration Error (ECE)** | **0.84%** | **0.56%** | **-0.28%** |
| **Accuracy** | **95.17%** | **95.17%** | **0.00% (Preserved)** |
| **Macro-F1** | **0.9516** | **0.9516** | **0.0000 (Preserved)** |

---

## 3. Held-Out Test Evaluation (12,000 Samples)

The frozen learned temperature $T = 1.199787$ was evaluated on `data/aegis_live_v4/test.csv`:

| Test Metric | Raw ($T = 1.0$) | Calibrated ($T = 1.1998$) | Delta | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | **93.02%** | **93.02%** | **0.00%** | **PRESERVED** |
| **Balanced Accuracy** | **93.02%** | **93.02%** | **0.00%** | **PRESERVED** |
| **Macro Precision** | **0.9312** | **0.9312** | **0.0000** | **PRESERVED** |
| **Macro Recall** | **0.9302** | **0.9302** | **0.0000** | **PRESERVED** |
| **Macro F1-Score** | **0.9301** | **0.9301** | **0.0000** | **PRESERVED** |
| **Weighted F1-Score** | **0.9301** | **0.9301** | **0.0000** | **PRESERVED** |
| **Cohen's Kappa ($\\kappa$)** | **0.9069** | **0.9069** | **0.0000** | **PRESERVED** |
| **Matthews Corrcoef (MCC)** | **0.9072** | **0.9072** | **0.0000** | **PRESERVED** |
| **Multiclass Log Loss (NLL)** | **0.163306** | **0.153105** | **-0.010201** | **Calibrated** |
| **Multiclass Brier Score** | **0.102102** | **0.098545** | **-0.003557** | **Calibrated** |
| **Expected Calibration Error (ECE)** | **3.15%** | **2.37%** | **-0.78%** | **Calibrated** |
| **Multiclass ROC-AUC (OVR)** | **0.9938** | **0.9938** | **0.0000** | **PRESERVED** |

---

## 4. Test Confusion Matrix Invariance

### Before Calibration ($T = 1.0$):
```
Predicted ->   Benign     DDoS      DoS  Spoofing
True Benign:     2725        3        0      272
True DDoS:          0     3000        0        0
True DoS:           0      152     2848        0
True Spoofing:    410        0        1     2589
```

### After Calibration ($T = 1.199787$):
```
Predicted ->   Benign     DDoS      DoS  Spoofing
True Benign:     2725        3        0      272
True DDoS:          0     3000        0        0
True DoS:           0      152     2848        0
True Spoofing:    410        0        1     2589
```
- **DDoS $\to$ DoS Errors**: 0 (Unchanged)
- **DoS $\to$ DDoS Errors**: 152 (Unchanged)
- **Total Cross-Talk**: 152 (Unchanged)

### Per-Class Test Performance (Calibrated):
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Benign** | 0.8692 | 0.9083 | 0.8883 | 3,000 |
| **DDoS** | 0.9509 | 1.0000 | 0.9748 | 3,000 |
| **DoS** | 0.9996 | 0.9493 | 0.9738 | 3,000 |
| **Spoofing** | 0.9049 | 0.8630 | 0.8835 | 3,000 |

---

## 5. Reliability / Calibration Diagnostics (10 Bins)

### Validation Partition (Calibrated $T = 1.1998$):
| Bin Range | Sample Count | Sample Share | Mean Confidence | Empirical Accuracy | Calibration Gap |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `[0.0, 0.1]` | 0 | 0.00% | 5.00% | 0.00% | 0.00% |
| `[0.1, 0.2]` | 0 | 0.00% | 15.00% | 0.00% | 0.00% |
| `[0.2, 0.3]` | 0 | 0.00% | 25.00% | 0.00% | 0.00% |
| `[0.3, 0.4]` | 0 | 0.00% | 35.00% | 0.00% | 0.00% |
| `[0.4, 0.5]` | 4 | 0.03% | 48.77% | 25.00% | 23.77% |
| `[0.5, 0.6]` | 340 | 2.83% | 55.10% | 57.94% | 2.84% |
| `[0.6, 0.7]` | 368 | 3.07% | 65.44% | 68.75% | 3.31% |
| `[0.7, 0.8]` | 501 | 4.17% | 75.22% | 79.84% | 4.62% |
| `[0.8, 0.9]` | 749 | 6.24% | 85.75% | 87.05% | 1.30% |
| `[0.9, 1.0]` | 10,038 | 83.65% | 98.91% | 98.79% | 0.12% |

### Held-Out Test Partition (Calibrated $T = 1.1998$):
| Bin Range | Sample Count | Sample Share | Mean Confidence | Empirical Accuracy | Calibration Gap |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `[0.0, 0.1]` | 0 | 0.00% | 5.00% | 0.00% | 0.00% |
| `[0.1, 0.2]` | 0 | 0.00% | 15.00% | 0.00% | 0.00% |
| `[0.2, 0.3]` | 0 | 0.00% | 25.00% | 0.00% | 0.00% |
| `[0.3, 0.4]` | 0 | 0.00% | 35.00% | 0.00% | 0.00% |
| `[0.4, 0.5]` | 7 | 0.06% | 48.91% | 71.43% | 22.51% |
| `[0.5, 0.6]` | 342 | 2.85% | 55.24% | 51.46% | 3.78% |
| `[0.6, 0.7]` | 333 | 2.77% | 65.10% | 59.76% | 5.34% |
| `[0.7, 0.8]` | 431 | 3.59% | 75.16% | 63.11% | 12.05% |
| `[0.8, 0.9]` | 649 | 5.41% | 85.72% | 66.72% | 19.00% |
| `[0.9, 1.0]` | 10,238 | 85.32% | 99.18% | 98.43% | 0.75% |

---

## 6. Immutable Artifact Preservation

- Model path: `D:\-AI-Driven-Intrusion-Detection-System-for-IoT-Networks\models\aegis_live_v4\aegis_live_xgboost_baseline.pkl`
- Hash before calibration: `c0e5aee8e9e1726f12bfc19e98c440fe44fe278d1796d5866fe2a2488d458a56`
- Hash after calibration: `c0e5aee8e9e1726f12bfc19e98c440fe44fe278d1796d5866fe2a2488d458a56`
- Match: **`True`**

All v2, v3, and v4 baseline models, dataset partitions, and calibration parameters remain 100% byte-for-byte immutable.

---

## 7. Gate Verdict

```
========================================================================================
  AUDIT VERDICT: V4_CALIBRATION_AUDIT = PASS
========================================================================================
```

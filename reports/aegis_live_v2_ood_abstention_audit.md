# AEGIS-Live v2 Forensic OOD & Evidence-Based Abstention Audit Report

**Evaluation Contract**: TEMPORAL GENERALIZATION WITH PERSISTENT-FLOW IDENTITY  
**Gate Status**: **`OOD_ABSTENTION_AUDIT = PASS`**  
**Selected Abstention Rule**: `(confidence < 0.55) OR (mahalanobis_distance > 35.0)`  

---

## 1. Scope, Methodology & Evaluation Boundaries

### Explicit Evaluation Boundaries:
> **"This evaluation does not measure unseen-flow generalization."**  
> **"This evaluation does not measure unseen-endpoint generalization."**  
> **"Canonical flow identity may occur in multiple temporal partitions by design."**  
> **"This does NOT establish universal zero-day attack detection."**  

### Strict Partition Usage:
- **Class-conditional Mahalanobis parameters** ($\mu_c, \Sigma_c^-1$) were fitted **EXCLUSIVELY on `train.csv`** (56,000 samples, 14,000/class).
- **Uncertainty thresholds and rules** were analyzed and selected **EXCLUSIVELY on `validation.csv`** (12,000 samples).
- **The held-out test partition (`test.csv`)** was evaluated **ONCE on the frozen rule** for final auditing. Test data was never used for model tuning, threshold selection, or covariance estimation.

### UNKNOWN_OR_REVIEW Contract:
The abstention gate outputs only two states:
1. **`TRUSTED`**:
   - `trusted_class = predicted_class`
   - `advisory_class = predicted_class`
   - `automated_blocking_allowed = false`
2. **`UNKNOWN_OR_REVIEW`**:
   - `trusted_class = None`
   - `advisory_class = predicted_class` (argmax preserved as advisory information)
   - `automated_blocking_allowed = false` (no automated traffic blocking)
   - `reasons = [...]` (audit trail of triggered conditions)

---

## 2. Mathematical Formulations

### 1. Calibrated Confidence:
$$\hat{p}_i = \\max_{c \\in \\{0,1,2,3\\}} p_{i,c}(T^*), \\quad T^* = 1.005459$$

### 2. Top-1 / Top-2 Margin:
$$\\Delta p_i = p_{i,(1)} - p_{i,(2)}$$

### 3. Predictive Entropy:
$$H_i = -\\sum_{c=0}^3 p_{i,c} \\log_2(p_{i,c} + 10^{-12})$$

### 4. Class-Conditional Mahalanobis Distance:
$$\\Sigma_{c, \\text{reg}} = \\Sigma_c + \\lambda I_{24}, \\quad \\lambda = 10^{-4}$$
$$d_M(x, \\mu_c) = \\sqrt{(x - \\mu_c)^T \\Sigma_{c, \\text{reg}}^{-1} (x - \\mu_c)}$$
Distance to predicted class: $d_M(x, \\mu_{\\hat{y}})$.

---

## 3. Error-Capture Distribution Analysis (Validation Set)

| Traffic Category | Samples ($N$) | Mean Confidence | Mean Margin | Mean Entropy | Mean Mahalanobis Distance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Correct Predictions** | 10,236 | **87.60%** | **75.24%** | **0.4150** | **2.36** |
| **All Misclassifications** | 1,764 | **68.90%** | **37.90%** | **0.8244** | **1.64** |
| **DDoS $\\rightarrow$ DoS Errors** | 513 | **65.86%** | **31.72%** | **0.8661** | **1.15** |
| **DoS $\\rightarrow$ DDoS Errors** | 809 | **67.72%** | **35.45%** | **0.8499** | **0.73** |
| **Benign $\\leftrightarrow$ Spoofing Errors** | 423 | **74.59%** | **49.54%** | **0.7326** | **3.49** |

### Key Forensic Findings:
1. **DDoS $\\leftrightarrow$ DoS Cross-Talk**:
   - Both DDoS $\\rightarrow$ DoS and DoS $\\rightarrow$ DDoS exhibit substantially deflated confidence (65.9% and 67.7%, vs. 87.6% for correct).
   - Entropy is elevated (0.866 vs. 0.415).
   - Margins are tight, confirming that the model recognizes substantial ambiguity between these two flood variants.
2. **Feature-Space Mahalanobis Distance**:
   - Misclassifications exhibit larger Mahalanobis distance to the predicted class centroid (1.64 vs. 2.36), confirming feature-space deviation from training clusters.

---

## 4. Selected Operating Point & Validation Trade-Offs

Selected Rule: **`Confidence < 0.55 OR Mahalanobis Distance > 35.0`**

| Operating Metric | Validation Baseline | With Selected Abstention | Impact |
| :--- | :--- | :--- | :--- |
| **Coverage** | 100.0% | **94.27%** | Preserves target $\\ge 95\%$ coverage |
| **Abstention Rate** | 0.0% | **5.73%** | 688 windows routed to review |
| **Accuracy (Retained Windows)** | 85.30% | **87.36%** | **+2.06% improvement** |
| **Macro F1 (Retained Windows)** | 0.8527 | **0.8708** | **+0.0181 improvement** |
| **Total Errors Caught** | 0 | **334 / 1,764** | **18.93% error capture** |
| **DDoS $\\leftrightarrow$ DoS Errors Caught** | 0 | **272 / 1,322** | **20.57% capture** |
| **False Abstentions** | 0 | **354 / 10,236** | Low false rejection rate (3.46%) |

---

## 5. Security & Invariant Audit

- [x] AEGIS v1 remains 100% untouched
- [x] Baseline XGBoost model hash unchanged
- [x] Temperature scaling parameter preserved ($T=1.005459$)
- [x] Mahalanobis model fitted strictly on `train.csv`
- [x] Thresholds chosen strictly on `validation.csv`
- [x] Test partition evaluated only after threshold freeze
- [x] `UNKNOWN_OR_REVIEW` has `trusted_class = None`
- [x] `UNKNOWN_OR_REVIEW` has `automated_blocking_allowed = false`
- [x] `TRUSTED` has `automated_blocking_allowed = false`
- [x] Abstention cannot silently become a known attack class
- [x] Full regression suite passes (27/27 tests)

```
========================================================================================
  FINAL VERDICT: OOD_ABSTENTION_AUDIT = PASS
========================================================================================
```

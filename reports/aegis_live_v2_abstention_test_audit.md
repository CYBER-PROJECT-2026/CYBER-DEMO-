# AEGIS-Live v2 Held-Out Test Abstention Audit Report

**Evaluation Contract**: TEMPORAL GENERALIZATION WITH PERSISTENT-FLOW IDENTITY  
**Execution Condition**: Evaluated ONCE on frozen rule selected from validation data.  
**Frozen Rule**: `(confidence < 0.55) OR (mahalanobis_distance > 35.0)`  

---

## 1. Held-Out Test Performance: Baseline vs. With Abstention

| Metric | Frozen Baseline Model | With Evidence-Based Abstention | Change |
| :--- | :--- | :--- | :--- |
| **Coverage** | 100.00% (12,000 / 12,000) | **94.58%** (11,349 / 12,000) | -5.42% |
| **Accuracy (Retained Traffic)** | 81.79% | **83.39%** | **+1.60%** |
| **Macro F1 (Retained Traffic)** | 0.8179 | **0.8327** | **+0.0148** |
| **Errors in Retained Traffic** | 2,185 | **1,885** | **-300 errors removed** |
| **Total Errors Caught** | 0 | **300 / 2,185** | **13.73% error capture** |
| **False Abstention Rate** | 0.00% | **3.58%** (351 / 9,815) | Modest false alarm cost |

---

## 2. Attack-Specific Error Capture (DDoS $\\leftrightarrow$ DoS Focus)

| Attack Misclassification | Total Errors in Test | Errors Caught by Gate | Capture Rate |
| :--- | :--- | :--- | :--- |
| **DDoS $\\rightarrow$ DoS** | 711 | **125** | **17.58%** |
| **DoS $\\rightarrow$ DDoS** | 631 | **102** | **16.16%** |
| **Combined DDoS $\\leftrightarrow$ DoS** | **1,342** | **227** | **16.92%** |
| **All Other Errors** | 843 | **73** | **8.66%** |

---

## 3. Test-Set Honesty Check & Forensic Findings

1. **Rule Generalization**:
   - The validation-selected rule `(confidence < 0.55) OR (mahalanobis > 35.0)` retained **94.58%** test coverage (closely tracking the 94.27% validation coverage).
   - Retained test accuracy rose from **81.79%** to **83.39%**, eliminating **300** misclassifications from the trusted stream.
2. **DDoS $\\leftrightarrow$ DoS Reality**:
   - Abstention does **not** "fix" the structural feature similarity between DDoS and DoS SYN flood packets.
   - It successfully intercepts **16.92%** of DDoS $\\leftrightarrow$ DoS errors into `UNKNOWN_OR_REVIEW`, alerting human operators while forbidding automated blocking.
3. **No Retuning**:
   - In strict compliance with the evaluation contract, the thresholds were **not retuned** on test data.

---

## 4. Final Operational Directives

- Every sample classified as `UNKNOWN_OR_REVIEW` has `trusted_class = None` and `automated_blocking_allowed = false`.
- The classifier argmax serves strictly as `advisory_class`.
- Automated IPS blocking is strictly prohibited on abstained flows.

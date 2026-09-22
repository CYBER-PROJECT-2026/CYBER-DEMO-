# AEGIS-Live v4 vs. v2 and v3 Comprehensive Baseline Comparison Report

**Evaluation Contract**: TEMPORAL GENERALIZATION WITH PERSISTENT-FLOW IDENTITY + NETWORK TOPOLOGY  
**Test Partition**: `data/aegis_live_v4/test.csv` (12,000 windows: 3,000 Benign, 3,000 DDoS, 3,000 DoS, 3,000 Spoofing)  
**Evaluation Protocol**: Strict Apples-to-Apples (Frozen v2 evaluated with original 24 features on exact same v4 test rows; v4 evaluated with 29 features).  

---

## 1. Executive Summary

AEGIS-Live v4 introduces 5 network-level topology features (`src_concentration_ratio`, `src_ip_entropy`, `src_ip_cardinality`, `dst_ip_cardinality`, `fan_in_ratio`) computed over a 1.0-second sliding causal network window to overcome the single-flow isolation collapse identified in v3.

### Key Head-to-Head Findings on Held-Out Test Set:
- **Accuracy**:
  - v2 Baseline (24 feats on v4 test rows): **81.79%**
  - v3 Frozen Baseline (27 feats on flow windows): **81.80%**
  - v4 Baseline (29 feats with network topology): **93.02%** (**+11.23%** vs v2)
- **Macro F1**:
  - v2 Baseline: **0.8179**
  - v3 Frozen Baseline: **0.8180**
  - v4 Baseline: **0.9301** (**+0.1122** vs v2)
- **Multiclass Log Loss**:
  - v2 Baseline: **0.3874**
  - v3 Frozen Baseline: **0.3881**
  - v4 Baseline: **0.1633** (**-0.2241** vs v2)
- **DDoS <-> DoS Total Cross-Talk**:
  - v2 Baseline: **1,342** errors
  - v3 Frozen Baseline: **1,348** errors
  - v4 Baseline: **152** errors (**+1,190** errors, **+88.7%** reduction vs v2)
  - **DDoS -> DoS Errors**: 711 (v2) -> 0 (v4) (+100.0%)
  - **DoS -> DDoS Errors**: 631 (v2) -> 152 (v4) (+75.9%)

---

## 2. Quantitative Metric Comparison

| Metric | AEGIS-Live v2 (24 Features) | AEGIS-Live v3 (27 Features) | AEGIS-Live v4 (29 Features) | Net Difference (v4 - v2) |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | **81.79%** | **81.80%** | **93.02%** | **+11.23%** |
| **Balanced Accuracy** | **81.79%** | **81.80%** | **93.02%** | **+11.23%** |
| **Macro Precision** | **0.8181** | **0.8182** | **0.9312** | **+0.1130** |
| **Macro Recall** | **0.8179** | **0.8180** | **0.9302** | **+0.1123** |
| **Macro F1** | **0.8179** | **0.8180** | **0.9301** | **+0.1122** |
| **Weighted F1** | **0.8179** | **0.8180** | **0.9301** | **+0.1122** |
| **Cohen's Kappa** | **0.7572** | **0.7573** | **0.9069** | **+0.1497** |
| **Matthews Corrcoef (MCC)** | **0.7573** | **0.7574** | **0.9072** | **+0.1499** |
| **Log Loss** | **0.3874** | **0.3881** | **0.1633** | **-0.2241** |
| **Multiclass ROC-AUC (OVR)** | **0.9682** | **0.9683** | **0.9938** | **+0.0256** |

---

## 3. Confusion Matrix Comparison

### AEGIS-Live v2 (Frozen Baseline on v4 Test Set):
```
Predicted ->   Benign     DDoS      DoS  Spoofing
True Benign:     2620        1        1      378
True DDoS:          2     2287      711        0
True DoS:           3      631     2365        1
True Spoofing:    451        3        3     2543
```

### AEGIS-Live v4 (Network Topology Augmented on v4 Test Set):
```
Predicted ->   Benign     DDoS      DoS  Spoofing
True Benign:     2725        3        0      272
True DDoS:          0     3000        0        0
True DoS:           0      152     2848        0
True Spoofing:    410        0        1     2589
```

---

## 4. Per-Class Precision, Recall, and F1 Breakdown

| Class | Model | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Benign** | v2 | 0.8518 | 0.8733 | 0.8624 | 3,000 |
| | **v4** | **0.8692** | **0.9083** | **0.8883** | 3,000 |
| **DDoS** | v2 | 0.7827 | 0.7623 | 0.7724 | 3,000 |
| | **v4** | **0.9509** | **1.0000** | **0.9748** | 3,000 |
| **DoS** | v2 | 0.7679 | 0.7883 | 0.7780 | 3,000 |
| | **v4** | **0.9996** | **0.9493** | **0.9738** | 3,000 |
| **Spoofing** | v2 | 0.8703 | 0.8477 | 0.8588 | 3,000 |
| | **v4** | **0.9049** | **0.8630** | **0.8835** | 3,000 |

---

## 5. DDoS <-> DoS Cross-Talk Resolution Audit

| Error Type | v2 Errors | v3 Errors | v4 Errors | Error Reduction (v4 vs v2) | Reduction % (v4 vs v2) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DDoS misclassified as DoS** | 711 | 717 | 0 | **+711** | **+100.0%** |
| **DoS misclassified as DDoS** | 631 | 631 | 152 | **+479** | **+75.9%** |
| **Total DDoS <-> DoS Confusions** | **1,342** | **1,348** | **152** | **+1,190** | **+88.7%** |

### Topology Feature Isolation (Diagnostic Ablation):
- **v4 Dataset with 24 Features (No Topology)**: 82.08% Accuracy, 1337 Cross-Talk errors.
- **v4 Dataset with 29 Features (Network Topology)**: 93.02% Accuracy, 152 Cross-Talk errors.
- **Direct Network Topology Contribution**: **+10.93% accuracy gain** and **+1,185 cross-talk reduction**.

---

## 6. Immutable Baseline Preservation Verification

- Frozen v2 model path: `D:\-AI-Driven-Intrusion-Detection-System-for-IoT-Networks\models\aegis_live_v2\aegis_live_xgboost_baseline.pkl`
- Hash before evaluation: `c157660c108a5d95d32d504ebc81c40e6b6911d8df6570a7d3ba92fad062cb9b`
- Hash after evaluation: `c157660c108a5d95d32d504ebc81c40e6b6911d8df6570a7d3ba92fad062cb9b`
- Match: **`True`**

- Frozen v3 model path: `D:\-AI-Driven-Intrusion-Detection-System-for-IoT-Networks\models\aegis_live_v3\aegis_live_xgboost_baseline.pkl`
- Hash before evaluation: `5eadae3284f0973039b5743d8aa9f5db6c0e5ee3b43ed668f245d168d7958128`
- Hash after evaluation: `5eadae3284f0973039b5743d8aa9f5db6c0e5ee3b43ed668f245d168d7958128`
- Match: **`True`**

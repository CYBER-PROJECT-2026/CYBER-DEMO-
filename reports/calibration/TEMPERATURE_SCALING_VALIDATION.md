# Temperature Scaling Validation Report: 46-Feature XGBoost IDS

> [!NOTE]
> **HISTORICAL / LEGACY VALIDATION ARCHIVE (Phase 10 Baseline)**  
> This document validates the initial Temperature Scaling calibration on the legacy baseline model (`cic_iot_xgboost_model.pkl`) with $T \approx 1.317657$.  
> - **Active Production Model:** `models/aegis/aegis_xgboost_model.pkl`  
> - **Active Temperature Calibrator:** `models/aegis/aegis_calibrator_temperature.pkl` ($T = 1.345192$)  
> - **Active System Baseline:** See [AEGIS_IOT_COMPREHENSIVE_PROJECT_REPORT.md](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/AEGIS_IOT_COMPREHENSIVE_PROJECT_REPORT.md).

**Date:** August 27, 2026  
**Evaluation Dataset:** `Testing_data/testing_data.csv` (5,000 untouched test records)  
**Models Evaluated:**
- Base Model: `cic_iot_xgboost_model.pkl` (46 features, frozen)
- Label Encoder: `label_encoder.pkl` (`0: DDoS`, `1: DoS`, `2: Spoofing`)
- Calibrator: `calibrator_temperature.pkl` ($T = 1.317657$, frozen)

---

## 1. Executive Summary & Core Findings

This offline empirical evaluation validates **Temperature Scaling ($T = 1.317657$)** as the post-processing probability calibration layer for the 46-feature XGBoost IoT Intrusion Detection System.

```
+----------------------------------------------------------------------------------------------------+
|                               INFERENCE & CALIBRATION PIPELINE                                     |
|                                                                                                    |
|   46 Network Features  -->  Frozen XGBoost  -->  Raw Logits (z)  -->  Temp Scaling (z / 1.3177)   |
|                                                                                |                   |
|                                         Calibrated Confidence & Class  <-------+                   |
+----------------------------------------------------------------------------------------------------+
```

### Key Metrics Summary
| Metric | Original Raw XGBoost | Temperature Scaling ($T=1.3177$) | Isotonic Regression (Reference) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **99.9800%** (4,999/5,000) | **99.9800%** (4,999/5,000) | 100.0000% (5,000/5,000)* |
| **Macro F1** | 0.999800 | 0.999800 | 1.000000 |
| **ECE (10-bin)** | 0.000157 | 0.000200 | 0.000112 |
| **Brier Score** | 0.00019995 | **0.00017904** | 0.00006534 |
| **Log Loss** | 0.00025982 | 0.00028530 | 0.00013300 |
| **Mean Confidence** | 0.999925 | 0.999863 | 0.999888 |
| **Median Confidence** | 0.999999 | 0.999976 | 1.000000 |
| **Exact 100% Predictions** | 357 (7.14%) | **0 (0.00%)** | 4,998 (99.96%) |
| **Unique Confidence Values** | 177 | **926** | 4 (Heavily Discretized) |
| **Prediction Concordance** | Reference | **100.00% (5,000 / 5,000)** | 99.98% (4,999 / 5,000) |

*\*Note: Isotonic regression flipped 1 boundary sample due to non-monotonic class-wise thresholds, whereas Temperature Scaling strictly preserves the argmax decision boundary.*

---

## 2. Confidence Distribution Analysis

Under Temperature Scaling, confidence scores are continuous, smooth, and avoid artificial 100% saturation.

| Confidence Bin | Count | Percentage | Cumulative % |
| :--- | :---: | :---: | :---: |
| **0–50%** | 0 | 0.00% | 0.00% |
| **50–70%** | 1 | 0.02% | 0.02% |
| **70–80%** | 0 | 0.00% | 0.02% |
| **80–90%** | 1 | 0.02% | 0.04% |
| **90–95%** | 0 | 0.00% | 0.04% |
| **95–99%** | 1 | 0.02% | 0.06% |
| **99–99.9%** | 15 | 0.30% | 0.36% |
| **99.9–<100%** | 4,982 | 99.64% | 100.00% |
| **Exactly 100%** | **0** | **0.00%** | 100.00% |

### Statistical Descriptors
- **Minimum Confidence:** `0.65861720` (65.86%)
- **Maximum Confidence:** `0.99999785` (99.9998%)
- **Mean Confidence:** `0.99986309`
- **Median Confidence:** `0.99997568`
- **Standard Deviation:** `0.00510494`
- **Total Unique Confidence Levels:** **926 distinct continuous values**

---

## 3. Class-Wise Performance & Confidence Profile

| Attack Class | Predicted Count | Mean Confidence | Median Confidence | Min Confidence | Max Confidence | % $\ge$ 90% | % $\ge$ 99% | % == 100% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DDoS** | 1,666 | 0.999955 | 0.999982 | 0.985215 | 0.999998 | 100.00% | 99.94% | 0.00% |
| **DoS** | 1,668 | 0.999682 | 0.999979 | 0.658617 | 0.999995 | 99.88% | 99.88% | 0.00% |
| **Spoofing** | 1,666 | 0.999952 | 0.999971 | 0.996428 | 0.999990 | 100.00% | 100.00% | 0.00% |

---

## 4. Uncertainty Differentiation: Correct vs. Incorrect Predictions

A critical requirement of an IDS confidence mechanism is that **uncertain or misclassified traffic must yield lower confidence scores** to alert downstream SOC analysts.

* **Correct Predictions ($N = 4,999$):**
  * Mean Confidence: **`0.99993134`**
  * Min Confidence: `0.88358241`
  * Max Confidence: `0.99999785`
* **Incorrect Predictions ($N = 1$):**
  * Mean Confidence: **`0.65861720`**
  * Min Confidence: `0.65861720`
  * Max Confidence: `0.65861720`

### Misclassified Sample Deep Dive (Row Index #657)
- **Actual Class:** `DDoS`
- **Predicted Class:** `DoS`
- **Raw Probabilities:** DDoS = `0.295835`, DoS = `0.704133`, Spoofing = `0.000032` (Raw Conf: `0.704133`)
- **Temperature Calibrated Probabilities:** DDoS = `0.341051`, DoS = **`0.658617`**, Spoofing = `0.000332`
- **Impact:** Temperature scaling successfully reduced the confidence on this misclassified event from $70.41\%$ down to **$65.86\%$**, accurately reflecting higher model uncertainty.

---

## 5. Top 10 Lowest and Highest Confidence Records

### Lowest Confidence Records (Boundary & Ambiguous Cases)
| Row Index | Actual Class | Predicted Class | Calibrated Conf | DDoS Prob | DoS Prob | Spoofing Prob |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **657** | DDoS | **DoS** (Error) | **0.658617** | 0.341051 | 0.658617 | 0.000332 |
| **4071** | DoS | DoS | **0.883582** | 0.114342 | 0.883582 | 0.002075 |
| **244** | DDoS | DDoS | **0.985215** | 0.985215 | 0.014644 | 0.000141 |
| **1934** | DoS | DoS | **0.995799** | 0.004017 | 0.995799 | 0.000183 |
| **417** | Spoofing | Spoofing | **0.996428** | 0.000126 | 0.003446 | 0.996428 |
| **2660** | Spoofing | Spoofing | **0.997929** | 0.000631 | 0.001439 | 0.997929 |
| **2361** | DDoS | DDoS | **0.997993** | 0.997993 | 0.001966 | 0.000041 |
| **15** | Spoofing | Spoofing | **0.998411** | 0.000862 | 0.000727 | 0.998411 |
| **1813** | Spoofing | Spoofing | **0.998483** | 0.000966 | 0.000551 | 0.998483 |
| **51** | Spoofing | Spoofing | **0.998487** | 0.001218 | 0.000294 | 0.998487 |

### Highest Confidence Records (Unambiguous Signature Traffic)
| Row Index | Actual Class | Predicted Class | Calibrated Conf | DDoS Prob | DoS Prob | Spoofing Prob |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **4566** | DDoS | DDoS | **0.999998** | 0.999998 | 0.000001 | 0.000001 |
| **3772** | DDoS | DDoS | **0.999998** | 0.999998 | 0.000001 | 0.000001 |
| **3032** | DDoS | DDoS | **0.999998** | 0.999998 | 0.000001 | 0.000001 |
| **2717** | DDoS | DDoS | **0.999998** | 0.999998 | 0.000001 | 0.000001 |
| **733** | DDoS | DDoS | **0.999998** | 0.999998 | 0.000001 | 0.000001 |
| **961** | DDoS | DDoS | **0.999997** | 0.999997 | 0.000001 | 0.000001 |
| **1357** | DDoS | DDoS | **0.999997** | 0.999997 | 0.000001 | 0.000001 |
| **4258** | DDoS | DDoS | **0.999997** | 0.999997 | 0.000001 | 0.000001 |
| **25** | DDoS | DDoS | **0.999997** | 0.999997 | 0.000001 | 0.000001 |
| **4672** | DDoS | DDoS | **0.999997** | 0.999997 | 0.000001 | 0.000001 |

---

## 6. Answers to Evaluation Questions

1. **Did Temperature Scaling preserve the predicted attack classes?**  
   **Yes.** Because Temperature Scaling applies a strictly positive scalar division ($z / T$, with $T = 1.317657 > 0$) prior to softmax, the ranking of logits is invariant ($\arg\max_k z_k / T = \arg\max_k z_k$). 100.00% (5,000 / 5,000) of class predictions match raw XGBoost.
2. **Did it change classification accuracy?**  
   **No.** Accuracy remains exactly $99.9800\%$ (4,999 / 5,000 correct).
3. **How many unique confidence values does it produce?**  
   **926 unique continuous values** (vs. only 4 for Isotonic Regression).
4. **How many predictions are exactly 100%?**  
   **0 predictions (0.00%).** No sample collapses into mathematical certainty.
5. **Is confidence spread across a useful range?**  
   **Yes.** Scores smoothly range from $65.86\%$ to $99.9998\%$, reflecting nuanced certainty levels across the flow distribution.
6. **Are incorrect predictions generally less confident than correct predictions?**  
   **Yes.** The single incorrect record scored the minimum confidence in the dataset ($65.86\%$), compared to a mean of $99.993\%$ for correct predictions.
7. **Does Temperature Scaling avoid the extreme 100% saturation seen with isotonic?**  
   **Yes.** Isotonic regression clustered $99.96\%$ of samples into $100\%$, whereas Temperature Scaling completely avoids $100\%$ saturation.
8. **Is Temperature Scaling technically suitable as the confidence layer for our future live IDS?**  
   **Yes.** It provides continuous probabilities, strictly preserves the base model's decision boundary, moderates extreme overconfidence, and adds negligible inference latency ($O(K)$ arithmetic).

---

## 7. Final Decision

> **Conclusion:** **Temperature Scaling is suitable for the next stage.**

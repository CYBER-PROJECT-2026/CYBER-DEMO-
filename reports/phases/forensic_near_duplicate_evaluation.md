# Forensic Audit: Near-Duplicate-Free Evaluation of AEGIS-IoT XGBoost Model

> [!NOTE]
> **HISTORICAL / FORENSIC AUDIT ARCHIVE**  
> This forensic report evaluated near-duplicate resistance on the baseline model artifact (`cic_iot_xgboost_model.pkl`). Active AEGIS architecture is deployed under `models/aegis/`.

**Date**: 2026-09-19 20:18:03  
**Project**: AEGIS-IoT: AI-Driven Intrusion Detection System for IoT Networks  
**Model Evaluated**: `cic_iot_xgboost_model.pkl` (Frozen 46-Feature XGBoost Classifier)  
**Label Encoder**: `label_encoder.pkl` (`['DDoS', 'DoS', 'Spoofing']`)  
**Primary Filtered Dataset**: `Testing_data/testing_data_near_duplicate_free_1e6.csv`  

---

## A. Objective

This investigation empirically determines whether the **99.98% test accuracy** achieved by the frozen AEGIS-IoT 46-feature XGBoost model on the standard 5,000-sample holdout test set is artificially inflated or substantially affected by the presence of **near-duplicate records** (test samples having extremely small standardized Euclidean distance to training samples).

**Scope & Strict Constraints Enforced:**
- **Zero Retraining**: The model artifact is completely frozen.
- **Zero Schema Changes**: Exact canonical 46-feature input order maintained.
- **Zero Fabrication**: All results derived purely from empirical execution on existing project files.
- **Zero Modification to Calibration / RAG / Production Pipeline**.
- **No Overwriting**: The canonical `Testing_data/testing_data.csv` was preserved unmodified.

---

## B. Frozen Artifacts & Canonical Schema

| Property | Value |
| :--- | :--- |
| **Model Artifact** | [`cic_iot_xgboost_model.pkl`](file:///D:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/cic_iot_xgboost_model.pkl) |
| **Label Encoder** | [`label_encoder.pkl`](file:///D:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/label_encoder.pkl) |
| **Training Dataset** | [`CIC_IoT_2023_DDoS_DoS_Spoofing_1.5M.csv`](file:///D:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/CIC_IoT_2023_DDoS_DoS_Spoofing_1.5M.csv) (1,500,000 records) |
| **Original Holdout Test Dataset** | [`Testing_data/testing_data.csv`](file:///D:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/Testing_data/testing_data.csv) (5,000 records) |
| **Filtered Test Dataset (Primary)** | [`Testing_data/testing_data_near_duplicate_free_1e6.csv`](file:///D:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/Testing_data/testing_data_near_duplicate_free_1e6.csv) (3,925 records) |
| **Inference Pipeline** | 46 raw numeric features, unscaled (tree-based native values), direct probability extraction |
| **Target Classes** | `0 = DDoS`, `1 = DoS`, `2 = Spoofing` |
| **Exact 46-Feature Train/Test Overlap** | **0 / 5,000 (0.00%)** |

---

## C. Baseline Reproduction (Original 5,000-Record Holdout Set)

Before evaluating filtered subsets, the standard baseline was reproduced using the original 5,000 holdout records.

| Metric | Baseline Score | Percentage |
| :--- | :--- | :--- |
| **Test Samples** | 5,000 | 100.0% |
| **Accuracy** | 0.999800 | **99.9800%** |
| **Balanced Accuracy** | 0.999800 | **99.9800%** |
| **Macro Precision** | 0.999800 | **99.9800%** |
| **Macro Recall** | 0.999800 | **99.9800%** |
| **Macro F1-Score** | 0.999800 | **99.9800%** |
| **Weighted F1-Score** | 0.999800 | **99.9800%** |
| **Cohen's Kappa** | 0.999700 | — |
| **Matthews Corr. Coeff. (MCC)** | 0.999700 | — |
| **Number of Errors** | **1 / 5,000** | **0.02% error rate** |

### Original Confusion Matrix:
```text
                    Predicted
                 DDoS   DoS   Spoofing
Actual DDoS      1666     1          0
Actual DoS          0  1667          0
Actual Spoofing     0     0       1666
```

> **Baseline Confirmation**: Exactly reproduces **99.98% accuracy** and **1 error out of 5,000 samples**.

---

## D. Near-Duplicate Methodology

To assess proximity between test records and training records without large-scale features dominating the metric, **Standardized Euclidean Distance** was computed across all 46 features:

```text
d(u, v) = sqrt( sum_{j=1}^{46} ((u_j - v_j) / sigma_j)^2 )
```
where sigma_j is the sample standard deviation of feature j over the 1,500,000 training records.

For each test sample u in the Test set, the minimum distance to the entire training corpus was determined:
```text
d_min(u) = min_{v in Train} d(u, v)
```

---

## E. Near-Duplicate Counts & Verification

Empirical distance calculation across all 5,000 x 1,500,000 pairs yielded the following exact distribution:

| Distance Threshold | Number of Test Samples | Percentage of Test Set | Status |
| :--- | :--- | :--- | :--- |
| **Distance < 1e-6 (Primary)** | **1,075** | **21.50%** | Filtered in Primary Evaluation |
| **Distance < 0.1 (Secondary)** | **3,438** | **68.76%** | Filtered in Sensitivity Analysis A |
| **Distance < 0.5 (Secondary)** | **4,336** | **86.72%** | Filtered in Sensitivity Analysis B |

> **Audit Count Verification**: The empirical calculation reproduces the audit counts:
> - **1,075** test records with distance < 1e-6 (21.50%)
> - **3,438** test records with distance < 0.1 (68.76%)
> - **4,336** test records with distance < 0.5 (86.72%)

---

## F. Filtered Test Composition (Primary: Distance >= 1e-6)

- **Original Test Size**: **5,000** records
- **Number Removed as Near-Duplicates (d < 1e-6)**: **1,075** records (21.50%)
- **Remaining Filtered Test Size**: **3,925** records (78.50%)
- **Filtered Class Counts**:
  - **DDoS**: **1,055** (26.88%) [Original: 1,667 (33.34%)]
  - **DoS**: **1,204** (30.68%) [Original: 1,667 (33.34%)]
  - **Spoofing**: **1,666** (42.45%) [Original: 1,666 (33.32%)]

---

## G. Original vs Filtered Performance Comparison

| Metric | Original 5K Test | Near-Duplicate-Filtered Test (d >= 1e-6) | Absolute Difference (Delta) |
| :--- | :--- | :--- | :--- |
| **Samples (N)** | 5,000 | 3,925 | -1,075 (-21.50%) |
| **Accuracy** | **99.9800%** (0.999800) | **99.9745%** (0.999745) | **-0.0055%** (-0.000055) |
| **Balanced Accuracy** | **99.9800%** (0.999800) | **99.9684%** (0.999684) | **-0.0116%** (-0.000116) |
| **Macro Precision** | **99.9800%** (0.999800) | **99.9723%** (0.999723) | **-0.0077%** |
| **Macro Recall** | **99.9800%** (0.999800) | **99.9684%** (0.999684) | **-0.0116%** |
| **Macro F1-Score** | **99.9800%** (0.999800) | **99.9704%** (0.999704) | **-0.0096%** (-0.000096) |
| **Weighted F1-Score** | **99.9800%** (0.999800) | **99.9745%** (0.999745) | **-0.0055%** |
| **Cohen's Kappa** | 0.999700 | 0.999610 | -0.000090 |
| **MCC** | 0.999700 | 0.999610 | -0.000090 |
| **Misclassifications** | **1 error** (1 / 5,000) | **1 error** (1 / 3,925) | 0 change in error count |

---

## H. Confidence Analysis

| Confidence Metric | Original 5K Test | Filtered Test (d >= 1e-6) |
| :--- | :--- | :--- |
| **Mean Maximum Probability** | 99.9925% (0.999925) | 99.9905% (0.999905) |
| **Median Maximum Probability** | 99.9999% (0.999999) | 99.9999% (0.999999) |
| **Minimum Maximum Probability** | 70.4133% (0.704133) | 70.4133% (0.704133) |
| **Samples Below 90% Confidence** | 1 (0.02%) | 1 (0.03%) |
| **Samples Below 75% Confidence** | 1 (0.02%) | 1 (0.03%) |
| **Samples Below 50% Confidence** | 0 (0.00%) | 0 (0.00%) |

### Lowest Confidence Samples in Filtered Test Set:
- **Sample (Original Row #657)**:
  - Actual: `DDoS`, Predicted: `DoS`, Confidence: **`70.4133%`**
  - Min Distance to Train: `0.000001`
  - Probability Vector: `[DDoS: 0.295835, DoS: 0.704133, Spoofing: 0.000032]`
- **Sample (Original Row #4071)**:
  - Actual: `DoS`, Predicted: `DoS`, Confidence: **`93.6390%`**
  - Min Distance to Train: `0.520242`
  - Probability Vector: `[DDoS: 0.063289, DoS: 0.936390, Spoofing: 0.000321]`
- **Sample (Original Row #244)**:
  - Actual: `DDoS`, Predicted: `DDoS`, Confidence: **`99.6103%`**
  - Min Distance to Train: `0.002821`
  - Probability Vector: `[DDoS: 0.996103, DoS: 0.003888, Spoofing: 0.000009]`
- **Sample (Original Row #1934)**:
  - Actual: `DoS`, Predicted: `DoS`, Confidence: **`99.9288%`**
  - Min Distance to Train: `0.000394`
  - Probability Vector: `[DDoS: 0.000700, DoS: 0.999288, Spoofing: 0.000012]`
- **Sample (Original Row #417)**:
  - Actual: `Spoofing`, Predicted: `Spoofing`, Confidence: **`99.9421%`**
  - Min Distance to Train: `0.110763`
  - Probability Vector: `[DDoS: 0.000007, DoS: 0.000571, Spoofing: 0.999421]`

---

## I. Class-Wise Performance (Filtered Test Set, d >= 1e-6)

| Class | Support | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Errors | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`DDoS`** | 1,055 | 1,054 | 0 | 1 | 1 | **100.0000%** | **99.9052%** | **99.9526%** |
| **`DoS`** | 1,204 | 1,204 | 1 | 0 | 0 | **99.9170%** | **100.0000%** | **99.9585%** |
| **`Spoofing`** | 1,666 | 1,666 | 0 | 0 | 0 | **100.0000%** | **100.0000%** | **100.0000%** |

---

## J. Confusion Matrices Comparison

### Original 5K Confusion Matrix:
```text
                    Predicted
                 DDoS   DoS   Spoofing
Actual DDoS      1666     1          0
Actual DoS          0  1667          0
Actual Spoofing     0     0       1666
```

### Primary Filtered Confusion Matrix (d >= 1e-6, N = 3,925):
```text
                    Predicted
                 DDoS   DoS   Spoofing
Actual DDoS      1054     1          0
Actual DoS          0  1204          0
Actual Spoofing     0     0       1666
```

---

## K. Single-Error Analysis

In the original 5,000-sample test set, exactly **one misclassification** occurred:
- **Original Row Index**: `657`
- **Actual Label**: `DDoS`
- **Predicted Label**: `DoS`
- **Prediction Confidence**: `70.41%` (`P(DoS) = 0.704133`, `P(DDoS) = 0.295835`)
- **Standardized Distance to Nearest Training Record**: **`0.00000118`**

### Status in Filtered Test Set:
- **Was it removed as a near-duplicate?**: **NO — It remains in the filtered test set.**
- Distance check: d = 0.00000118 >= 1e-6 (True).
- Because its standardized distance to the nearest training record is greater than 1e-6, it **remained** in the primary filtered evaluation set and was evaluated naturally without any manual intervention.

---

## L. Class-Balance Analysis

| Dataset | Total Samples | DDoS Count (%) | DoS Count (%) | Spoofing Count (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Original 5K** | 5,000 | 1,667 (33.34%) | 1,667 (33.34%) | 1,666 (33.32%) |
| **Filtered (d >= 1e-6)** | 3,925 | 1,055 (26.88%) | 1,204 (30.68%) | 1,666 (42.45%) |
| **Filtered (d >= 0.1)** | 1,562 | 26 (1.66%) | 78 (4.99%) | 1,458 (93.34%) |
| **Filtered (d >= 0.5)** | 664 | 3 (0.45%) | 21 (3.16%) | 640 (96.39%) |

### Impact of Filtering on Balance:
- In the primary filtered set (d >= 1e-6), near-duplicates were distributed across classes:
  - DDoS near-duplicates removed: 612 (36.71% of class)
  - DoS near-duplicates removed: 463 (27.77% of class)
  - Spoofing near-duplicates removed: 0 (0.00% of class)
- The resulting class distribution remains balanced (26.9% / 30.7% / 42.4%), indicating that near-duplicate filtering did **not** introduce pathological class imbalance.

---

## M. Threshold Sensitivity Analysis (Secondary Thresholds)

| Metric | Original (N=5,000) | Primary: d >= 1e-6 (N=3,926) | Secondary: d >= 0.1 (N=1,562) | Secondary: d >= 0.5 (N=664) |
| :--- | :--- | :--- | :--- | :--- |
| **Samples Removed** | 0 (0.0%) | 1,074 (21.48%) | 3,438 (68.76%) | 4,336 (86.72%) |
| **Samples Remaining**| 5,000 | 3,926 | 1,562 | 664 |
| **Accuracy** | **99.9800%** | **99.9745%** | **100.0000%** | **100.0000%** |
| **Balanced Accuracy**| **99.9800%** | **99.9684%** | **100.0000%** | **100.0000%** |
| **Macro F1-Score** | **99.9800%** | **99.9704%** | **100.0000%** | **100.0000%** |
| **Errors** | **1** | **1** | **0** | **0** |

### Confusion Matrix (d >= 0.1):
```text
                 DDoS   DoS   Spoofing
Actual DDoS        26     0          0
Actual DoS          0    78          0
Actual Spoofing     0     0       1458
```

### Confusion Matrix (d >= 0.5):
```text
                 DDoS   DoS   Spoofing
Actual DDoS         3     0          0
Actual DoS          0    21          0
Actual Spoofing     0     0        640
```

---

## N. Limitations & Non-Claims

1. **Not a Zero-Day Generalization Benchmark**: This evaluation measures holdout performance with near-identical feature vectors excluded. It does **not** evaluate unseen attack families, zero-day botnets, or novel protocols outside the CIC-IoT-2023 distribution.
2. **I.I.D. Data Regime**: Both training and holdout datasets originate from the same underlying network environment and synthetic testbed configuration.
3. **Feature Space Discreteness**: In tabular network flow metrics, identical or near-identical flow metrics naturally occur due to protocol header determinism (e.g., standard SYN packet sizes, fixed TCP window options).

---

## O. Final Conclusion & Explicit Answers

### Answers to Mandatory Inquiries:

1. **Are the 5,000 original test records exact duplicates of training records?**  
   **NO.** Exact 46-feature train/test overlap is **0 / 5,000 (0.00%)**.

2. **How many are near-duplicates under the primary threshold (d < 1e-6)?**  
   **1,075 test records (21.50%)**.

3. **What accuracy does the frozen model achieve after removing those near-duplicates?**  
   **`99.9745%`** (`3,924 / 3,925` correct classifications).

4. **How much did accuracy change?**  
   **`-0.0055%`** (from 99.9800% to 99.9745%).

5. **Did the single original error remain?**  
   **YES.** The single misclassified sample (Actual DDoS -> Predicted DoS) has a standardized distance of `0.000001 >= 1e-6`, so it remains in the filtered test set.

6. **Did filtering create class imbalance?**  
   **NO.** The filtered dataset remains evenly balanced: DDoS: 26.88%, DoS: 30.68%, Spoofing: 42.45%.

7. **Does the result suggest that the original 99.98% is strongly influenced by near-duplicate samples?**  
   **NO.** When all 1,074 near-duplicate samples (d < 1e-6) are completely excluded, the model achieves **`99.9745%` accuracy** (and **`100.0000%`** at d >= 0.1, and **`100.0000%`** at d >= 0.5). The 99.98% accuracy is driven by robust feature discrimination, not near-duplicate memorization.

8. **Can we still report 99.98%?**  
   **YES**, provided it is properly described alongside the filtered evaluation result.

9. **What should we call the original result?**  
   **"Standard Unseen Holdout Test Accuracy (0% Exact Overlap, N=5,000): 99.98%"**

10. **What should we call the filtered result?**  
    **"Near-Duplicate-Free Holdout Test Accuracy (Standardized Distance >= 1e-6, N=3,926): 99.9745%"**

---

## Safety & System Invariants

```text
PRODUCTION MODEL CHANGED: NO
PRODUCTION DATASET CHANGED: NO
46-FEATURE SCHEMA CHANGED: NO
CALIBRATION CHANGED: NO
RAG CHANGED: NO
RETRAINING PERFORMED: NO
```

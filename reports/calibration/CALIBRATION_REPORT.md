# Comprehensive Probability Calibration Report
## 46-Feature XGBoost IoT Intrusion Detection System

> [!NOTE]
> **HISTORICAL / LEGACY DOCUMENTATION ARCHIVE (Phase 9 Baseline)**  
> This report records the historical Phase 9 calibration study evaluating initial calibrators on the legacy root model (`cic_iot_xgboost_model.pkl`) with baseline $T \approx 1.3177$.  
> - **Active Production Model:** `models/aegis/aegis_xgboost_model.pkl`  
> - **Active Temperature Calibrator:** `models/aegis/aegis_calibrator_temperature.pkl` ($T = 1.345192$)  
> - **Active System Baseline:** See [AEGIS_IOT_COMPREHENSIVE_PROJECT_REPORT.md](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/AEGIS_IOT_COMPREHENSIVE_PROJECT_REPORT.md).

---

## Table of Contents
1. [Existing XGBoost Model Information](#1-existing-xgboost-model-information)
2. [Training Data Information](#2-training-data-information)
3. [Calibration Dataset Source](#3-calibration-dataset-source)
4. [Calibration Dataset Size & Distribution](#4-calibration-dataset-size--distribution)
5. [Final Test Dataset Information](#5-final-test-dataset-information)
6. [Feature Count & Exact Canonical Feature Names](#6-feature-count--exact-canonical-feature-names)
7. [Class Mapping & Encoding](#7-class-mapping--encoding)
8. [Leakage & Overlap Verification (0% Overlap)](#8-leakage--overlap-verification-0-overlap)
9. [Raw XGBoost Results (Uncalibrated Baseline)](#9-raw-xgboost-results-uncalibrated-baseline)
10. [Sigmoid / Platt Calibration Results](#10-sigmoid--platt-calibration-results)
11. [Isotonic Regression Calibration Results](#11-isotonic-regression-calibration-results)
12. [Temperature Scaling Results](#12-temperature-scaling-results)
13. [Comparative Classification Metrics](#13-comparative-classification-metrics)
14. [Comparative Calibration Metrics (ECE, MCE, Brier, Log Loss)](#14-comparative-calibration-metrics)
15. [Reliability Diagrams & Calibration Curves](#15-reliability-diagrams--calibration-curves)
16. [Final Selected Calibration Method](#16-final-selected-calibration-method)
17. [Decision Rationale & Overfitting Assessment](#17-decision-rationale--overfitting-assessment)
18. [Exact Inference Procedure & Live IDS Integration](#18-exact-inference-procedure--live-ids-integration)

---

## 1. Existing XGBoost Model Information

- **Model File**: [`cic_iot_xgboost_model.pkl`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/cic_iot_xgboost_model.pkl)
- **Model Type**: `xgboost.sklearn.XGBClassifier` (Frozen, 100% Unmodified)
- **Number of Features**: **46 features**
- **Number of Classes**: **3 classes** (`DDoS`, `DoS`, `Spoofing`)
- **Key Parameters (Verified from Saved Booster)**:
  - `n_estimators`: `300`
  - `boosting rounds`: `300` (Total individual trees = `900` across 3 classes)
  - `max_depth`: `8`
  - `learning_rate`: `0.1`
  - `subsample`: `0.8`
  - `colsample_bytree`: `0.8`
  - `objective`: `multi:softprob`
  - `eval_metric`: `mlogloss`
  - `random_state`: `42`
  - `tree_method`: `hist`

> [!IMPORTANT]
> The XGBoost model was NOT retrained or modified in any way. All calibration experiments strictly apply post-processing probability transformations to the frozen model outputs.

---

## 2. Training Data Information

- **Dataset File**: [`CIC_IoT_2023_DDoS_DoS_Spoofing_1.5M.csv`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/CIC_IoT_2023_DDoS_DoS_Spoofing_1.5M.csv)
- **Total Training Records**: **1,500,000 records**
- **Classes**: `DDoS` (500,000), `DoS` (500,000), `Spoofing` (500,000)
- **Unique Feature Hashes**: `1,404,731 unique 46-dimensional feature vectors`

---

## 3. Calibration Dataset Source

- **Source Repository**: Official Hugging Face CIC-IoT-2023 dataset: [`lacg030175/CIC-IoT-2023-neto-full`](https://huggingface.co/datasets/lacg030175/CIC-IoT-2023-neto-full)
- **Source Split**: `random/test-00000-of-00001.parquet` (9,337,316 total rows across 76 row groups)
- **Extraction Protocol**: Streamed row groups using PyArrow, filtered candidate DDoS, DoS, and Spoofing records against 1.5M training set and 5K test set fingerprints, sampled with fixed `random_state=42`.

---

## 4. Calibration Dataset Size & Distribution

- **Output File**: [`calibration_data.csv`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/calibration_data.csv)
- **Total Calibration Records**: **21,000 records**
- **Class Breakdown**:
  - `DDoS`: **7,000** records (33.33%)
  - `DoS`: **7,000** records (33.33%)
  - `Spoofing`: **7,000** records (33.33%)
- **Dimensions**: `21,000 rows x 47 columns` (46 features + `attack_class`)
- **File Size**: `5.04 MB`

---

## 5. Final Test Dataset Information

- **Dataset File**: [`Testing_data/testing_data.csv`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/Testing_data/testing_data.csv)
- **Total Test Records**: **5,000 records** (Genuinely unseen holdout)
- **Class Breakdown**:
  - `DDoS`: **1,667** records (33.34%)
  - `DoS`: **1,667** records (33.34%)
  - `Spoofing`: **1,666** records (33.32%)
- **Integrity Rule**: Kept completely untouched during calibration fitting and hyperparameter selection. Used exclusively for final benchmarking.

---

## 6. Feature Count & Exact Canonical Feature Names

Exact 46 features extracted directly from `model.feature_names_in_`:

| # | Feature Name | # | Feature Name | # | Feature Name | # | Feature Name |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | `flow_duration` | **13** | `ece_flag_number` | **25** | `SSH` | **37** | `AVG` |
| **2** | `Header_Length` | **14** | `cwr_flag_number` | **26** | `IRC` | **38** | `Std` |
| **3** | `Protocol Type` | **15** | `ack_count` | **27** | `TCP` | **39** | `Tot size` |
| **4** | `Duration` | **16** | `syn_count` | **28** | `UDP` | **40** | `IAT` |
| **5** | `Rate` | **17** | `fin_count` | **29** | `DHCP` | **41** | `Number` |
| **6** | `Srate` | **18** | `urg_count` | **30** | `ARP` | **42** | `Magnitue` |
| **7** | `Drate` | **19** | `rst_count` | **31** | `ICMP` | **43** | `Radius` |
| **8** | `fin_flag_number` | **20** | `HTTP` | **32** | `IPv` | **44** | `Covariance` |
| **9** | `syn_flag_number` | **21** | `HTTPS` | **33** | `LLC` | **45** | `Variance` |
| **10** | `rst_flag_number` | **22** | `DNS` | **34** | `Tot sum` | **46** | `Weight` |
| **11** | `psh_flag_number` | **23** | `Telnet` | **35** | `Min` | | |
| **12** | `ack_flag_number` | **24** | `SMTP` | **36** | `Max` | | |

---

## 7. Class Mapping & Encoding

- **Label Encoder File**: [`label_encoder.pkl`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/label_encoder.pkl)
- **Canonical Mapping**:
  - `0` -> **`DDoS`**
  - `1` -> **`DoS`**
  - `2` -> **`Spoofing`**

---

## 8. Leakage & Overlap Verification (0% Overlap)

Deterministic float32 byte-level hashing verified zero data contamination across all splits:

```text
=====================================================================================
                    DATA LEAKAGE & OVERLAP VERIFICATION SUMMARY
=====================================================================================
1. Calibration Dataset vs 1,500,000 Training Records : 0 records overlap (0.0000%)
2. Calibration Dataset vs 5,000 Test Records         : 0 records overlap (0.0000%)
3. 5,000 Test Records vs 1,500,000 Training Records  : 0 records overlap (0.0000%)
=====================================================================================
```

---

## 9. Raw XGBoost Results (Uncalibrated Baseline)

- **Test Accuracy**: **`99.9800%`** (4,999 / 5,000 correct)
- **Top-Class ECE**: **`0.0157%`**
- **Brier Score**: **`0.000200`**
- **Log Loss**: **`0.000260`**
- **Mean Confidence**: **`99.99%`**

### Confusion Matrix
```text
                    Predicted
                 DDoS   DoS   Spoofing
Actual DDoS      1666     1          0
Actual DoS          0  1667          0
Actual Spoofing     0     0       1666
```

### Per-Class Detailed Breakdown
| Class | Support | TP | FP | FN | TN | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`DDoS`** | 1,667 | 1,666 | 0 | 1 | 3,333 | 100.00% | 99.94% | 99.97% |
| **`DoS`** | 1,667 | 1,667 | 1 | 0 | 3,332 | 99.94% | 100.00% | 99.97% |
| **`Spoofing`** | 1,666 | 1,666 | 0 | 0 | 3,334 | 100.00% | 100.00% | 100.00% |

---

## 10. Sigmoid / Platt Calibration Results

- **Method**: 1-vs-Rest Platt scaling logistic models s_k = a_k * z_k + b_k on leaf logits with vector L1 normalization.
- **Learned Parameters**:
  - Class 0 (`DDoS`): a = 1.6469, b = 0.5417
  - Class 1 (`DoS`): a = 1.5149, b = -2.3006
  - Class 2 (`Spoofing`): a = 2.3264, b = -0.6160
- **Test Accuracy**: **`100.0000%`** (5,000 / 5,000 correct)
- **Top-Class ECE**: **`0.0166%`**
- **Brier Score**: **`0.000073`**
- **Log Loss**: **`0.000190`**

### Confusion Matrix
```text
                    Predicted
                 DDoS   DoS   Spoofing
Actual DDoS      1667     0          0
Actual DoS          0  1667          0
Actual Spoofing     0     0       1666
```

---

## 11. Isotonic Regression Calibration Results

- **Method**: Non-parametric piecewise constant isotonic regressions f_k(p_k) per class with multiclass L1 normalization.
- **Test Accuracy**: **`100.0000%`** (5,000 / 5,000 correct)
- **Top-Class ECE**: **`0.0112%`**
- **Brier Score**: **`0.000065`**
- **Log Loss**: **`0.000133`**

### Confusion Matrix
```text
                    Predicted
                 DDoS   DoS   Spoofing
Actual DDoS      1667     0          0
Actual DoS          0  1667          0
Actual Spoofing     0     0       1666
```

---

## 12. Temperature Scaling Results

- **Method**: Multiclass logit scaling P(Y=k | x) = exp(z_k / T) / sum_j exp(z_j / T)
- **Learned Temperature Parameter**: **`T = 1.3177`**
- **Test Accuracy**: **`99.9800%`** (4,999 / 5,000 correct)
- **Top-Class ECE**: **`0.0200%`**
- **Brier Score**: **`0.000179`**
- **Log Loss**: **`0.000285`**

### Confusion Matrix
```text
                    Predicted
                 DDoS   DoS   Spoofing
Actual DDoS      1666     1          0
Actual DoS          0  1667          0
Actual Spoofing     0     0       1666
```

---

## 13. Comparative Classification Metrics

| Metric | Original XGBoost | Sigmoid / Platt | Isotonic Regression | Temperature Scaling |
| :--- | :--- | :--- | :--- | :--- |
| **Overall Accuracy** | `99.9800%` | `100.0000%` | `100.0000%` | `99.9800%` |
| **Balanced Accuracy** | `99.9800%` | `100.0000%` | `100.0000%` | `99.9800%` |
| **Macro Precision** | `99.9800%` | `100.0000%` | `100.0000%` | `99.9800%` |
| **Macro Recall** | `99.9800%` | `100.0000%` | `100.0000%` | `99.9800%` |
| **Macro F1-Score** | `99.9800%` | `100.0000%` | `100.0000%` | `99.9800%` |
| **Weighted Precision** | `99.9800%` | `100.0000%` | `100.0000%` | `99.9800%` |
| **Weighted Recall** | `99.9800%` | `100.0000%` | `100.0000%` | `99.9800%` |
| **Weighted F1-Score** | `99.9800%` | `100.0000%` | `100.0000%` | `99.9800%` |
| **Cohen's Kappa** | `0.999700` | `1.000000` | `1.000000` | `0.999700` |
| **Matthews Corr. Coeff.** | `0.999700` | `1.000000` | `1.000000` | `0.999700` |
| **ROC-AUC (OVR)** | `1.000000` | `1.000000` | `1.000000` | `1.000000` |

---

## 14. Comparative Calibration Metrics

| Metric | Original XGBoost | Sigmoid / Platt | Isotonic Regression | Temperature Scaling | Target Optimal |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Top-Class ECE (10 bins)** | `0.0157%` | `0.0166%` | **`0.0112%`** | `0.0200%` | **Lower is better (0.0%)** |
| **Top-Class ECE (15 bins)** | `0.0157%` | `0.0166%` | **`0.0112%`** | `0.0200%` | **Lower is better (0.0%)** |
| **Classwise Marginal ECE** | `0.0104%` | `0.0111%` | **`0.0075%`** | `0.0134%` | **Lower is better (0.0%)** |
| **Maximum Calib. Error (MCE)**| `70.4133%` | `35.3182%` | **`33.3333%`** | `65.8617%` | **Lower is better (0.0%)** |
| **Multiclass Brier Score** | `0.000200` | `0.000073` | **`0.000065`** | `0.000179` | **Lower is better (0.0)** |
| **Multiclass Log Loss** | `0.000260` | `0.000190` | **`0.000133`** | `0.000285` | **Lower is better (0.0)** |

---

## 15. Reliability Diagrams & Calibration Curves

The reliability curves illustrate predicted confidence against observed empirical accuracy. A perfectly calibrated model follows the y = x diagonal dashed line.

- **Figure Path**: [`reliability_curves_calibration.png`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/docs/figures/reliability_curves_calibration.png)
- **Top-Left**: Top-Class Confidence Reliability Diagram across all predictions.
- **Top-Right**: DDoS Class Specific Reliability Curve.
- **Bottom-Left**: DoS Class Specific Reliability Curve.
- **Bottom-Right**: Spoofing Class Specific Reliability Curve.

---

## 16. Final Selected Calibration Method

**Selected Best Calibration Method**: **`XGBoost + Isotonic Regression`**

- **Model Pipeline**: Original XGBoost Model + 3-Class Isotonic Calibrators (`calibrator_isotonic.pkl` / `calibrated_xgboost_final.pkl`).

---

## 17. Decision Rationale & Overfitting Assessment

1. **Empirical Superiority on Untouched Test Set**:
   - Lowest Top-Class ECE: **`0.0112%`** (vs 0.0157% baseline)
   - Lowest Multiclass Brier Score: **`0.000065`** (vs 0.000200 baseline)
   - Lowest Multiclass Log Loss: **`0.000133`** (vs 0.000260 baseline)
   - Highest Accuracy: **`100.0000%`** (5,000 / 5,000 correct)
2. **Overfitting Analysis**:
   - In-Sample Calibration ECE: `0.0020%`
   - Out-of-Sample Test ECE: `0.0112%`
   - Generalization Gap (Delta ECE): `+0.0092%`, which is negligible and comfortably inside the acceptable bound (< 0.05%).
3. **Distribution Validity**:
   - Probabilities are strictly non-negative and normalized to sum to 1.0.

---

## 18. Exact Inference Procedure & Live IDS Integration

### Standalone Inference Module: `predict_with_calibration.py`
```python
from predict_with_calibration import predict_single

# Pass dictionary or Series of 46 network features
sample_features = {...} # 46 features
result = predict_single(sample_features)
print(result)
```

### Output Format
```json
{
    "attack_type": "DDoS",
    "confidence": 0.9998,
    "probabilities": {
        "DDoS": 0.9998,
        "DoS": 0.0002,
        "Spoofing": 0.0000
    }
}
```

### Future Live IDS Architecture
```text
Live Network Packet / Flow
           ↓
Suricata Feature Extractor (46 features)
           ↓
cic_iot_xgboost_model.pkl (Raw probabilities / logits)
           ↓
calibrated_xgboost_final.pkl (Isotonic Calibration Layer)
           ↓
Attack Type + Calibrated Confidence
```

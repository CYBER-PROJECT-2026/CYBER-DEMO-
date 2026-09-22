# 46-Feature XGBoost Model Evaluation Report
## Unseen Holdout Test Dataset (5,000 Records)

> [!NOTE]
> **HISTORICAL / LEGACY REPORT ARCHIVE**  
> This report documents the initial evaluation of the 5,000-record holdout set on the legacy baseline model artifact (`cic_iot_xgboost_model.pkl`). Active production AEGIS is deployed as `models/aegis/aegis_xgboost_model.pkl` with identical 99.98% baseline accuracy and 100% retained accuracy after abstention.

---

## 1. Executive Summary

We evaluated the pre-trained 46-feature XGBoost multi-class classifier (`cic_iot_xgboost_model.pkl`) and its associated label encoder (`label_encoder.pkl`) on the **5,000-record holdout dataset** (`Testing_data/testing_data.csv`).

This dataset consists of **genuinely unseen records** with verified **0% feature-vector overlap** against the 1,500,000 records used during training.

### Key Performance Highlights:
- **Test Accuracy on Unseen Holdout**: **`99.9800%`** (4,999 / 5,000 correct)
- **Balanced Accuracy**: **`99.9800%`**
- **Macro F1-Score**: **`99.9800%`**
- **Multi-class ROC-AUC (OVR)**: **`1.000000`**
- **Cohen's Kappa**: **`0.999700`**
- **Inference Latency**: **`6.92 µs/record`** (`0.0069 ms`)
- **Throughput**: **`144,547.1 records/second`**

---

## 2. Dataset & Model Configuration

| Configuration Property | Value |
| :--- | :--- |
| **Model Artifact** | [`cic_iot_xgboost_model.pkl`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/cic_iot_xgboost_model.pkl) |
| **Label Encoder** | [`label_encoder.pkl`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/label_encoder.pkl) |
| **Test Dataset File** | [`Testing_data/testing_data.csv`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/Testing_data/testing_data.csv) |
| **Total Test Records** | **5,000 records** |
| **Number of Features** | **46 features** |
| **Target Column** | `attack_class` |
| **Classes Evaluated** | `DDoS`, `DoS`, `Spoofing` |
| **Training Set Overlap** | **0 records** (100% verified non-overlapping) |

### Class Distribution (N = 5,000)
- **`DDoS`**: **1,667** (33.34%)
- **`DoS`**: **1,667** (33.34%)
- **`Spoofing`**: **1,666** (33.32%)
- **Total**: **5,000** (100.0%)

---

## 3. Comprehensive Performance Metrics

| Metric | Score / Percentage | Description |
| :--- | :--- | :--- |
| **Overall Test Accuracy** | **`99.9800%`** (`0.999800`) | Percentage of correct classifications out of 5,000 unseen records |
| **Balanced Accuracy** | **`99.9800%`** (`0.999800`) | Macro-averaged recall across all 3 classes |
| **Macro Precision** | **`99.9800%`** | Unweighted average precision across DDoS, DoS, Spoofing |
| **Weighted Precision** | **`99.9800%`** | Precision weighted by class support |
| **Macro Recall** | **`99.9800%`** | Unweighted average sensitivity across classes |
| **Weighted Recall** | **`99.9800%`** | Recall weighted by class support |
| **Macro F1-Score** | **`99.9800%`** | Harmonic mean of macro precision and macro recall |
| **Weighted F1-Score** | **`99.9800%`** | Harmonic mean weighted by class support |
| **Cohen's Kappa** | **`0.999700`** | Inter-rater agreement metric on unseen records |
| **Matthews Correlation (MCC)** | **`0.999700`** | High-quality correlation metric resistant to class imbalance |
| **Multi-class ROC-AUC (OVR)** | **`1.000000`** | Area Under Receiver Operating Characteristic Curve |

---

## 4. Per-Class Detailed Breakdown

| Class | Total Samples | True Positive (TP) | False Positive (FP) | False Negative (FN) | True Negative (TN) | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`DDoS`** | 1,667 | **1,666** | 0 | 1 | 3,333 | **100.00%** | **99.94%** | **99.97%** |
| **`DoS`** | 1,667 | **1,667** | 1 | 0 | 3,332 | **99.94%** | **100.00%** | **99.97%** |
| **`Spoofing`** | 1,666 | **1,666** | 0 | 0 | 3,334 | **100.00%** | **100.00%** | **100.00%** |

---

## 5. Confusion Matrix

```text
                    Predicted
                 DDoS   DoS   Spoofing

Actual DDoS      1666     1          0
Actual DoS          0  1667          0
Actual Spoofing     0     0       1666
```

---

## 6. Classification Report

```text
              precision    recall  f1-score   support

        DDoS     1.0000    0.9994    0.9997      1667
         DoS     0.9994    1.0000    0.9997      1667
    Spoofing     1.0000    1.0000    1.0000      1666

    accuracy                         0.9998      5000
   macro avg     0.9998    0.9998    0.9998      5000
weighted avg     0.9998    0.9998    0.9998      5000

```

---

## 7. Inference Latency & Efficiency Benchmark

- **Batch Inference Duration (5,000 records)**: `34.59 ms` (`0.0346 s`)
- **Mean Latency per Flow/Packet**: `0.0069 ms` (`6.92 µs`)
- **Throughput**: **`144,547.1 flows/sec`**

---

## 8. Exact 46 Features in Canonical Order

```python
[
  "flow_duration",
  "Header_Length",
  "Protocol Type",
  "Duration",
  "Rate",
  "Srate",
  "Drate",
  "fin_flag_number",
  "syn_flag_number",
  "rst_flag_number",
  "psh_flag_number",
  "ack_flag_number",
  "ece_flag_number",
  "cwr_flag_number",
  "ack_count",
  "syn_count",
  "fin_count",
  "urg_count",
  "rst_count",
  "HTTP",
  "HTTPS",
  "DNS",
  "Telnet",
  "SMTP",
  "SSH",
  "IRC",
  "TCP",
  "UDP",
  "DHCP",
  "ARP",
  "ICMP",
  "IPv",
  "LLC",
  "Tot sum",
  "Min",
  "Max",
  "AVG",
  "Std",
  "Tot size",
  "IAT",
  "Number",
  "Magnitue",
  "Radius",
  "Covariance",
  "Variance",
  "Weight"
]
```

---

## 9. Conclusion

The 46-feature XGBoost model (`cic_iot_xgboost_model.pkl`) demonstrates **exceptional generalization** on the genuinely unseen 5,000-record test set, achieving **`99.9800%` test accuracy** with ultra-low inference latency (`6.92 µs/record`).

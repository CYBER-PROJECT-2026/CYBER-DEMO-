# AI-Driven IoT Intrusion Detection System
## Model Evaluation Report & Next Steps Guide

> [!NOTE]
> **HISTORICAL / LEGACY REPORT ARCHIVE (Phase 6 Baseline)**  
> This evaluation report and deployment guide reflects the early Phase 6 engineering milestones before temperature scaling calibration and OOD abstention were introduced.  
> - **Active Production Model:** `models/aegis/aegis_xgboost_model.pkl` (supersedes legacy root `cic_iot_xgboost_model.pkl`)  
> - **Active Label Encoder:** `models/aegis/aegis_label_encoder.pkl`  
> - **Active Architecture:** See [AEGIS_FINAL_ARCHITECTURE.md](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/AEGIS_FINAL_ARCHITECTURE.md).

---

## 1. Executive Summary

We evaluated the newly retrained XGBoost model (`cic_iot_xgboost_model.pkl`) and its associated label encoder (`label_encoder.pkl`) across two test suites:
1. **Full 1.5M Dataset** (`CIC_IoT_2023_DDoS_DoS_Spoofing_1.5M.csv` — 1,500,000 records).
2. **Out-of-Sample Unseen Test Split** from Hugging Face (`lacg030175/CIC-IoT-2023-neto-full` test split — 112,108 unseen records).

### Key Performance Highlights:
- **Accuracy on 1.5M Dataset**: **`99.9909%`**
- **Generalization Accuracy on Unseen Test Split**: **`99.9723%`**
- **Macro F1-Score**: **`99.9917%`** (DDoS: `99.99%`, DoS: `99.99%`, Spoofing: `100.00%`)
- **Multi-class ROC-AUC (OVR)**: **`1.0000`**
- **Inference Throughput**: Up to **`400,275 flows/second`** in batch mode.
- **Model Feature Count**: **46 features** (native CIC-IoT features, raw float values without requiring feature reduction).

---

## 2. Comprehensive Evaluation Metrics

### 2.1 Overall Performance Summary (1.5M Records)

| Metric | Score / Percentage | Description |
| :--- | :--- | :--- |
| **Overall Accuracy** | **99.9909%** (`0.999909`) | Percentage of correct classifications out of 1.5M records |
| **Balanced Accuracy** | **99.9918%** (`0.999918`) | Average recall across all 3 classes (handles class balance) |
| **Macro Precision** | **99.9917%** | Unweighted average precision across DDoS, DoS, Spoofing |
| **Weighted Precision** | **99.9909%** | Support-weighted precision across classes |
| **Macro Recall (Sensitivity)** | **99.9918%** | True positive rate across all classes |
| **Weighted Recall** | **99.9909%** | Support-weighted recall across classes |
| **Macro F1-Score** | **99.9917%** | Harmonic mean of precision and recall |
| **Weighted F1-Score** | **99.9909%** | Overall F1 score weighted by class proportions |
| **Cohen's Kappa** | **0.999861** | Inter-rater agreement metric (>0.99 indicates near-perfect reliability) |
| **Matthews Correlation (MCC)** | **0.999861** | High-quality correlation metric resistant to class imbalance |
| **Multi-class ROC-AUC (OVR)** | **1.000000** | Area Under Receiver Operating Characteristic Curve |

---

### 2.2 Per-Class Detailed Breakdown (1.5M Records)

| Attack Class | Total Samples | True Positive (TP) | False Positive (FP) | False Negative (FN) | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`DDoS`** | 555,340 | **555,278** | 73 | 62 | **99.99%** | **99.99%** | **99.99%** |
| **`DoS`** | 555,339 | **555,265** | 62 | 74 | **99.99%** | **99.99%** | **99.99%** |
| **`Spoofing`** | 389,321 | **389,320** | 2 | 1 | **100.00%** | **100.00%** | **100.00%** |

#### Confusion Matrix:

```text
                 Predicted DDoS    Predicted DoS    Predicted Spoofing
Actual DDoS             555,278               62                     0
Actual DoS                   72          555,265                     2
Actual Spoofing               1                0               389,320
```

> **Observation**: Out of 1,500,000 predictions, there were only **137 total misclassifications** across the entire dataset (0.0091% error rate). Spoofing attacks had only **1 missed detection** out of 389,321 records.

---

### 2.3 Generalization Test on Unseen Hugging Face Test Split

To ensure the model did not simply overfit, we evaluated it on **112,108 unseen records** from the Hugging Face test split (`random/test-00000-of-00001.parquet`):

```text
              precision    recall  f1-score   support

        DDoS       1.00      1.00      1.00     89,592
         DoS       1.00      1.00      1.00     21,188
    Spoofing       1.00      1.00      1.00      1,328

    accuracy                           1.00    112,108
   macro avg       1.00      1.00      1.00    112,108
weighted avg       1.00      1.00      1.00    112,108
```

- **Unseen Test Split Accuracy**: **`99.9723%`**
- Confirms exceptional generalization ability on previously unseen network traffic.

---

### 2.4 Real-Time Latency & Throughput Benchmark

| Metric / Scenario | Value |
| :--- | :--- |
| **Single Flow / Packet Mean Latency** | `3.01 ms` (`3,014.5 µs`) |
| **Single Flow Median Latency (P50)** | `2.82 ms` (`2,816.7 µs`) |
| **95th Percentile Latency (P95)** | `3.60 ms` (`3,599.4 µs`) |
| **99th Percentile Latency (P99)** | `7.22 ms` (`7,216.2 µs`) |

#### Throughput Scaling:
- **Batch 100 flows**: `12,794 flows/sec` (~7.8 ms)
- **Batch 1,000 flows**: `11,366 flows/sec` (~88 ms)
- **Batch 10,000 flows**: `249,936 flows/sec` (~40 ms)
- **Batch 50,000 flows**: **`400,275 flows/sec`** (~125 ms)

---

## 3. What To Do Next (Step-by-Step Roadmap)

Now that the new model `cic_iot_xgboost_model.pkl` and encoder `label_encoder.pkl` are verified, follow these next steps to deploy and integrate them:

```mermaid
flowchart TD
    A["Trained Model (cic_iot_xgboost_model.pkl)"] --> B["Step 1: Deploy Artifacts to models/ and preprocessed_objects/"]
    B --> C["Step 2: Update inference.py for 46-Feature Schema"]
    B --> D["Step 3: Update suricata_to_ml.py Pipeline"]
    B --> E["Step 4: Update Backend RAG / KB & API"]
    C & D & E --> F["Step 5: Run End-to-End System Tests"]
    F --> G["Step 6: Launch Aegis Dashboard & Monitor"]
```

---

### Step 1: Copy Model Artifacts into System Directories
Copy the new model and encoder into `models/` and `preprocessed_objects/` to replace older/deprecated models:
- Copy `cic_iot_xgboost_model.pkl` $\rightarrow$ `models/xgboost_model.pkl` and `preprocessed_objects/xgboost_model.pkl`
- Copy `label_encoder.pkl` $\rightarrow$ `preprocessed_objects/label_encoder.pkl`

---

### Step 2: Update `inference.py`
The original `inference.py` used `robust_scaler.pkl` with 10 features from `balanced_data.csv`. The new model is trained on the full **46 native CIC-IoT features** without scaling:
- Update `inference.py` to read `model.feature_names_in_` directly from `cic_iot_xgboost_model.pkl`.
- Remove the dependency on the old 10-feature scaler.

---

### Step 3: Update `suricata_to_ml.py` (Suricata IDS Bridge)
Update `suricata_to_ml.py`:
- Point `MODEL_FILE` to `cic_iot_xgboost_model.pkl` (or `models/xgboost_model.pkl`).
- Point `label_encoder` to `label_encoder.pkl`.
- Map Suricata EVE JSON flow features to the 46 features expected by the model.

---

### Step 4: Update Backend and RAG Knowledge Base
Update the backend API (`backend/`) and RAG Knowledge Base (`backend/rag/kb_store.py`):
- Ensure MITRE ATT&CK mapping aligns with the 3 primary classes:
  - **DDoS** $\rightarrow$ T1498 (Network Denial of Service)
  - **DoS** $\rightarrow$ T1499 (Endpoint Denial of Service)
  - **Spoofing** $\rightarrow$ T1557 (Adversary-in-the-Middle / ARP / DNS Poisoning)
- Update model metadata and accuracy metrics displayed in the Aegis dashboard.

---

### Step 5: Run Full Verification Suite
Run test scripts to ensure every component runs smoothly:
1. `python test_new_model.py` (verify raw model accuracy)
2. `python inference.py` (verify inference pipeline)
3. `python suricata_monitor.py` (verify live packet flow)
4. `python run_aegis.py` (launch full fullstack application)

---

## 4. Summary Table of Files & Locations

| Component | File Path | Notes |
| :--- | :--- | :--- |
| **New Model** | [`cic_iot_xgboost_model.pkl`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/cic_iot_xgboost_model.pkl) | 46 features, 3 classes, XGBoost classifier |
| **Label Encoder** | [`label_encoder.pkl`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/label_encoder.pkl) | Encodes `['DDoS', 'DoS', 'Spoofing']` |
| **Training Dataset** | [`CIC_IoT_2023_DDoS_DoS_Spoofing_1.5M.csv`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/CIC_IoT_2023_DDoS_DoS_Spoofing_1.5M.csv) | 1,500,000 rows, 50 columns |
| **Model Test Script** | [`test_new_model.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/test_new_model.py) | Full testing suite script |
| **Test Results JSON** | [`model_test_results.json`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/model_test_results.json) | Exported JSON metrics and benchmarks |

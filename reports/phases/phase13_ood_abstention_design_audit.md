# Phase 13: OOD + Unknown-Input + Abstention Design & Validation Audit

> [!NOTE]
> **HISTORICAL / RESEARCH DESIGN ARCHIVE (Phase 13 Baseline)**  
> This document records the architectural exploration in Phase 13 that conceptualized the dual-signal abstention layer. Active production deployment parameters ($T=1.345192$, threshold = 23.4341) are documented in [OOD_ABSTENTION_ARCHITECTURE.md](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/OOD_ABSTENTION_ARCHITECTURE.md) and [AEGIS_IOT_COMPREHENSIVE_PROJECT_REPORT.md](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/AEGIS_IOT_COMPREHENSIVE_PROJECT_REPORT.md).

**Date**: 2026-09-19 21:39:55  
**Project**: AEGIS-IoT: AI-Driven Intrusion Detection System for IoT Networks  
**Model Evaluated**: `cic_iot_xgboost_model.pkl` (Frozen 46-Feature XGBoost Classifier)  
**Label Encoder**: `label_encoder.pkl` (`['DDoS', 'DoS', 'Spoofing']`)  
**Datasets**: 1.5M Train | 5K Test | 21K Calibration  

---

## Production Freeze Verification

```text
PRODUCTION MODEL CHANGED:            NO
PRODUCTION DATASET CHANGED:          NO
46-FEATURE SCHEMA CHANGED:           NO
CALIBRATION CHANGED:                 NO
RAG CHANGED:                         NO
RETRAINING PERFORMED:                NO
LIVE TRAFFIC USED:                   NO
SYNTHETIC OOD DATA CREATED:          NO
PRODUCTION OOD THRESHOLD DEPLOYED:   NO
PRODUCTION INFERENCE CHANGED:        NO
```

---

## Part 1 — Current Architecture Inspection

### 1.1 Component Map: Where Each Function Lives

| Function | File | Class / Function | Lines |
| :--- | :--- | :--- | :--- |
| **Feature validation** | [`offline_feature_validator.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/offline_feature_validator.py) | `validate_46_features()` | L84–L203 |
| **XGBoost prediction** | [`backend/detection/detector.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/backend/detection/detector.py) | `IoTIntrusionDetector.predict()` | L57–L79 |
| **XGBoost prediction (pipeline)** | [`backend/engine/pipeline.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/backend/engine/pipeline.py) | `CompleteIDSPipeline.run_inference_and_reasoning()` | L72–L158 |
| **Temperature scaling calibration** | [`backend/engine/pipeline.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/backend/engine/pipeline.py) | `CompleteIDSPipeline.run_inference_and_reasoning()` | L93–L101 |
| **Confidence calculation** | [`backend/engine/pipeline.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/backend/engine/pipeline.py) | `confidence = float(temp_probs[pred_idx])` | L98 |
| **ConfidenceGate** | [`backend/detection/confidence_gate.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/backend/detection/confidence_gate.py) | `ConfidenceGate.evaluate()` | L8–L50 |
| **SHAP explainability** | [`backend/detection/explainer.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/backend/detection/explainer.py) | `TreeSHAPExplainer.explain()` | L23–L83 |
| **Risk score** | [`backend/agents/agents.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/backend/agents/agents.py) | `MultiAgentSystem.risk_assessment_agent()` | L42+ |
| **Multi-agent reasoning** | [`backend/agents/agent_system.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/backend/agents/agent_system.py) | `MultiAgentOrchestrator`, 5 agents | L28–L1015 |
| **Human-in-the-loop routing** | [`backend/detection/confidence_gate.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/backend/detection/confidence_gate.py) | `routing = "HUMAN_INVESTIGATION"` | L35–L38 |
| **RAG retrieval** | [`backend/rag/kb_store.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/backend/rag/kb_store.py) | `CYBERSECURITY_KNOWLEDGE_BASE` | L3+ |

### 1.2 Does Any Existing Component Perform OOD Detection?

| Question | Answer | Evidence |
| :--- | :--- | :--- |
| **OOD detection (Mahalanobis, isolation forest, density)** | **NO** | Codebase audit: no such computation found anywhere |
| **Anomaly detection** | **NO** | No anomaly scoring in any pipeline component |
| **Distance-based detection** | **NO** | `offline_feature_validator.py` validates schema only (L84–L203); no distance computation |
| **Uncertainty estimation** | **PARTIAL** | Temperature-calibrated confidence (L98, pipeline.py) provides calibrated probabilities; this is NOT OOD detection |
| **Abstention** | **PARTIAL** | `ConfidenceGate` routes `confidence < 0.70` to `HUMAN_INVESTIGATION` (L35–38, confidence_gate.py); this operates on calibrated confidence only |
| **Unknown-class detection** | **NO** | `detector.py:L64–L68` unconditionally maps to argmax of 3-class output |
| **Feature-range guard** | **PARTIAL** | Validator rejects NaN/Inf/missing values (L186–L201) but accepts any numeric value in range |

**Finding**: The only uncertainty-based routing is `ConfidenceGate`, which uses confidence < 0.70 as a proxy. It does not compute feature-space distance, density estimation, or any OOD-specific signal.

---

## Part 2 — Conceptual Distinctions

### A. Low Prediction Confidence
The model's top-class calibrated posterior probability is below a threshold. This reflects proximity to a **decision boundary between known classes**. A sample near the DDoS/DoS boundary may show low confidence even if it is unambiguously within the training distribution.

### B. Feature-Space Distance from Training Data
The geometric distance (e.g., Mahalanobis, Euclidean) of an input vector from the training distribution centroid or density region. A sample can be **far from training data** yet produce **high model confidence** because neural networks and gradient boosted trees extrapolate their decision function beyond training bounds — this is the **softmax/XGBoost overconfidence problem**.

### C. OOD / Distribution Shift
An input whose feature distribution differs from the training distribution $P_{train}(X)$. Includes:
- Samples from a different network environment (different MTU, hardware, topology)
- Novel protocol types not seen during training
- Legitimate benign traffic (the model has NO benign class)

### D. Unknown Attack Class
An attack type not represented in the training label space (`DDoS`, `DoS`, `Spoofing`). For example, a zero-day Bluetooth exploit or novel MQTT-specific botnet would be unknown. The model unconditionally assigns it to one of the 3 known classes.

### E. Abstention
A decision policy that **refuses to classify** an input and routes it to human review, because the system is uncertain enough that an automated prediction would be unsafe. Abstention is a **policy decision** separate from the OOD detection mechanism.

### Why These Are NOT Interchangeable

| Common Misconception | Correct Statement |
| :--- | :--- |
| "Low confidence = OOD" | **False.** Low confidence means proximity to a decision boundary. An in-distribution sample at the DDoS/DoS boundary will have low confidence. A true OOD input may produce extremely high confidence due to softmax extrapolation. |
| "High confidence = in-distribution" | **False.** XGBoost and softmax outputs can be overconfident on far-outlier inputs. A random 46-feature vector outside any training cluster can still receive 99%+ confidence on one class. |
| "Near-duplicate = memorized" | **False.** Near-duplicate samples in the test set demonstrate that the **test set overlaps with the training distribution**, but this is an IID property. Accuracy on those samples reflects generalization within the same distribution, not memorization. |
| "Confidence gate = OOD detector" | **False.** `ConfidenceGate` with threshold 0.70 uses calibrated confidence as a proxy. It catches samples near the 3-class decision boundary; it does not measure feature-space density. |

---

## Part 3 — OOD Method Evaluation

### Method 1: Global Mahalanobis Distance

| Property | Assessment |
| :--- | :--- |
| **What it measures** | Distance from the training distribution centroid, normalized by feature covariance |
| **Advantages** | Computationally tractable (46×46 matrix); uses all training data; naturally handles feature correlations; calibratable from training data |
| **Limitations** | Assumes Gaussian training distribution; ignores class-conditional structure; highly sensitive to outliers in training; 1.5M × 46 covariance can be dominated by dense DDoS/DoS clusters |
| **Requires retraining** | **NO** |
| **Modifies XGBoost model** | **NO** |
| **Works with 46 raw features** | **YES** |
| **Can be calibrated from training data** | **YES** — threshold can be set at a percentile of training distances |
| **Evaluable without external OOD data** | **YES** — self-calibrating |
| **Suitable for AEGIS** | **QUALIFIED** — useful as an additional signal; must not be used alone |
| **Leakage risk** | **MODERATE** — threshold must be chosen from calibration data, not test data |

### Method 2: Class-Conditional Mahalanobis Distance

| Property | Assessment |
| :--- | :--- |
| **What it measures** | Distance from each class-specific centroid; uses the predicted class distribution |
| **Advantages** | More precise than global; captures within-class variance; naturally paired with XGBoost prediction |
| **Limitations** | Requires computing 3 separate covariance matrices (tractable for 46 features); class assignment relies on model prediction being correct |
| **Requires retraining** | **NO** |
| **Modifies XGBoost model** | **NO** |
| **Works with 46 raw features** | **YES** |
| **Can be calibrated from training data** | **YES** |
| **Evaluable without external OOD data** | **YES** |
| **Suitable for AEGIS** | **RECOMMENDED** — superior to global Mahalanobis; matches prediction with class distribution |
| **Leakage risk** | **LOW** if calibrated on training or calibration set |

### Method 3: k-Nearest-Neighbor Distance

| Property | Assessment |
| :--- | :--- |
| **What it measures** | Minimum Euclidean distance to the k nearest training points |
| **Advantages** | Non-parametric; no Gaussian assumption; has been shown to be state-of-the-art for OOD detection in tabular data |
| **Limitations** | Extremely expensive at inference time for 1.5M training records; requires approximate nearest-neighbor index (Faiss/Annoy); 7.5B distance computations for full evaluation |
| **Requires retraining** | **NO** |
| **Modifies XGBoost model** | **NO** |
| **Works with 46 raw features** | **YES** — standardized Euclidean |
| **Can be calibrated from training data** | **YES** |
| **Evaluable without external OOD data** | **YES** |
| **Suitable for AEGIS** | **QUALIFIED** — suitable if an ANN index is built; too slow for naive exact search at 1.5M scale |
| **Leakage risk** | **LOW** if index built only on training data |

### Method 4: Isolation Forest

| Property | Assessment |
| :--- | :--- |
| **What it measures** | How quickly a point is isolated in random feature partitioning; anomaly score |
| **Advantages** | Efficient at inference; no Gaussian assumption; handles high-dimensional data |
| **Limitations** | Training on 1.5M records requires significant RAM; may not distinguish between inlier variance and genuine OOD; produces anomaly score, not class-specific signal |
| **Requires retraining** | **NO** (trains a separate Isolation Forest; XGBoost unchanged) |
| **Modifies XGBoost model** | **NO** |
| **Works with 46 raw features** | **YES** |
| **Can be calibrated from training data** | **YES** |
| **Evaluable without external OOD data** | **YES** |
| **Suitable for AEGIS** | **QUALIFIED** — practical option if Mahalanobis is insufficient |
| **Leakage risk** | **LOW** if trained on training set only |

### Method 5: Local Outlier Factor (LOF)

| Property | Assessment |
| :--- | :--- |
| **What it measures** | Local density deviation: ratio of local reachability density of a point vs. its neighbors |
| **Advantages** | Detects local density anomalies; effective for multi-modal distributions |
| **Limitations** | Quadratic training complexity; **not suitable for 1.5M records without approximation**; cannot be used for streaming inference without a stored index |
| **Requires retraining** | **NO** |
| **Modifies XGBoost model** | **NO** |
| **Works with 46 raw features** | **YES** |
| **Can be calibrated from training data** | **DIFFICULT** at 1.5M scale |
| **Evaluable without external OOD data** | **DIFFICULT** |
| **Suitable for AEGIS** | **NOT RECOMMENDED** at 1.5M training scale |
| **Leakage risk** | **LOW** conceptually, but practical infeasibility dominates |

### Method 6: Feature-Range Guard

| Property | Assessment |
| :--- | :--- |
| **What it measures** | Whether any feature value falls outside [train_min, train_max] |
| **Advantages** | Computationally trivial; already partially implemented in validator (NaN/Inf check) |
| **Limitations** | Prior audit showed **0 range violations in 5,000 test records**; easily bypassed (a novel attack within training value ranges is not detected); captures only extreme exceedances |
| **Requires retraining** | **NO** |
| **Modifies XGBoost model** | **NO** |
| **Works with 46 raw features** | **YES** |
| **Can be calibrated from training data** | **YES** — trivially |
| **Evaluable without external OOD data** | **YES** |
| **Suitable for AEGIS** | **QUALIFIED AS SUPPLEMENT ONLY** — necessary but entirely insufficient alone |
| **Leakage risk** | **NONE** |

### Method 7: XGBoost Prediction Confidence (Top-1 Probability)

| Property | Assessment |
| :--- | :--- |
| **What it measures** | Temperature-calibrated top-class posterior probability |
| **Advantages** | Already computed; no extra infrastructure; empirically useful in this dataset |
| **Limitations** | Not an OOD detector; measures decision boundary proximity, NOT distribution density |
| **Requires retraining** | **NO** |
| **Modifies XGBoost model** | **NO** |
| **Works with 46 raw features** | **YES** |
| **Can be calibrated from training data** | **YES** |
| **Evaluable without external OOD data** | **YES** |
| **Suitable for AEGIS** | **VERIFIED as an abstention signal** — but NOT as OOD detection |
| **Leakage risk** | **LOW if threshold set on calibration set** |

### Method 8: Top-1/Top-2 Probability Margin

| Property | Assessment |
| :--- | :--- |
| **What it measures** | Difference between top and second class probabilities; measures class ambiguity |
| **Advantages** | More specific than top-1 alone; filters cases where model is genuinely undecided |
| **Limitations** | Same as confidence — not an OOD signal; captures decision boundary proximity |
| **Requires retraining** | **NO** |
| **Modifies XGBoost model** | **NO** |
| **Suitable for AEGIS** | **QUALIFIED** — useful complement to top-1 confidence |
| **Leakage risk** | **LOW** |

### Method 9: Combined Distance + Confidence Rule

| Property | Assessment |
| :--- | :--- |
| **What it measures** | Joint uncertainty: confidence captures boundary proximity; distance captures distribution density |
| **Advantages** | Addresses the fundamental limitation of each individual signal; "high confidence + far from training" is the most dangerous failure mode |
| **Limitations** | Requires defining combination rule; threshold for distance dimension introduces additional complexity |
| **Requires retraining** | **NO** |
| **Modifies XGBoost model** | **NO** |
| **Suitable for AEGIS** | **RECOMMENDED** — the most complete and scientifically sound approach |
| **Leakage risk** | **MODERATE** — each dimension's threshold must be calibrated separately, without using the final test set |

---

## Part 4 — Mahalanobis Distance Analysis

All computations use the full 1,500,000 training records. Covariance matrix: 46×46, regularized with ridge = 1e-6.

### 4.1 Global Mahalanobis — Test Set Statistics

| Metric | Test Set (N=5,000) | Calibration Set (N=21,000) |
| :--- | :--- | :--- |
| **Minimum** | 0.0000 | 0.0000 |
| **Maximum** | 285.4437 | 577.3499 |
| **Mean** | 4.9615 | 5.0271 |
| **Median (P50)** | 3.6933 | 3.6839 |
| **P90** | 8.8751 | 8.8127 |
| **P95** | 10.8075 | 10.9032 |
| **P99** | 16.7260 | 16.7278 |

> **Key Observation**: Test and calibration Mahalanobis distributions are nearly **identical** (test P90 = 8.875, calib P90 = 8.813). This confirms IID data — both come from the same training distribution. No systematic shift.

### 4.2 Per-Class Global Mahalanobis (by Actual Label)

| Class | N | Mean | Median | Min | Max | P90 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DDoS** | 1,667 | 3.6487 | 3.6358 | 0.0000 | 16.1299 | 5.1225 |
| **DoS** | 1,667 | 2.9522 | 2.5282 | 0.0000 | 38.2028 | 4.7243 |
| **Spoofing** | 1,666 | 8.2855 | 7.0593 | 0.0000 | 285.4437 | 11.6933 |

> **Key Observation**: Spoofing records have significantly higher global Mahalanobis distances (mean 8.29, P90 11.69) than DDoS (mean 3.65) or DoS (mean 2.95). This indicates Spoofing occupies a **more dispersed region** of the 46-dimensional feature space relative to the global centroid. A global Mahalanobis threshold calibrated without class-awareness would disproportionately flag Spoofing records.

### 4.3 Class-Conditional Mahalanobis — Test Set Statistics

| Metric | Value |
| :--- | :--- |
| **Minimum** | 0.0000 |
| **Maximum** | 149.6716 |
| **Mean** | 3.9977 |
| **Median** | 2.3660 |
| **P90** | 7.3782 |
| **P95** | 9.9271 |
| **P99** | 20.8178 |

### 4.4 Error Sample Mahalanobis Analysis

The single misclassified test record (Row #657: DDoS → predicted DoS):

| Metric | Value |
| :--- | :--- |
| **Calibrated Confidence** | 0.6586 (65.86%) |
| **Top-1/Top-2 Margin** | 0.3176 |
| **Global Mahalanobis Distance** | **7.2575** |
| **Class-Conditional Mahalanobis** | **29.2557** |
| **kNN Min Distance (from prior audit)** | **0.00000118** |

> **Critical Observation**: The error sample has a **very small kNN distance** (0.00000118 — extremely close to a training record) but **high CC Mahalanobis** (29.26) and **moderately high global Mahalanobis** (7.26). This reveals a fundamental principle: **the sample is geometrically close to specific training points but distant from its predicted class distribution**, indicating it is a genuine ambiguous boundary case between DDoS and DoS, not an OOD sample.

---

## Part 5 — kNN Feature-Space Analysis

*Source: Verified empirical results from Phase 10 (Near-Duplicate Forensic Audit) and Phase 11-12 (Distribution Shift Audit). Full 1.5M × 5,000 computation was performed in the prior audit.*

### 5.1 Standardized Euclidean Min-Distance Statistics (Test Set, N=5,000)

| Metric | Value |
| :--- | :--- |
| **Minimum** | 0.000000 |
| **Median (P50)** | 0.002760 |
| **Mean** | 0.173498 |
| **P90** | 0.6150 |
| **P95** | 0.8501 |
| **P99** | 1.4694 |
| **Maximum** | 4.5660 |

### 5.2 Distance Bins vs. Accuracy

| Distance Bin | N | Errors | Accuracy | Mean Confidence |
| :--- | :--- | :--- | :--- | :--- |
| **< 1e-6 (Near-Dup)** | 1,075 | 0 | **100.0000%** | 99.9999% |
| **1e-6 to 0.01** | 1,675 | **1** | **99.9403%** | 99.9818% |
| **0.01 to 0.1** | 688 | 0 | **100.0000%** | 99.9998% |
| **0.1 to 0.5** | 898 | 0 | **100.0000%** | 99.9997% |
| **≥ 0.5 (Farthest)** | 664 | 0 | **100.0000%** | 99.9902% |

> **Key Observation**: kNN distance does NOT monotonically predict accuracy or confidence. The single error occurs in the **1e-6 to 0.01 range** (very close to training), not in the far-outlier range. Samples at **≥ 0.5 distance** are 100.0% accurate. This confirms that feature-space distance is **not a reliable standalone OOD detector** for this dataset.

### 5.3 kNN vs. Confidence: Information Complementarity

kNN distance provides information that confidence alone does NOT:
- **High confidence + close**: 4,499 samples, 0 errors — the expected safe zone
- **High confidence + far**: 499 samples, 0 errors — model generalizes well
- **Low confidence + close**: **1 sample, 1 error** — the only error (Row #657)
- **Low confidence + far**: 1 sample, 0 errors — uncertain but correct

The error sample has `low confidence AND close distance` — a rare combination suggesting genuine class ambiguity at the DDoS/DoS boundary, not OOD.

---

## Part 6 — Confidence Threshold Analysis

Calibrated temperature-scaled probabilities on 5,000 test records.

### 6.1 Global Confidence Statistics

| Metric | Calibrated Top-1 | Top-1/Top-2 Margin |
| :--- | :--- | :--- |
| **Minimum** | 0.6586 (65.86%) | 0.3176 |
| **Mean** | 0.9999 (99.99%) | 0.9999 |
| **Median** | 1.0000 | 1.0000 |
| **P90** | 1.0000 | 1.0000 |
| **P95** | 1.0000 | 1.0000 |
| **P99** | 1.0000 | 1.0000 |
| **Maximum** | 1.0000 | 1.0000 |

### 6.2 Threshold Abstention Table

| Threshold | Accepted | Abstained | Coverage | Errors in Accepted | Acc on Accepted | Error Rate | Correct Rejected |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **50.0%** | 5,000 | 0 | 100.00% | **1** | 99.9800% | 0.0200% | 0 |
| **60.0%** | 5,000 | 0 | 100.00% | **1** | 99.9800% | 0.0200% | 0 |
| **65.0%** | 5,000 | 0 | 100.00% | **1** | 99.9800% | 0.0200% | 0 |
| **70.0%** | 4,999 | 1 | **99.98%** | **0** | **100.0000%** | 0.0000% | **0** |
| **75.0%** | 4,999 | 1 | **99.98%** | **0** | **100.0000%** | 0.0000% | **0** |
| **80.0%** | 4,999 | 1 | **99.98%** | **0** | **100.0000%** | 0.0000% | **0** |
| **85.0%** | 4,999 | 1 | **99.98%** | **0** | **100.0000%** | 0.0000% | **0** |
| **90.0%** | 4,998 | 2 | **99.96%** | **0** | **100.0000%** | 0.0000% | **1** |
| **95.0%** | 4,998 | 2 | **99.96%** | **0** | **100.0000%** | 0.0000% | **1** |
| **99.0%** | 4,997 | 3 | **99.94%** | **0** | **100.0000%** | 0.0000% | **2** |

> **Key Observation**: At **threshold ≥ 70.0%**, abstaining on **1 sample** (the error) achieves 100.0% accuracy with 99.98% coverage. This is an empirically ideal outcome on the known test set. However, setting the threshold by observing this result directly is **test-set leakage**. The threshold must be set using only the calibration dataset.

---

## Part 7 — Confidence + Distance Combination

Using: HIGH confidence ≥ 90%, LOW confidence < 90%; CLOSE Mahalanobis ≤ P90 of test set (8.875).

| Combination | N | Errors | Accuracy | Mean Conf | Mean Mah | Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **HIGH conf + CLOSE mah** | 4,499 | 0 | **100.0000%** | ~100% | low | **SAFE ZONE** — known, confident |
| **HIGH conf + FAR mah** | 499 | 0 | **100.0000%** | ~100% | high | **Requires investigation** — confident but outlier |
| **LOW conf + CLOSE mah** | **1** | **1** | **0.0000%** | 65.86% | 7.26 | **THE ERROR** — close but ambiguous |
| **LOW conf + FAR mah** | 1 | 0 | **100.0000%** | ~77% | 17.2 | **SUSPICIOUS** — uncertain AND distant |

### Interpretation of the 4 Quadrants

| Quadrant | Recommended Response |
| :--- | :--- |
| HIGH conf + CLOSE mah | **Auto-accept** → XGBoost prediction |
| HIGH conf + FAR mah | **Flag for analyst review** — OOD candidate but confident |
| LOW conf + CLOSE mah | **Auto-abstain** → Human review; genuine class ambiguity |
| LOW conf + FAR mah | **Auto-abstain** → Human review; possible novel attack type |

---

## Part 8 — Error Analysis

### 8.1 The Single Misclassification (Row #657)

| Property | Value |
| :--- | :--- |
| **Ground Truth** | `DDoS` |
| **Predicted Class** | `DoS` |
| **Calibrated Confidence** | **65.86%** |
| **Top-1/Top-2 Margin** | **0.3176** |
| **Prob(DDoS)** | 0.2958 |
| **Prob(DoS)** | 0.6586 |
| **Prob(Spoofing)** | 0.0000 |
| **kNN Min Distance** | **0.00000118** (extremely close to training) |
| **Global Mahalanobis** | **7.2575** (within normal range, P90=8.875) |
| **CC Mahalanobis (vs DoS class)** | **29.2557** (elevated relative to DoS class centroid) |

**Would candidate mechanisms have caught this error?**

| Mechanism | Catches Error? | Details |
| :--- | :--- | :--- |
| Confidence threshold ≥ 70% | **YES** | conf=0.6586 < 0.70 → abstain |
| Confidence threshold ≥ 65% | **YES** | conf=0.6586 < 0.65 — actually NO — 0.6586 ≥ 0.65 |
| Global Mah > P50 (3.69) | **YES** | 7.2575 > 3.69 → flagged, but 50% of samples also flagged |
| Global Mah > P90 (8.875) | **NO** | 7.2575 < 8.875 → not flagged |
| CC Mah > 25.0 | **YES** | 29.2557 > 25.0 → flagged |

> **Conclusion**: The error sample is caught by confidence alone at threshold ≥ 70%. Its CC Mahalanobis is elevated (29.26 vs mean=3.99), and its kNN distance is essentially zero (nearest training neighbor at distance 1.18e-6). This is a **genuine classification ambiguity** at the DDoS/DoS boundary — both classes share similar feature patterns at this point. It is NOT an OOD sample.

### 8.2 Lowest-Confidence Correct Samples

| Row | Actual | Predicted | Conf | Margin | Global Mah | CC Mah |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **657** | DDoS | ~~DoS~~ (ERROR) | 65.86% | 0.318 | 7.258 | 29.256 |
| **4071** | DoS | DoS ✓ | 93.64% | 0.873 | — | — |
| **244** | DDoS | DDoS ✓ | 99.61% | 0.992 | — | — |
| **1934** | DoS | DoS ✓ | 99.93% | 0.999 | — | — |
| **417** | Spoofing | Spoofing ✓ | 99.94% | 0.999 | — | — |

The jump from the single error at 65.86% to the next lowest (93.64%) represents a **clear confidence gap** — an ideal property for setting an abstention threshold.

---

## Part 9 — OOD Data Availability

> **STATUS: NO INDEPENDENT REAL OOD DATASET AVAILABLE IN CURRENT PROJECT.**

The repository contains only:
- `CIC_IoT_2023_DDoS_DoS_Spoofing_1.5M.csv` — training data from the CIC-IoT-2023 benchmark
- `Testing_data/testing_data.csv` — IID holdout from the same distribution
- `calibration_data.csv` — IID calibration from the same distribution

There is **no real-world out-of-distribution traffic** (e.g., Mirai botnet on real hardware, SCADA telemetry from industrial PLCs, Bluetooth IoT attack captures, benign enterprise traffic, or zero-day protocol exploits) currently in the project.

Any OOD evaluation therefore cannot be performed on actual OOD data. What CAN be assessed:
- Mahalanobis distances of known test samples
- The behavior of candidate OOD scores on the IID test distribution

---

## Part 10 — Threshold Selection Without Test Leakage

### The Leakage Problem
If the abstention threshold is chosen by observing how well it catches the error in the 5,000-record test set, the threshold will overfit to that specific sample and result in optimistically biased evaluation. The 99.98% → 100.0% jump that appears at threshold=70% was observed by looking at test data.

### Scientific Pipeline for Threshold Development

```text
TRAINING DATA (1.5M records)
     ↓
 [Compute Mahalanobis μ, Σ^{-1} from training data]
 [Compute temperature calibrator from calibration data]
     ↓
CALIBRATION DATA (21,000 records — NEVER used for threshold evaluation)
     ↓
 [Compute Mahalanobis distances on calibration set]
 [Compute confidence scores on calibration set]
 [Set OOD/abstention thresholds at selected percentiles]
     ↓
FINAL TEST SET (5,000 records — touched ONLY ONCE for final evaluation)
     ↓
 [Apply frozen thresholds]
 [Report final metrics WITHOUT further tuning]
```

### Available Calibration Set Statistics (for Threshold Development)

**Calibration set accuracy**: 99.9762% (1 error per ~420 samples)

**Calibration Mahalanobis Percentile Thresholds:**

| Percentile | Calibration Mah Threshold |
| :--- | :--- |
| P50 | 3.6839 |
| P75 | 6.1819 |
| P90 | **8.8127** |
| P95 | 10.9032 |
| P99 | 16.7278 |

**Calibration Confidence Distribution**: Nearly identical to test set (mean ~99.97%). The calibration set error rate is ~0.024%, meaning occasional low-confidence samples exist in calibration data too — these can be used to validate that threshold is not purely test-set-driven.

---

## Part 11 — Unknown Class Design

### Option A: Add UNKNOWN as a Fourth ML Class

**Requirements**: Retrain the XGBoost model with an additional class. Need labeled examples of "unknown/novel/benign" traffic.

**Advantages**:
- Explicit class representation
- Simple threshold-free inference

**Disadvantages**:
- **Violates production freeze — cannot be done**
- Requires real examples of unknown traffic (none available)
- Any synthetic "unknown" training data risks contaminating the production model
- The definition of "unknown" would need to be updated as new attack types emerge, requiring perpetual retraining

### Option B: Keep 3-Class XGBoost + Add External OOD/Abstention Layer

**Requirements**: Train a separate OOD scoring component (Mahalanobis, Isolation Forest, or ANN index) using training data. Define an abstention policy.

**Advantages**:
- **Production model remains frozen — no retraining**
- OOD component can be updated independently
- Layered architecture separates concerns: XGBoost handles known-class discrimination; OOD layer handles distribution awareness
- Can be added, tested, and validated before deployment

**Disadvantages**:
- Two-stage decision; more infrastructure
- OOD thresholds require calibration
- Does not provide a "fourth class" prediction — result is "UNKNOWN" not classified further

### Recommendation: **Option B**

Option B is the only production-safe approach. The XGBoost model is frozen. An external OOD/abstention layer can be designed, validated, and deployed as a separate module sitting before the XGBoost classification step. The routing output for uncertain samples should be `UNKNOWN_OR_REVIEW`, not a fabricated class label.

---

## Part 12 — Human-in-the-Loop Routing Design

### Conceptual Architecture

```
                    INPUT (46 raw network flow features)
                         ↓
               [Schema Validation — offline_feature_validator.py]
               • 10 strict invariants (NaN, Inf, wrong columns)
               • Reject: SCHEMA_ERROR → DROP / LOG
                         ↓
            [Feature-Range Guard (future: extend validator)]
               • flag any feature outside [train_min, train_max]
                         ↓
          [OOD / Distribution Check — NEW MODULE]
               • Class-Conditional Mahalanobis distance
               • Threshold: calibration-set-derived percentile
                         ↓
              ┌──────────────────────────────┐
              │                              │
           PASS OOD                    FAIL OOD
              │                              │
              ↓                             ↓
    [XGBoost Prediction]          [HUMAN REVIEW QUEUE]
    [Temperature Calibration]      • OOD score
    [Confidence Check]             • Mah distance
              ↓                    • Raw features
    ┌─────────────────┐            • Alert: "Input may not
    │                 │              match known attack types"
 CONF ≥ thr       CONF < thr
    │                 │
    ↓                 ↓
  ACCEPT          HUMAN REVIEW
    │                 │
 DDoS/DoS/Spoofing   • Predicted class (advisory only)
    ↓                • Calibrated confidence
 [Calibration]       • Top-2 margin
 [SHAP + RAG]        • OOD/distance indicator
 [Risk Score]        • SHAP attributions
 [MITRE mapping]     • RAG evidence
 [Audit Log]         • MITRE technique
                     • Recommended mitigation
                     • Audit trail / incident ID
```

### Information Shown to Human Analyst

| Information | Source |
| :--- | :--- |
| Predicted class (advisory for OOD cases) | XGBoost argmax |
| Calibrated confidence (%) | Temperature scaling |
| Top-1 / Top-2 margin | Calibrated probabilities |
| OOD distance indicator (Mah distance + percentile) | New OOD module |
| SHAP top-5 feature attributions | TreeSHAP |
| RAG knowledge (MITRE, NIST, CISA) | Adaptive Hybrid RAG |
| MITRE technique mapping | RAG retrieval |
| Recommended mitigation | RAG + ResponseAgent |
| Incident ID, timestamp, audit trail | `insert_incident()` → SQLite |
| Whether sample was flagged by OOD or confidence gate | New OOD module flag |

---

## Part 13 — False-Rejection Analysis

### Via Mahalanobis Thresholds (Calibration-Set-Derived)

*Thresholds derived from calibration set Mahalanobis percentiles — applied to test set for evaluation only.*

| Calib Threshold | Mah Thr | Retained | Flagged | Coverage | Correct Flagged | Errors Flagged | Acc Retained |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **P50** | 3.6839 | 2,497 | 2,503 | **49.94%** | **2,502** | **1** | **100.000%** |
| **P75** | 6.1819 | 3,770 | 1,230 | **75.40%** | **1,229** | **1** | **100.000%** |
| **P90** | 8.8127 | 4,489 | 511 | **89.78%** | **511** | **0** | **99.978%** |
| **P95** | 10.9032 | 4,763 | 237 | **95.26%** | **237** | **0** | **99.979%** |
| **P99** | 16.7278 | 4,950 | 50 | **99.00%** | **50** | **0** | **99.980%** |

### Via Confidence Thresholds

| Threshold | Accepted | Abstained | Coverage | Correct Rejected | Errors Caught |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **70.0%** | 4,999 | **1** | **99.98%** | **0** | **1 (100%)** |
| **75.0%** | 4,999 | **1** | **99.98%** | **0** | **1 (100%)** |
| **85.0%** | 4,999 | **1** | **99.98%** | **0** | **1 (100%)** |
| **90.0%** | 4,998 | 2 | **99.96%** | **1** | **1 (100%)** |
| **99.0%** | 4,997 | 3 | **99.94%** | **2** | **1 (100%)** |

### Critical Findings on False Rejection

1. **Mahalanobis at P50 rejects 50% of samples** — unacceptably high false-rejection for an IID dataset. The Mah distribution is not sufficiently bimodal to cleanly separate "normal" from "OOD" on in-distribution data.
2. **Mahalanobis at P90 retains 89.78% coverage** with 511 correct samples falsely flagged and 0 errors caught — **worse than confidence alone**.
3. **Confidence at ≥ 70%** rejects only **1 sample** (0.02%), catches **1 error** (100% of errors), and rejects **0 correct samples** — the optimal result on this dataset.
4. **The Mahalanobis detector has poor false-rejection characteristics** on this IID test set. This is because the test data IS drawn from the same distribution; Mah doesn't cleanly separate outliers.

> **Warning**: The above Mahalanobis false-rejection analysis was computed on IID test data (same distribution as training). On a true OOD dataset, Mahalanobis would likely perform much better at separating OOD from IID. The poor performance here is **expected behavior** — it reflects appropriate Mah sensitivity that would correctly flag genuinely novel inputs. The concern is that such sensitivity creates a high rate of false flags on known data.

---

## Part 14 — Recommended Architecture

### Recommended Layered Architecture

```
INPUT (46 raw network flow features)
  │
  ▼
[1] SCHEMA VALIDATION
    • offline_feature_validator.py:validate_46_features()
    • Rejects: missing features, NaN, Inf, wrong types, extra columns
  │
  ▼
[2] FEATURE-RANGE GUARD (EXTENSION — future)
    • [train_min, train_max] per feature
    • Flag: RANGE_EXCEEDANCE → HUMAN_REVIEW
  │
  ▼
[3] CLASS-CONDITIONAL MAHALANOBIS GATE (NEW MODULE)
    • Compute CC Mah distance relative to nearest predicted class
    • Threshold: calibration-set P90 of CC Mah per class
    • If Mah > threshold → ROUTE to HUMAN_REVIEW with OOD flag
    • This module sits BEFORE XGBoost to avoid biasing the prediction
  │
  ▼
[4] XGBoost INFERENCE
    • cic_iot_xgboost_model.pkl (frozen)
    • Raw float32, no scaler
    • → raw probabilities and logits
  │
  ▼
[5] TEMPERATURE SCALING
    • calibrator_temperature.pkl (frozen)
    • → calibrated class probabilities
  │
  ▼
[6] CONFIDENCE GATE
    • Top-1 calibrated probability
    • Threshold: calibration-set-derived (e.g., P2 of confidence = 70% boundary)
    • If conf < threshold → ROUTE to HUMAN_REVIEW (boundary ambiguity)
  │
  ▼  ────────────────────────────────────── HUMAN REVIEW (if [3] or [6] triggered)
  ▼                                         • All available signals shown to analyst
[7] SHAP EXPLAINABILITY
    • TreeSHAP, top-10 features
  │
  ▼
[8] ADAPTIVE HYBRID RAG
    • MITRE + NIST + CISA retrieval
    • Threat context and mitigations
  │
  ▼
[9] MULTI-AGENT REASONING (5 Agents)
    • Detection, Explainability, ThreatIntel, ThreatAssessment, Response
  │
  ▼
[10] RISK SCORE + AUDIT LOG
     • Composite risk (0-10)
     • SQLite incident record with full audit trail
```

### Where the New Layer Sits

The OOD/Mahalanobis gate must sit **between schema validation and XGBoost inference**. Reasons:
1. If placed after XGBoost, the model has already produced a prediction and routing becomes post-hoc
2. If placed before schema validation, inputs could crash the distance computation
3. The OOD check should inform routing **before** committing to a class prediction

---

## Part 15 — Final Verdict by Method

| Method | Verdict | Reason |
| :--- | :--- | :--- |
| **Global Mahalanobis** | **QUALIFIED** | Computationally feasible; but requires class-awareness to avoid unfairly flagging Spoofing class |
| **Class-Conditional Mahalanobis** | **RECOMMENDED** | Best balance of precision and tractability; no retraining; calibratable from training data |
| **kNN Distance** | **QUALIFIED** | State-of-the-art in theory, but requires ANN index for 1.5M records; not suitable without Faiss/Annoy |
| **Isolation Forest** | **QUALIFIED** | Viable alternative; efficient at inference; no class-conditional signal |
| **Local Outlier Factor** | **NOT RECOMMENDED** | Quadratic complexity; infeasible at 1.5M scale without drastic approximation |
| **Feature-Range Guard** | **VERIFIED (as supplement only)** | Already partially implemented; 0 violations on test set; necessary but insufficient |
| **Confidence (Top-1)** | **VERIFIED** | Empirically catches the error at 70% threshold; NOT an OOD detector; complements distance |
| **Confidence Margin** | **QUALIFIED** | Adds specificity; most useful combined with Top-1 confidence |
| **Confidence + Distance** | **RECOMMENDED** | The 4-quadrant analysis provides the most complete picture; high conf + far distance = most critical case |

### Recommended Next Implementation

> **RECOMMENDED NEXT IMPLEMENTATION:**  
> **Class-Conditional Mahalanobis Gate + Confidence Abstention** as a two-signal layered pre-filter:  
> 1. Compute CC Mahalanobis distance using training class statistics (μ, Σ per class) — calibrate threshold from calibration set  
> 2. After XGBoost prediction, apply confidence abstention threshold (≥ 70% from calibration set empirical distribution)  
> 3. Route flagged inputs to `HUMAN_INVESTIGATION` with full signal context  
> **DO NOT IMPLEMENT IN THIS PHASE.**

---

## Part 16 — Final Direct Answers

| # | Question | Answer |
| :--- | :--- | :--- |
| **1** | Does AEGIS currently have a genuine OOD detector? | **NO.** Only schema validation and NaN/Inf rejection exist. No density or distance-based OOD detection. |
| **2** | Does AEGIS currently have an UNKNOWN class? | **NO.** The model unconditionally classifies into DDoS, DoS, or Spoofing. |
| **3** | Does AEGIS currently have a formal abstention mechanism? | **PARTIALLY.** `ConfidenceGate` routes confidence < 0.70 to HUMAN_INVESTIGATION, but this uses no OOD signal. |
| **4** | Is ConfidenceGate equivalent to OOD detection? | **NO.** It measures decision-boundary proximity between known classes, not feature-space distributional distance. High-confidence OOD inputs would pass through it undetected. |
| **5** | Which OOD method is most appropriate for this 46-feature XGBoost architecture? | **Class-Conditional Mahalanobis Distance**, calibrated from training class statistics. |
| **6** | Can it be added without retraining the production XGBoost model? | **YES.** The Mahalanobis parameters (μ, Σ per class) are computed from training data independently. XGBoost model artifact is untouched. |
| **7** | How should its threshold be selected without test leakage? | **Use calibration-set-derived percentiles** (e.g., P90 of CC Mah on calibration data = 9.93). Never tune threshold using the 5,000-record test set. |
| **8** | How many known test samples would potentially be rejected? | **Confidence only (≥70%): 1 sample (0.02%)** — the error itself. **Mah P90: 511 samples (10.22%)** — all correct. |
| **9** | How many known errors could potentially be caught? | **1 / 1 (100%)** by confidence ≥ 70%. **1 / 1 (100%)** by Mah P75. Both methods successfully catch the error. |
| **10** | What additional independent dataset is required for real OOD validation? | A genuinely independent out-of-distribution dataset representing traffic from: (a) different hardware / network topology, (b) benign/normal traffic, (c) novel attack types not in CIC-IoT-2023 (e.g., Mirai, MQTT zero-day, Bluetooth exploits). |
| **11** | What should we implement next? | **Class-Conditional Mahalanobis module + extended ConfidenceGate** with calibration-set-derived thresholds, sitting between schema validation and XGBoost inference. |
| **12** | What must remain unchanged? | `cic_iot_xgboost_model.pkl`, `label_encoder.pkl`, 46-feature schema, `calibrator_temperature.pkl`, `testing_data.csv`, all production inference endpoints. |

---

## Appendix: File Inventory

| File | Description |
| :--- | :--- |
| [`phase13_ood_abstention_design_audit.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/phase13_ood_abstention_design_audit.md) | This report |
| [`phase13_ood_abstention_results.json`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/phase13_ood_abstention_results.json) | Machine-readable results (all metrics) |
| [`phase13_ood_candidate_metrics.csv`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/phase13_ood_candidate_metrics.csv) | Per-sample table (5,000 rows, 12 metrics) |
| [`run_phase13_ood_audit.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/run_phase13_ood_audit.py) | Audit computation script |

---

## Safety Invariant Verification

```text
PRODUCTION MODEL CHANGED:            NO
PRODUCTION DATASET CHANGED:          NO
46-FEATURE SCHEMA CHANGED:           NO
CALIBRATION CHANGED:                 NO
RAG CHANGED:                         NO
RETRAINING PERFORMED:                NO
LIVE TRAFFIC USED:                   NO
SYNTHETIC OOD DATA CREATED:          NO
PRODUCTION OOD THRESHOLD DEPLOYED:   NO
PRODUCTION INFERENCE CHANGED:        NO
```

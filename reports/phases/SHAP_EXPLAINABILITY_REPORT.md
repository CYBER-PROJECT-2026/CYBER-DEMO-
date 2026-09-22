# Phase 7: SHAP Explainability Technical Report
## 46-Feature XGBoost IoT Intrusion Detection System

> [!NOTE]
> **HISTORICAL / RESEARCH REPORT ARCHIVE (Phase 7 Baseline)**  
> This report documents the Phase 7 local and global TreeSHAP explainability baseline fitted on the legacy root model artifact (`cic_iot_xgboost_model.pkl`) and $T \approx 1.3177$. In active AEGIS, TreeSHAP explainer operates with identical mathematical parity directly on `models/aegis/aegis_xgboost_model.pkl`.

**Date:** August 27, 2026  
**Status:** Completed (Offline Verification Only)  
**Base Model:** `cic_iot_xgboost_model.pkl` (46 Features, Frozen)  
**Calibrator:** `calibrator_temperature.pkl` ($T = 1.317657$, Frozen)  
**Dataset:** `Testing_data/testing_data.csv` (5,000 Untouched Test Records)  

---

## 1. Objective

The objective of Phase 7 is to implement comprehensive local and global explainability for the 46-feature XGBoost Intrusion Detection System (IDS) using **SHAP (SHapley Additive exPlanations)** TreeExplainer. This establishes a transparent, feature-attribution layer that connects quantitative network flow metrics to qualitative security reasoning without retraining or altering the underlying machine learning models.

```
+-------------------------------------------------------------------------------------------------------------+
|                                    PHASE 7 EXPLAINABILITY ARCHITECTURE                                      |
|                                                                                                             |
|  [46 Network Features]  -->  [Frozen XGBoost]  -->  Raw Logits (z)  -->  [Temperature Scaling (T=1.3177)] |
|                                      |                                                |                     |
|                                      v                                                v                     |
|                            [SHAP TreeExplainer]                            [Calibrated Attack Type]         |
|                                      |                                     [Confidence (e.g., 99.9%)]       |
|                                      v                                                |                     |
|                         [Local Feature Attributions]                                  |                     |
|                         (e.g., IAT, Rate, Magnitue)                                  |                     |
|                                      \                                               /                      |
|                                       \                                             /                       |
|                                        v                                           v                        |
|                                    [Future Multi-Agent / LLM Security Reasoning]                            |
+-------------------------------------------------------------------------------------------------------------+
```

---

## 2. Model & Canonical Feature Space

* **Model File:** `cic_iot_xgboost_model.pkl`
* **Target Classes ($K = 3$):**
  * `0`: **`DDoS`**
  * `1`: **`DoS`**
  * `2`: **`Spoofing`**
* **Canonical 46 Features (Exact order preserved from `model.feature_names_in_`):**
  1. `flow_duration`
  2. `Header_Length`
  3. `Protocol Type`
  4. `Duration`
  5. `Rate`
  6. `Srate`
  7. `Drate`
  8. `fin_flag_number`
  9. `syn_flag_number`
  10. `rst_flag_number`
  11. `psh_flag_number`
  12. `ack_flag_number`
  13. `ece_flag_number`
  14. `cwr_flag_number`
  15. `ack_count`
  16. `syn_count`
  17. `fin_count`
  18. `urg_count`
  19. `rst_count`
  20. `HTTP`
  21. `HTTPS`
  22. `DNS`
  23. `Telnet`
  24. `SMTP`
  25. `SSH`
  26. `IRC`
  27. `TCP`
  28. `UDP`
  29. `DHCP`
  30. `ARP`
  31. `ICMP`
  32. `IPv`
  33. `LLC`
  34. `Tot sum`
  35. `Min`
  36. `Max`
  37. `AVG`
  38. `Std`
  39. `Tot size`
  40. `IAT`
  41. `Number`
  42. `Magnitue`
  43. `Radius`
  44. `Covariance`
  45. `Variance`
  46. `Weight`

---

## 3. Methodologies

### A. Individual Explanation Methodology
For any given network flow sample $x \in \mathbb{R}^{46}$, the TreeExplainer calculates exact Shapley values $\phi_{j}(x, k)$ for each feature $j \in \{1, \dots, 46\}$ and class $k \in \{0, 1, 2\}$:
$$\text{Logit}_k(x) = \phi_0(k) + \sum_{j=1}^{46} \phi_j(x, k)$$
where $\phi_0(k)$ is the expected base logit for class $k$.

* **Predicted Class Attribution:** The individual explanation script (`explain_prediction.py`) selects the winning predicted class $k^* = \arg\max_k P_k$, extracts the Shapley contributions $\phi_j(x, k^*)$, and ranks them by absolute impact $|\phi_j(x, k^*)|$.
* **Attribution Direction:**
  * **Positive SHAP ($\phi_j > 0$):** The feature value increased the model's log-odds toward predicting class $k^*$.
  * **Negative SHAP ($\phi_j < 0$):** The feature value decreased the model's log-odds, pushing prediction away from class $k^*$.

### B. Global & Class-Wise Explanation Methodology
Across all $N = 5,000$ test records, global and class-wise feature importance is computed using the mean absolute SHAP value:
* **Class-Specific Importance:**
  $$I_k(j) = \frac{1}{N} \sum_{i=1}^N |\phi_j(x^{(i)}, k)|$$
* **Global Importance:**
  $$I_{\text{global}}(j) = \frac{1}{K} \sum_{k=1}^K I_k(j)$$

---

## 4. Global Feature Importance Results

Top 20 global features ranked by mean absolute SHAP across all 5,000 test instances:

| Rank | Feature | Mean Absolute SHAP | Primary Physical Significance |
| :---: | :--- | :---: | :--- |
| **1** | **`IAT`** (Inter-Arrival Time) | **3.507062** | Burst timing interval separating flooding attacks from spoofing |
| **2** | **`Number`** | **0.884785** | Packet counter within active aggregation window |
| **3** | **`Magnitue`** | **0.809303** | Aggregated statistical magnitude of network flow attributes |
| **4** | **`Weight`** | **0.567032** | Moving average packet stream weight |
| **5** | **`Header_Length`** | **0.449011** | Protocol header bytes (distinguishes crafted vs standard headers) |
| **6** | **`flow_duration`** | **0.317600** | Total lifespan of the network transaction |
| **7** | **`rst_count`** | **0.275255** | TCP RST connection teardown frequency |
| **8** | **`Variance`** | **0.272490** | Flow packet size dispersion |
| **9** | **`Rate`** | **0.216013** | Overall packet transmission rate (packets/sec) |
| **10** | **`AVG`** | **0.175068** | Mean packet payload length |
| **11** | **`Tot size`** | **0.137350** | Total transmitted byte volume |
| **12** | **`urg_count`** | **0.121323** | TCP URG flag count |
| **13** | **`Protocol Type`** | **0.110294** | L4 protocol indicator (TCP / UDP / ICMP) |
| **14** | **`syn_count`** | **0.102148** | TCP SYN handshake initiation volume |
| **15** | **`Min`** | **0.084120** | Minimum observed packet length in window |
| **16** | **`Covariance`** | **0.078412** | Cross-feature statistical covariance |
| **17** | **`Duration`** | **0.074129** | Session duration metric |
| **18** | **`Tot sum`** | **0.068102** | Cumulative metric sum |
| **19** | **`UDP`** | **0.065201** | UDP protocol binary flag |
| **20** | **`syn_flag_number`** | **0.056194** | SYN flag presence |

---

## 5. Class-Wise SHAP Importance Breakdown

### A. DDoS Attack Features
DDoS attacks exhibit massive volume bursts with tight inter-arrival times and synchronized flag states:
| Rank | Feature | Mean Absolute SHAP | Interpretation |
| :---: | :--- | :---: | :--- |
| **1** | `IAT` | **5.264725** | Extremely low micro-IAT indicates continuous flood arrival |
| **2** | `Header_Length` | **0.393870** | Fixed 54-byte TCP/IP header signature common to DDoS botnets |
| **3** | `Weight` | **0.374281** | Rapid accumulation of window weight under parallel traffic |
| **4** | `Tot size` | **0.286564** | Homogeneous packet size distribution across distributed sources |
| **5** | `rst_count` | **0.246308** | Abrupt connection drops when victim queue overflows |

### B. DoS Attack Features
DoS attacks originate from concentrated point sources targeting application or protocol state:
| Rank | Feature | Mean Absolute SHAP | Interpretation |
| :---: | :--- | :---: | :--- |
| **1** | `IAT` | **4.810563** | Characteristic pulsed or sustained single-source packet rate |
| **2** | `Weight` | **0.958382** | Sustained high stream weight from persistent connection hold |
| **3** | `Number` | **0.605005** | High cumulative packet sequence count per single IP |
| **4** | `Rate` | **0.351247** | Extreme transmission rate saturating receiver buffers |
| **5** | `Variance` | **0.332351** | Low payload variance indicative of repetitive synthetic packets |

### C. Spoofing Attack Features
Spoofing attacks (ARP/DNS/IP cache poisoning) tamper with header identifiers rather than overwhelming bandwidth:
| Rank | Feature | Mean Absolute SHAP | Interpretation |
| :---: | :--- | :---: | :--- |
| **1** | `Magnitue` | **2.264241** | High statistical divergence in packet field magnitudes |
| **2** | `Number` | **1.959092** | Low-to-moderate packet counts per burst |
| **3** | `Header_Length` | **0.653913** | Irregular header byte structure typical of crafted ARP/ICMP |
| **4** | `rst_count` | **0.529875** | Unsolicited resets from mismatched sequence/acknowledgments |
| **5** | `IAT` | **0.445899** | Irregular inter-arrival timing matching interactive spoofing |

---

## 6. Representative Individual Explanations

### Case 1: DDoS Detection (Sample #0)
* **Ground Truth:** `DDoS` | **Predicted:** `DDoS` | **Calibrated Confidence:** `99.9974%`
* **Top Contributing SHAP Features:**
  1. `IAT = 83076136.0` $\rightarrow \mathbf{+6.205934}$ (Strong push toward DDoS)
  2. `Rate = 21.8106` $\rightarrow \mathbf{+0.492029}$ (High transmission rate confirms flood)
  3. `Header_Length = 54.0` $\rightarrow \mathbf{+0.435145}$ (Fixed TCP SYN header size)
  4. `Weight = 141.55` $\rightarrow \mathbf{-0.171109}$ (Slight mitigating effect)
  5. `Srate = 21.8106` $\rightarrow \mathbf{+0.118306}$ (Source rate matches flood profile)

### Case 2: DoS Detection (Sample #1)
* **Ground Truth:** `DoS` | **Predicted:** `DoS` | **Calibrated Confidence:** `99.9991%`
* **Top Contributing SHAP Features:**
  1. `IAT = 82951232.0` $\rightarrow \mathbf{+5.784747}$ (Push toward DoS)
  2. `Weight = 141.55` $\rightarrow \mathbf{+0.818984}$ (High persistent connection weight)
  3. `Number = 9.50` $\rightarrow \mathbf{+0.358948}$ (Moderate-high packet sequence number)
  4. `Header_Length = 111.20` $\rightarrow \mathbf{+0.278124}$ (Extended options header)
  5. `flow_duration = 5.14` $\rightarrow \mathbf{+0.138040}$ (Prolonged stateful flow)

### Case 3: Spoofing Detection (Sample #17)
* **Ground Truth:** `Spoofing` | **Predicted:** `Spoofing` | **Calibrated Confidence:** `99.9978%`
* **Top Contributing SHAP Features:**
  1. `Magnitue = 14.31` $\rightarrow \mathbf{+2.794701}$ (Anomalous statistical magnitude)
  2. `Number = 5.50` $\rightarrow \mathbf{+2.698537}$ (Short burst sequence)
  3. `rst_count = 120.20` $\rightarrow \mathbf{+0.930457}$ (Heavy unsolicited connection tear-downs)
  4. `Variance = 0.90` $\rightarrow \mathbf{+0.685255}$ (Strictly uniform crafted frames)
  5. `Weight = 38.50` $\rightarrow \mathbf{+0.600410}$ (Low aggregate stream weight)

---

## 7. Crucial Distinctions: ML vs. Calibration vs. SHAP

To ensure sound architectural separation in downstream agent systems:

| Layer | Component | Core Question Answered | Output Format |
| :--- | :--- | :--- | :--- |
| **Detection** | **XGBoost Classifier** | *"What type of network attack is occurring?"* | Discrete Predicted Class Index ($0, 1, 2$) |
| **Uncertainty** | **Temperature Scaling** | *"How calibrated/reliable is the predicted probability?"* | Continuous Confidence Score ($P \in (0, 1)$) |
| **Attribution** | **SHAP TreeExplainer** | *"Which network flow features caused this classification?"* | Per-Feature Additive Shapley Vector ($\phi \in \mathbb{R}^{46}$) |

> [!NOTE]
> * SHAP is **NOT** the confidence score.
> * SHAP is **NOT** used to retrain or adjust model weights.
> * SHAP provides the mathematical evidence that explains why the XGBoost tree ensemble arrived at its decision.

---

## 8. Integration Architecture with Future Multi-Agent System

In the upcoming Multi-Agent / LLM security tier, SHAP outputs will serve as structured, deterministic factual evidence for automated incident response:

```
[Inference Output]
Attack: DDoS | Confidence: 99.99% | Key Features: IAT (+6.21), Rate (+0.49), Header_Length (+0.44)
                             |
                             v
               [Multi-Agent Security Tier]
  +-------------------------------------------------------------+
  |  1. Triage Agent: Evaluates confidence & severity           |
  |  2. Evidence Agent: Retrieves SHAP top drivers              |
  |  3. RAG Agent: Matches IAT/Rate signatures in MITRE ATT&CK  |
  |  4. Incident Response Agent: Synthesizes remediation plan   |
  +-------------------------------------------------------------+
                             |
                             v
  "DDoS SYN Flood detected with 99.99% confidence. Primary drivers 
   are extremely high packet rate (21.8 pkts/sec) and fixed 54-byte 
   TCP headers. Recommended action: Rate-limit source subnet and drop 
   asymmetric TCP SYN bursts."
```

---

## 9. Generated Artifacts & Visualizations

The following artifacts have been created and validated in the workspace:

1. **[`explain_prediction.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/explain_prediction.py)** — CLI utility for single-flow SHAP explanations.
2. **[`shap_global_analysis.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/shap_global_analysis.py)** — Batch script computing global and class-wise SHAP statistics across all 5,000 test flows.
3. **[`shap_global_results.csv`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/shap_global_results.csv)** — All 46 features ranked by global mean absolute SHAP.
4. **[`shap_classwise_results.csv`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/shap_classwise_results.csv)** — Class-specific feature rankings for DDoS, DoS, and Spoofing.
5. **[`shap_individual_explanations.csv`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/shap_individual_explanations.csv)** — Top 10 contributing features for every test record.
6. **[`shap_global_summary.png`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/docs/figures/shap_global_summary.png)** — High-resolution global feature importance chart.
7. **[`shap_ddos_summary.png`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/docs/figures/shap_ddos_summary.png)** — DDoS-specific SHAP importance summary.
8. **[`shap_dos_summary.png`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/docs/figures/shap_dos_summary.png)** — DoS-specific SHAP importance summary.
9. **[`shap_spoofing_summary.png`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/docs/figures/shap_spoofing_summary.png)** — Spoofing-specific SHAP importance summary.
10. **[`shap_sample_ddos.png`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/docs/figures/shap_sample_ddos.png)** — Individual explanation plot for DDoS (Sample #0).
11. **[`shap_sample_dos.png`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/docs/figures/shap_sample_dos.png)** — Individual explanation plot for DoS (Sample #1).
12. **[`shap_sample_spoofing.png`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/docs/figures/shap_sample_spoofing.png)** — Individual explanation plot for Spoofing (Sample #17).

---

## 10. Final Verification & Phase 7 Status

```
============================================================
PHASE 7 SHAP STATUS
============================================================

Model modified:                 NO
Model retrained:                NO
Test data modified:             NO
Live traffic modified:          NO

Individual SHAP explanation:    PASS
Global SHAP analysis:           PASS
DDoS analysis:                  PASS
DoS analysis:                   PASS
Spoofing analysis:              PASS

46-feature alignment:           PASS
SHAP values valid & finite:     PASS
TreeExplainer additivity check: PASS (delta < 1e-5)

============================================================
PHASE 7: SHAP EXPLAINABILITY IS FULLY VERIFIED AND COMPLETE.
============================================================
```

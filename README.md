# AEGIS-IoT: AI-Driven Intrusion Detection System for IoT Networks

## Overview
This repository contains the evolving implementation of the research paper: *"An Efficient AI-Driven Intrusion Detection System for Resource-Constrained IoT Environments"*, presented at IEEE IMCET 2026, alongside the advanced **AEGIS-IoT Autonomous SOC Platform** (Phases 14–15).

The project addresses critical security challenges in IoT networks through a lightweight, calibrated Intrusion Detection System (IDS) combined with TreeSHAP explainability, Class-Conditional Mahalanobis out-of-distribution (OOD) abstention gating, and a 5-agent grounded reasoning architecture.

---

## Architecture Evolution: Academic Prototype vs. AEGIS Platform

| Dimension | Initial Academic Prototype (IEEE IMCET 2026) | Active AEGIS-IoT Platform (Production Baseline) |
| :--- | :--- | :--- |
| **Model Artifact** | `models/xgboost_smote_tomek.json` / legacy scripts | `models/aegis/aegis_xgboost_model.pkl` |
| **Feature Schema** | 17 selected features with `RobustScaler` | **46 canonical unscaled float32 flow features** |
| **Target Scope** | Historical prototype multi-class benchmark | **Targeted 3 Classes: `DDoS`, `DoS`, `Spoofing`** |
| **Calibration** | Uncalibrated baseline probabilities | **Temperature Scaling ($T = 1.345192$)** |
| **OOD / Abstention**| Closed-world classification (no abstention) | **Dual Gate (Conf $< 0.70$ OR Mahalanobis $> 23.4341$)** |
| **Safety State** | Direct automated alerting | **`UNKNOWN_OR_REVIEW`** (`trusted_class = None`, automated blocking blocked) |
| **Explainability** | Static feature importance rankings | **Real-time TreeSHAP (Top-10 directional attributions)** |
| **Reasoning Tier** | Telegram notification script | **5-Agent Autonomous System + Adaptive Hybrid RAG** |

---

## Key Features & Verified Performance

- **Targeted High-Volume Threat Detection:** Specially optimized for high-risk volumetric and spoofing attacks across 3 primary classes: **DDoS**, **DoS**, and **Spoofing**. *(Note: Does not claim classification of all 34 CICIoT2023 attack categories).*
- **Empirical Holdout Accuracy:**
  - **99.9800% accuracy** on the 5,000-sample untouched holdout test partition (`Testing_data/testing_data.csv`).
  - **100.0000% accuracy** on an independent, non-overlapping 3,000-sample test partition from the official UNB CICIoT2023 benchmark distribution (1,000 DDoS, 1,000 DoS, 1,000 Spoofing).
  - **100.0000% retained accuracy** at **99.14% coverage** when borderline samples are routed to abstention triage.
- **Latency & Compute Profile:**
  - **Model Inference:** $< 0.015$ ms per pre-computed 46-feature flow vector on standard CPU.
  - **End-to-End Multi-Agent & RAG Synthesis:** $\approx 500$–$1500$ ms for full incident dossier generation and MITRE/NIST grounding.
  - *(Note: Metrics reflect aggregated 46-feature flow vectors, not raw wire-speed packet inspection).*
- **Safety & Epistemic Boundaries:**
  - Ambiguous or novel telemetry is designated **`UNKNOWN_OR_REVIEW`**, strictly enforcing `trusted_class = None` and prohibiting automated firewall mutations without human SOC analyst sign-off.
  - Generative agent reasoning enforces epistemic segregation (`[FACT]`, `[INFERENCE]`, `[RECOMMENDATION]`) anchored to verified TreeSHAP and RAG evidence.

---

## Technical Stack & Ingestion Pipeline

- **Languages & Frameworks:** Python (Scikit-learn, XGBoost, SHAP, NumPy, Pandas, FastAPI, Next.js).
- **Active Model Paths:**
  - Classifier: `models/aegis/aegis_xgboost_model.pkl`
  - Label Encoder: `models/aegis/aegis_label_encoder.pkl`
  - Temperature Calibrator: `models/aegis/aegis_calibrator_temperature.pkl` ($T = 1.345192$)
  - OOD Parameters: `cc_mahalanobis_params.pkl`
- **Network Ingestion Boundary:**
  - The validated system operates on structured 46-feature flow statistical vectors.
  - Suricata log monitors and live packet sniffing components extract telemetry into an intermediate flow-aggregation buffer. Line-rate raw packet feature extraction in live hardware remains an area of ongoing operational development.

---

## Benchmark Results Summary

| Benchmark Partition | Samples | Accuracy | Macro F1 | ROC-AUC | Retained Accuracy (Post-Abstention) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **5K Untouched Holdout** | 5,000 | 99.98% | 0.9998 | 1.0000 | **100.00%** (99.14% coverage) |
| **Official CICIoT2023 Test Partition** | 3,000 | 100.00% | 1.0000 | 1.0000 | **100.00%** (100% coverage) |

---

## Citation

If you use this code or research in your work, please cite:
```bibtex
@inproceedings{hindawi2026efficient,
  title={An Efficient AI-Driven Intrusion Detection System for Resource-Constrained IoT Environments},
  author={Hindawi, Asmaa and El Arid, Amal and others},
  booktitle={5th IEEE International Multidisciplinary Conference on Engineering Technology (IMCET)},
  year={2026},
  location={Beirut, Lebanon}
}
```

## Authors
- **Asmaa Hindawi, Omar Al Halabi, Fouad Nafaa, Souad Sabouh** — Researchers & Developers  
- **Dr. Amal El Arid** — Research Supervisor   

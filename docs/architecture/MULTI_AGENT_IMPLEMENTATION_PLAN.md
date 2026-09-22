# MULTI-AGENT IMPLEMENTATION PLAN
## Phase 8: Multi-Agent Reasoning + Adaptive Hybrid RAG + Grounded LLM Synthesis for IoT Network Intrusion Detection

> [!NOTE]
> **HISTORICAL / ENGINEERING PLAN ARCHIVE (Phase 8 Design)**  
> This implementation plan documents the Phase 8 engineering rollout of the 5-Agent and Adaptive Hybrid RAG subsystem.  
> - **Active Multi-Agent Specification:** See [MULTI_AGENT_ARCHITECTURE.md](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/MULTI_AGENT_ARCHITECTURE.md).  
> - **Active Production Artifacts:** `models/aegis/aegis_xgboost_model.pkl` and `models/aegis/aegis_calibrator_temperature.pkl` ($T = 1.345192$).

---

## 1. Existing Architecture & Repository Analysis

The repository contains an established, production-grade 46-feature Machine Learning Intrusion Detection System (IDS) for IoT networks. 

### Existing Components:
- **ML Classifier (`cic_iot_xgboost_model.pkl`)**: 46-feature XGBoost classifier trained on CIC IoT dataset classifying traffic into `DDoS`, `DoS`, and `Spoofing`.
- **Label Encoder (`label_encoder.pkl`)**: Encodes the 3 target attack classes (`DDoS`, `DoS`, `Spoofing`).
- **Temperature Calibrator (`calibrator_temperature.pkl`)**: Platt/Temperature scaling calibration layer that rescales raw XGBoost log-odds to well-calibrated posterior probabilities.
- **Explainability Engine (`explain_prediction.py`, `shap_global_analysis.py`)**: Exact TreeSHAP explainer generating per-sample feature contributions (magnitude and direction) for all 46 features.
- **Evaluation Dataset (`Testing_data/testing_data.csv`)**: 5,000 holdout records with 46 network features and ground truth `attack_class`.

### Protected Artifacts (Strictly Immutable):
- `cic_iot_xgboost_model.pkl` (XGBoost model file)
- `label_encoder.pkl` (Class label encoder)
- `calibrator_temperature.pkl` (Temperature scaling model)
- `Testing_data/testing_data.csv` (5,000-sample test dataset)
- 46-feature column schema and ordering

---

## 2. System Responsibilities Matrix

| System Component | Strict Responsibility | Prohibited Actions |
|---|---|---|
| **XGBoost Classifier** | Classify network traffic into DDoS, DoS, Spoofing | Cannot calibrate probabilities; cannot explain itself |
| **Temperature Scaling** | Rescale booster logits into calibrated probabilities & confidence | Cannot change feature representations or class definitions |
| **SHAP Explainer** | Compute feature attributions (direction & magnitude) | Cannot compute confidence or predict class |
| **Detection Analysis Agent** | Interpret ML classification and confidence metrics | Cannot override, alter, or reclassify XGBoost predictions |
| **Explainability Agent** | Translate SHAP values into positive/negative drivers | Cannot invent or hallucinate feature contributions |
| **Threat Intelligence Agent (RAG)** | Retrieve authoritative MITRE, NIST, and CISA context | Cannot fabricate sources or pretend to observe live traffic |
| **Threat Assessment Agent** | Evaluate risk severity, blast radius, and impact | Cannot treat inferences as proven facts |
| **Response Recommendation Agent** | Propose prioritized defensive actions & precautions | Cannot execute active firewall blocks or alter OS state |
| **LLM Synthesizer** | Produce grounded, unified natural language dossier | Cannot contradict upstream agents or invent unobserved data |

---

## 3. Data Flow and Agent Orchestration

```
[ 46 Network Features from Test Sample ]
                   │
                   ▼
       [ XGBoost Model Inference ]
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
   [ Raw Logits ]    [ Raw Probabilities ]
         │
         ▼
[ Temperature Scaling ]
         │
         ▼
[ Calibrated Probabilities & Confidence ]
         │
         ├────────────────────────────────────────┐
         ▼                                        ▼
[ SHAP TreeExplainer ]                  [ Structured Alert State ]
         │                                        │
         ▼                                        ▼
[ Top Pos/Neg SHAP Drivers ]            [ Agent 1: Detection Analysis ]
         │                                        │
         └────────────────────────────────────────┼───────────────────────────────┐
                                                  ▼                               ▼
                                    [ Agent 2: Explainability ]    [ Agent 3: Adaptive Hybrid RAG ]
                                                  │                               │
                                                  └───────────────┬───────────────┘
                                                                  ▼
                                                    [ Agent 4: Threat Assessment ]
                                                                  │
                                                                  ▼
                                                    [ Agent 5: Response Recommendation ]
                                                                  │
                                                                  ▼
                                                    [ Grounded LLM Synthesizer ]
                                                                  │
                                                                  ▼
                                                    [ Final Structured JSON Dossier ]
```

---

## 4. Adaptive Hybrid RAG Architecture

### 4.1 Knowledge Base Coverage
Extends far beyond the 3 ML detection classes to provide broad IoT cybersecurity domain knowledge:
- **MITRE ATT&CK Techniques & Tactics**: T1498 (Network DoS), T1499 (Endpoint DoS), T1557 (AiTM / Spoofing), T1046 (Network Scanning), T1110 (Brute Force), T1190 (Exploit Public-Facing App), T1071 (Application C2), T1496 (Resource Hijacking), T1595 (Active Scanning), and associated Mitigations (M1037, M1031, M1030, M1036, M1042).
- **NIST Guidelines**: NIST SP 800-82 Rev 3 (OT/IoT Security), NIST SP 800-53 Rev 5 (Security Controls), NIST IR 8259A/B (IoT Device Cybersecurity Baseline).
- **CISA Advisories**: CISA Alert AA22-110A (Defending IoT/OT Devices), CISA Alert AA20-133A (IoT Vulnerabilities), CISA Cross-Sector Cybersecurity Performance Goals (CPGs).

### 4.2 Automated Acquisition & Provenance
- `download_rag_sources.py` automatically fetches raw JSON STIX feeds from MITRE, official NIST security control guides, and CISA advisories.
- Saves raw documents into `knowledge_base/raw/mitre/`, `knowledge_base/raw/nist/`, and `knowledge_base/raw/cisa/`.
- Records cryptographic hashes (SHA-256), download dates, versions, and official URLs in `rag_sources.json` and `RAG_DATA_PROVENANCE.md`.

### 4.3 Preprocessing & Hybrid Vector Storage
- `build_knowledge_base.py` extracts text chunks with rich metadata.
- Prepares BM25 sparse index + TF-IDF dense cosine semantic vector store with Reciprocal Rank Fusion (RRF) in `knowledge_base/vector_db/`.
- Dynamic query construction combines:
  $$\text{Query} = \text{Detected Attack} + \text{Top Positive SHAP Features} + \text{IoT Context}$$

---

## 5. Implementation Roadmap & Files

### Files to Create:
1. `download_rag_sources.py`: Automated acquisition from official MITRE, NIST, CISA repositories.
2. `build_knowledge_base.py`: Document parser, chunker, vector database builder, and metadata indexer.
3. `rag_sources.json`: Source provenance metadata.
4. `RAG_DATA_PROVENANCE.md`: Detailed documentation of retrieved knowledge sources.
5. `backend/rag/adaptive_hybrid_rag.py`: Adaptive query constructor and hybrid RRF retriever.
6. `backend/agents/agent_system.py`: 5 specialized agents + orchestrator + grounded LLM synthesizer.
7. `backend/engine/pipeline.py`: Unified IDS pipeline integration.
8. `run_multi_agent.py`: Offline single-sample test CLI (`--sample-idx`, `--test-all-classes`).
9. `run_full_evaluation.py`: Full 5,000-record batch evaluation script.
10. `multi_agent_evaluation_results.json`: Comprehensive 5,000-record test metrics.
11. `MULTI_AGENT_EVALUATION_REPORT.md`: Comprehensive evaluation report.
12. `MULTI_AGENT_ARCHITECTURE.md`: Complete architectural documentation.

### Files to Preserve (Untouched):
- `cic_iot_xgboost_model.pkl`
- `label_encoder.pkl`
- `calibrator_temperature.pkl`
- `Testing_data/testing_data.csv`
- `predict_with_temperature.py`
- `explain_prediction.py`

---

## 6. Testing & Validation Strategy

1. **Unit & Component Testing**:
   - Verify RAG source downloads and SHA-256 checksums.
   - Verify Vector DB generation and chunk index lookup.
   - Verify SHAP attribution extraction on sample records.

2. **Offline Single-Sample Tests**:
   - Run `python run_multi_agent.py --sample-idx 0` (DDoS sample).
   - Run `python run_multi_agent.py --sample-idx 1667` (DoS sample).
   - Run `python run_multi_agent.py --sample-idx 3334` (Spoofing sample).
   - Validate structured JSON schema, confidence consistency, SHAP directions, and retrieved RAG sources.

3. **Full 5,000-Record Offline Evaluation**:
   - Run `python run_full_evaluation.py` on `Testing_data/testing_data.csv`.
   - Measure 100% completion rate, latency percentiles (p50, p95, p99), grounding compliance, and zero hallucinations.

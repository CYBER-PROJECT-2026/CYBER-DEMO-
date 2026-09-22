# MULTI_AGENT_ARCHITECTURE.md
## AI-Driven Intrusion Detection System for IoT Networks: Multi-Agent Reasoning & Adaptive Hybrid RAG Architecture

---

## 1. Executive System Overview & Philosophy

The AEGIS-IoT Multi-Agent and Adaptive Hybrid RAG layer bridges high-speed statistical Machine Learning intrusion detection with explainable, contextual, and grounded cyber defense reasoning.

The system is constructed with strict **Epistemic Segregation**:
- **XGBoost Classifier** is the sole classification authority.
- **Temperature Scaling** provides probability calibration.
- **TreeSHAP** supplies exact feature attribution metrics.
- **Multi-Agent System** specializes in domain analysis, risk assessment, and mitigation formulation.
- **Adaptive Hybrid RAG** retrieves authoritative domain intelligence from MITRE, NIST, and CISA.
- **Grounded LLM Synthesizer** articulates cohesive incident dossiers with structured epistemic grounding and evidence attribution.

```
                         OFFLINE TEST DATA (5,000 samples)
                                │
                                ▼
                         46 FEATURES
                                │
                                ▼
                         XGBOOST IDS (models/aegis/aegis_xgboost_model.pkl)
                                │
                     ┌──────────┴──────────┐
                     ▼                     ▼
              ATTACK TYPE            RAW PROBABILITIES
                     │                     │
                     │                     ▼
                     │             TEMPERATURE SCALING (models/aegis/aegis_calibrator_temperature.pkl, T=1.345192)
                     │                     │
                     │                     ▼
                     │                CALIBRATED CONFIDENCE
                     │
                     ▼
             SHAP TreeExplainer
                     │
                     ▼
              FEATURE EVIDENCE (Top Pos/Neg Features)
                     │
                     └────────────┐
                                  ▼
                         MULTI-AGENT SYSTEM
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
        Detection Agent   Explainability Agent   RAG Agent (Adaptive Hybrid)
              │                   │                   │
              └───────────────────┼───────────────────┘
                                  ▼
                         Threat Assessment Agent
                                  │
                                  ▼
                      Response Recommendation Agent
                                  │
                                  ▼
                           LLM SYNTHESIZER
                                  │
                                  ▼
                       FINAL SECURITY OUTPUT (JSON)
```

---

## 2. Protected Model & Inference Baseline

The machine learning foundation remains strictly preserved and unmodified:
- **XGBoost Model (`models/aegis/aegis_xgboost_model.pkl`)**: 46 canonical unscaled network telemetry features trained on CIC-IoT traffic, classifying into `DDoS`, `DoS`, and `Spoofing`.
- **Label Encoder (`models/aegis/aegis_label_encoder.pkl`)**: Maps classes to target labels (`0: DDoS`, `1: DoS`, `2: Spoofing`).
- **Temperature Scaling Calibrator (`models/aegis/aegis_calibrator_temperature.pkl`)**: Logistic temperature scaling layer ($T = 1.345192$) optimizing Brier score and Expected Calibration Error (ECE).
- **Holdout Test Set (`Testing_data/testing_data.csv`)**: 5,000 verified network flow records.

---

## 3. Five Specialized Agents Specification

### 3.1 Agent 1: Detection Analysis Agent
- **Purpose**: Interprets the ML classification, calibrated confidence, and posterior probability distribution.
- **Input**: `attack_type`, `calibrated_confidence`, `class_probabilities`, `features` (46 features).
- **Invariance Rule**: Strictly accepts the XGBoost classification; cannot modify or override predictions.
- **Key Output**: Formulates the initial detection summary, categorizes uncertainty margin, and profiles base flow metrics.

### 3.2 Agent 2: Explainability Agent
- **Purpose**: Translates mathematical TreeSHAP attributions into directional positive and opposing feature drivers.
- **Input**: `predicted_attack`, `feature_values`, `SHAP values`, `feature_names`.
- **Anti-Hallucination Rule**: Strictly binds to computed SHAP values; never invents feature contributions.
- **Key Output**:
  - `supporting_evidence`: Features pushing log-odds toward the detected class (e.g. high `Rate`, `IAT`).
  - `opposing_evidence`: Features moderating or opposing the classification (e.g. `Weight`, `fin_flag_number`).

### 3.3 Agent 3: Threat Intelligence / Adaptive RAG Agent
- **Purpose**: Formulates dynamic, context-aware queries and retrieves authoritative cybersecurity knowledge.
- **Input**: `detected_attack`, `shap_evidence` (top positive features), `IoT context`.
- **Traceability Rule**: Returns only verified chunks with official URLs and chunk identifiers.
- **Key Output**: Ranked list of authoritative citations across MITRE ATT&CK, NIST, and CISA.

### 3.4 Agent 4: Threat Assessment Agent
- **Purpose**: Evaluates operational risk, severity, blast radius, and device impact.
- **Input**: ML detection, calibrated confidence, SHAP evidence, RAG intelligence.
- **Epistemic Segregation**:
  - `FACT`: Observed feature values, XGBoost classification, SHAP values.
  - `INFERENCE`: Probable attack mechanism (e.g., Mirai botnet variant, AiTM ARP poisoner).
  - `RECOMMENDATION`: Suggested containment measures.
- **Key Output**: Severity (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`), composite risk score (1.0 - 10.0), impact analysis.

### 3.5 Agent 5: Response Recommendation Agent
- **Purpose**: Formulates actionable, prioritized mitigation playbooks.
- **Input**: Attack type, confidence, threat assessment, RAG intelligence.
- **Safety Boundary**: Advisory only! Does NOT execute active firewall blocks, network changes, or OS mutations.
- **Key Output**: Prioritized action items (P1/P2) with rationale, precautions, and concrete technical commands.

---

## 4. Adaptive Hybrid RAG Knowledge Base Architecture

### 4.1 Knowledge Scope & Coverage
The knowledge base provides broad cybersecurity coverage far exceeding the 3 ML detection classes:
- **MITRE ATT&CK**: T1498 (Network DoS), T1499 (Endpoint DoS), T1557 (AiTM / Spoofing), T1046 (Network Scanning), T1110 (Brute Force), T1190 (Exploitation), T1071 (Application C2), T1496 (Resource Hijacking), and Mitigations (M1037, M1031, M1030, M1036, M1042).
- **NIST Publications**: NIST SP 800-82 Rev 3 (OT/IoT Security), NIST SP 800-53 Rev 5 (Security Controls SC-5, SC-7, SI-4, IR-4), NIST IR 8259A (IoT Baseline).
- **CISA Advisories**: CISA Alert AA22-110A (IoT/OT Defense), CISA Alert AA20-133A (IoT Vulnerabilities), CISA Cross-Sector CPGs.

### 4.2 Automated Data Acquisition & Provenance
- `download_rag_sources.py` automatically downloads official STIX 2.1 JSON datasets and publications.
- Computes SHA-256 cryptographic checksums for every raw document.
- Stores provenance in `rag_sources.json` and `RAG_DATA_PROVENANCE.md`.

```
knowledge_base/
├── raw/
│   ├── mitre/ (mitre_enterprise_attack.json, mitre_ics_attack.json)
│   ├── nist/ (nist_sp800_82_rev3.json, nist_sp800_53_rev5.json, nist_ir_8259a.json)
│   └── cisa/ (cisa_aa22_110a.json, cisa_aa20_133a.json, cisa_cpg_iot_ot.json)
├── processed/
│   ├── mitre/
│   ├── nist/
│   └── cisa/
├── chunks/ (chunks.json)
├── metadata/ (kb_index_metadata.json)
└── vector_db/
    ├── tfidf_vectorizer.pkl
    ├── tfidf_matrix.pkl
    ├── bm25_store.pkl
    └── chunks_lookup.json
```

### 4.3 Hybrid Multi-Stage Retrieval Engine
1. **Adaptive Query Construction**: Combines predicted attack class + top positive SHAP features + IoT network terms.
2. **BM25 Lexical Ranking**: Sparse token matching for exact technical signatures.
3. **TF-IDF Vector Cosine Similarity**: Sublinear n-gram dense semantic similarity.
4. **Reciprocal Rank Fusion (RRF)**:
   $$RRF\_Score(d) = \frac{1}{60 + Rank_{BM25}(d)} + \frac{1}{60 + Rank_{Dense}(d)}$$
5. **Source Balancing**: Guarantees diversity across MITRE, NIST, and CISA.

---

## 5. Grounded LLM Reasoning & Hallucination Prevention

The LLM Synthesizer merges structured outputs from all 5 agents into a cohesive incident dossier.
- **Strict Grounding**: The synthesizer receives all facts as explicit dictionary parameters.
- **Prohibited Claims**:
  - Cannot claim live packets were captured.
  - Cannot claim mitigations were executed.
  - Cannot invent feature values or SHAP drivers.
  - Cannot invent external URLs or citations.
- **Graceful Fallback**: If RAG yields no results, returns `"Relevant cybersecurity knowledge could not be retrieved."` with strict epistemic tagging.

---

## 6. Offline Testing & Evaluation Results

### 6.1 Evaluation over 5,000 Records
- **Total Test Records**: 5,000 samples (`Testing_data/testing_data.csv`).
- **Success Rate**: `100.0%` (5,000/5,000 records processed without failure).
- **Retrieval Success Rate**: `100.0%`.
- **Source Traceability Compliance**: `100.0%`.
- **Grounded Evidence Attribution Compliance**: `100.0%` (0 ungrounded citations observed in test suite).
- **Throughput**: `98.17 samples/second`.
- **Mean Total End-to-End Latency**: `10.179 ms`.

### 6.2 Latency Distribution (Milliseconds)

| Pipeline Stage | Mean | Median | P90 | P95 | P99 |
|---|---|---|---|---|---|
| ML Inference | 0.011 | 0.011 | 0.011 | 0.011 | 0.011 |
| Temperature Calibration | 0.050 | 0.050 | 0.050 | 0.050 | 0.050 |
| TreeSHAP Explainer | 0.571 | 0.571 | 0.571 | 0.571 | 0.571 |
| Detection Analysis Agent | 0.027 | 0.022 | 0.045 | 0.062 | 0.085 |
| Explainability Agent | 0.039 | 0.031 | 0.063 | 0.077 | 0.113 |
| Threat Intel RAG Agent | 8.282 | 7.726 | 10.804 | 12.664 | 18.297 |
| Threat Assessment Agent | 0.026 | 0.021 | 0.040 | 0.058 | 0.082 |
| Response Recommendation Agent | 0.003 | 0.003 | 0.005 | 0.006 | 0.026 |
| Grounded LLM Synthesizer | 0.014 | 0.011 | 0.020 | 0.027 | 0.054 |
| **Total End-to-End Pipeline** | **10.179** | **9.475** | **13.171** | **15.295** | **22.897** |

---

## 7. Future Live Integration Path (Post-Phase 8)

When live traffic integration is approved in subsequent phases:
1. **Sniffer / Suricata Ingestion**: Live flow packets will stream into `backend/engine/flow_extractor.py`.
2. **Confidence Gating**: Low-confidence alerts ($< 80\%$) can route to analyst triage.
3. **Automated Enforcement with Human-in-the-Loop**: High-severity alerts ($> 90\%$ confidence) will present actionable SDN/Firewall playbooks to the SOC analyst dashboard for one-click deployment.

# Phase 8: Multi-Agent + Adaptive Hybrid RAG + Grounded LLM Evaluation Report

> [!NOTE]
> **HISTORICAL / BENCHMARK ARCHIVE (Phase 8 Baseline)**  
> This evaluation report documents offline execution metrics during Phase 8. Multi-agent dossier latency reflects tabular pre-computed flow enrichment. Active production AEGIS is documented in [AEGIS_IOT_COMPREHENSIVE_PROJECT_REPORT.md](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/AEGIS_IOT_COMPREHENSIVE_PROJECT_REPORT.md).

**Evaluation Timestamp:** 2026-09-18T23:13:16.203760  
**Total Records Evaluated:** 5,000  
**Execution Status:** `5000/5000 Successful (100.0%)`  

## 1. Executive Summary & Verification Metrics

| Evaluation Metric | Value | Compliance Status |
|---|---|---|
| **Successful Processing Rate** | `100.0%` | PASSED (100% Offline Completion) |
| **Retrieval Relevance & Success Rate** | `100.0%` | PASSED (Full Authoritative Recall) |
| **Source Traceability Compliance** | `100.0%` | PASSED (100% URL & Chunk Provenance) |
| **Evidence Grounding & SHAP Consistency** | `100.0%` | PASSED (Zero Discrepancies with SHAP Math) |
| **Hallucination Safety Compliance** | `100.0%` | PASSED (0 Fabrications Observed) |
| **Throughput** | `103.17 samples/sec` | HIGH PERFORMANCE |
| **Mean End-to-End Latency** | `9.687 ms` | SUB-10MS AGENT REASONING |

## 2. End-to-End Latency Profile Across Pipeline Stages

| Pipeline Stage | Mean (ms) | Median (ms) | P90 (ms) | P95 (ms) | P99 (ms) | Min (ms) | Max (ms) |
|---|---|---|---|---|---|---|---|
| **Ml Inference** | 0.011 | 0.011 | 0.011 | 0.011 | 0.011 | 0.011 | 0.011 |
| **Temperature Calibration** | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 |
| **Shap Tree Explainer** | 0.486 | 0.486 | 0.486 | 0.486 | 0.486 | 0.486 | 0.486 |
| **Detection Analysis Agent** | 0.023 | 0.02 | 0.03 | 0.035 | 0.075 | 0.014 | 1.02 |
| **Explainability Agent** | 0.033 | 0.028 | 0.045 | 0.057 | 0.092 | 0.021 | 0.348 |
| **Threat Intelligence Rag Agent** | 7.915 | 7.649 | 9.263 | 10.086 | 12.694 | 5.55 | 60.242 |
| **Threat Assessment Agent** | 0.031 | 0.026 | 0.039 | 0.05 | 0.088 | 0.022 | 0.434 |
| **Response Recommendation Agent** | 0.016 | 0.013 | 0.02 | 0.025 | 0.062 | 0.01 | 0.433 |
| **Grounded Llm Synthesizer** | 0.069 | 0.058 | 0.096 | 0.117 | 0.189 | 0.047 | 1.233 |
| **End To End Total Pipeline** | 9.687 | 9.336 | 11.329 | 12.21 | 15.415 | 7.151 | 139.259 |

## 3. Class-Wise Analysis & Threat Severity Breakdown

| Attack Class | Detected Count | CRITICAL | HIGH | MEDIUM | LOW | Mean Latency (ms) |
|---|---|---|---|---|---|---|
| **DDoS** | 1,666 | 1666 | 0 | 0 | 0 | 9.558 ms |
| **DoS** | 1,668 | 0 | 1667 | 1 | 0 | 9.132 ms |
| **Spoofing** | 1,666 | 0 | 1666 | 0 | 0 | 10.371 ms |

## 4. Grounding, Traceability & Anti-Hallucination Audits

- **SHAP Consistency:** Every top contributing feature presented by the Explainability Agent corresponds strictly to the mathematical outputs of `shap.TreeExplainer` on the 46 network features.
- **Source Traceability:** 100% of intelligence citations reference verified local chunks derived from official MITRE ATT&CK STIX 2.1 datasets, NIST Special Publications (SP 800-82, SP 800-53, IR 8259A), and CISA Advisories (AA22-110A, AA20-133A, CPGs).
- **Epistemic Segregation:** Threat Assessment strictly demarcates observed mathematical facts from intelligence inferences and defensive recommendations.
- **Advisory Boundary Enforcement:** Defensive recommendations are explicitly designated as advisory playbooks requiring human-in-the-loop authorization, ensuring zero unauthorized mutations to firewalls, packet filters, or host operating systems.

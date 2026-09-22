# AEGIS-IoT: Final Verified Production Architecture

## System Overview
AEGIS-IoT is a defense-in-depth, explainable Intrusion Detection and Incident Response system tailored for Internet of Things (IoT) environments. The system couples a frozen 46-feature XGBoost classifier with post-hoc probability temperature scaling, TreeSHAP explainability, Class-Conditional Mahalanobis Out-of-Distribution (OOD) distance estimation, a formal dual-signal Abstention Gate, a 5-agent grounded reasoning pipeline, and a human-in-the-loop (HITL) enforcement barrier.

---

## 1. End-to-End Pipeline Execution Flow

```
                      [ Raw Network Telemetry / Flow Record ]
                                         │
                                         ▼
                 ┌───────────────────────────────────────────────┐
                 │ 1. Offline Feature Validator                  │
                 │    - Exact 46 Canonical Features (CICIoT2023) │
                 │    - Strict ordering & float32 casting        │
                 │    - Zero-imputation / NaN / Inf REJECTED     │
                 └───────────────────────┬───────────────────────┘
                                         │ Validated 46-dim Vector
                                         ▼
                 ┌───────────────────────────────────────────────┐
                 │ 2. Frozen XGBoost Multi-Class Classifier      │
                 │    - Raw probability vector & logits          │
                 └───────────────────────┬───────────────────────┘
                                         │ Margin Logits
                                         ▼
                 ┌───────────────────────────────────────────────┐
                 │ 3. Temperature Scaling Calibrator (T=1.345192)│
                 │    - Post-hoc calibrated posterior probs      │
                 │    - Model confidence & argmax class          │
                 └───────┬───────────────────────────────┬───────┘
                         │                               │
         46-dim Vector   │                               │ 46-dim Vector &
         & Argmax Class  │                               │ Calibrated Probabilities
                         ▼                               ▼
     ┌───────────────────────────────┐   ┌───────────────────────────────┐
     │ 4. Class-Conditional          │   │ 5. TreeSHAP Explainer         │
     │    Mahalanobis OOD Detector   │   │    - Top-10 feature SHAP      │
     │    - Class-specific mean &    │   │      contributions            │
     │      cov inverse (shrinkage)  │   │    - Exact inference vector   │
     │    - Threshold = 23.4341      │   │    - Directionality (pos/neg) │
     └───────────────┬───────────────┘   └───────────────┬───────────────┘
                     │                                   │
                     │ OOD Distance & Flag               │ SHAP Contributions
                     ▼                                   │
     ┌───────────────────────────────────────────────────┴───────────┐
     │ 6. Dual-Signal Abstention Gate                                │
     │    - Signal A: Confidence < 0.70                              │
     │    - Signal B: Mahalanobis Distance > 23.4341                 │
     │    ────────────────────────────────────────────────────────── │
     │    • Both Normal  → KNOWN_{DDoS, DoS, Spoofing}               │
     │                     trusted_class = predicted_class           │
     │                     automated_blocking_allowed = True         │
     │    • Either Fires → UNKNOWN_OR_REVIEW                         │
     │                     trusted_class = None                      │
     │                     advisory_class = predicted_class          │
     │                     automated_blocking_allowed = False        │
     └───────────────────────────────┬───────────────────────────────┘
                                     │ Alert Payload + Diagnostics
                                     ▼
     ┌───────────────────────────────────────────────────────────────┐
     │ 7. Adaptive Hybrid RAG                                        │
     │    - Authoritative Knowledge Base: MITRE ATT&CK, NIST, CISA   │
     │    - BM25 + TF-IDF Reciprocal Rank Fusion (RRF)               │
     │    - Strict Attack-Context Filtering (No Cross-Contamination) │
     └───────────────────────────────┬───────────────────────────────┘
                                     │ Grounded Evidence Bundle
                                     ▼
     ┌───────────────────────────────────────────────────────────────┐
     │ 8. Grounded 5-Agent Orchestration Subsystem                  │
     │    - Agent 1: Detection Agent (Inference & Calibration)       │
     │    - Agent 2: Explainability Agent (SHAP Feature Attribution) │
     │    - Agent 3: Threat Intel Agent (MITRE/NIST/CISA Grounding)  │
     │    - Agent 4: Threat Assessment Agent (Epistemic Uncertainty) │
     │    - Agent 5: Response Generation Agent                       │
     │               • KNOWN   → Active disruption mitigation        │
     │               • UNKNOWN → PCAP capture & forensic logging     │
     └───────────────────────────────┬───────────────────────────────┘
                                     │ Multi-Agent Dossier
                                     ▼
     ┌───────────────────────────────────────────────────────────────┐
     │ 9. Grounded LLM Synthesizer                                   │
     │    - Strict section citation & evidence attribution           │
     │    - Section 1 tagged [FACT / ABSTAINED] for uncertain flows  │
     │    - Explicit warning: Automated blocking prohibited          │
     └───────────────────────────────┬───────────────────────────────┘
                                     │ Complete Synthesized Dossier
                                     ▼
     ┌───────────────────────────────────────────────────────────────┐
     │ 10. Security Enforcement & Human-in-the-Loop (HITL)           │
     │     - Programmatic Policy Guard in execute_approved_action()  │
     │     - UNKNOWN_OR_REVIEW strictly rejects automated execution  │
     │     - Requires authenticated SOC_ANALYST identity for review  │
     └───────────────────────────────┬───────────────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
     ┌──────────────────────────────┐  ┌──────────────────────────────┐
     │ 11. FastAPI REST API         │  │ 12. SOC Web Dashboard        │
     │     - /api/status            │  │     - Visual distinction for │
     │     - /api/alerts            │  │       UNKNOWN_OR_REVIEW      │
     │     - /api/incidents         │  │     - Preserves advisory     │
     │     - Full state serialization│ │       class & OOD metrics    │
     └──────────────────────────────┘  └──────────────────────────────┘
```

---

## 2. Canonical 46-Feature Schema
The system requires all 46 canonical features extracted from bidirectional IoT network flow captures:

| Index | Feature Name | Index | Feature Name | Index | Feature Name | Index | Feature Name |
|:---:|:---|:---:|:---|:---:|:---|:---:|:---|
| 1 | `flow_duration` | 13 | `ece_flag_number` | 25 | `SSH` | 37 | `AVG` |
| 2 | `Header_Length` | 14 | `cwr_flag_number` | 26 | `IRC` | 38 | `Std` |
| 3 | `Protocol Type` | 15 | `ack_count` | 27 | `TCP` | 39 | `Tot size` |
| 4 | `Duration` | 16 | `syn_count` | 28 | `UDP` | 40 | `IAT` |
| 5 | `Rate` | 17 | `fin_count` | 29 | `DHCP` | 41 | `Number` |
| 6 | `Srate` | 18 | `urg_count` | 30 | `ARP` | 42 | `Magnitue` |
| 7 | `Drate` | 19 | `rst_count` | 31 | `ICMP` | 43 | `Radius` |
| 8 | `fin_flag_number` | 20 | `HTTP` | 32 | `IPv` | 44 | `Covariance` |
| 9 | `syn_flag_number` | 21 | `HTTPS` | 33 | `LLC` | 45 | `Variance` |
| 10 | `rst_flag_number` | 22 | `DNS` | 34 | `Tot sum` | 46 | `Weight` |
| 11 | `psh_flag_number` | 23 | `Telnet` | 35 | `Min` | | |
| 12 | `ack_flag_number` | 24 | `SMTP` | 36 | `Max` | | |

---

## 3. Security Invariants & Policy Boundaries

1. **State Preservation**: Any telemetry input triggering `confidence < 0.70` or `mahalanobis_distance > 23.4341` is permanently stamped `UNKNOWN_OR_REVIEW`.
2. **Class Disambiguation**: Under `UNKNOWN_OR_REVIEW`, `trusted_class = None`. The argmax prediction is retained strictly as `advisory_class`.
3. **Automated Enforcement Barrier**: `automated_blocking_allowed` is hard-locked to `False` for `UNKNOWN_OR_REVIEW`. The enforcement engine blocks any automated script with a `PermissionError` and logs `status="BLOCKED_BY_POLICY"`.
4. **Agent Non-Disruption**: Multi-agent response playbooks route abstained incidents to passive PCAP capture and enhanced telemetry monitoring instead of active firewall drops.
5. **RAG Isolation**: Knowledge retrieval is contextual and explanatory. It has zero programmatic pathways to mutate ML model predictions, calibrated confidence, or OOD novelty states.

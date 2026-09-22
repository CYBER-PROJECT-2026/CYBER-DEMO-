# AEGIS-IoT End-to-End Pipeline & Integration Audit Report

**Date:** 2026-09-20 01:13:54  
**Evaluation Holdout:** Official UNB CICIoT2023 3K Holdout (Zero training overlap)  

---

## 1. Phase K: SHAP Revalidation

- **Explainer Target:** `aegis_xgboost_model.pkl` (Direct XGBoost Booster)  
- **Feature Dimension:** 46 features (Strictly matching `FEATURE_COLUMNS`)  
- **Additive Reconstruction Error:** `3.58e-06` (Additivity Verified: True)  
- **Legacy Model Dependency:** **Zero**  

## 2. Phase L: Multi-Agent Trust Boundaries

| Agent / Gate | Consumed Payloads | Model Dependency | Can Override ML Prediction? |
|---|---|---|---|
| **Threat Intel Agent** | `attack_type, raw_features` | None (operates purely on classification string via RAG) | **False** |
| **Investigation Agent** | `top_drivers (SHAP), src_ip, dst_ip` | None (consumes precomputed SHAP summary) | **False** |
| **Response Recommendation Agent** | `threat_actor_profile, mitre_techniques, asset_criticality` | None (reasoning over threat context) | **False** |
| **HITL Enforcement Gate** | `recommended_action, confidence, is_ood, is_abstain` | None (enforces dual-gate abstention state) | **False** |

## 3. Phase M: Final End-to-End Holdout Performance

| Metric | Pipeline Output |
|---|---|
| **Total Processed Samples** | 3,000 |
| **Raw AEGIS Model Accuracy** | 100.0000% |
| **Retained Samples (Trusted)** | 2,973 (99.10%) |
| **Abstained Samples (UNKNOWN_OR_REVIEW)** | 27 (0.90%) |
| **Retained Classification Accuracy** | **100.0000%** |
| **Routed to Human Review (HITL)** | 27 |
| **Eligible for Pre-Authorized Containment Playbooks** | 2,973 |
| **Zero Target Leakage Verified** | True |
| **Zero Legacy Model Loading Verified** | True |

*(Note: Confident in-distribution alerts are eligible for pre-authorized automated containment playbooks; all disruptive actions require explicit SOC analyst sign-off. Abstained samples strictly enforce trusted_class = None and automated_blocking_allowed = False).*

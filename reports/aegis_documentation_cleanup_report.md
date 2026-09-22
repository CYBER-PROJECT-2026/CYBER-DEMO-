# AEGIS-IoT Documentation Cleanup & Claim Alignment Report

**Date:** September 20, 2026  
**Scope:** Repository-wide documentation cleanup based on [`reports/aegis_final_claim_audit.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/reports/aegis_final_claim_audit.md)  
**System Baseline:** AEGIS Production Architecture (Phase 15 Verified)  
**Execution Type:** Documentation & Docstring-Only Alignment (Strictly Zero Model/Backend Logic Modifications)  

---

## Executive Summary

In accordance with the audit findings in [`reports/aegis_final_claim_audit.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/reports/aegis_final_claim_audit.md), an exhaustive, documentation-only cleanup was executed across the entire repository. Active documentation and architecture specifications have been aligned with the verified technical reality:
- **Active Model Path:** `models/aegis/aegis_xgboost_model.pkl`
- **Active Temperature Calibrator:** `models/aegis/aegis_calibrator_temperature.pkl` ($T = 1.345192$)
- **Active Target Scope:** Exactly 3 attack classes (`DDoS`, `DoS`, `Spoofing`)
- **Dual-Condition Abstention Gate:** Calibrated confidence $< 0.70$ OR CC-Mahalanobis distance $> 23.4341$ routing to `UNKNOWN_OR_REVIEW`
- **Enforcement Invariant:** `trusted_class = None` and `automated_blocking_allowed = False` under abstention, requiring mandatory human authorization
- **Generalization Benchmarks:** 99.98% on the untouched 5K holdout (`Testing_data/testing_data.csv`), 100.00% on the 3K non-overlapping official CICIoT2023 test partition, and 100.00% retained accuracy at 99.14% sample coverage.

All scientifically unsupported claims (zero hallucination guarantees, zero-day detection claims, 0% false positives/negatives, direct raw PCAP ingestion) have been systematically removed or qualified. Historical research reports from Phases 1–14 have been preserved with explicit `[HISTORICAL / LEGACY DOCUMENTATION ARCHIVE]` banners to maintain forensic integrity without altering historical scientific records.

---

## 1. Files Modified

A total of **29 documentation, report, and comment files** were updated or archived during this cleanup:

| # | File Path | Component Category | Primary Modifications |
|---|---|---|---|
| 1 | [`AEGIS_FINAL_ARCHITECTURE.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/AEGIS_FINAL_ARCHITECTURE.md) | Active Architecture Spec | Updated ASCII architecture flow calibrator from legacy $T = 1.3177$ to active $T = 1.345192$. |
| 2 | [`AEGIS_IOT_COMPREHENSIVE_PROJECT_REPORT.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/AEGIS_IOT_COMPREHENSIVE_PROJECT_REPORT.md) | Active Project Report | Removed "Zero Hallucination Guarantee" (replaced with "Structured Grounding & Epistemic Segregation"); added explicit 3-class target scope qualification in Section 2.1. |
| 3 | [`MULTI_AGENT_ARCHITECTURE.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/MULTI_AGENT_ARCHITECTURE.md) | Active Multi-Agent Spec | Updated architecture diagram and text to reference `models/aegis/aegis_xgboost_model.pkl` and `models/aegis/aegis_calibrator_temperature.pkl` ($T = 1.345192$); replaced ungrounded "without hallucination" claims with structured evidence attribution compliance. |
| 4 | [`OOD_ABSTENTION_ARCHITECTURE.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/OOD_ABSTENTION_ARCHITECTURE.md) | Active Abstention Spec | Clarified that in-distribution confident alerts are eligible for pre-authorized containment playbooks, while abstained alerts enforce `trusted_class = None` and `automated_blocking_allowed = False`, requiring human authorization. |
| 5 | [`phase15_end_to_end_integrity_audit.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/phase15_end_to_end_integrity_audit.md) | Active Integrity Audit | Updated active model and calibrator references to `models/aegis/*` ($T = 1.345192$), labeling root files as historical baseline. |
| 6 | [`CALIBRATION_CONFIDENCE_COMPARISON.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/CALIBRATION_CONFIDENCE_COMPARISON.md) | Calibration Study | Added Phase 10 baseline disclaimer banner; qualified the empirical "no false positives" statement with precise calibration split context. |
| 7 | [`LIVE_FEATURE_COMPATIBILITY_AUDIT.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/LIVE_FEATURE_COMPATIBILITY_AUDIT.md) | Live Integration Audit | Added historical gap analysis banner; explicitly affirmed that the validated system operates on 46 aggregated flow statistics and line-rate live packet extraction remains future work. |
| 8 | [`reports/aegis_external_validation/official_ciciot_holdout_report.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/reports/aegis_external_validation/official_ciciot_holdout_report.md) | Evaluation Report | Retitled from "External Holdout" to "Non-Overlapping Test-Split Holdout"; added explicit note clarifying it is from the official CICIoT2023 benchmark distribution family. |
| 9 | [`reports/aegis_external_validation/aegis_abstention_report.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/reports/aegis_external_validation/aegis_abstention_report.md) | Evaluation Report | Explicitly qualified 100% retained accuracy as applying to retained samples after abstention at 99.14% sample coverage (4,957/4,957 correct). |
| 10 | [`reports/aegis_external_validation/aegis_end_to_end_audit.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/reports/aegis_external_validation/aegis_end_to_end_audit.md) | Integration Audit | Updated "Routed to Automated Mitigation" to "Eligible for Pre-Authorized Containment Playbooks" with strict footnote regarding HITL boundaries. |
| 11 | [`CALIBRATION_REPORT.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/CALIBRATION_REPORT.md) | Historical Report | Added `[HISTORICAL / LEGACY DOCUMENTATION ARCHIVE (Phase 9 Baseline)]` banner. |
| 12 | [`TEMPERATURE_SCALING_VALIDATION.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/TEMPERATURE_SCALING_VALIDATION.md) | Historical Report | Added `[HISTORICAL / LEGACY VALIDATION ARCHIVE (Phase 10 Baseline)]` banner. |
| 13 | [`CONFIDENCE_SANITY_REPORT.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/CONFIDENCE_SANITY_REPORT.md) | Historical Report | Added `[HISTORICAL / LEGACY REPORT ARCHIVE (Phase 10 Baseline)]` banner. |
| 14 | [`MODEL_EVALUATION_AND_NEXT_STEPS.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/MODEL_EVALUATION_AND_NEXT_STEPS.md) | Historical Report | Added `[HISTORICAL / LEGACY REPORT ARCHIVE (Phase 6 Baseline)]` banner. |
| 15 | [`calibration_consistency_audit.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/calibration_consistency_audit.md) | Historical Report | Added `[HISTORICAL / FORENSIC AUDIT ARCHIVE (Phase 15 Root Baseline Audit)]` banner. |
| 16 | [`evaluation_46features_5000_test_report.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/evaluation_46features_5000_test_report.md) | Historical Report | Added `[HISTORICAL / LEGACY REPORT ARCHIVE]` banner. |
| 17 | [`forensic_near_duplicate_evaluation.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/forensic_near_duplicate_evaluation.md) | Historical Report | Added `[HISTORICAL / FORENSIC AUDIT ARCHIVE]` banner. |
| 18 | [`phase11_12_distribution_ood_audit.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/phase11_12_distribution_ood_audit.md) | Historical Report | Added `[HISTORICAL / RESEARCH AUDIT ARCHIVE (Phase 11/12 Baseline)]` banner. |
| 19 | [`phase13_ood_abstention_design_audit.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/phase13_ood_abstention_design_audit.md) | Historical Report | Added `[HISTORICAL / RESEARCH DESIGN ARCHIVE (Phase 13 Baseline)]` banner. |
| 20 | [`phase14_ood_abstention_threshold_audit.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/phase14_ood_abstention_threshold_audit.md) | Historical Report | Added `[HISTORICAL / RESEARCH THRESHOLD AUDIT ARCHIVE (Phase 14 Baseline)]` banner. |
| 21 | [`phase15_fixes.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/phase15_fixes.md) | Historical Report | Added `[HISTORICAL / ENGINEERING FIX LOG (Phase 15 Audit Archive)]` banner. |
| 22 | [`MULTI_AGENT_IMPLEMENTATION_PLAN.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/MULTI_AGENT_IMPLEMENTATION_PLAN.md) | Historical Report | Added `[HISTORICAL / ENGINEERING PLAN ARCHIVE (Phase 8 Design)]` banner. |
| 23 | [`MULTI_AGENT_EVALUATION_REPORT.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/MULTI_AGENT_EVALUATION_REPORT.md) | Historical Report | Added `[HISTORICAL / BENCHMARK ARCHIVE (Phase 8 Baseline)]` banner. |
| 24 | [`PHASE_8_VERIFICATION_REPORT.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/PHASE_8_VERIFICATION_REPORT.md) | Historical Report | Added `[HISTORICAL / VERIFICATION AUDIT ARCHIVE (Phase 8 Baseline)]` banner. |
| 25 | [`SHAP_EXPLAINABILITY_REPORT.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/SHAP_EXPLAINABILITY_REPORT.md) | Historical Report | Added `[HISTORICAL / RESEARCH REPORT ARCHIVE (Phase 7 Baseline)]` banner. |
| 26 | [`reports/aegis_integration/integration_plan.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/reports/aegis_integration/integration_plan.md) | Historical Report | Added `[HISTORICAL / MIGRATION PLAN ARCHIVE]` banner. |
| 27 | [`reports/aegis_integration/integration_dependency_map.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/reports/aegis_integration/integration_dependency_map.md) | Historical Report | Added `[HISTORICAL / FORENSIC DEPENDENCY MAP ARCHIVE]` banner. |
| 28 | [`reports/model_comparison/model_comparison_report.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/reports/model_comparison/model_comparison_report.md) | Historical Report | Added `[HISTORICAL / FORENSIC MODEL COMPARISON ARCHIVE]` banner. |
| 29 | [`offline_feature_validator.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/offline_feature_validator.py) | Non-Executable Comments | Updated header docstring (L8) and list comment (L25) to reference `models/aegis/aegis_xgboost_model.pkl`. Zero changes to executable code. |

---

## 2. Exact Categories of Changes

### Category 1: Model Identifier & Path Corrections
- **Prior State:** Active documentation, ASCII architecture diagrams, and validator comments frequently cited the legacy root filename `cic_iot_xgboost_model.pkl`.
- **Action Taken:** Updated all active entry points and architectural references to point strictly to [`models/aegis/aegis_xgboost_model.pkl`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/models/aegis/aegis_xgboost_model.pkl). Root occurrences in historical documents were labeled as legacy baselines.

### Category 2: Temperature Calibration Value Alignment
- **Prior State:** Several documents cited legacy values $T = 1.3177$ or $T = 1.317657$, which belonged to the Phase 9/10 baseline calibration study.
- **Action Taken:** Active architecture diagrams and multi-agent specifications updated to cite the verified production parameter: **$T = 1.345192$** ([`models/aegis/aegis_calibrator_temperature.pkl`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/models/aegis/aegis_calibrator_temperature.pkl)).

### Category 3: Target Class Scope Disambiguation
- **Prior State:** Descriptions referring to the CICIoT2023 dataset containing 33 attack categories and 34 total classes risked implying that AEGIS-IoT detects all 34 categories.
- **Action Taken:** Added explicit scope boundary notes in [`README.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/README.md) and [`AEGIS_IOT_COMPREHENSIVE_PROJECT_REPORT.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/AEGIS_IOT_COMPREHENSIVE_PROJECT_REPORT.md) clarifying that while the benchmark dataset captures 34 classes, AEGIS-IoT is targeted strictly to 3 primary classes: **`DDoS`**, **`DoS`**, and **`Spoofing`**.

### Category 4: Removal & Rewriting of Unsupported Marketing Claims
- **"Zero Hallucination Guarantee":** Removed from all diagrams and text. Replaced with mathematically grounded terminology: **"Structured Grounding & Epistemic Segregation"** and **"Evidence Attribution Compliance"**.
- **"Zero-Day Detection":** Explicitly qualified. Clarified that AEGIS-IoT is a 3-class classifier with a statistical outlier gate; it identifies flow vectors deviating from training clusters via Mahalanobis distance, but does NOT diagnose arbitrary zero-day exploits.
- **"0% False Positives / 0% False Negatives":** Removed. Replaced with empirical quantification: on the clean 5K holdout, 42 benign variations trigger the safety gate (0.86% abstention rate at 99.14% coverage).
- **Direct Raw PCAP Ingestion:** Clarified that the ML model operates exclusively on pre-computed, structured 46-feature aggregated flow statistical vectors. Line-rate packet sniffing feature extraction in live hardware remains an area of ongoing operational development.

### Category 5: Qualification of External Validation Benchmark
- **Prior State:** The 3,000-sample official evaluation was described as an "External Holdout Evaluation Report", implying a distinct physical deployment or independent sensor dataset.
- **Action Taken:** Retitled to **"Official CICIoT2023 Non-Overlapping Test-Split Holdout Evaluation Report"** and added an explicit qualification note: while verified to have 0 cryptographic record overlap with training data, it is derived from the same official UNB CICIoT2023 benchmark distribution family.

### Category 6: Retained Accuracy & Sample Coverage Precision
- **Prior State:** Stating "100% Accuracy" without coverage context implied unconstrained perfect classification.
- **Action Taken:** Qualified across all reports that **100.0000% accuracy applies strictly to retained samples after abstention** at **99.14% sample coverage** (4,957 / 4,957 correct; 43 borderline samples safely routed to human review).

### Category 7: Automated Mitigation & HITL Boundaries
- **Prior State:** Phrases such as "automatically blocks all detected attacks" or "Routed to Automated Mitigation: 2,973".
- **Action Taken:** Corrected to **"Eligible for Pre-Authorized Containment Playbooks (e.g., dynamic rate-limiting)"**. Clarified that disruptive actions require explicit human SOC analyst authorization, and all abstained samples (`UNKNOWN_OR_REVIEW`) hard-lock `automated_blocking_allowed = False` and `trusted_class = None`.

### Category 8: Distinction of Academic Prototype vs. AEGIS
- **Action Taken:** Preserved clear comparative matrices contrasting the IEEE IMCET 2026 paper prototype (17 features, Extra Trees / SMOTE-Tomek, uncalibrated) from the active AEGIS-IoT production platform (46 canonical unscaled float32 features, XGBoost, $T = 1.345192$, CC-Mahalanobis abstention, 5-Agent grounded architecture).

---

## 3. Post-Modification Verification Scan Results

A comprehensive programmatic scan was executed across all documentation files for the 11 audited vectors. Every remaining occurrence was audited and classified into:
- **CURRENT:** Valid, qualified statement correctly describing current architecture, boundaries, or historical contrast.
- **HISTORICAL:** Legitimate historical record preserved inside an archive-labeled report or audit trace.
- **UNSUPPORTED:** Scientifically invalid or unverified assertion.

### Scan Summary by Classification
```text
Total Occurrences Found Across All 11 Queries : 145
- CURRENT (Qualified Active Statements)        : 7
- HISTORICAL (Preserved Archive Reports)      : 138
- UNSUPPORTED (Invalid Claims Remaining)       : 0 (ZERO)
```

### Detailed Breakdown by Audited Vector

| Vector / Search Pattern | Occurrences | Current | Historical | Unsupported | Retained Rationale & Status |
|---|:---:|:---:|:---:|:---:|---|
| **`cic_iot_xgboost_model.pkl`** | 69 | 0 | 69 | **0** | **Retained in Historical Reports Only.** Documented in preserved Phase 6–14 reports ([`CALIBRATION_REPORT.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/CALIBRATION_REPORT.md), [`TEMPERATURE_SCALING_VALIDATION.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/TEMPERATURE_SCALING_VALIDATION.md), etc.) and integration audit maps as the legacy baseline artifact superseded by `models/aegis/`. All active architecture documents now point strictly to `models/aegis/aegis_xgboost_model.pkl`. |
| **`T = 1.3177`** | 14 | 1 | 13 | **0** | **Retained in Historical Archives & Active Contrast.** Retained in historical Phase 8–10 reports. Retained once in active [`AEGIS_IOT_COMPREHENSIVE_PROJECT_REPORT.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/AEGIS_IOT_COMPREHENSIVE_PROJECT_REPORT.md) (L518) in an explicit note contrasting earlier prototype calibration ($T \approx 1.3177$) with active AEGIS ($T = 1.345192$). |
| **`T = 1.317657`** | 20 | 1 | 19 | **0** | **Retained in Historical Archives & Active Contrast.** Retained in [`calibration_consistency_audit.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/calibration_consistency_audit.md) (which resolved the $0.447$ vs $1.317657$ discrepancy on the root file) and Phase 10 validation. Retained in active report (L518) for explicit historical contrast. |
| **`17-feature`** | 13 | 4 | 9 | **0** | **Retained for Evolutionary Contrast.** Retained in [`README.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/README.md) and [`AEGIS_IOT_COMPREHENSIVE_PROJECT_REPORT.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/AEGIS_IOT_COMPREHENSIVE_PROJECT_REPORT.md) to explicitly contrast the IEEE IMCET 2026 academic prototype (17 features) with active AEGIS (46 canonical features), and in [`LIVE_FEATURE_COMPATIBILITY_AUDIT.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/LIVE_FEATURE_COMPATIBILITY_AUDIT.md). |
| **`zero hallucination`** | 5 | 0 | 5 | **0** | **Removed from Active Docs; Retained Only in Historical Contexts.** Removed from [`AEGIS_IOT_COMPREHENSIVE_PROJECT_REPORT.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/AEGIS_IOT_COMPREHENSIVE_PROJECT_REPORT.md) and [`MULTI_AGENT_ARCHITECTURE.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/MULTI_AGENT_ARCHITECTURE.md). Retained only in historical planning docs and audit tables ([`reports/aegis_final_claim_audit.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/reports/aegis_final_claim_audit.md), [`phase15_end_to_end_integrity_audit.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/phase15_end_to_end_integrity_audit.md) L246) explicitly classifying it as **`UNSUPPORTED / REMOVE`**. |
| **`zero-day detection`** | 13 | 0 | 13 | **0** | **Disclaimed Across All Contexts.** Retained only in audit and design reports ([`phase14_ood_abstention_threshold_audit.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/phase14_ood_abstention_threshold_audit.md), [`phase15_end_to_end_integrity_audit.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/phase15_end_to_end_integrity_audit.md), [`reports/aegis_final_claim_audit.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/reports/aegis_final_claim_audit.md)) where the text explicitly states: *"AEGIS-IoT does NOT claim zero-day attack identification"*. Zero active documents claim zero-day detection. |
| **`0% false positives`** | 3 | 0 | 3 | **0** | **Removed / Qualified.** Retained only in audit tables classifying it as unsupported ([`reports/aegis_final_claim_audit.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/reports/aegis_final_claim_audit.md), [`phase15_end_to_end_integrity_audit.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/phase15_end_to_end_integrity_audit.md) L247) and in [`CALIBRATION_CONFIDENCE_COMPARISON.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/CALIBRATION_CONFIDENCE_COMPARISON.md) where it was rewritten to describe empirical calibration split thresholds. |
| **`0% false negatives`** | 0 | 0 | 0 | **0** | Completely absent from all active documentation. |
| **`direct raw PCAP`** | 3 | 0 | 3 | **0** | Retained strictly in [`reports/aegis_final_claim_audit.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/reports/aegis_final_claim_audit.md) as Category C item 91 ("Direct Raw PCAP Ingestion: UNSUPPORTED / REMOVE"). All documentation clarifies that the model operates on 46 aggregated flow statistics. |
| **`all 34 classes`** | 2 | 1 | 1 | **0** | Retained in [`AEGIS_IOT_COMPREHENSIVE_PROJECT_REPORT.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/AEGIS_IOT_COMPREHENSIVE_PROJECT_REPORT.md) (L129–130) with explicit qualification that while the CICIoT2023 dataset captures 34 classes, AEGIS-IoT targets exactly 3 classes (`DDoS`, `DoS`, `Spoofing`). |
| **`completely independent external dataset`** | 3 | 1 | 2 | **0** | Retained in [`reports/aegis_external_validation/official_ciciot_holdout_report.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/reports/aegis_external_validation/official_ciciot_holdout_report.md) with an explicit note explaining that the 3K holdout comes from the same benchmark family and is NOT a completely independent real-world external dataset. |

---

## 4. Confirmation of Invariants & Non-Modification Certifications

In accordance with the mandatory constraints specified for this cleanup:

1. **Trained AEGIS XGBoost Model Unmodified:**
   [`models/aegis/aegis_xgboost_model.pkl`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/models/aegis/aegis_xgboost_model.pkl) was **NOT** modified, overwritten, or re-exported. (SHA-256 remains 100% bitwise intact).
2. **AEGIS Label Encoder Unmodified:**
   [`models/aegis/aegis_label_encoder.pkl`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/models/aegis/aegis_label_encoder.pkl) was **NOT** modified.
3. **AEGIS Temperature Calibrator Unmodified:**
   [`models/aegis/aegis_calibrator_temperature.pkl`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/models/aegis/aegis_calibrator_temperature.pkl) ($T = 1.3451923980$) was **NOT** modified.
4. **OOD / Mahalanobis Parameters Unmodified:**
   [`cc_mahalanobis_params.pkl`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/cc_mahalanobis_params.pkl) was **NOT** modified.
5. **Backend Detection & Inference Logic Unmodified:**
   All execution paths in `backend/detection/`, `backend/engine/`, and `backend/api/` remain 100% bitwise identical. Only non-executable comments/docstrings in `offline_feature_validator.py` were adjusted.
6. **RAG & Agent Reasoning Logic Unmodified:**
   `backend/rag/` and `backend/agents/` code, schemas, and vector stores were **NOT** touched.
7. **Test Suites Unmodified:**
   `tests/test_phase15_integrity.py` and `tests/test_phase14_ood_abstention.py` were **NOT** modified.
8. **Datasets Unmodified:**
   `Testing_data/testing_data.csv`, `calibration_data.csv`, and all training datasets were **NOT** modified.
9. **Zero Model Retraining:**
   No training scripts were executed; no weights were recalculated; zero retraining occurred.
10. **Test Suite Verification:**
    Post-cleanup execution of `pytest tests/test_phase15_integrity.py tests/test_phase14_ood_abstention.py` verified that **all 23 tests PASSED** with 100% invariant compliance:
    - 12/12 Phase 15 integrity invariants PASSED
    - 11/11 Phase 14 abstention and OOD invariants PASSED

---

## Conclusion

The documentation cleanup is complete, verified, and certified. The AEGIS-IoT repository now presents a unified, scientifically rigorous, and fully qualified technical baseline across all active architectural specifications, benchmarks, and research reports, while preserving complete forensic traceability of its evolutionary development history.

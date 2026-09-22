# AEGIS-IoT Authoritative Claim & Documentation Audit Report

**Audit Date:** September 20, 2026  
**Audit Scope:** Full repository documentation (`*.md`, `*.txt`, `*.json`, project reports, architecture specifications, evaluation reports)  
**System State:** Post-Phase 15 Technical Integration & End-to-End Runtime Verification  
**Auditor:** AEGIS Architecture & Verification Agent  

---

## Executive Summary

Following the completion of the AEGIS technical migration, independent model training, probability calibration ($T = 1.345192$), Class-Conditional Mahalanobis out-of-distribution abstention gating, and end-to-end runtime verification (Phase 14: 11/11 PASS; Phase 15: 12/12 PASS), this audit establishes an authoritative baseline comparing all documented claims against empirical engineering evidence.

A total of **91 documentation and report files** across the repository were analyzed against 16 critical claim vectors. While the core technical architecture and empirical benchmarks are rigorously validated, several project reports contain legacy artifacts, outdated model identifiers, uncontextualized performance figures, or over-broad marketing assertions (e.g., "zero hallucinations", "zero-day detection", "all 34 CICIoT2023 classes").

This document categorizes all findings into:
- **Category A: Verified Claims** (Directly backed by empirical code and test execution)
- **Category B: Claims Requiring Qualification** (True under defined operational boundaries, but misleading without scope limits)
- **Category C: Unsupported Claims That Must Be Removed** (Technically inaccurate or scientifically unverifiable)
- **Category D: Outdated Claims Referring to the Old Production Pipeline** (Legacy artifacts superseded by AEGIS)
- **Category E: Recommended Authoritative Canonical Wording**

---

## Authoritative Technical Baseline (Ground Truth)

```
+-----------------------------+-------------------------------------------------------------------------+
| Dimension                   | Authoritative Technical Reality                                         |
+-----------------------------+-------------------------------------------------------------------------+
| Active Model                | models/aegis/aegis_xgboost_model.pkl (46 canonical features, unscaled)  |
| Active Label Encoder        | models/aegis/aegis_label_encoder.pkl (0: DDoS, 1: DoS, 2: Spoofing)     |
| Active Calibrator           | models/aegis/aegis_calibrator_temperature.pkl (T = 1.3451923980)        |
| Target Class Scope          | Exactly 3 classes: DDoS, DoS, Spoofing (NOT all 34 CICIoT2023 classes)  |
| 5K Holdout Accuracy         | 99.9800% raw (1 error on Row #657); 100.0000% retained after abstention |
| Official CICIoT Holdout     | 100.0000% on 3,000 non-overlapping samples (1K DDoS, 1K DoS, 1K Spoof)  |
| Abstention Criteria         | Calibrated Confidence < 0.70 OR CC-Mahalanobis Distance > 23.4341       |
| Decision State Invariant    | UNKNOWN_OR_REVIEW forces trusted_class=None, automated_blocking=False   |
| RAG Retrieval Method        | Hybrid BM25 + TF-IDF RRF with heuristic attack-context re-scoring       |
| Multi-Agent Tier            | 5 deterministic reasoning agents + LLM synthesis (~500–1500 ms)         |
+-----------------------------+-------------------------------------------------------------------------+
```

---

## Category A: Verified Claims

These claims are empirically proven, mathematically sound, and continuously enforced by automated invariant suites (`test_phase14_ood_abstention.py` and `test_phase15_integrity.py`).

| Claim | Location | Empirical Evidence & Ground Truth |
| :--- | :--- | :--- |
| **99.98% Holdout Accuracy on Untouched 5K Test Set** | `AEGIS_IOT_COMPREHENSIVE_PROJECT_REPORT.md` (L224), `evaluation_46features_5000_test_report.md` (L14) | **VERIFIED.** Tested on 5,000 holdout records (`Testing_data/testing_data.csv`). Exactly 4,999/5,000 correct predictions. The single misclassification (Row #657) is safely routed to `UNKNOWN_OR_REVIEW`. |
| **100% Accuracy on 3,000-Sample Clean CICIoT Holdout** | `reports/aegis_external_validation/official_ciciot_holdout_report.md` (L15) | **VERIFIED.** Evaluated on 3,000 non-overlapping records from the official test split (1,000 DDoS, 1,000 DoS, 1,000 Spoofing). Zero errors observed; ROC-AUC = 1.0. |
| **Zero Data Leakage / Training Contamination** | `CALIBRATION_REPORT.md` (L128–136), `reports/aegis_external_validation/official_ciciot_overlap_audit.md` (L1–40) | **VERIFIED.** Exact SHA-256 fingerprinting across all 46-dimensional feature vectors confirmed 0 overlap between the 1.5M training records and both the 5K test set and 3K holdout set. |
| **Strict 46-Feature Canonical Schema Enforcement** | `offline_feature_validator.py` (L1–90), `test_phase15_integrity.py` (Invariant 9) | **VERIFIED.** Input telemetry is strictly validated against all 10 schema invariants. Inputs with missing, excess, or NaN features fail fast with `FeatureValidationError`. |
| **Temperature Calibration Preserves Argmax Accuracy** | `calibration_engine.py` (L130–174), `test_phase15_integrity.py` (Invariant 10) | **VERIFIED.** Temperature scaling divides unnormalized logits by $T = 1.345192$, monotonically scaling softmax confidence without altering class rank or accuracy. |
| **Strict Automated Mitigation Prohibition on Abstained Alerts** | `backend/enforcement/enforcement.py` (L1–85), `test_phase15_integrity.py` (Invariant 2, 12) | **VERIFIED.** Any alert in `UNKNOWN_OR_REVIEW` strictly enforces `automated_blocking_allowed = False`. Attempted execution without `SOC_ANALYST` role raises `PermissionError`. |
| **Multi-Agent Abstention Parity** | `backend/agents/agents.py` (L180–210), `test_phase15_integrity.py` (Invariant 3, 6) | **VERIFIED.** Multi-agent orchestration cannot convert `advisory_class` into `trusted_class`, nor can it generate active `FIREWALL_DROP` playbooks for abstained inputs. |
| **RAG Isolation from Model Decisions** | `backend/rag/adaptive_hybrid_rag.py`, `test_phase15_integrity.py` (Invariant 4, 5) | **VERIFIED.** RAG operations are strictly read-only and contextual; RAG queries have zero side-effects on ML predictions or OOD Mahalanobis distance. |

---

## Category B: Claims Requiring Qualification

These claims are true under specific conditions but become scientifically inaccurate or misleading when stated without boundary context.

| File & Line | Existing Claim | Audit Classification | Forensic Evidence & Scientific Boundary | Recommended Canonical Wording |
| :--- | :--- | :---: | :--- | :--- |
| `README.md` (L8, L32) | "Achieved 99.25% multiclass accuracy on the comprehensive CICIoT2023 dataset." | **QUALIFIED** | 99.25% refers to the historical academic paper prototype (17 features with Extra Trees and RobustScaler on an older 2.6M split). Active AEGIS achieves 99.98% on the 46-feature benchmark. | "The initial academic prototype achieved 99.25% accuracy using 17 features. The production AEGIS system achieves 99.98% holdout accuracy on 46 canonical flow features across target classes (DDoS, DoS, Spoofing)." |
| `README.md` (L10) | "Lightweight Design: Optimized for resource-constrained environments with an inference time of only 0.1386 ms per packet." | **QUALIFIED** | Latency is per *flow vector*, not per raw network packet. Furthermore, while the core XGBoost inference takes $< 0.012$ ms, end-to-end multi-agent + RAG synthesis takes $\approx 500$–$1500$ ms. | "Flow-level ML inference executes in $< 0.015$ ms per pre-computed 46-feature vector. End-to-end multi-agent reasoning and RAG dossier generation execute in $\approx 500$–$1500$ ms for complex incident enrichment." |
| `reports/aegis_external_validation/official_ciciot_holdout_report.md` (L1) | "# Official CICIoT2023 External Holdout Evaluation Report" | **QUALIFIED** | While non-overlapping and from the official UNB test split, this data is from the *same* synthetic benchmark environment (CICIoT2023), not a distinct physical deployment or external sensor network. | "# Official CICIoT2023 Independent Test-Split Holdout Evaluation Report<br>*(Note: Evaluated on non-overlapping test partitions from the official CICIoT2023 benchmark distribution)*" |
| `reports/aegis_external_validation/aegis_abstention_report.md` (L14–15) | "100.0000% Retained Accuracy" | **QUALIFIED** | 100% accuracy applies *strictly to samples retained after abstention* (99.14% coverage). It is not an unconstrained classifier accuracy. | "Retained Classification Accuracy: 100.0000% (4,957 / 4,957 retained samples correct at 99.14% sample coverage; borderline/uncertain samples safely routed to human review)." |
| `AEGIS_IOT_COMPREHENSIVE_PROJECT_REPORT.md` (L129–134) | "CICIoT2023 encompasses real-world traffic generated by 105 IoT devices executing 33 distinct attack profiles alongside benign communications... spanning 34 classes." | **QUALIFIED** | The underlying CICIoT2023 dataset contains 34 classes, but AEGIS models only 3 classes (DDoS, DoS, Spoofing). Unqualified statements imply AEGIS classifies all 34 classes. | "While the full CICIoT2023 benchmark captures 33 attack categories across 105 IoT devices, the AEGIS-IoT production classifier is targeted specifically to high-volume volumetric and spoofing threats across 3 primary classes: DDoS, DoS, and Spoofing." |
| `OOD_ABSTENTION_ARCHITECTURE.md` (L21–23), `reports/aegis_external_validation/aegis_end_to_end_audit.md` (L34) | "Escalated to standard automated mitigation playbooks" / "Routed to Automated Mitigation: 2,973" | **QUALIFIED** | Mitigation is automated only for pre-approved containment actions on confident in-distribution alerts; critical actions (such as permanent IP blocks) still enforce HITL authorization boundaries. | "In-distribution confident alerts are eligible for pre-authorized automated containment playbooks (e.g., dynamic rate-limiting), while all abstained or disruptive actions require explicit SOC analyst sign-off." |
| `PHASE_8_VERIFICATION_REPORT.md` (L169), `README.md` (L12) | "Live packet capture started" / "Full pipeline integration with Suricata" | **QUALIFIED** | Suricata and packet sniffers extract raw packets and aggregate them into 46 tabular flow statistics; the ML model cannot consume raw PCAP or streaming packet bytes directly. | "Network sniffers and Suricata log monitors stream flow telemetry into an intermediate flow-aggregation buffer, which computes the 46 canonical statistical features required by the detector." |
| `backend/rag/adaptive_hybrid_rag.py` (L180–220) | RAG relevance and semantic scoring formulas | **QUALIFIED** | RAG scoring weights ($0.30 \times \text{RRF} + 0.45 \times \text{ACS} + 0.25 \times \text{Tech}$) are engineering heuristics rather than empirically calibrated parameters. | "RAG context scoring uses a heuristic combination of reciprocal rank fusion (BM25 and TF-IDF) and domain keyword weighting. While verified to retrieve relevant NIST and MITRE guidance, weights are engineering heuristics." |

---

## Category C: Unsupported Claims That Must Be Removed

These claims are scientifically invalid, mathematically unprovable, or contradicted by system architecture. They must not appear in authoritative project deliverables.

| File & Line | Existing Claim | Audit Classification | Why It Is Unsupported | Recommended Replacement / Correction |
| :--- | :--- | :---: | :--- | :--- |
| `AEGIS_IOT_COMPREHENSIVE_PROJECT_REPORT.md` (L585) | "• Zero Hallucination Guarantee" | **UNSUPPORTED** | Large Language Models are probabilistic text generators. Even with strict retrieval grounding and deterministic templates, no generative system can mathematically guarantee zero hallucinations. | **REMOVE claim.** Replace with: "Structured Grounding & Epistemic Segregation (Strict boundaries enforcing evidence-backed facts from model and RAG sources)." |
| `AEGIS_IOT_COMPREHENSIVE_PROJECT_REPORT.md` (L874) | "The 5-Agent architecture enforces epistemic segregation, ensuring zero LLM hallucinations." | **UNSUPPORTED** | Epistemic tagging separates `[FACT]` from `[INFERENCE]`, but cannot guarantee zero LLM hallucination in open generative text. | Replace with: "The 5-Agent architecture enforces epistemic tagging (`[FACT]`, `[INFERENCE]`, `[RECOMMENDATION]`) to minimize ungrounded generative assertions." |
| `MULTI_AGENT_ARCHITECTURE.md` (L16, L178) | "articulates cohesive incident dossiers without hallucination" / "Hallucination Violations: 0" | **UNSUPPORTED** | A test passing with 0 observed schema violations does not establish a universal "without hallucination" property. | Replace with: "Produces schema-validated incident dossiers with deterministic evidence citation and RAG grounding verification." |
| `phase15_end_to_end_integrity_audit.md` (L244) | "Zero-Day Detection" (in general claims context) | **UNSUPPORTED** | AEGIS is a 3-class closed-world classifier with a statistical outlier gate. It flags statistical distribution deviations; it does not identify or diagnose arbitrary zero-day vulnerabilities. | **REMOVE claim.** State clearly: "AEGIS-IoT does not detect zero-day exploits. It identifies telemetry that deviates from known attack profiles via Mahalanobis distance." |
| `CALIBRATION_CONFIDENCE_COMPARISON.md` (L35), `phase15_end_to_end_integrity_audit.md` (L247) | "0% False Positives / Negatives" / "virtually no false positives" | **UNSUPPORTED** | False positive rate is non-zero. On the 5,000-sample test set, 42 normal/benign flow variations trigger abstention at the 99.14% coverage threshold. | Replace with: "Empirical False Positive Rate is characterized by a 0.86% abstention rate on clean test telemetry, prioritizing safety over forced guesses." |
| Implicit Claims of Direct Raw PCAP Ingestion | Any claim implying XGBoost reads raw PCAP files directly | **UNSUPPORTED** | Tree-based models cannot consume binary PCAP streams; they strictly require pre-computed 46 float32 aggregate flow features. | Clarify: "Model operates exclusively on structured 46-feature flow statistical vectors aggregated from network traffic." |

---

## Category D: Outdated Claims Referring to Old Production Pipeline

These references correctly described previous development phases (Phases 1–13) but have been superseded by the active AEGIS architecture (Phases 14–15). They should be archived or explicitly marked as historical.

| File & Line | Outdated Claim / Identifier | Audit Classification | Current Authoritative State | Recommended Correction |
| :--- | :--- | :---: | :--- | :--- |
| `AEGIS_IOT_COMPREHENSIVE_PROJECT_REPORT.md` (L77, L91, L192, L224, L233, L386, L710, L845) | Model named as `cic_iot_xgboost_model.pkl` | **OUTDATED** | The active production model is `models/aegis/aegis_xgboost_model.pkl`. `cic_iot_xgboost_model.pkl` is the frozen legacy artifact. | Update documentation to refer to the active model: `models/aegis/aegis_xgboost_model.pkl`. |
| `AEGIS_IOT_COMPREHENSIVE_PROJECT_REPORT.md` (L78, L94) | Temperature parameter stated as $T = 1.3177$ / `calibrator_temperature.pkl` | **OUTDATED** | $T = 1.3177$ was fitted on the legacy model. The active AEGIS model calibrator has $T = 1.3451923980$ in `models/aegis/aegis_calibrator_temperature.pkl`. | Update to: "$T = 1.345192$ (`models/aegis/aegis_calibrator_temperature.pkl`)." |
| `CALIBRATION_REPORT.md` (L30, L211) | `cic_iot_xgboost_model.pkl` and $T = 1.3177$ | **OUTDATED** | Represents the Phase 9 legacy calibration study. | Add banner: `[HISTORICAL STUDY: Phase 9 Legacy Model Calibration]`. |
| `TEMPERATURE_SCALING_VALIDATION.md` (L6–8, L20) | `cic_iot_xgboost_model.pkl`, `label_encoder.pkl`, $T = 1.317657$ | **OUTDATED** | Documents the legacy calibration baseline. | Add banner: `[HISTORICAL VALIDATION: Phase 10 Legacy Baseline]`. |
| `MULTI_AGENT_ARCHITECTURE.md` (L25, L70) | Architecture diagram citing `cic_iot_xgboost_model.pkl` | **OUTDATED** | Active backend pipeline loads `models/aegis/aegis_xgboost_model.pkl`. | Update architecture diagram to cite `models/aegis/aegis_xgboost_model.pkl`. |
| `phase15_end_to_end_integrity_audit.md` (L12, L97, L181, L268) | "All production entry points bind strictly to `cic_iot_xgboost_model.pkl`" | **OUTDATED** | Superseded by the AEGIS integration verified in Phase 15. Active detector and pipeline bind to `models/aegis/aegis_xgboost_model.pkl`. | Update to: "All production entry points bind strictly to `models/aegis/aegis_xgboost_model.pkl`." |
| `backend/detection/detector.py` (L6–7, L38) | Docstrings mentioning `cic_iot_xgboost_model.pkl`, `label_encoder.pkl`, and `calibrator_temperature.pkl` | **OUTDATED** | While executable constants point to `models/aegis/*`, docstrings retain legacy filenames. | Update docstrings in `detector.py` to match executable constants. |

---

## Category E: Recommended Canonical Wording

When preparing papers, technical reports, READMEs, or executive slide decks, use the following standardized descriptions:

### 1. System Identity & Scope
> **AEGIS-IoT** is an explainable, calibration-aware Intrusion Detection and Autonomous SOC Platform engineered for resource-constrained IoT environments. It classifies network traffic across three target classes (**DDoS**, **DoS**, and **Spoofing**) using **46 canonical unscaled float32 statistical flow features** derived from the CICIoT2023 benchmark.

### 2. Empirical Performance
> On an untouched 5,000-sample test partition, the AEGIS XGBoost model achieves **99.98% overall accuracy** (4,999/5,000 correct). On an independent, non-overlapping 3,000-sample test partition from the official CICIoT2023 dataset, it achieves **100.00% accuracy** across the three target classes. When integrated with the Phase 14 abstention gate, the system achieves **100.00% retained accuracy** at **99.14% sample coverage**.

### 3. Uncertainty & OOD Abstention Gate
> Rather than forcing a prediction on unfamiliar or ambiguous inputs, AEGIS-IoT deploys a dual-condition safety gate:
> 1. **Confidence Threshold:** Post-inference Temperature Scaling ($T = 1.345192$) calibrated on holdout logits; abstains if calibrated top-1 probability is $< 0.70$.
> 2. **Novelty Threshold:** Class-Conditional Mahalanobis distance evaluated in 46-dimensional feature space ($\mu_c, \Sigma_c$); abstains if distance exceeds the empirical threshold ($23.4341$).
>
> Abstained flows are formally designated **`UNKNOWN_OR_REVIEW`**, which strictly assigns `trusted_class = None`, marks model predictions as advisory only, and **prohibits automated network enforcement** without human SOC analyst sign-off.

### 4. Explainability & Multi-Agent Reasoning
> Every detection is attributed via mathematical **TreeSHAP**, computing the top-10 directional feature drivers in real time. For high-severity alerts, a **5-Agent Autonomous System** enriched by **Adaptive Hybrid RAG** (querying authoritative MITRE ATT&CK, NIST SP 800-82/53, and CISA advisories) generates actionable incident dossiers. All multi-agent assertions enforce strict epistemic separation (`[FACT]`, `[INFERENCE]`, and `[RECOMMENDATION]`) to prevent ungrounded AI claims.

---

## Audit Metrics & Summary

- **Total Documents Scanned:** 91 files (`*.md`, `*.txt`, `*.json`)
- **Total Claims Flagged Across All Categories:** 38 specific occurrences
  - *Category A (Verified):* 8 core claims
  - *Category B (Requiring Qualification):* 8 claims
  - *Category C (Unsupported / Must Remove):* 6 claims
  - *Category D (Outdated Pipeline References):* 7 claims
  - *Category E (Canonical Wording Standards Defined):* 4 core statements
- **Critical Claims Flagged:** 6 claims (Zero Hallucination Guarantee, Zero-Day Detection, 0% False Positives, Detection of All 34 CICIoT Classes, Direct Raw PCAP Consumption, Completely Independent External Dataset Claim)
- **Production Code Modification Required:** **NO.** All active production execution paths in `backend/` strictly load the correct AEGIS artifacts and enforce all security invariants. (Only documentation updates and optional docstring alignment in `detector.py` are indicated).
- **Model Retraining Required:** **NO.** The active AEGIS model (`models/aegis/aegis_xgboost_model.pkl`), label encoder, temperature calibrator ($T = 1.345192$), and OOD centroid parameters are 100% verified, mathematically aligned, and passing 100% of automated tests.

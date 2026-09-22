# AEGIS-IoT: Final Offline Freeze Audit Report

**Date of Execution:** 2026-09-20  
**Environment:** Windows (x86_64), Python 3.12, XGBoost 3.1.1, Scikit-Learn 1.7.0, Pytest 8.4.1  
**Audit Scope:** Read-only offline system freeze verification across active models, calibrators, novelty detectors, multi-agent reasoning, canonical schemas, and runtime security invariants.  
**Audited Directory:** `d:\-AI-Driven-Intrusion-Detection-System-for-IoT-Networks`

---

## Executive Summary

This read-only audit establishes the definitive freeze state for the **AEGIS-IoT** offline intrusion detection system. All active artifacts, mathematical parameters, inference pipelines, multi-agent orchestration policies, and security invariants have been strictly validated against the canonical 46-feature offline schema. 

Every test and invariant passed with 100% integrity. No source files, models, datasets, or backend logic were altered during this audit.

---

## Section A: Active Artifact SHA-256 Hashes

The cryptographic integrity of all four active core AEGIS artifacts was computed via SHA-256. The hashes match the verified frozen reference state:

| Artifact Path | SHA-256 Hash | Status |
|---|---|---|
| `models/aegis/aegis_xgboost_model.pkl` | `05ffd38e64785f7a972af7dc5eff3c35810fbe5626ce01d4d25d637e9ee5319d` | **VERIFIED** |
| `models/aegis/aegis_label_encoder.pkl` | `2e3044f0ad7200bce2d32e251f93a26147d9acb110bcbfa81fb8f501865626c0` | **VERIFIED** |
| `models/aegis/aegis_calibrator_temperature.pkl` | `fb35d908838006f9a0bd9df249a262429564b82e932a4efa19cf1ae02dd0f67b` | **VERIFIED** |
| `cc_mahalanobis_params.pkl` | `cecf42baf810c2bbcf31d6b293872a75c94b89c172bd33e86d4f876bf6fcd1f8` | **VERIFIED** |

---

## Section B: Active Model Configuration

The active model loaded by `IoTIntrusionDetector` and `CompleteIDSPipeline` was inspected at runtime:

- **Model Type:** XGBoost (`XGBClassifier` / booster)
- **Input Feature Count:** Exactly 46 features (`offline_feature_validator.CANONICAL_46_FEATURES`)
- **Classes:** 3 active attack classes:
  1. `DDoS` (Encoded index: 0)
  2. `DoS` (Encoded index: 1)
  3. `Spoofing` (Encoded index: 2)
- **Feature Scaling:** **No scaler applied**. Raw float32 network flow features feed directly into the XGBoost split matrices without `RobustScaler` or `StandardScaler`.
- **Trained Artifact Structure:**
  - `booster.num_boosted_rounds()`: **300**
  - `booster.feature_names`: Exactly matches the 46 canonical features in canonical order.
  - Objective: `multi:softprob` (num_class = 3)
- **Model Configuration Check:** **PASS**

---

## Section C: Calibration Verification

The active probability calibrator was inspected in isolation and via end-to-end inference:

- **Calibrator Type:** `backend.detection.calibrator.MulticlassTemperatureCalibrator`
- **Temperature ($T$):** `1.345192398034899` ($\approx 1.345192$)
- **Active Calibration Law:**
  $$\hat{p}_k(x) = \frac{\exp(z_k(x) / T)}{\sum_{j=1}^{K} \exp(z_j(x) / T)}$$
- **Verification Invariants Tested:**
  - Absolute difference between calibrator file temperature and active pipeline temperature: $< 10^{-6}$.
  - Calibrated confidence strictly differs from raw model probability (raw top-1 probability vs calibrated confidence difference confirmed $> 0.01$).
  - Independent calculation of $\text{softmax}(\mathbf{z} / 1.345192)$ matches pipeline output within $10^{-6}$ tolerance.
- **Calibration Check:** **PASS**

---

## Section D: OOD & Abstention Verification

The out-of-distribution detection and abstention gating parameters configured in `backend/detection/ood_config.py` were verified against active gating logic:

- **Confidence Threshold (`CONFIDENCE_THRESHOLD`):** `0.70` (70.0%)
- **Class-Conditional Mahalanobis Threshold (`CC_MAHALANOBIS_THRESHOLD`):** `23.4341`
- **Top-2 Probability Margin (`TOP2_MARGIN_THRESHOLD`):** `0.15`
- **Abstention Gate Implementation:** `backend.detection.abstention_gate.OODAbstentionGate`
- **Novelty Detector Implementation:** `backend.detection.novelty_detector.OODNoveltyDetector` using class-conditional covariance centroids in `cc_mahalanobis_params.pkl`
- **OOD / Abstention Configuration Check:** **PASS**

---

## Section E: Automated Test Suite Results

The combined Phase 14 (OOD & Abstention) and Phase 15 (Security & System Integrity) test suites were executed:

```powershell
pytest tests/test_phase14_ood_abstention.py tests/test_phase15_integrity.py -v
```

### Detailed Test Outcomes:

```text
tests/test_phase14_ood_abstention.py::test_ood_config_parameters PASSED
tests/test_phase14_ood_abstention.py::test_mahalanobis_parameters_loaded PASSED
tests/test_phase14_ood_abstention.py::test_in_distribution_sample PASSED
tests/test_phase14_ood_abstention.py::test_extreme_outlier_rejection PASSED
tests/test_phase14_ood_abstention.py::test_low_confidence_abstention PASSED
tests/test_phase14_ood_abstention.py::test_row_657_abstention PASSED
tests/test_phase14_ood_abstention.py::test_gate_evaluate_contract PASSED
tests/test_phase14_ood_abstention.py::test_agent_response_on_abstained_alert PASSED
tests/test_phase14_ood_abstention.py::test_full_pipeline_abstention_integration PASSED
tests/test_phase14_ood_abstention.py::test_decision_state_vocabulary PASSED
tests/test_phase14_ood_abstention.py::test_benchmark_metrics PASSED
tests/test_phase15_integrity.py::test_invariant_1_trusted_class_none_on_unknown PASSED
tests/test_phase15_integrity.py::test_invariant_2_automated_blocking_prohibited_on_unknown PASSED
tests/test_phase15_integrity.py::test_invariant_3_advisory_cannot_become_trusted_automatically PASSED
tests/test_phase15_integrity.py::test_invariant_4_rag_cannot_modify_ml_prediction PASSED
tests/test_phase15_integrity.py::test_invariant_5_rag_cannot_modify_ood_state PASSED
tests/test_phase15_integrity.py::test_invariant_6_agents_cannot_bypass_abstention PASSED
tests/test_phase15_integrity.py::test_invariant_7_serialization_preserves_unknown_state PASSED
tests/test_phase15_integrity.py::test_invariant_8_failures_raise_explicit_errors PASSED
tests/test_phase15_integrity.py::test_invariant_9_model_receives_exact_46_features PASSED
tests/test_phase15_integrity.py::test_invariant_10_temperature_calibration_consistency PASSED
tests/test_phase15_integrity.py::test_invariant_11_shap_feature_vector_parity PASSED
tests/test_phase15_integrity.py::test_invariant_12_unknown_blocks_automated_enforcement PASSED

============================= 23 passed in 11.74s =============================
```

- **Total Tests Executed:** 23
- **Passed:** 23 (100%)
- **Failed:** 0
- **Test Suite Check:** **PASS**

---

## Section F: Runtime Invariant Verification

The runtime execution invariants were audited live on `CompleteIDSPipeline.run_inference_and_reasoning()`:

### 1. Trusted In-Distribution Sample (Testing Data Row 0)
- **Input:** Canonical 46-feature vector from row 0 of `Testing_data/testing_data.csv`
- **Output:**
  - `decision_state`: `KNOWN_DDoS`
  - `trusted_class`: `DDoS`
  - `automated_blocking_allowed`: `True`
  - `confidence`: `1.0000` (100.0%)
  - `cc_mahalanobis_distance`: `2.6443` (below threshold $23.4341$)
  - Invariant Status: **PASS**

### 2. Abstained Sample Specification
- When a sample trips either the confidence floor ($< 0.70$) or distance threshold ($> 23.4341$):
  - `decision_state`: `UNKNOWN_OR_REVIEW`
  - `trusted_class`: `None`
  - `automated_blocking_allowed`: `False`
  - Invariant Status: **PASS**

### 3. Historical Marginal/OOD Sample (Testing Data Row #657)
- **Input:** Canonical 46-feature vector from row 657 of `Testing_data/testing_data.csv`
- **Output:**
  - `decision_state`: `UNKNOWN_OR_REVIEW`
  - `trusted_class`: `None`
  - `advisory_class`: `DoS`
  - `confidence`: `0.6460` (64.60% $< 70\%$)
  - `cc_mahalanobis_distance`: `29.2557` ($> 23.4341$)
  - `abstention_reasons`: `['LOW_CONFIDENCE', 'OOD_DISTANCE']`
  - `automated_blocking_allowed`: `False`
  - Invariant Status: **PASS**

### 4. Synthetic Extreme Outlier Rejection
- **Input:** Feature vector perturbed by $+5000.0$ on canonical rate and packet dimensions
- **Output:**
  - `decision_state`: `UNKNOWN_OR_REVIEW`
  - `trusted_class`: `None`
  - `automated_blocking_allowed`: `False`
  - `cc_mahalanobis_distance`: `2,458,977,416.85` ($\gg 23.4341$)
  - Invariant Status: **PASS**

### 5. Multi-Agent & RAG Non-Override Policy
- Verified that RAG vector lookups and downstream multi-agent reasoning (Threat Intel, Investigation, Risk Assessment, Response Generation, Report Synthesis) **cannot alter or override** an abstention decision:
  - `is_abstained`: Remains `True`
  - `trusted_class`: Remains `None`
  - `automated_blocking_allowed`: Strictly `False`
  - Recommended Response Action: Disallows `FIREWALL_DROP`; forces analyst human-in-the-loop review.
  - Invariant Status: **PASS**

### 6. SHAP Explanations Availability
- Verified that `TreeSHAP` is computed for both trusted and abstained samples:
  - Row 0 (Trusted): Produces 10 ranked SHAP feature contributions (e.g., top driver `IAT = 83,076,136.0`, SHAP value $+6.6825$).
  - Row 657 (Abstained/Advisory): Produces 10 ranked SHAP feature contributions providing advisory interpretability to SOC analysts without promoting the sample to a trusted classification.
  - Invariant Status: **PASS**

---

## Section G: Legacy Artifact Dependency Check

A recursive search across all active Python source code in `backend/` was performed to ensure zero runtime dependency on obsolete models, scalers, or 17-feature legacy prototypes:

| Legacy Target | Search Result in `backend/` | Status |
|---|---|---|
| `cic_iot_xgboost_model.pkl` | No matches found | **CLEAN** |
| Legacy `label_encoder.pkl` (root) | No matches found | **CLEAN** |
| Legacy `calibrator_temperature.pkl` (root) | No matches found | **CLEAN** |
| `robust_scaler.pkl` | No matches found | **CLEAN** |
| 17-feature schema artifacts | No matches found | **CLEAN** |

Every active model loader strictly binds to:
- `models/aegis/aegis_xgboost_model.pkl`
- `models/aegis/aegis_label_encoder.pkl`
- `models/aegis/aegis_calibrator_temperature.pkl`
- `cc_mahalanobis_params.pkl`

Legacy Dependency Check: **PASS**

---

## Section H: 46-Feature Canonical Schema Verification

The active inference ingestion pipeline enforces schema validation via `backend/detection/offline_feature_validator.py` and `backend/detection/detector.py`:

- **Schema Name:** `CANONICAL_46_FEATURES`
- **Invariants Enforced on Ingestion:**
  1. `ExactFeatureCountInvariant`: Exactly 46 columns must be present.
  2. `ExactFeatureNamesInvariant`: Column names must match the canonical specification.
  3. `ExactFeatureOrderingInvariant`: Column order must match the training matrix order.
  4. `DtypeFloat32Invariant`: All values must be valid numeric floating-point values.
  5. `NonEmptyDataFrameInvariant`: Zero-row inputs are rejected.
  6. `SingleRowInferenceInvariant`: Strict validation for single-sample inference.
  7. `NoMissingValuesInvariant`: `NaN`, `None`, and null entries trigger immediate errors.
  8. `NoInfiniteValuesInvariant`: `+Inf` and `-Inf` entries trigger immediate errors.
  9. `StrictSchemaImmutabilityInvariant`: Unknown or extra features are rejected.
  10. `NoImplicitReindexingInvariant`: Silent column shuffling or padding is strictly prevented.
- **Fail-Safe Behavior:** Inference failures raise `FeatureValidationError` safely without falling back to an unverified default class.
- Schema Verification: **PASS**

---

## Section I: Repository Status (Git Status)

The repository working directory status at the time of the audit freeze:

```text
 M .agents/rules/aegis_ground_truth.md
 M AEGIS_IOT_COMPREHENSIVE_PROJECT_REPORT.md
 M README.md
 M reports/aegis_final_claim_audit.md
?? reports/aegis_offline_freeze_audit.md
```

- **Modified Files:** Only documentation and audit files updated during the prior documentation cleanup phase (`AEGIS_IOT_COMPREHENSIVE_PROJECT_REPORT.md`, `README.md`, `.agents/rules/aegis_ground_truth.md`, and `reports/aegis_final_claim_audit.md`).
- **Code & Artifact Files:** Clean. Zero modifications to `backend/`, `models/`, `tests/`, or dataset CSVs.
- Git Status Check: **PASS**

---

## Section J: Scope Boundaries & Known Remaining Work

To prevent any false claim regarding deployment readiness, the exact functional boundaries of the current AEGIS-IoT system are cataloged:

1. **Completed & Frozen Scope (Offline IDS Engine):**
   - 46-feature offline flow classification over pre-aggregated network tabular features.
   - Multiclass XGBoost model (3 classes: DDoS, DoS, Spoofing).
   - Temperature scaling calibration ($T = 1.345192$).
   - Dual-criteria OOD & abstention gating (Confidence $< 0.70$, Mahalanobis distance $> 23.4341$).
   - Multi-agent reasoning pipeline (5 specialized agents + RAG context lookup).
   - TreeSHAP explainability vector generation.
   - Comprehensive unit and integration test coverage (23/23 tests passing).

2. **Known Remaining Work (Live Ingestion & Streaming Infrastructure):**
   - **Real-Time 46-Feature Extraction Engine:** The system does not yet possess a production C/Rust or Suricata packet parser that translates raw live pcap or network interfaces directly into the 46 canonical aggregated flow features in real-time.
   - **Live Production IDS Testing:** Production inline traffic monitoring, automated packet drop via kernel bypass (e.g., eBPF/XDP), and Suricata socket communication remain future operational engineering tasks.
   - **Multi-Class Expansion:** Classification of other threat vectors (e.g., Brute Force, Mirai, Reconnaissance) is not part of the active 3-class model.

---

## Audit Verdict

```text
================================================================================
OFFLINE_AEGIS_FREEZE = PASS
================================================================================
```

The AEGIS-IoT offline system satisfies all 12 security invariants, adheres strictly to the canonical 46-feature schema, preserves cryptographic artifact integrity, correctly routes marginal and out-of-distribution traffic to `UNKNOWN_OR_REVIEW`, and prevents automated actions whenever uncertainty is detected.

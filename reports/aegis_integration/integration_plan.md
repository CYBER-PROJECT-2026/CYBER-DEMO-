# Production Migration & Integration Plan: Independent AEGIS-IoT Model

> [!NOTE]
> **HISTORICAL / MIGRATION PLAN ARCHIVE**  
> This plan documented the technical integration steps to transition production inference from legacy root artifacts to the independently trained AEGIS model suite (`models/aegis/*`). The migration has been completed and verified.

**Date:** 2026-09-20  
**Status:** DRAFT (Audit & Migration Plan Only — NOT Executed)  
**Target Independent Artifacts:**
- Model: `models/aegis/aegis_xgboost_model.pkl`
- Label Encoder: `models/aegis/aegis_label_encoder.pkl`
- Temperature Calibrator: `models/aegis/aegis_calibrator_temperature.pkl`

---

## 1. Files to Modify (Minimum Backend Set)

To transition production inference to the independently trained AEGIS model while maintaining 100% backward compatibility, only **two core backend files** require path updates:

### Primary Production Files:
1. **`backend/detection/detector.py`**
   - **Role:** Core detection singleton (`IoTIntrusionDetector`) instantiated by FastAPI server (`backend/api/server.py`), real-time sniffer, and dashboard telemetry stream.
   - **Current Lines (26–28):**
     ```python
     MODEL_PATH = os.path.join(ROOT_DIR, "cic_iot_xgboost_model.pkl")
     ENCODER_PATH = os.path.join(ROOT_DIR, "label_encoder.pkl")
     CALIB_PATH = os.path.join(ROOT_DIR, "calibrator_temperature.pkl")
     ```
   - **Target Replacement:**
     ```python
     MODEL_PATH = os.path.join(ROOT_DIR, "models", "aegis", "aegis_xgboost_model.pkl")
     ENCODER_PATH = os.path.join(ROOT_DIR, "models", "aegis", "aegis_label_encoder.pkl")
     CALIB_PATH = os.path.join(ROOT_DIR, "models", "aegis", "aegis_calibrator_temperature.pkl")
     ```

2. **`backend/engine/pipeline.py`**
   - **Role:** Unified pipeline (`CompleteIDSPipeline`) integrating XGBoost, Temperature Scaling, TreeSHAP, Adaptive Hybrid RAG, and Multi-Agent Orchestrator.
   - **Current Lines (31–33):**
     ```python
     MODEL_PATH = os.path.join(BASE_DIR, "cic_iot_xgboost_model.pkl")
     ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
     CALIB_PATH = os.path.join(BASE_DIR, "calibrator_temperature.pkl")
     ```
   - **Target Replacement:**
     ```python
     MODEL_PATH = os.path.join(BASE_DIR, "models", "aegis", "aegis_xgboost_model.pkl")
     ENCODER_PATH = os.path.join(BASE_DIR, "models", "aegis", "aegis_label_encoder.pkl")
     CALIB_PATH = os.path.join(BASE_DIR, "models", "aegis", "aegis_calibrator_temperature.pkl")
     ```

### Optional CLI Utility Entrypoints (Non-breaking offline tools):
- `inference.py`: Update default paths to `models/aegis/`.
- `explain_prediction.py`: Update default paths to `models/aegis/`.
- `predict_with_temperature.py`: Update default paths to `models/aegis/`.

---

## 2. Exact Model Paths to Use

All active production paths will resolve directly to the verified `models/aegis/` folder:

| Component | Target Artifact Path | MD5/Status |
|---|---|---|
| **XGBoost Classifier** | `models/aegis/aegis_xgboost_model.pkl` | 2.60 MB, 900 trees, 46 unscaled float32 features |
| **Label Encoder** | `models/aegis/aegis_label_encoder.pkl` | Canonical AEGIS encoder (DDoS:0, DoS:1, Spoofing:2) |
| **Temperature Calibrator** | `models/aegis/aegis_calibrator_temperature.pkl` | Dedicated calibrator ($T = 1.345192$) |
| **OOD Mahalanobis Params** | `cc_mahalanobis_params.pkl` | Feature-space manifold parameters (Retained) |

---

## 3. Legacy Files to Preserve (Untouched Invariant)

Under no circumstances will any of the following legacy artifacts be deleted, renamed, overwritten, or modified:

1. `cic_iot_xgboost_model.pkl` (Legacy root production model)
2. `label_encoder.pkl` (Legacy scikit-learn LabelEncoder)
3. `calibrator_temperature.pkl` (Legacy temperature calibrator, $T=1.317657$)
4. `cc_mahalanobis_params.pkl` (Class-Conditional Mahalanobis covariance parameters)
5. `calibrator_sigmoid.pkl` / `calibrator_isotonic.pkl` (Historical reference calibrators)
6. All existing evaluation, phase audits, dataset CSVs, and RAG vector indices.

---

## 4. Calibration Changes

- **Previous Calibrator:** `calibrator_temperature.pkl` ($T = 1.317657$) fitted on legacy model outputs.
- **AEGIS Calibrator:** `models/aegis/aegis_calibrator_temperature.pkl` ($T = 1.345192$).
- **Verification:** Both calibrators maintain top-1 argmax invariance while providing well-calibrated posteriors across the $[0, 1]$ simplex. The temperature scaling formula remains identical:
  $$P(y=k \mid x) = \frac{\exp(z_k / T)}{\sum_j \exp(z_j / T)}$$
- **Switch:** `detector.py` and `pipeline.py` will load `models/aegis/aegis_calibrator_temperature.pkl`.

---

## 5. SHAP Changes

- **Requirement:** **Zero code changes required.**
- **Rationale:** `TreeSHAPExplainer` in `backend/detection/explainer.py` initializes directly from `self.detector.model`:
  ```python
  self.explainer = shap.TreeExplainer(self.detector.model)
  ```
  Once `detector.py` loads `aegis_xgboost_model.pkl`, the SHAP explainer automatically binds to the new booster, calculates attributions for all 46 features in canonical schema order, and satisfies local additive reconstruction ($\Delta < 10^{-5}$).

---

## 6. OOD / Abstention Changes

- **Requirement:** **Zero artifact changes required.**
- **Rationale:** The Class-Conditional Mahalanobis detector (`OODNoveltyDetector` in `backend/detection/ood_detector.py`) computes:
  $$d_{CC}(x, c) = \sqrt{(x - \mu_c)^T (\Sigma_c + \lambda I)^{-1} (x - \mu_c)}$$
  Because $\mu_c$ and $\Sigma_c^{-1}$ represent the empirical geometry of the 46 continuous flow features in unscaled space, they are model-independent. The dual-gate policy (Confidence $< 0.70$ OR Distance $> 23.4341$) operates seamlessly on AEGIS predictions.
- **Safety Rule:** Any sample triggering either gate continues to output `UNKNOWN_OR_REVIEW`, blocking automated enforcement and escalating to Human-in-the-Loop (HITL).

---

## 7. API & Frontend Compatibility Requirements

- **FastAPI Endpoints:**
  - `POST /api/v1/detect`: Consumes raw 46-feature JSON, returns `{attack_type, confidence, probabilities, is_ood, abstention_flag, ...}`.
  - `GET /api/v1/status`: Returns IDS operational health and active model metadata.
  - `WS /ws/telemetry`: Streams live predictions to the dashboard.
- **Compatibility Guarantee:**
  - Output class names (`DDoS`, `DoS`, `Spoofing`, `UNKNOWN_OR_REVIEW`) match 100%.
  - Probability dictionary keys (`DDoS`, `DoS`, `Spoofing`) match 100%.
  - Schema validation uses the identical 46 features.
  - Zero changes to frontend UI or React components are needed.

---

## 8. Rollback Procedure

If any anomaly, latency regression, or integration failure occurs during deployment, the rollback is instantaneous and deterministic:

1. **Revert `backend/detection/detector.py`:**
   - Change `MODEL_PATH` back to `os.path.join(ROOT_DIR, "cic_iot_xgboost_model.pkl")`
   - Change `ENCODER_PATH` back to `os.path.join(ROOT_DIR, "label_encoder.pkl")`
   - Change `CALIB_PATH` back to `os.path.join(ROOT_DIR, "calibrator_temperature.pkl")`
2. **Revert `backend/engine/pipeline.py`:**
   - Change paths back to root legacy files.
3. **Restart Services:**
   - Restart FastAPI backend (`uvicorn backend.api.server:app`).
4. **Verification:**
   - Run `pytest tests/test_phase15_integrity.py` to confirm legacy baseline restoration.

---

## 9. Tests That Must Pass After Integration

Before declaring the integration complete, the following comprehensive test suite must be executed and pass 100%:

1. **Unit & Integrity Tests:**
   - `pytest tests/test_phase15_integrity.py` (End-to-end system and invariant tests).
   - `pytest tests/test_phase14_ood_abstention.py` (Dual-gate abstention tests).
2. **Holdout Verification:**
   - Run `python compare_production_vs_aegis.py` to ensure 0 prediction discrepancies.
   - Run `python validate_official_ciciot.py` to verify official 3K benchmark evaluation.
3. **API Smoke Test:**
   - Start backend server: `python -m uvicorn backend.api.server:app --port 8000`.
   - Send single test telemetry packet to `POST /api/v1/detect`.
   - Verify HTTP 200, valid probabilities summing to 1.0, and SHAP top drivers present.
4. **Abstention Smoke Test:**
   - Send synthetic OOD / high-variance packet to `POST /api/v1/detect`.
   - Verify response assigned `UNKNOWN_OR_REVIEW` and auto-blocking is disabled.

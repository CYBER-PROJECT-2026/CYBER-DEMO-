# AEGIS-IoT: Production Integration Dependency Map

> [!NOTE]
> **HISTORICAL / FORENSIC DEPENDENCY MAP ARCHIVE**  
> This dependency map cataloged all file locations loading models and calibrators across the repository prior to migration to `models/aegis/`.

**Date:** 2026-09-20  
**Total Model/Artifact Loading Locations Identified:** 73  

---

## 1. Executive Dependency Breakdown

| Category | Count | Scope & Role |
|---|---|---|
| **MODEL_CORE** | 6 | Primary detection logic and XGBoost inference loaders |
| **LABEL_ENCODING** | 6 | Label transformation and class name decoding |
| **CALIBRATION** | 18 | Temperature calibrator loading and probability scaling |
| **OOD** | 1 | Class-Conditional Mahalanobis novelty detector |
| **SHAP** | 3 | TreeSHAP explainer loading |
| **API** | 0 | FastAPI server (indirectly via detector singleton) |
| **AGENT** | 0 | Multi-Agent orchestrator (indirectly via pipeline) |
| **RAG** | 0 | Knowledge retriever (operates on threat context, no model loading) |
| **OTHER** | 39 | Offline audit scripts, evaluation suites, and verification tools |

## 2. Core Backend Production Loading Locations

These files constitute the **actual active backend inference path**:

### `backend/detection/detector.py`

| Line | Artifact | Category | Code |
|---|---|---|---|
| 26 | `cic_iot_xgboost_model.pkl` | `MODEL_CORE` | `MODEL_PATH = os.path.join(ROOT_DIR, "cic_iot_xgboost_model.pkl")` |
| 27 | `label_encoder.pkl` | `LABEL_ENCODING` | `ENCODER_PATH = os.path.join(ROOT_DIR, "label_encoder.pkl")` |
| 28 | `calibrator_temperature.pkl` | `CALIBRATION` | `CALIB_PATH = os.path.join(ROOT_DIR, "calibrator_temperature.pkl")` |

### `backend/engine/pipeline.py`

| Line | Artifact | Category | Code |
|---|---|---|---|
| 31 | `cic_iot_xgboost_model.pkl` | `MODEL_CORE` | `MODEL_PATH = os.path.join(BASE_DIR, "cic_iot_xgboost_model.pkl")` |
| 32 | `label_encoder.pkl` | `LABEL_ENCODING` | `ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")` |
| 33 | `calibrator_temperature.pkl` | `CALIBRATION` | `CALIB_PATH = os.path.join(BASE_DIR, "calibrator_temperature.pkl")` |

### `backend/detection/ood_detector.py`

| Line | Artifact | Category | Code |
|---|---|---|---|
| 37 | `cc_mahalanobis_params.pkl` | `OOD` | `DEFAULT_PARAMS_PATH = os.path.join(BASE_DIR, "cc_mahalanobis_params.pkl")` |

### `inference.py`

| Line | Artifact | Category | Code |
|---|---|---|---|
| 34 | `cic_iot_xgboost_model.pkl` | `MODEL_CORE` | `MODEL_PATH = os.path.join(BASE_DIR, "cic_iot_xgboost_model.pkl")` |
| 35 | `label_encoder.pkl` | `LABEL_ENCODING` | `ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")` |
| 36 | `calibrator_temperature.pkl` | `CALIBRATION` | `CALIB_PATH = os.path.join(BASE_DIR, "calibrator_temperature.pkl")` |

### `explain_prediction.py`

| Line | Artifact | Category | Code |
|---|---|---|---|
| 32 | `cic_iot_xgboost_model.pkl` | `MODEL_CORE` | `MODEL_PATH = os.path.join(BASE_DIR, "cic_iot_xgboost_model.pkl")` |
| 33 | `label_encoder.pkl` | `LABEL_ENCODING` | `ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")` |
| 34 | `calibrator_temperature.pkl` | `CALIBRATION` | `TEMP_CALIB_PATH = os.path.join(BASE_DIR, "calibrator_temperature.pkl")` |

### `predict_with_temperature.py`

| Line | Artifact | Category | Code |
|---|---|---|---|
| 33 | `cic_iot_xgboost_model.pkl` | `MODEL_CORE` | `MODEL_PATH = os.path.join(BASE_DIR, "cic_iot_xgboost_model.pkl")` |
| 34 | `label_encoder.pkl` | `LABEL_ENCODING` | `ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")` |
| 35 | `calibrator_temperature.pkl` | `CALIBRATION` | `TEMP_CALIB_PATH = os.path.join(BASE_DIR, "calibrator_temperature.pkl")` |

## 3. Minimum Set of Backend Files to Switch to AEGIS

To make AEGIS the active production ML model across the FastAPI server, WebSocket stream, and Multi-Agent SOC, **only 2 backend files** must have their default artifact paths updated:

1. **`backend/detection/detector.py`** (Used by `backend/api/server.py`, FastAPI endpoints, real-time sniffer)
   - `MODEL_PATH` -> `models/aegis/aegis_xgboost_model.pkl`
   - `ENCODER_PATH` -> `models/aegis/aegis_label_encoder.pkl`
   - `CALIB_PATH` -> `models/aegis/aegis_calibrator_temperature.pkl`

2. **`backend/engine/pipeline.py`** (Used by end-to-end testing, multi-agent evaluation)
   - `MODEL_PATH` -> `models/aegis/aegis_xgboost_model.pkl`
   - `ENCODER_PATH` -> `models/aegis/aegis_label_encoder.pkl`
   - `CALIB_PATH` -> `models/aegis/aegis_calibrator_temperature.pkl`

*(Optional CLI utility scripts for offline testing: `inference.py`, `explain_prediction.py`, `predict_with_temperature.py`)*

## 4. Invariant Preservations

- `backend/detection/ood_detector.py`: **NO CHANGE**. `cc_mahalanobis_params.pkl` remains active because it is purely feature-space based and model-independent.
- `backend/detection/explainer.py`: **NO CHANGE**. Takes `IoTIntrusionDetector` instance and automatically inherits `detector.model`.
- `backend/detection/confidence_gate.py`: **NO CHANGE**. Delegates to `OODAbstentionGate`.
- `backend/agents/`: **NO CHANGE**. Consumes generic JSON dictionaries.
- `backend/rag/`: **NO CHANGE**. Pure vector retrieval.
- `backend/api/server.py`: **NO CHANGE**. Instantiates `IoTIntrusionDetector()`.

# Forensic Calibration Consistency Audit

> [!NOTE]
> **HISTORICAL / FORENSIC AUDIT ARCHIVE (Phase 15 Root Baseline Audit)**  
> This document resolves the historical clerical discrepancy regarding the Phase 15 root calibrator (`calibrator_temperature.pkl`, $T \approx 1.317657$).  
> - **Active Production Calibrator:** `models/aegis/aegis_calibrator_temperature.pkl` ($T = 1.345192$)  
> - **Active Production Model:** `models/aegis/aegis_xgboost_model.pkl`

## Objective
Investigate the reported discrepancy between the Phase 15 documentation ($T \approx 0.447$) and the canonical calibration audit records ($T = 1.317657$). Determine the true stored parameter, the mathematical role of each value, the parameterization used in production inference, the validity of Phase 14/15 confidence thresholds and abstention states, and whether any code change is required.

---

## 1. Executive Summary & Forensic Findings

1. **Artifact Verification**:
   - File: [`calibrator_temperature.pkl`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/calibrator_temperature.pkl)
   - SHA-256: `61fad75aaecad0bd92e45c12c6af399334e79a066d38b2edc46baa77881115a8`
   - File Size: `99 bytes` (Identical to Phase 8, Phase 13, and Phase 14 frozen snapshots)
   - Object Type: `calibration_engine.MulticlassTemperatureCalibrator`
   - Stored Attribute: `calib.temperature = 1.317656719700599`
   
2. **Discrepancy Category**:
   **Category C**: $T = 1.317657$ is the actual, frozen, and only temperature value in the artifact and production codebase. The mention of $T \approx 0.447$ in the Phase 15 documentation was a **clerical mislabeling / documentation error in the Markdown text**.

3. **Mathematical Equivalence**:
   - The probability outputs of `calib.predict_proba(logits)` and manual temperature scaling $\text{softmax}(z / 1.317656719700599)$ are **mathematically and bit-for-bit identical** ($\text{max } |\Delta| = 0.0$) across all 5,000 test set samples $\times$ 3 classes (15,000 probability values).
   - If $T$ were hypothetically $0.447$, temperature scaling would have severely over-sharpened probabilities, driving Row #657's confidence from $0.6586$ up to $0.8743$, which would have prevented Row #657 from triggering the $0.70$ low-confidence threshold.
   - Because Row #657 produced confidence $0.6586$ in all Phase 14 and Phase 15 test runs, this empirically confirms that the entire codebase has been executing strictly with $T = 1.317657$.

4. **Code Change Assessment**:
   **NO CODE CHANGE REQUIRED**. The model, label encoder, calibrator, pipeline, and test suites are 100% consistent, mathematically sound, and untouched.

---

## 2. In-Depth Artifact & Code Inspection

### A. The Calibrator Object (`calibrator_temperature.pkl`)
Inspection of the serialized object confirms:
```python
import joblib
calib = joblib.load('calibrator_temperature.pkl')
print(type(calib))          # <class 'calibration_engine.MulticlassTemperatureCalibrator'>
print(vars(calib))          # {'temperature': 1.317656719700599}
```
The calibrator does not contain logistic regression weights ($a, b$), intercepts, or vector arrays. It contains a single float scalar: `self.temperature = 1.317656719700599`.

### B. Mathematical Implementation in `calibration_engine.py`
The `MulticlassTemperatureCalibrator` implements standard temperature scaling (Guo et al., 2017):
```python
class MulticlassTemperatureCalibrator:
    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits, y_true):
        # Optimizes temperature T > 0 by minimizing multiclass negative log likelihood (cross-entropy)
        ...
        res = minimize(nll_loss, x0=[1.0], method='L-BFGS-B', bounds=[(0.01, 50.0)])
        self.temperature = float(res.x[0])
        return self

    def predict_proba(self, logits):
        scaled_logits = logits / self.temperature
        return softmax(scaled_logits, axis=1)
```

### C. Production Inference Execution Path
Every production entry point invokes `calib.predict_proba(logits)`:
- `backend/engine/pipeline.py` (Line 97):
  ```python
  temp_probs = self.calib_temp.predict_proba(logits)[0]
  ```
- `backend/detection/detector.py` (Line 74):
  ```python
  probabilities = self.calibrator.predict_proba(logits)[0]
  ```
- `predict_with_temperature.py` (Line 92):
  ```python
  temp_probs = self.calib_temp.predict_proba(logits)[0]
  ```

There are no alternative model-loading branches, no hardcoded probability multipliers, and no secondary calibrator files used by the production detection stack.

---

## 3. Mathematical Verification Across Test Samples

To prove mathematical consistency, raw XGBoost margin logits $z$ were extracted for test samples and processed through both the serialized calibrator and manual formula evaluation:

### Sample 0 (Known DDoS)
- Raw Margin Logits: `[7.6025743, -6.537554, -8.660055]`
- `calib.predict_proba(logits)`:
  - DDoS: `0.99997377`
  - DoS: `0.00002185`
  - Spoofing: `0.00000436`
- Manual $\text{softmax}(z / 1.317657)$:
  - DDoS: `0.99997377`
  - DoS: `0.00002185`
  - Spoofing: `0.00000436`
- Absolute Difference: **`0.00000000`**

### Sample 1 (Known DoS)
- Raw Margin Logits: `[-7.6185803, 8.286808, -8.445805]`
- `calib.predict_proba(logits)`:
  - DDoS: `0.00000572`
  - DoS: `0.99999118`
  - Spoofing: `0.00000305`
- Manual $\text{softmax}(z / 1.317657)$:
  - DDoS: `0.00000572`
  - DoS: `0.99999118`
  - Spoofing: `0.00000305`
- Absolute Difference: **`0.00000000`**

### Sample 657 (Historical Near-Tie Error Sample)
- Raw Margin Logits: `[0.4646767, 1.3318405, -8.673511]`
- `calib.predict_proba(logits)` ($T = 1.317657$):
  - DDoS: `0.34105098`
  - DoS: `0.65861720`
  - Spoofing: `0.00033182`
  - Calibrated Confidence: **`0.6586`**
- Hypothetical Scaling if $T$ were $0.447$:
  - DDoS: `0.12565182`
  - DoS: `0.87434816`
  - Spoofing: `0.00000000`
  - Hypothetical Confidence: **`0.8743`**

### Key Diagnostic Implication
If $T = 0.447$ had been used, Row #657 would have produced confidence $0.8743$, which is well above the $0.70$ threshold. Row #657 would have failed to trigger the low-confidence abstention rule!
However, in Phase 14 and Phase 15, Row #657 produced confidence **$0.6586$**, successfully triggering the low-confidence rule. This confirms that Phase 14 and Phase 15 executed with $T = 1.317657$.

---

## 4. Dataset-Wide Confidence Threshold Verification

Across the entire 5,000-sample test set:
- **Calibrated probabilities using `calibrator_temperature.pkl`**:
  - Exactly **1 sample** exhibits confidence $< 0.70$ (Row #657 at $0.6586$).
- **Manual calculation using $T = 1.317656719700599$**:
  - Exactly **1 sample** exhibits confidence $< 0.70$ (Row #657 at $0.6586$).
- **Hypothetical calculation using $T = 0.447$**:
  - **0 samples** exhibit confidence $< 0.70$.

This proves that the Phase 14 confidence threshold ($0.70$) was established, tested, and validated against the exact probabilities produced by $T = 1.317657$.

---

## 5. Answers to Definitive Questions

### 1. What is the actual calibration parameter?
**$T = 1.317656719700599$** (commonly reported as $T \approx 1.3177$). It is stored as a float attribute `self.temperature` inside `calibrator_temperature.pkl`.

### 2. What does 0.447 represent?
$0.447$ is a **clerical / transcription typo** in the Phase 15 Markdown audit report. It corresponds to no artifact, no fitted parameter, and no mathematical formula in the AEGIS-IoT codebase.

### 3. What does 1.317657 represent?
It represents the **optimal post-hoc temperature scaling parameter** derived by minimizing multiclass negative log-likelihood (cross-entropy) over the 21,000-record calibration dataset (`calibration_data.csv`).

### 4. Which value is actually used in production?
**$T = 1.317656719700599$**. All production inference components (`CompleteIDSPipeline`, `IoTIntrusionDetector`, `predict_with_temperature.py`) dynamically load `calibrator_temperature.pkl` and evaluate `logits / self.temperature`.

### 5. Are the Phase 14/15 confidence and abstention results valid?
**YES, 100% VALID**. All empirical metrics, confidence scores (such as Row #657's $0.6586$ confidence), abstention evaluations, and invariant tests were computed using the actual frozen calibrator ($T = 1.317657$).

### 6. Is any code change required?
**NO CODE CHANGE REQUIRED**. The model, label encoder, calibrator artifact, and inference code are completely intact and mathematically verified. The only rectification required is the text correction in the Phase 15 Markdown documentation.

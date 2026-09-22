# Probability Calibration & Confidence Analysis Report
## Comprehensive Investigation: Why Isotonic Calibration Produces 100% Confidence

> [!NOTE]
> **Calibration Comparison Study Archive (Phase 10 Baseline)**: This technical investigation evaluates why isotonic calibration over-saturates compared to Temperature Scaling on the 5,000-record test partition. Active AEGIS integrates Temperature Scaling ($T = 1.345192$) in `models/aegis/aegis_calibrator_temperature.pkl` with `models/aegis/aegis_xgboost_model.pkl`.

**Date:** August 27, 2026  
**Scope:** Strict Analysis of Pre-Trained Artifacts on 5,000-Record Test Dataset (`Testing_data/testing_data.csv`)  
**Artifacts Analyzed:**
- Model: `models/aegis/aegis_xgboost_model.pkl` (46 features, frozen; root historical baseline: `cic_iot_xgboost_model.pkl`)
- Label Encoder: `label_encoder.pkl` (`['DDoS', 'DoS', 'Spoofing']`)
- Calibrator Artifacts: `calibrator_sigmoid.pkl`, `calibrator_isotonic.pkl`, `calibrator_temperature.pkl`, `calibrated_xgboost_final.pkl`

---

## 1. Direct Method Comparison Table

Below is the empirical evaluation across all 5,000 test records without refitting any model or calibrator.

| Method | Accuracy | Macro F1 | ECE (10-bin) | Brier Score | Log Loss | Mean Confidence | Median Confidence | Exact / Effective 100% Count | Exact 100% (%) | Unique Confidence Values |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original XGBoost** | 99.9800% | 0.999800 | 0.000157 | 0.00019995 | 0.00025982 | 0.999925 | 0.999999 | 357 (exact) / 891 (eff) | 7.14% / 17.82% | 177 |
| **Sigmoid / Platt** | 100.0000% | 1.000000 | 0.000101 | 0.00009312 | 0.00013615 | 0.999899 | 1.000000 | 0 (exact) / 4,981 (eff) | 0.00% / 99.62% | 3,561 |
| **Isotonic Regression** | 100.0000% | 1.000000 | 0.000112 | 0.00006534 | 0.00013300 | 0.999888 | 1.000000 | 4,998 (eff: $1.0 - 2\times 10^{-12}$) | 99.96% | **4** |
| **Temperature Scaling** | 99.9800% | 0.999800 | 0.000200 | 0.00017904 | 0.00028530 | 0.999863 | 0.999976 | 0 (exact) / 0 (eff) | 0.00% | 926 |

*Note on 100% Count:* In `calibration_engine.py`, `calib_probs = np.clip(calib_probs, 1e-12, 1.0)` is applied before L1 vector normalization. When the isotonic mapping assigns `1.0` to the winning class and `0.0` to the two losing classes, the resulting probability vector is $\left[\frac{1.0}{1.0 + 2\times 10^{-12}}, \frac{10^{-12}}{1.0 + 2\times 10^{-12}}, \frac{10^{-12}}{1.0 + 2\times 10^{-12}}\right]$, yielding a winning class confidence of $0.999999999998$ (which rounds to $1.000000$ / $100.0\%$).

---

## 2. Executive Summary of Root Causes

The phenomenon where **4,998 out of 5,000 (99.96%) test samples produce 100% confidence** under Isotonic Calibration is caused by three compounding mathematical and architectural factors:

1. **Extreme Separability of the Raw Model:**
   The frozen 46-feature XGBoost model already exhibits near-perfect class separation. On the 5,000 test records, 4,997 records have raw XGBoost confidence $\ge 99.61\%$, and 4,106 records have raw confidence $\ge 99.9\%$.
2. **Learned Step-Function Plateaus in Isotonic Calibration:**
   Because the calibration split exhibited high class separation with zero observed empirical errors above moderate threshold values, the non-parametric Pool Adjacent Violators (PAV) algorithm in Isotonic Regression assigned an empirical probability of $y = 1.000000$ to all raw probabilities exceeding:
   - **DDoS:** Raw probability $\ge 0.91750312 \implies \hat{P}_{\text{iso}} = 1.000000$
   - **DoS:** Raw probability $\ge 0.99918669 \implies \hat{P}_{\text{iso}} = 1.000000$
   - **Spoofing:** Raw probability $\ge 0.99672478 \implies \hat{P}_{\text{iso}} = 1.000000$
3. **Severe Output Discretization (Quantization):**
   Isotonic regression maps the continuous probability spectrum into a coarse step function. Across all 5,000 test samples, the isotonic calibrator outputs **only 4 unique confidence values**:
   - $0.9999999999979998 \ (\approx 100.0\%)$: 3,332 samples (66.64%)
   - $0.9999999999980000 \ (\approx 100.0\%)$: 1,666 samples (33.32%)
   - $0.7714285582904279 \ (\approx 77.14\%)$: 1 sample (0.02%)
   - $0.6666666666659999 \ (\approx 66.67\%)$: 1 sample (0.02%)

---

## 3. Detailed Inspection of the Learned Isotonic Mapping

The saved `calibrator_isotonic.pkl` artifact implements **One-vs-Rest (OvR)** binary isotonic regression per class, followed by vector L1 normalization.

### Class 0: DDoS (9 Thresholds)
| Threshold Index | Raw Input Probability ($X$) | Calibrated Output Probability ($Y$) |
| :---: | :---: | :---: |
| 1 | $3.221862 \times 10^{-8}$ | $0.000000$ |
| 2 | $8.110221 \times 10^{-4}$ | $0.000000$ |
| 3 | $1.385096 \times 10^{-3}$ | $0.222222$ |
| 4 | $1.052174 \times 10^{-1}$ | $0.222222$ |
| 5 | $2.958354 \times 10^{-1}$ | $0.666667$ |
| 6 | $5.721394 \times 10^{-1}$ | $0.750000$ |
| 7 | $8.659407 \times 10^{-1}$ | $0.750000$ |
| **8** | **$0.91750312$** | **$1.000000$** |
| **9** | **$0.99999940$** | **$1.000000$** |

### Class 1: DoS (8 Thresholds)
| Threshold Index | Raw Input Probability ($X$) | Calibrated Output Probability ($Y$) |
| :---: | :---: | :---: |
| 1 | $1.182430 \times 10^{-8}$ | $0.000000$ |
| 2 | $8.246335 \times 10^{-2}$ | $0.000000$ |
| 3 | $1.340040 \times 10^{-1}$ | $0.333333$ |
| 4 | $7.041328 \times 10^{-1}$ | $0.333333$ |
| 5 | $8.947025 \times 10^{-1}$ | $0.750000$ |
| 6 | $9.985687 \times 10^{-1}$ | $0.750000$ |
| **7** | **$0.99918669$** | **$1.000000$** |
| **8** | **$0.99999905$** | **$1.000000$** |

### Class 2: Spoofing (4 Thresholds)
| Threshold Index | Raw Input Probability ($X$) | Calibrated Output Probability ($Y$) |
| :---: | :---: | :---: |
| 1 | $1.365201 \times 10^{-8}$ | $0.000000$ |
| 2 | $1.767385 \times 10^{-1}$ | $0.000000$ |
| **3** | **$0.99672478$** | **$1.000000$** |
| **4** | **$0.99999964$** | **$1.000000$** |

---

## 4. Confidence Distribution: Raw vs. Isotonic

| Confidence Bin | Raw XGBoost Count | Raw XGBoost (%) | Isotonic Count | Isotonic (%) |
| :--- | :---: | :---: | :---: | :---: |
| **0–50%** | 0 | 0.00% | 0 | 0.00% |
| **50–70%** | 0 | 0.00% | 1 | 0.02% |
| **70–80%** | 1 | 0.02% | 1 | 0.02% |
| **80–90%** | 0 | 0.00% | 0 | 0.00% |
| **90–95%** | 1 | 0.02% | 0 | 0.00% |
| **95–99%** | 0 | 0.00% | 0 | 0.00% |
| **99–99.9%** | 1 | 0.02% | 0 | 0.00% |
| **99.9–<100%** | 4,640 | 92.80% | 0 | 0.00% |
| **Effective 100% ($\approx 1.0$)** | 357 (exact float 1.0) | 7.14% | **4,998** | **99.96%** |

---

## 5. Detailed Analysis of the Two Non-100% Records

Exactly **two** records out of 5,000 have calibrated confidence $< 1.0$:

### Record #657 (Prediction Corrected by Isotonic Calibration)
- **Actual Ground Truth:** `DDoS`
- **Raw XGBoost Probabilities:** DDoS = `0.29583541`, DoS = `0.70413280`, Spoofing = `0.00003180`
- **Raw Predicted Class:** `DoS` (Raw Confidence: `0.70413280`) $\rightarrow$ **Incorrect raw prediction**
- **Isotonic Mapping:**
  - DDoS calibrator on $0.295835 \implies 0.666667$
  - DoS calibrator on $0.704133 \implies 0.333333$
  - Spoofing calibrator on $0.000032 \implies 0.000000$ (clipped to $10^{-12}$)
- **Calibrated Probabilities:** DDoS = `0.66666667`, DoS = `0.33333333`, Spoofing = `0.00000000`
- **Calibrated Predicted Class:** `DDoS` (Calibrated Confidence: `0.66666667`) $\rightarrow$ **Corrected to True Class**

### Record #4071
- **Actual Ground Truth:** `DoS`
- **Raw XGBoost Probabilities:** DDoS = `0.06328854`, DoS = `0.93639004`, Spoofing = `0.00032143`
- **Raw Predicted Class:** `DoS` (Raw Confidence: `0.93639004`)
- **Isotonic Mapping:**
  - DDoS calibrator on $0.063289 \implies 0.222222$
  - DoS calibrator on $0.936390 \implies 0.750000$
  - Spoofing calibrator on $0.000321 \implies 0.000000$
  - Vector normalization: $0.222222 + 0.750000 = 0.972222$
- **Calibrated Probabilities:** DDoS = `0.22857144`, DoS = `0.77142856`, Spoofing = `0.00000000`
- **Calibrated Predicted Class:** `DoS` (Calibrated Confidence: `0.77142856`)

---

## 6. Technical Interpretation

1. **Classification Accuracy vs. Probability Calibration:**
   - **Classification accuracy** measures the fraction of discrete class assignments ($\arg\max_k P_k$) that match ground truth. The model achieves $99.98\%$ (raw) and $100.00\%$ (calibrated) accuracy because the 46 network flow features cleanly partition DDoS, DoS, and Spoofing attacks in feature space.
   - **Probability calibration** measures whether predicted confidence scores correspond to true empirical frequencies (i.e., among instances predicted with confidence $p$, exactly a fraction $p$ belong to that class).
2. **Why Isotonic Regression Outputs 1.0:**
   - Isotonic regression is a non-parametric, monotonic step function fitted using isotonic least squares (PAV algorithm). When all samples in the calibration dataset with raw probabilities above a given threshold have ground truth label $y=1$, the PAV algorithm sets the calibrated output value for that interval to exactly $y = 1.0$.
3. **Assessment of Isotonic Behavior for Production IDS:**
   - While Isotonic calibration achieves a very low Brier score ($0.000065$) and low ECE on this static holdout set, its output is **excessively discretized** (only 4 distinct confidence values across 5,000 samples).
   - In live deployment, an intrusion detection system requires nuanced confidence values to distinguish borderline anomalies from unambiguous signatures. Isotonic calibration collapses almost all incoming traffic into flat 100% plateaus, destroying the uncertainty signals needed for threshold tuning, tier-2 SOC escalation, and drift detection.

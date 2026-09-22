# AEGIS-IoT: OOD-Aware Abstention & Decision Architecture

This document describes the external decision and abstention layer introduced in Phase 14 to wrap the frozen 46-feature XGBoost classifier.

---

## 1. Architectural Motivation

Prior to Phase 14, AEGIS-IoT operated as a closed-world 3-class classifier (`DDoS`, `DoS`, `Spoofing`). While schema validation ensured that inputs conformed to the 46 canonical float32 columns, unfamiliar, anomalous, or out-of-distribution inputs were unconditionally assigned to one of the 3 classes. Furthermore, gradient boosted trees extrapolate constant predictions into peripheral regions of feature space, leading to softmax overconfidence on far outliers.

Phase 14 resolves this by wrapping the frozen model with an **External OOD-Aware Abstention Layer**.

---

## 2. Decision State Taxonomy

Every processed telemetry vector transitions through the gate and is assigned one of four definitive states:

| Decision State | Meaning | Operational Action |
| :--- | :--- | :--- |
| `KNOWN_DDoS` | In-distribution, calibrated confidence $\ge 0.70$, classified as DDoS (`trusted_class = DDoS`, `automated_blocking_allowed = True`). | Eligible for pre-authorized automated containment playbooks (e.g., rate-limiting); disruptive actions require SOC analyst authorization. |
| `KNOWN_DoS` | In-distribution, calibrated confidence $\ge 0.70$, classified as DoS (`trusted_class = DoS`, `automated_blocking_allowed = True`). | Eligible for pre-authorized automated containment playbooks (e.g., connection throttling); disruptive actions require SOC analyst authorization. |
| `KNOWN_Spoofing` | In-distribution, calibrated confidence $\ge 0.70$, classified as Spoofing (`trusted_class = Spoofing`, `automated_blocking_allowed = True`). | Eligible for pre-authorized automated containment playbooks (e.g., port isolation); disruptive actions require SOC analyst authorization. |
| `UNKNOWN_OR_REVIEW` | Lacks sufficient evidence to trust prediction (`trusted_class = None`, `automated_blocking_allowed = False`). | **Strictly blocks automated network disruptions.** Routes complete diagnostic evidence to SOC analyst for mandatory review. |

---

## 3. Dual-Trigger Evaluation Engine

The gate evaluates two complementary dimensions of uncertainty:

```
                          ┌───────────────────────────┐
                          │   46 Validated Features   │
                          └─────────────┬─────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
              [XGBoost + Calibrator]        [Class Centroid & VI]
                         │                             │
                         ▼                             ▼
             Calibrated Confidence (P)      CC Mahalanobis Distance (d_CC)
                         │                             │
                         ▼                             ▼
                     P < 0.70 ?                   d_CC > 23.4341 ?
                (Boundary Ambiguity)            (Geometric Atypicality)
                         │                             │
                         └──────────────┬──────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         │   Any Condition Triggered?  │
                         └──────────────┬──────────────┘
                                        │
                       YES ─────────────┴───────────── NO
                        │                               │
                        ▼                               ▼
               UNKNOWN_OR_REVIEW                   KNOWN_Class
          (Automated Blocking Prohibited)   (Eligible for Pre-Authorized Playbooks)
```

1. **Classification Boundary Uncertainty**:
   - Metric: Temperature-calibrated posterior probability $P(\hat{y} \mid x)$.
   - Threshold: $T_{\text{conf}} = 0.70$.
   - Rationale: Catches inputs near decision boundaries between known classes.
2. **Feature-Space Atypicality / Novelty**:
   - Metric: Class-Conditional Mahalanobis Distance $d_{CC}(x, \hat{y}) = \sqrt{(x - \mu_{\hat{y}})^T (\Sigma_{\hat{y}} + 10^{-6}I)^{-1} (x - \mu_{\hat{y}})}$.
   - Threshold: $T_{\text{mah}} = 23.4341$ (P99.0 empirical cutoff on 21,000 calibration records).
   - Rationale: Detects geometric outliers that tree models overconfidently classify.

---

## 4. Human-In-The-Loop Safety Guarantees

For any sample marked `UNKNOWN_OR_REVIEW`:
1. **Network Mutation Lock**: The automated execution engine (`backend/enforcement/enforcement.py`) rejects automated firewall drop execution.
2. **Advisory Distinction**: The model's argmax prediction is marked advisory: `"[ADVISORY / UNVERIFIED] Potential DoS (UNKNOWN_OR_REVIEW)"`.
3. **Forensic Evidence Dossier**: SOC analysts receive:
   - Primary predicted class and top-2 margin.
   - Calibrated posterior probabilities across all classes.
   - CC Mahalanobis distance and calibration percentile reference.
   - TreeSHAP feature attributions explaining what drove the advisory prediction.
   - MITRE ATT&CK and NIST guidelines from Adaptive Hybrid RAG.
   - Explicit reason for abstention (`LOW_CONFIDENCE`, `OOD_DISTANCE`, or `BOTH`).

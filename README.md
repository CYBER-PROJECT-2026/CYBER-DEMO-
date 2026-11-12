# AEGIS-IoT: AI-Driven Intrusion Detection System for IoT Networks

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14.2%2B-black)](https://nextjs.org/)
[![Model Freeze](https://img.shields.io/badge/Model%20Freeze-SHA--256%20Verified-brightgreen)](file:///models/aegis_live_v4)
[![Safety Invariants](https://img.shields.io/badge/Automated%20Blocking-Disabled-red)](file:///backend/detection/evidence_fusion.py)

---

## 1. System Overview & Architecture

**AEGIS-IoT** is an enterprise-grade autonomous Security Operations Center (SOC) platform designed for resource-constrained IoT edge environments. The system transitions beyond standard closed-world ML classification by integrating:

1. **V4 Authoritative Detection Pipeline (29 Canonical Features):** Lightweight, calibrated gradient boosting model targeting high-risk volumetric and spoofing attacks (`Benign`, `DDoS`, `DoS`, `Spoofing`).
2. **Epistemic Abstention & Out-of-Distribution (OOD) Gating:** Class-Conditional Mahalanobis distance gating combined with temperature-scaled Platt calibration to safely abstain on borderline or novel inputs (`UNKNOWN_OR_REVIEW`).
3. **Behavioral Anomaly Residual Detection:** Baseline reconstruction and drift monitoring providing continuous anomaly scores and z-score feature deviations.
4. **Online TreeSHAP Directional Explainability:** Real-time exact feature attribution on 29 flow dimensions to expose model drivers for human analysts.
5. **Multi-Agent RAG Threat Reasoning:** 5-agent grounded reasoning architecture (Detection Specialist, Threat Intel, Forensics, Enforcement, Lead SOC) grounded in 949 MITRE ATT&CK, NIST SP 800-61, and CISA advisories.
6. **Unknown Attack Knowledge Base (UAKB):** Structured persistence and similarity retrieval for novel, suspicious, and OOD observations.
7. **Controlled Feedback & Research Curation:** Safe, non-mutating feedback workflow (Phase 5) allowing human analysts to curate novel attack candidates without altering frozen models (`training_eligible = 0`).
8. **V6 Staging Candidate Sandbox (35 Features):** Parallel, isolated staging pipeline evaluating topological network extensions (Config B) strictly segregated from authoritative production decisions.
9. **Next.js Real-Time SOC Dashboard:** Live WebSocket incident feeds, TreeSHAP force displays, interactive Human-in-the-Loop (HITL) triage, and simulated mitigation approval.

```
                                 [ Network Packets / PCAP Stream ]
                                                 │
                                 ┌───────────────┴───────────────┐
                                 ▼                               ▼
                      [ V4 Authoritative Ingestion ]   [ V6 Staging Sandbox ]
                        29 Canonical Flow Features       35 Features (Config B)
                                 │                               │
                                 ▼                               ▼
                      [ V4 XGBoost Baseline ]           [ V6 Staging Model ]
                      Temperature Scaling (T=1.345)     (Isolated Candidate)
                                 │                               │
                                 ▼                               │
                      [ Evidence Fusion Engine ]                 │
                        • Mahalanobis OOD Gating                 │
                        • Behavioral Anomaly Residual            │
                        • Online TreeSHAP Drivers                │
                                 │                               │
                     ┌───────────┴───────────┐                   │
                     ▼                       ▼                   │
             [ TRUSTED STATE ]     [ UNKNOWN_OR_REVIEW ]         │
                     │                       │                   │
                     │                       ▼                   │
                     │             [ Unknown Attack KB ]         │
                     │             Similarity Retrieval          │
                     │                       │                   │
                     └───────────┬───────────┘                   │
                                 ▼                               │
                      [ Multi-Agent RAG System ]                 │
                        5 Grounded SOC Agents                    │
                                 │                               │
                                 ▼                               ▼
                      [ FastAPI Local Backend ] ◄────────────────┘
                        Port 8000 | WebSocket Stream
                                 │
                                 ▼
                      [ Next.js SOC Dashboard ]
                        Human-in-the-Loop Review
                        Simulated Mitigation Only
```

---

## 2. Cryptographic Model Freeze & Startup Integrity

All model weights and threshold parameters are cryptographically sealed with SHA-256 hashes. Upon startup, `verify_startup_integrity()` enforces a strict fail-closed contract before exposing API endpoints:

| Artifact Name | Scope | SHA-256 Digest | Status |
| :--- | :--- | :--- | :---: |
| `aegis_live_xgboost_baseline.pkl` | V4 Authoritative Classifier | `c0e5aee8e9e1726f12bfc19e98c440fe44fe278d1796d5866fe2a2488d458a56` | Sealed |
| `temperature_scaling.json` | V4 Platt Calibration | `c3ac0662ca896d06d4b9f62035a637fe04c8e01c00c466790d2ec0f03e581f99` | Sealed |
| `ood_parameters.pkl` | V4 Mahalanobis Covariances | `579063cfc4dda3e396ab0d026198f14f8ef593ebf9910f290157aa0f669712c3` | Sealed |
| `abstention_config.json` | V4 Threshold Contracts | `7d49f05bef592cb838ef69f813a1c297beca9e0a007bbad1aad6f5f0f4ac397f` | Sealed |
| `config_b_model.pkl` | V6 Staging Candidate | `09afb03cbe74201e4e3768e367fb35c70d727bc41a5b9ffc6180c3305fdf1f84` | Sealed |
| `temperature_scaling_b.json` | V6 Staging Calibration | `2cedf27975b4c28aba2bbf52b84b0a5c5c49caaf1ef3fa861ac6660885aa3b6f` | Sealed |
| `abstention_config_b.json` | V6 Staging Abstention | `e30bc4571c5045c3374aae930145821f1277e1c9aba56e4f419f7b10b39e2039` | Sealed |

If any authoritative artifact hash does not match, the application terminates immediately with `RuntimeError: FATAL STARTUP INTEGRITY ERROR`.

---

## 3. Verified Performance & Independent Evaluation Summary

### Known-Attack Holdout Evaluation (CIC-IoT-2023)
- **Macro-F1 (Trusted Denominator):** **0.9505** (95.05%)
- **Macro-F1 (Total Denominator, Penalizing Abstentions):** **0.9118** (91.18%)
- **Benign False Positive Rate:** **3.67%** (0.0367)
- **Mean Inference Latency:** **10.28 ms** (p99: 40.47 ms)

### Independent Unseen-Attack Evaluation (Phases 6 & 6.1)
- **Kitsune Mirai Botnet Capture (NDSS 2018):**
  * Supervised V4 Zero-Shot Recall: **1.75%** (demonstrating why closed-world supervised models fail on novel malware).
  * AEGIS Evidence Fusion Flagging Rate: **82.64%** flagged as suspicious/OOD and safely routed to SOC triage.
  * Zero Mirai attack windows escaped as Benign.
- **Kitsune SYN DoS Capture:**
  * 55,544 primary windows ($55,509$ benign, $35$ DoS) + $118$ attack onset transition windows ($55,662$ total).
  * Staging candidate V6 captured $100\%$ of SYN DoS windows under total capture criteria ($82.86\%$ direct DoS/DDoS classification).
- **CTU-IoT-Malware-Capture-8-1 (IoT-23 IRC Botnet C&C):**
  * $891$ extracted flow windows ($483$ Benign, $408$ C&C Malware).
  * Classifier Raw Predictions (Advisory Only): $483$ Spoofing, $408$ DDoS.
  * Final Evidence Fusion Decisions: $483$ Spoofing, $408$ `UNKNOWN_OR_REVIEW`.
  * Malware Flagging Rate: **100.0%** ($408 / 408$). Zero C&C windows escaped as Benign.

---

## 4. Local Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.10–3.13)
- Node.js 18+ and npm
- Local SQLite (bundled with Python)

### Step 1: Clone Repository
```bash
git clone https://github.com/<your-org>/AEGIS-IoT.git
cd AEGIS-IoT
```

### Step 2: Set Up Python Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Set Up and Build Frontend
```bash
cd frontend
npm install
npm run build
cd ..
```

---

## 5. Running the Application

### 1. Launch Backend API Server
Start the local FastAPI server with startup integrity verification:
```bash
python -m uvicorn backend.api.server:app --host 127.0.0.1 --port 8000 --reload
```
Upon startup, the server logs:
```text
[INFO] AEGIS-IoT startup integrity verified successfully: PASS
[INFO] Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

Verify integrity endpoint:
```bash
curl http://127.0.0.1:8000/api/integrity
# {"status":"PASS","authoritative_v4_model":"VERIFIED_FROZEN_29_FEATURES","pipeline_mode":"PRODUCTION_V4_AUTHORITATIVE"}
```

### 2. Launch Next.js SOC Dashboard
In a separate terminal:
```bash
cd frontend
npm run start
# Or for local development:
# npm run dev
```
Open your browser at `http://localhost:3000` to access the AEGIS SOC Dashboard.

---

## 6. Running the End-to-End Demonstration

AEGIS includes a fully automated, reproducible local demonstration script:
```bash
python eval/run_phase7_end_to_end_demo.py
```

This script exercises all 10 pipeline stages using authentic IoT PCAP traffic (`CTU-IoT-Malware-Capture-8-1`):
1. **Startup Integrity:** Verifies SHA-256 hashes of all frozen V4 and V6 artifacts.
2. **Packet Ingestion:** Extracts streaming flow windows across 29 canonical features.
3. **V4 Authoritative Detection:** Executes calibrated XGBoost inference and computes online TreeSHAP feature attributions.
4. **Evidence Fusion:** Evaluates behavioral anomaly residual and class-conditional Mahalanobis distance.
5. **UAKB Persistence:** Automatically stores novel/suspicious observations with strict safety invariants.
6. **V6 Staging Evaluation:** Runs candidate Config B (35 features) in isolated staging mode.
7. **SOC Incident Generation:** Synthesizes multi-agent RAG advisory report.
8. **Human-in-the-Loop Review:** Simulates SOC analyst review and mitigation approval.
9. **Research Curation:** Promotes verified finding to curated research candidates (`training_eligible = 0`).
10. **Audit Trail Logging:** Verifies immutable SQLite audit log entries.

---

## 7. Running Verification Tests

Run the full project regression test suite (57 tests covering all phases and architecture evolution):
```bash
pytest tests/ -v
```

Run targeted Architecture Evolution tests (Auth/RBAC, Docker, Evolution Harness, Edge Export):
```bash
pytest tests/test_architecture_evolution.py -v
```

Run targeted Phase 7 verification tests:
```bash
pytest tests/test_phase7_integration_packaging_demo.py -v
```

Run Phase 6 evaluation audit tests:
```bash
pytest tests/test_phase6_unseen_attack_eval.py -v
```

---

## 8. Architecture Evolution & Enterprise Extensibility

AEGIS-IoT includes the complete post-Phase 7 enterprise architecture stack:

### 1. Enterprise Authentication & RBAC (`backend/security/auth.py`)
- Standard RFC 7519 HS256 JWT tokens with role-based access tiers:
  * `SOC_ANALYST_L1`: Read-only triage and alert monitoring.
  * `SOC_ANALYST_L2`: Incident review, HITL decision approval, observation verification.
  * `SOC_LEAD_ADMIN`: Full governance, research candidate promotion, dataset export.
- Rejection of automated/system identities (`SYSTEM`, `AUTOMATED_*`) across all credential and review paths.
- Endpoints: `POST /api/auth/login` and `GET /api/auth/me`.

### 2. Containerized Deployment (`docker-compose.yml`)
Deploy the entire platform (backend, Next.js frontend, SQLite, ML models) with a single command:
```bash
docker-compose up --build
```
- Multi-stage `Dockerfile.backend` (Python 3.11-slim, unprivileged `aegis` user, fail-closed healthcheck).
- Multi-stage `Dockerfile.frontend` (Node 18-alpine, standalone production runner).

### 3. Governed Offline Model Evolution Harness (`backend/curation/evolution_harness.py`)
- Serves as the governed bridge between analyst-curated observations (Phase 5) and future candidate models.
- Air-gapped execution: candidates train strictly outside the live path (`models/candidates/`).
- Automated Regression Guardrails:
  * Known-attack Macro-F1 $\ge 0.9500$.
  * Benign False Positive Rate $\le 5.0\%$.
  * Degradation compared to baseline $\le 1.0\%$.
- Generates tamper-evident evaluation audit reports in `reports/model_evolution/`.

### 4. Edge Model Optimization & Lightweight Tree Export (`backend/detection/edge_exporter.py`)
- Exports 29-feature V4 model into portable JSON tree definitions for low-power C/C++/Rust edge runtimes (Raspberry Pi, NVIDIA Jetson).
- Verified mathematical output parity ($0.0$ max absolute divergence) compared to Python XGBoost predictions.

### 5. Wire-Speed Ingestion & Packet Filtering (`backend/engine/packet_filter.py`)
- Simulates kernel-level eBPF / XDP zero-copy packet pre-filtering.
- Verified line-rate filtering throughput ($> 500,000$ packets/sec).

### 6. Device-Adaptive Topology Normalization (`aegis_live/adaptive_topology_normalizer.py`)
- Resolves cross-topology false positive spikes on novel IoT profiles (e.g., surveillance cameras).
- Normalizes link-layer fan-out and concentration features against learned device-profile baselines.

---

## 9. Security & Deployment Boundaries

> [!IMPORTANT]
> **Strict Operational Invariants:**
> 1. **Automated Blocking Prohibited (`automated_blocking_allowed = False`):**
>    The system operates strictly as an autonomous SOC advisor. Under no circumstances can detection models or agent recommendations trigger automated firewall, iptables, or network blocking without human analyst sign-off.
> 2. **Simulated Enforcement Only:**
>    Approved mitigation actions execute inside the `SecurityEnforcementEngine(dry_run=True)` simulation sandbox. No real network configuration, routing tables, or firewall rules are modified.
> 3. **Feedback Training Exclusion (`training_eligible = 0`):**
>    Analyst feedback, incident notes, and curated candidate records are strictly segregated from model weights. Automated retraining, dynamic fine-tuning, or online weight mutation is prohibited.
> 4. **Local Deployment Boundary:**
>    AEGIS-IoT backend binds to `127.0.0.1:8000`. It is not configured for public internet exposure. Production enterprise deployment requires an external reverse proxy (TLS/HTTPS termination) and enterprise identity provider (OIDC/SAML/RBAC).
> 5. **Reviewer Identity Contract:**
>    Reviewer IDs supplied by clients are validated against system and automated bot impersonation (rejecting `SYSTEM` and `AUTOMATED_*`). In this local demonstration release, production SSO/mTLS authentication is not implemented.

---

## 9. Citation & Academic Reference

If you reference this architecture or evaluation framework in academic work, please cite:
```bibtex
@inproceedings{hindawi2026efficient,
  title={An Efficient AI-Driven Intrusion Detection System for Resource-Constrained IoT Environments},
  author={Hindawi, Asmaa and El Arid, Amal and others},
  booktitle={5th IEEE International Multidisciplinary Conference on Engineering Technology (IMCET)},
  year={2026},
  location={Beirut, Lebanon}
}
```

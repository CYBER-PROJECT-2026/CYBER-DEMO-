# AEGIS Batch 29.1 — Shadow Output Isolation

**Scope:** Tag controlled-evaluation outputs and prevent routing into production incident/response paths. No evaluation was executed.

## Root cause

The evaluation harness accumulated V4/V6 rows and aggregate results without a durable evaluation identity or output-mode tag. Meanwhile, the production V4 service, API callback, and incident correlator treated incoming detection dictionaries as operational records and could persist, broadcast, correlate, notify, or escalate them without checking whether they came from evaluation. The incident classification contract also had no evaluation-only state.

## Files changed

- `aegis_live/v4_runtime.py`
- `aegis_live/evaluation/harness.py`
- `backend/engine/v4_service.py`
- `backend/api/server.py`
- `backend/detection/incident_correlator.py`
- `backend/detection/output_isolation.py` (new)
- `tests/test_shadow_output_isolation.py` (new)
- `reports/BATCH29_1_SHADOW_OUTPUT_ISOLATION.md` (this report)

## Isolation design

- V4 runtime output mode is explicit (`PRODUCTION` or `SHADOW_EVALUATION`). Shadow mode requires an evaluation ID; invalid modes and evaluation IDs attached to production mode are rejected. Both eligible detection results and partial-window provenance carry mode and evaluation ID.
- The evaluation harness creates a per-capture evaluation ID and tags its records and result summaries `SHADOW_EVALUATION`. Records identify `SHADOW_EVALUATION_V4` / `_V6`, set `classification_scope=EVALUATION_ONLY`, and explicitly set `production_incident_eligible=False`.
- A shared fail-closed routing helper allows only production-mode records without an evaluation ID or shadow record type into operational callbacks. The V4 service skips evidence enrichment and callbacks for non-production records. The API detection callback returns before in-memory production buffers, persistence, websocket broadcast, and agent/incident routing. The incident correlator independently rejects non-production records before database access; its classification contract labels them `SHADOW_EVALUATION` with no incident prefix.
- Production records with no new tags retain the existing production behavior. Non-50 packet windows remain diagnostic/advisory (`INCOMPLETE_WINDOW` / `INELIGIBLE_WINDOW`, `v4_eligible=False`) and are never emitted by the classifier gate. Automated blocking remains false.

## Tests and exact results

- `python -m py_compile aegis_live/v4_runtime.py aegis_live/evaluation/harness.py backend/engine/v4_service.py backend/api/server.py backend/detection/incident_correlator.py backend/detection/output_isolation.py tests/test_shadow_output_isolation.py` — **passed**.
- `python -m pytest -q tests/test_shadow_output_isolation.py tests/test_incident_correlation_contract.py --disable-warnings --basetemp=.pytest-b291-tmp` — **14 passed, 0 failed**, 3 warnings, 7.56 seconds.
- The focused tests check evaluation identity/mode, fail-closed callback dispatch and no callback for shadow records, unchanged production callback dispatch, direct correlator/classification suppression, immutable blocking false, and diagnostic-only 49-packet eligibility. Existing incident-contract cases in the same run provide production/advisory regression coverage.
- An initial run using pytest's default temp root had 3 fixture setup errors because that root could not create temp directories (11 passed). The same selection passed when `--basetemp` pointed inside the workspace.
- No classifier runtime was instantiated by these tests; no classifier was loaded and no inference was performed. API callback suppression was verified by the shared routing predicate and code-path inspection; the server module integration suite was not run because importing its singleton service can initialize the model runtime.
- `git diff --check` reported two trailing-whitespace lines elsewhere in the already-modified `backend/api/server.py` working copy (lines 571 and 604); neither line was changed for this work.

## Production behavior regression evidence

Production dispatch remains enabled when `output_mode` is omitted or `PRODUCTION` and no evaluation ID is present. The new focused test confirms a production callback still fires; existing incident classification/correlation regression tests passed in the 14-test run. The new API guard precedes production append, persistence, notification, and incident side effects. Direct incident-correlator calls are separately gated, so bypassing the API does not convert tagged evaluation outputs into incidents.

## Frozen hashes

All seven artifacts match their established Batch 25 SHA-256 values (7/7):

| Artifact | SHA-256 | Result |
|---|---|---|
| V4 `aegis_live_xgboost_baseline.pkl` | `C0E5AEE8E9E1726F12BFC19E98C440FE44FE278D1796D5866FE2A2488D458A56` | MATCH |
| V4 `temperature_scaling.json` | `C3AC0662CA896D06D4B9F62035A637FE04C8E01C00C466790D2EC0F03E581F99` | MATCH |
| V4 `ood_parameters.pkl` | `579063CFC4DDA3E396AB0D026198F14F8EF593EBF9910F290157AA0F669712C3` | MATCH |
| V4 `abstention_config.json` | `7D49F05BEF592CB838EF69F813A1C297BECA9E0A007BBAD1AAD6F5F0F4AC397F` | MATCH |
| V6.3 `config_b_model.pkl` | `09AFB03CBE74201E4E3768E367FB35C70D727BC41A5B9FFC6180C3305FDF1F84` | MATCH |
| V6.3 `temperature_scaling_b.json` | `2CEDF27975B4C28ABA2BBF52B84B0A5C5C49CAAF1EF3FA861AC6660885AA3B6F` | MATCH |
| V6.3 `abstention_config_b.json` | `E30BC4571C5045C3374AAE930145821F1277E1C9ABA56E4F419F7B10B39E2039` | MATCH |

## Remaining gate

Data-source authorization remains outstanding. Before any controlled evaluation, obtain and record the exact source/version, owner authorization, explicit unsealed confirmation, explicit exclusion of Partition C and Kitsune SSDP holdout, label/adjudication provenance, permitted purpose, and retention constraints. This report does not approve or execute evaluation.

No PCAP scans, dataset access/generation, training, calibration, threshold tuning, classifier loading/inference, or sealed holdout access occurred. Frozen artifacts were not modified. `automated_blocking_allowed=False`.

SHADOW OUTPUT ISOLATION VERIFIED

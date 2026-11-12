# AEGIS Batch 29 — Controlled Shadow Evaluation Preflight

**Review mode:** Focused, read-only inspection. No dataset was accessed. No classifier was loaded and no inference was run.

## Decision

**CONDITIONALLY READY — CODE OR SAFETY BLOCKER**

The frozen V4 feature isolation and 50-packet eligibility checks are supported by the inspected implementation and Batch 28/28.1 evidence. Do not start shadow evaluation yet: the V4 runtime result has no explicit evaluation/shadow-only marker, and no data-source authorization has been supplied. These gates must close before any evaluation run.

## V4 isolation findings

- `v4_runtime.py` constructs a feature mapping from 24 flow features plus five topology features and calls the frozen inference interface only inside `_process_completed_window()`.
- That method returns before feature classification unless `len(window.packets) == 50`. Timeout, EOF, and capacity paths route non-50 windows to diagnostic records (`INCOMPLETE_WINDOW` or `INELIGIBLE_WINDOW`) with `v4_eligible=False`; they do not append to `detection_records` or call the inference interface. Batch 28.1 reports focused tests of 1-, 49-, and 50-packet cases, including timeout, EOF, and capacity.
- `v4_inference.py` calls `validate_29_features()`, which selects and orders `CANONICAL_29_FEATURES` before model, calibration, and OOD processing. Runtime passes the 29-feature mapping. Extra input columns, if supplied directly, are discarded by this validator.
- `CANONICAL_39_FEATURES` is defined as the 29 plus 10 context features. Relevant references found are its schema validator and separate V4.3 dataset/training modules. No reference from the V4 runtime or V4 inference caller to `CANONICAL_39_FEATURES` was found. No 39-feature vector can enter the V4 model/calibration/OOD matrix through the inspected code path; V4 receives exactly the ordered 29.
- Diagnostic partial records are held in `incomplete_window_records`, distinct from `detection_records`. Their status and `v4_eligible=False` distinguish them in the runtime summary. The evaluation harness's inspected V4 path consumes detections emitted by the pipeline, not the incomplete diagnostic list.

**No feature-vector or partial-window bypass was found in the inspected V4 callers.** This is a scoped code trace, not an exhaustive repository security audit.

## Shadow-mode safety findings

- `DetectionResultV4` sets and serializes `automated_blocking_allowed=False`; the V4 inference result does likewise. Inspected V4 runtime/inference code is detection-only and contains no firewall, iptables/nftables, or eBPF enforcement call.
- The evaluation harness documents frozen zero-shot evaluation and checks blocking flags; it creates evaluation records/results separately from the runtime's incident-oriented result type. However, `DetectionResultV4` has no `SHADOW_ONLY`/evaluation-mode field or run identifier. The inspected harness's V4 result rows contain class/status/blocking fields but no explicit shadow-vs-production provenance tag. **This is an unresolved isolation/labeling gate**: before execution, evaluation outputs must be placed in a dedicated evaluation namespace and tagged with run ID and `SHADOW_ONLY`, with a verified no-ingestion path into production incident handling.
- The inspected V4 runtime has no training, recalibration, threshold-tuning, artifact-save, or model-promotion entry point. Inference loads the model/calibration/OOD/abstention artifacts. For the future run, those files must be mounted/read-only or otherwise protected and their seven hashes checked before and after. This review did not execute the harness or test filesystem permissions.
- Partial/ineligible records remain diagnostic/advisory only and must be excluded from confirmed V4 classification metrics.

## Required data-source authorization checklist

No authorization is assumed. Complete and retain evidence for every item before opening any source:

- [ ] Exact dataset/source identity, version, and immutable source identifier/hash.
- [ ] Owner and written authorization reference covering this evaluation purpose.
- [ ] Explicit source-owner confirmation that the selected portion is unsealed.
- [ ] Explicit confirmation that Partition C and the Kitsune SSDP holdout are excluded.
- [ ] Whether independent adjudicated labels exist; identify label owner, provenance, and known ambiguity.
- [ ] Permitted evaluation purpose, access scope, retention period, storage location, and deletion requirements.

## Proposed run manifest (unpopulated)

Do not replace `PENDING` with an assumption. Store this manifest separately from production incident records.

```yaml
run_id: PENDING
run_date_utc: PENDING
mode: SHADOW_ONLY
production_incident_ingestion: DISABLED_AND_VERIFIED
source:
  dataset_identity_version_hash: PENDING
  owner_authorization_reference: PENDING
  unsealed_confirmation_reference: PENDING
  partition_c_excluded_confirmation: PENDING
  kitsune_ssdp_holdout_excluded_confirmation: PENDING
  permitted_purpose: PENDING
  retention_constraints: PENDING
  independently_adjudicated_labels: PENDING
  label_adjudication_provenance: PENDING
artifacts_sha256:
  v4_model: C0E5AEE8E9E1726F12BFC19E98C440FE44FE278D1796D5866FE2A2488D458A56
  v4_temperature: C3AC0662CA896D06D4B9F62035A637FE04C8E01C00C466790D2EC0F03E581F99
  v4_ood: 579063CFC4DDA3E396AB0D026198F14F8EF593EBF9910F290157AA0F669712C3
  v4_abstention: 7D49F05BEF592CB838EF69F813A1C297BECA9E0A007BBAD1AAD6F5F0F4AC397F
  v6_3_model: 09AFB03CBE74201E4E3768E367FB35C70D727BC41A5B9FFC6180C3305FDF1F84
  v6_3_temperature: 2CEDF27975B4C28ABA2BBF52B84B0A5C5C49CAAF1EF3FA861AC6660885AA3B6F
  v6_3_abstention: E30BC4571C5045C3374AAE930145821F1277E1C9ABA56E4F419F7B10B39E2039
canonical_feature_schema_version: V4_CANONICAL_29_ORDERED
parser:
  accepted_packets_by_reason: PENDING
  rejected_packets_by_reason: PENDING
timestamps:
  regression_count: PENDING
  reorder_count: PENDING
  late_packet_count: PENDING
  overflow_count: PENDING
windows:
  full_eligible_50_packet_count: PENDING
  partial_ineligible_count: PENDING
  ineligible_other_count: PENDING
  protocol_direction_window_size_breakdown: PENDING
outputs:
  confidence_records: PENDING
  ood_records: PENDING
  abstention_records: PENDING
  evaluation_namespace_and_run_tag: PENDING
labels_metrics:
  label_availability: PENDING
  adjudication_provenance: PENDING
  metrics_and_confidence_intervals: PENDING
  exclusions_and_missing_labels: PENDING
integrity:
  pre_run_hashes_verified: PENDING
  post_run_hashes_verified: PENDING
  artifacts_read_only_verified: PENDING
  automated_blocking_allowed: false
```

## Unresolved gates and closure evidence

1. **Evaluation output isolation/tagging (code/safety):** Provide a dedicated evaluation-only output namespace, explicit `SHADOW_ONLY` plus run ID provenance, and evidence that no evaluation output can be ingested as a production incident. Verify with a non-inference synthetic plumbing check before data use.
2. **Artifact immutability (run safety):** Establish read-only artifact access and pre/post hashes for all seven files; demonstrate no training, calibration, threshold, or promotion step is invoked by the selected run procedure.
3. **Source authorization (data gate):** Complete all checklist items with documentary references. No source may be opened until ownership, unsealed status, exclusions, purpose, and retention are approved.
4. **Evaluation outputs/labels (analysis gate):** Predeclare label adjudication and metric/exclusion rules. If independently adjudicated labels are absent, limit conclusions to operational telemetry and do not claim model accuracy or compatibility.

The code-level 29-feature and exact-50 gates are supported by Batch 28/28.1 and this scoped trace. Batch 28.1's targeted result was 49 passed, 0 failed; this preflight ran no tests. Synthetic builder/runtime parity is evidence of implementation agreement only and does **not** prove real-world feature-distribution compatibility.

**Safety confirmation:** No PCAP scans, real dataset access/generation, classifier loading/inference, training, calibration, threshold changes, frozen artifact modification, or sealed holdout access occurred. Automated blocking remains disabled.

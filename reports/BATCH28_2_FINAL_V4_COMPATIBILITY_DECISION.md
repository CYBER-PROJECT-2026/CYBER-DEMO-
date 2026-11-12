# AEGIS Batch 28.2 — Final Frozen V4 Compatibility Decision

**Review mode:** Read-only; current artifact hashes recalculated. No implementation or frozen artifacts modified.

## Executive decision

**CONDITIONALLY APPROVED — SPECIFIC GATES REQUIRED**

Frozen V4 is suitable only for a staged, non-blocking controlled evaluation after the gates below are documented and met. This is not an approval for general deployment, sealed holdout evaluation, or automated response. `automated_blocking_allowed=False` remains in force.

## Verified original training contract

The Batch 25 audit, checked-in builder/training source evidence summarized in Batch 28, model metadata, and Batch 26E flow identity investigation support this frozen contract:

- Canonical vector: 29 ordered features: 24 flow features followed by 5 topology features.
- Window identity: symmetric `(min(src_ip, dst_ip), max(src_ip, dst_ip), ip_proto)`; ports do not split flow windows. Topology separately uses a 5-tuple for a topology-only count excluded from the canonical vector.
- Windows: non-overlapping, exactly 50 packets. On a `>=2.0s` inactivity gap, only a full window is retained; builder EOF does not flush active partial windows.
- Packet size: captured Ethernet buffer length `len(buf)`, including Ethernet header and captured padding.
- Direction: `is_inbound = _is_local(dst_ip)` with default local subnet `192.168.0.0/16`.
- Topology: packet-global causal 1-second sliding window; the five canonical topology features are source concentration, source entropy, source/destination cardinality, and fan-in ratio as documented in Batch 28.

The exact training execution is not fully reproducible: metadata does not bind the model to a builder source hash or record parser rejection counts, fragment frequency, timestamp policy, or window-count histogram. Original parsing was less strict than corrected validation. Those are material distribution uncertainties.

## Compatibility evidence

- **Causality and timestamps:** Batch 26B1 review and Batch 26D1 audit record causal per-window topology snapshots, stable handling of equal timestamps, monotonic timestamp guards, bounded offline reorder, EOF drain, and fail-closed overflow tests. Batch 28.1 reran the directly related timestamp-ordering and temporal-causality selections.
- **Malformed packets:** Batch 28.1 places shared full Ethernet/IPv4/L4 validation before offline timestamp conversion and reorder insertion in builder/runtime replay. Rejected packets are counted and skipped before ordering, flow, topology, or feature state. Synthetic cases covered malformed frames before/between/after valid frames and invalid timestamp inputs.
- **Eligibility:** Batch 28.1 gates frozen classification to exactly 50 packets. Timeout, EOF, and capacity-flush partials remain diagnostic records with `INCOMPLETE_WINDOW` (or `INELIGIBLE_WINDOW` for other non-50 counts), `v4_eligible=False`; advisory eligibility cannot enable V4 classification.
- **Valid feature parity:** Batch 28 reports per-feature TCP and UDP deltas of `0.0` across all 29 features. Batch 28.1 reconfirms exact full-window parity for synthetic TCP and UDP streams, maximum absolute difference `0.0`.
- **Length/direction:** Batch 27 and 28 document preserved `len(buf)` semantics and destination-local inbound formula. Direction-class diagnostics do not alter canonical features. The known subnet/tap sensitivity and east-west collapse remain inherent frozen-contract risks.
- **Schema/artifacts:** Reports preserve the canonical 29 ordering. The current working tree additionally defines a separate `CANONICAL_39_FEATURES` context schema; available reports state it is additive and does not replace the frozen 29-feature vector. No claim is made that this review independently audits every caller of that separate schema. Current canonical V4 model, temperature, OOD, and abstention hashes match the recorded values below.

## Tests and evidence gaps

Existing evidence used; no test suite was rerun for this read-only review. Batch 28.1 reports `49 passed, 0 failed`, 3 upstream SHAP deprecation warnings, plus successful `py_compile`. Its focused cases include malformed-packet ordering/counters/state isolation, non-finite and regressing timestamps, full/partial windows, timeout/EOF/capacity paths, fail-closed overflow, and related timestamp/causality regressions. Synthetic 50-packet TCP/UDP builder/runtime parity is exact across all 29 features.

Synthetic parity proves implementation agreement for tested synthetic valid traffic; it does **not** prove real-world feature-distribution compatibility or model performance. No PCAP was scanned and no real dataset was generated. Training-run parser/timestamp provenance and the frequency of corrected-parser exclusions remain unavailable. The new separate 39-feature schema's runtime usage is outside this final V4 evidence review.

## Frozen artifact hash verification

Current SHA-256 values were recalculated and match the recorded Batch 25 values (7/7):

| Frozen artifact | SHA-256 | Result |
|---|---|---|
| V4 model `aegis_live_xgboost_baseline.pkl` | `C0E5AEE8E9E1726F12BFC19E98C440FE44FE278D1796D5866FE2A2488D458A56` | MATCH |
| V4 `temperature_scaling.json` | `C3AC0662CA896D06D4B9F62035A637FE04C8E01C00C466790D2EC0F03E581F99` | MATCH |
| V4 `ood_parameters.pkl` | `579063CFC4DDA3E396AB0D026198F14F8EF593EBF9910F290157AA0F669712C3` | MATCH |
| V4 `abstention_config.json` | `7D49F05BEF592CB838EF69F813A1C297BECA9E0A007BBAD1AAD6F5F0F4AC397F` | MATCH |
| V6.3 `config_b_model.pkl` | `09AFB03CBE74201E4E3768E367FB35C70D727BC41A5B9FFC6180C3305FDF1F84` | MATCH |
| V6.3 `temperature_scaling_b.json` | `2CEDF27975B4C28ABA2BBF52B84B0A5C5C49CAAF1EF3FA861AC6660885AA3B6F` | MATCH |
| V6.3 `abstention_config_b.json` | `E30BC4571C5045C3374AAE930145821F1277E1C9ABA56E4F419F7B10B39E2039` | MATCH |

## Controlled evaluation gates and protocol

Before any evaluation, record data authorization and confirm the selected data are unsealed and are not Partition C or the Kitsune SSDP holdout. Use an isolated, non-blocking shadow run with frozen artifacts and the corrected exact-50 eligibility gate. Do not retrain, fine-tune, recalibrate, tune thresholds, or enable automated blocking. Keep confirmed detections (eligible full 50-packet V4 samples) separate from advisory/diagnostic partial-window records; partials must never be represented as V4 classifications.

Collect dataset/source identity and authorization, parser acceptance/rejection counts by reason, timestamp regression/reorder/overflow telemetry, eligible and partial-window counts, protocol/direction/window-size breakdowns, feature finiteness/schema/order checks, and model/OOD/abstention outputs with frozen artifact hashes. Where independently adjudicated labels are authorized, report confusion matrix, precision, recall, F1, false-positive rate, abstention/unknown rate, OOD rate, and confidence intervals, stratified by protocol, direction class, and capture/source. Document exclusions and missing labels; do not infer performance from unlabeled traffic. Review distribution shifts and parity/provenance limitations before any broader use.

**Safety:** No PCAP scans, dataset generation, training, classifier loading, inference, or sealed holdout access occurred in this batch. Automated blocking remains disabled.

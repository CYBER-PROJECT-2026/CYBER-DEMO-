# AEGIS Batch 28.1 — Compatibility Blocker Fixes

**Date:** 2026-11-12  
**Decision:** **READY FOR FINAL COMPATIBILITY REVIEW**  
**Scope:** Resolve only the malformed-packet ordering and sub-capacity V4 eligibility blockers from Batch 28. This decision does not approve controlled evaluation.

## Root causes

### Malformed packets and offline ordering

Runtime replay and builder ordering used `classify_ipv4_ethernet_frame()` before buffering. That helper checked Ethernet framing, EtherType, and a minimum IPv4 prefix, but did not perform the shared full IPv4/L4 validation. A candidate with an invalid IPv4 total length or truncated transport header could therefore be timestamp-converted and inserted into the reorder buffer before later parsing rejected it. Its timestamp could advance the watermark or use capacity.

### Runtime partial windows

Runtime timeout and EOF paths used the configurable `min_packets_to_extract` threshold (default 2), and capacity completion went directly to `_process_completed_window()`. That allowed packet counts 2–49, and non-50 configured capacities, to reach frozen V4 inference even though the training builder emits 50-packet windows.

## Behavior corrected

### Full packet validation before ordering

- Runtime `ingest_packet()` and bounded `replay_packet_stream()` now run `parse_ipv4_ethernet_frame()` before timestamp conversion/ordering. The builder's `_iter_ordered_packets()` does the same in both strict and bounded modes.
- Invalid, truncated, unsupported, VLAN, and non-IPv4 frames are counted by their parser rejection reason and skipped before reorder-buffer insertion, timestamp guards, flow windows, topology tracking, direction counters, or features.
- Valid frames with non-finite timestamps retain the existing non-finite timestamp telemetry. They do not occupy reorder capacity or advance accepted timestamp state.
- Valid-packet ordering, bounded reorder behavior, and fail-closed overflow behavior remain intact. A final exact-50-packet gate in `_process_completed_window()` prevents accidental direct calls from bypassing eligibility.

### Explicit 50-packet eligibility

- Frozen V4 classification now requires **exactly 50 packets**. Only those windows increment `windows_generated`, enter `detection_records`, or call the classifier interface.
- Timeout, EOF, and capacity-flush paths retain non-50-packet windows in `incomplete_window_records`, including their computed feature values and provenance. Records expose `status` (`INCOMPLETE_WINDOW` for fewer than 50; `INELIGIBLE_WINDOW` for other non-50 counts), `v4_eligible=False`, and a reason such as `incomplete_window_requires_50_packets`.
- The previous `min_packets_to_extract` setting now marks `advisory_eligible` on diagnostic records only; it cannot authorize frozen V4 classification. Summary statistics expose diagnostic statuses and reasons.
- Full 50-packet behavior and `automated_blocking_allowed=False` remain unchanged. The canonical 29 features were not modified.

## Files changed

- `aegis_live/build_v4_dataset.py`
- `aegis_live/v4_runtime.py`
- `tests/test_aegis_live_v4_compatibility_blockers.py` (new focused synthetic tests)
- `tests/test_aegis_live_v4_timestamp_ordering.py` (updated packet-count expectations and preserved fail-closed regression)
- `tests/test_aegis_live_v4_temporal_causality.py` (partial-window snapshots now checked as diagnostics)
- `reports/BATCH28_1_COMPATIBILITY_BLOCKER_FIXES.md` (this report)

## Tests and exact results

`python -m py_compile` passed for both modified runtime/builder modules and all modified/new tests.

Targeted pytest selection: **49 passed, 0 failed**, with 3 upstream SHAP deprecation warnings. It included:

- Malformed frames before, between, and after valid packets, with finite future/regressing timestamps and NaN/Inf timestamps.
- Ethernet-short, IPv4-short, invalid IHL/total length, truncated IPv4 payload, truncated TCP/UDP, VLAN, and non-IPv4 frames.
- Proof that invalid frames do not consume reorder capacity, overflow the buffer, advance timestamp ordering, or enter flow/topology aggregation; valid packets remain chronological.
- Valid-packet reorder overflow remains fail-closed for both runtime and builder.
- Runtime windows of 1 and 49 packets at EOF, a 49-packet timeout followed by a 1-packet EOF window, and a 49-packet capacity flush: all are diagnostic and never call the classifier stub.
- Full 50-packet TCP and UDP windows: both remain eligible; builder/runtime feature values match exactly for all 29 features (maximum absolute difference `0.0`). Inference was replaced with a deterministic stub for this eligibility-path test.
- Directly related timestamp-ordering and temporal-causality regressions, including timeout/EOF snapshots and full-window builder/runtime parity.

## Frozen hash comparison

All seven current SHA-256 values match the recorded Batch 25 values:

| Frozen artifact | SHA-256 | Result |
|---|---|---|
| `models/aegis_live_v4/aegis_live_xgboost_baseline.pkl` | `C0E5AEE8E9E1726F12BFC19E98C440FE44FE278D1796D5866FE2A2488D458A56` | unchanged |
| `models/aegis_live_v4/temperature_scaling.json` | `C3AC0662CA896D06D4B9F62035A637FE04C8E01C00C466790D2EC0F03E581F99` | unchanged |
| `models/aegis_live_v4/ood_parameters.pkl` | `579063CFC4DDA3E396AB0D026198F14F8EF593EBF9910F290157AA0F669712C3` | unchanged |
| `models/aegis_live_v4/abstention_config.json` | `7D49F05BEF592CB838EF69F813A1C297BECA9E0A007BBAD1AAD6F5F0F4AC397F` | unchanged |
| `models/aegis_live_v6_3/config_b_model.pkl` | `09AFB03CBE74201E4E3768E367FB35C70D727BC41A5B9FFC6180C3305FDF1F84` | unchanged |
| `models/aegis_live_v6_3/temperature_scaling_b.json` | `2CEDF27975B4C28ABA2BBF52B84B0A5C5C49CAAF1EF3FA861AC6660885AA3B6F` | unchanged |
| `models/aegis_live_v6_3/abstention_config_b.json` | `E30BC4571C5045C3374AAE930145821F1277E1C9ABA56E4F419F7B10B39E2039` | unchanged |

## Safety confirmation

No PCAPs were scanned. No existing real dataset was read or generated. No training, fine-tuning, classifier loading, or real inference occurred. No sealed Partition C or Kitsune SSDP holdout was opened. Automated blocking remained disabled. Model, calibration, OOD artifacts, thresholds, and canonical feature definitions were not changed.

**Final decision: READY FOR FINAL COMPATIBILITY REVIEW**

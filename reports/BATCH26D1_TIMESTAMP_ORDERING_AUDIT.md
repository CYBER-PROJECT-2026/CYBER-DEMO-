# BATCH 26D.1 — Independent Timestamp Ordering Audit

**Date:** 2026-11-12  
**Audit Mode:** Read-Only Independent Verification (0 source/test files modified)  
**Target Implementation:** [BATCH26D_TIMESTAMP_ORDERING_IMPLEMENTATION.md](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/reports/BATCH26D_TIMESTAMP_ORDERING_IMPLEMENTATION.md)  
**Policy Reference:** [BATCH26C_TIMESTAMP_ORDERING_POLICY.md](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/reports/BATCH26C_TIMESTAMP_ORDERING_POLICY.md)

---

## Executive Decision

**Final Decision: `APPROVED FOR BATCH 26E`**

Independent code inspection and deterministic synthetic verification confirm that the Batch 26D two-tier timestamp-ordering architecture (zero-buffer Policy A live default, bounded 250 ms min-heap Policy B offline PCAP replay and dataset builder, and shared strict chronological core guards) is sound, deterministic, fail-closed on overflow, free of confirmed defects, and preserves builder/runtime 29-feature parity. Five minor `DESIGN RISK` observations and two scope-bounded `INSUFFICIENT EVIDENCE` items are documented below for engineering awareness.

---

## 1. `BoundedReorderBuffer` Audit

Inspected [`BoundedReorderBuffer`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/packet_window.py#L68-L179) and [`ReorderBufferOverflowError`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/packet_window.py#L18-L19).

| Audit Check | Code Reference | Classification | Evidence & Analysis |
| :--- | :--- | :--- | :--- |
| Heap ordering uses `(timestamp, arrival_sequence)` | [`packet_window.py#L82, L163-L164`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/packet_window.py#L163-L164) | `CONFIRMED CORRECT` | Entries are pushed onto `self._heap` as `(ts, self._seq, payload)` with monotonically incrementing `self._seq`. |
| Equal timestamps are deterministic | [`packet_window.py#L163-L164, L170-L178`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/packet_window.py#L163-L178) | `CONFIRMED CORRECT` | `test_02_equal_timestamps_stable_arrival_ordering` and synthetic probe B/EOF test verified exact FIFO tie preservation across mixed timestamps (`['a0', 'a1', 'a2', 'x0', 'x1', 'x2']`) and sequential post-watermark arrival at equal `ts`. |
| Watermark calculation (`max_seen_ts - lateness_sec`) | [`packet_window.py#L134-L136`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/packet_window.py#L134-L136) | `CONFIRMED CORRECT` | `max_seen_ts` advances monotonically; `watermark_ts` tracks `max_seen_ts - lateness_sec`. |
| Packets at the watermark released correctly | [`packet_window.py#L140-L154`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/packet_window.py#L140-L154) | `CONFIRMED CORRECT` | Items at `ts <= watermark_ts + 1e-9` drain from heap first (`L140-L144`), and incoming items at `ts <= watermark_ts + 1e-9` emit immediately when `not self._heap or ts < self._heap[0][0]` (`L147-L154`). Verified at exact `10.25 - 0.25 == 10.00` equality (`(10.00, 'a')` released immediately). |
| Exactly 10,000 buffered packets handled | [`packet_window.py#L75, L156-L168`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/packet_window.py#L156-L168) | `CONFIRMED CORRECT` | `test_08_exactly_10000_buffered_packets_succeeds` buffered 10,000 reversed packets inside `[10.0, 10.20]` (`max_buffer_occupancy == 10000`) and drained all 10,000 in monotonic order. |
| Packet 10,001 causes explicit fail-closed overflow | [`packet_window.py#L156-L161`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/packet_window.py#L156-L161) | `CONFIRMED CORRECT` | `if len(self._heap) >= self.max_packets:` increments `overflow_count` and raises `ReorderBufferOverflowError` before pushing packet 10,001 (`test_09_buffer_overflow_fails_closed`). |
| Overflow cannot leave partial replay/build reported as successful | [`v4_runtime.py#L513-L515`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/v4_runtime.py#L513-L515), [`build_v4_dataset.py#L329-L331`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/build_v4_dataset.py#L329-L331) | `CONFIRMED CORRECT` / `DESIGN RISK` | Both `replay_packet_stream` and `extract_packet_stream_windows` re-raise `ReorderBufferOverflowError` without returning success or writing CSV/manifest outputs (`CONFIRMED CORRECT`). **Minor `DESIGN RISK`:** if windows were already emitted earlier in a replay stream before a later burst overflows, `AegisLiveV4Pipeline.detection_records`, `windows_generated`, and `active_windows` retain that partial pre-overflow state if caller catches `ReorderBufferOverflowError` without calling `pipeline.reset()`. |
| EOF drains remaining packets once; empty buffer safe | [`packet_window.py#L170-L178`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/packet_window.py#L170-L178) | `CONFIRMED CORRECT` | Verified `test_07_end_of_stream_buffer_draining` and empty-buffer `drain() == []` with `emitted_count == 0`. |
| Late arrivals behind emitted timestamp/watermark rejected & counted | [`packet_window.py#L126-L131, L107-L116`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/packet_window.py#L126-L131) | `CONFIRMED CORRECT` | Rejects `earlier_than_last_emitted` and `behind_emitted_watermark`, increments `rejected_late_count`, and caps `rejected_events` at 100 entries (`L109`) for bounded diagnostic memory (`test_05`, `test_06`). |
| No unbounded memory growth | [`packet_window.py#L109, L156`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/packet_window.py#L109) | `CONFIRMED CORRECT` | `_heap` bounded by `max_packets` (`10000`) and `rejected_events` capped at `100`. |
| Floating-point `1e-9` watermark tolerance boundary | [`packet_window.py#L129, L140, L147`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/packet_window.py#L129) | `DESIGN RISK` | `L129` rejects `ts < watermark_ts - 1e-9`, so an un-emitted packet falling within `1e-9` s behind `watermark_ts` can enter the buffer cleanly without overflowing/dropping due to binary float subtraction error (`10.30 - 0.25`). At libpcap microsecond resolution this prevents false boundary drops, while `L126` (`ts < last_emitted_ts`) preserves strict monotonicity if a later packet already emitted past it. |

---

## 2. Strict Monotonic Guard Audit

Inspected [`MonotonicTimestampGuard`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/packet_window.py#L22-L66), [`FlowWindow.add_packet`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/packet_window.py#L221-L247), and [`NetworkTopologyTracker.add_packet`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/network_topology_features.py#L222-L278).

| Audit Check | Code Reference | Classification | Evidence & Analysis |
| :--- | :--- | :--- | :--- |
| Equal timestamps accepted; earlier rejected | [`packet_window.py#L46-L65`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/packet_window.py#L46-L65) | `CONFIRMED CORRECT` | Rejects iff `self.last_accepted_ts is not None and ts < self.last_accepted_ts`; accepts `ts == last_accepted_ts` monotonically (`test_02`, `test_12`). |
| Rejected packets never enter `FlowWindow` or `NetworkTopologyTracker` | [`v4_runtime.py#L362-L380`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/v4_runtime.py#L362-L380), [`packet_window.py#L227-L231`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/packet_window.py#L227-L231), [`network_topology_features.py#L232-L236`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/network_topology_features.py#L232-L236) | `CONFIRMED CORRECT` | `ingest_packet()` returns `[]` at `L367` before constructing `NetworkTopologyPacket` (`L370`) or `PacketRecord` (`L383`). Direct module calls also raise `ValueError` prior to mutating `FlowWindow.packets` or `NetworkTopologyTracker.packet_queue` (`test_10`, `test_11`, `test_12`). |
| Guard state updated only after acceptance | [`packet_window.py#L48-L65`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/packet_window.py#L48-L65) | `CONFIRMED CORRECT` | Rejected packets return early at `L61` without touching `last_accepted_ts` or `accepted_count` (verified across consecutive replay calls in probe D). |
| Multiple guard instances / duplicate guard telemetry | [`v4_runtime.py#L155, L495-L507`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/v4_runtime.py#L495-L507), [`build_v4_dataset.py#L301-L338`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/build_v4_dataset.py#L301-L338) | `CONFIRMED CORRECT` | Pipeline maintains one persistent `self.timestamp_guard`. When `BoundedReorderBuffer` rejects a packet (`L499-L504` in runtime, `L320-L322` in builder), `continue` skips `guard.check_and_accept`, preventing double-counting; subsequent sequential un-reset `replay_packet_stream` calls are still protected by `self.timestamp_guard` (`probe D`). |
| Non-finite (`NaN`) timestamp handling | [`packet_window.py#L47-L48, L124-L135`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/packet_window.py#L47-L48) | `DESIGN RISK` | `ts < last_accepted_ts` evaluates `False` for `float('nan')`, so a synthetic Python `NaN` timestamp would latch `last_accepted_ts = nan` rather than rejecting. Standard PCAP `dpkt` readers emit finite floats, so exposure is restricted to unvalidated Python stream callers. |

---

## 3. Integration Audit

Traced all public ingestion paths in [`v4_runtime.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/v4_runtime.py) and [`build_v4_dataset.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/build_v4_dataset.py):

| Route | Line Reference | Classification | Verification Result |
| :--- | :--- | :--- | :--- |
| Direct `ingest_packet(ts, buf)` | [`v4_runtime.py#L305-L435`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/v4_runtime.py#L305-L435) | `CONFIRMED CORRECT` | Always operates with zero buffering (`MonotonicTimestampGuard` at `L362` only, even if `self.reorder_window_sec > 0.0`), emitting full windows immediately on packet 50 (`test_13`). |
| `replay_packet_stream(...)` | [`v4_runtime.py#L472-L528`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/v4_runtime.py#L472-L528) | `CONFIRMED CORRECT` | Defaults to `self.reorder_window_sec` (`0.0s` live Policy A), uses `BoundedReorderBuffer` when `lateness > 0.0`, drains remaining items at EOF unless stopped early by `max_windows`, and finishes with `flush(current_time=flush_base_ts + 5.0, force_all=True)`. |
| `replay_pcap(...)` | [`v4_runtime.py#L530-L578`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/v4_runtime.py#L530-L578) | `CONFIRMED CORRECT` | Defaults to `reorder_window_sec=0.25, max_reorder_packets=10000` (verified via `inspect.signature`) and delegates to `replay_packet_stream()`. |
| Dataset builder (`_iter_ordered_packets` + `extract_packet_stream_windows`) | [`build_v4_dataset.py#L295-L550`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/build_v4_dataset.py#L295-L550) | `CONFIRMED CORRECT` | Defaults to `reorder_window_sec=0.25, max_reorder_packets=10000` (`L106-L107`), applies `BoundedReorderBuffer` + `MonotonicTimestampGuard` prior to Train/Val/Test partition quarantine state checks (`L376-L416`) and feature extraction (`L458-L550`), preventing out-of-order packets from bypassing quarantine boundaries. |
| Direct core bypass protection | [`packet_window.py#L227`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/packet_window.py#L227), [`network_topology_features.py#L232`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/network_topology_features.py#L232) | `CONFIRMED CORRECT` | Direct calls to `FlowWindow.add_packet()` or `NetworkTopologyTracker.add_packet()` raise `ValueError` on regressions (`test_12`). |
| Pre-parse buffering vs post-parse guard (`replay_packet_stream` offline vs `ingest_packet` live) | [`v4_runtime.py#L314-L362, L497-L504`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/v4_runtime.py#L497-L504) | `DESIGN RISK` | In offline mode (`lateness > 0.0`), `BoundedReorderBuffer.push()` runs *before* Ethernet/IPv4/L4 parsing (matching `build_v4_dataset.py#_iter_ordered_packets`), whereas in live `ingest_packet()`, Ethernet/IPv4/L4 validation runs *before* `MonotonicTimestampGuard`. Consequently: (a) a late non-IPv4 ARP packet increments `non_ipv4_packet` live vs `out_of_order_timestamp` offline (verified in probe), and (b) non-IPv4 bursts count toward the `10000`-packet heap cap offline. |
| Summary telemetry `reorder_window_sec` reporting | [`v4_runtime.py#L606-L615`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/v4_runtime.py#L606-L615) | `DESIGN RISK` | `get_summary_statistics()["timestamp_guard"]["reorder_window_sec"]` reports `self.reorder_window_sec` from `__init__` (default `0.0`), not a per-call `0.25` override passed to `replay_pcap()` or `replay_packet_stream()`. |

---

## 4. Synthetic Verification Results

Executed read-only against existing suites (with `AEGISLiveV4Inference` stubbed in runtime unit tests so **zero frozen model inference** executed) and standalone inline Python synthetic probes (zero files edited):

1. **Pytest Verification Run (`python -m pytest`, 49 passed, 0 failed, 0 skipped in 13.30s):**
   - [test_aegis_live_v4_timestamp_ordering.py](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/tests/test_aegis_live_v4_timestamp_ordering.py): `15 passed` (`test_01` through `test_15`, covering equal timestamps, 50 ms jitter live vs offline, 250 ms buffering, >250 ms & watermark rejection, EOF drain, 10,000 exact cap, 10,001 overflow fail-closed in buffer/runtime/builder, topology & FlowWindow isolation, live zero-buffer immediate emit, builder/runtime 29-feature parity, and Batch 26B snapshot causality after reordering).
   - [test_aegis_live_v4_temporal_causality.py](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/tests/test_aegis_live_v4_temporal_causality.py): `18 passed`.
   - [test_aegis_live_network_topology.py](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/tests/test_aegis_live_network_topology.py): `8 passed`.
   - [test_aegis_live_v4_dataset.py](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/tests/test_aegis_live_v4_dataset.py): `8 passed`.

2. **Additional Read-Only Synthetic Edge-Case Probes (`python -c`, `ALL_AUDIT_PROBES_PASSED` & `SYNTHETIC_AUDIT_CHECKS_2_PASSED`):**
   - Empty `BoundedReorderBuffer.drain()` returns `[]` with `emitted_count == 0` and `accepted_count == 0`.
   - Exact watermark boundary `10.25 - 0.25 == 10.00` releases `(10.00, 'a')` immediately; subsequent equal-timestamp packet `10.00` also emits cleanly after `10.00` was emitted.
   - EOF drain across interleaved equal timestamps emits `['a0', 'a1', 'a2', 'x0', 'x1', 'x2']` deterministically.
   - Overflow propagates `ReorderBufferOverflowError` out of both `replay_packet_stream()` and `extract_packet_stream_windows()` without returning success (`overflows=1`, `reorder_buffer_overflow=1`).
   - Chronological stream across `W=0.0`, `W=0.25`, and `LeakageSafeV4DatasetBuilder` yields exact equality across all `CANONICAL_29_FEATURES`.
   - Sequential un-reset `replay_packet_stream` calls maintain `timestamp_guard.last_accepted_ts == 20.1` and reject an earlier `10.0..10.1` replay without corrupting accepted state.

3. **Evidence Gaps Outside Synthetic Audit Scope (`INSUFFICIENT EVIDENCE`):**
   - Real Edge-IIoTset PCAP regression magnitude distribution ($\Delta t$) and non-IPv4 packet fraction inside 250 ms bursts cannot be measured without prohibited PCAP scans.
   - Hardware NIC live capture latency and multi-queue RSS timestamp skew cannot be verified in offline unit execution.

---

## 5. Compatibility Review

| Invariant / Semantic | Code / Test Evidence | Classification |
| :--- | :--- | :--- |
| Canonical 29 feature names & ordering unchanged | `test_29_feature_schema_and_order`, `test_11`, `test_14` ([`v4_runtime.py#L234-L241`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/v4_runtime.py#L234-L241)) | `CONFIRMED CORRECT` |
| Existing 3-tuple flow identity `(min_ip, max_ip, proto)` unchanged | [`v4_runtime.py#L360-L361`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/v4_runtime.py#L360-L361), [`build_v4_dataset.py#L480-L481`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/build_v4_dataset.py#L480-L481) | `CONFIRMED CORRECT` |
| Packet-length semantics (`len(buf)`) unchanged | [`v4_runtime.py#L377, L385`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/v4_runtime.py#L377), [`build_v4_dataset.py#L466, L473`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/build_v4_dataset.py#L466) | `CONFIRMED CORRECT` |
| Directionality (`_is_local(dst_ip)`) unchanged | [`v4_runtime.py#L382`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/v4_runtime.py#L382), [`build_v4_dataset.py#L470`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/build_v4_dataset.py#L470) | `CONFIRMED CORRECT` |
| Chronological input produces unchanged feature values | Synthetic probe C (`d0 == d25 == w25`), `test_01`, `test_all_80k_v4_rows_temporal_invariants` | `CONFIRMED CORRECT` |
| Jittered input live vs offline drift documented | [`BATCH26D_TIMESTAMP_ORDERING_IMPLEMENTATION.md#L115`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/reports/BATCH26D_TIMESTAMP_ORDERING_IMPLEMENTATION.md#L115), `test_03` | `CONFIRMED CORRECT` |
| Frozen model SHA-256 checksums preserved (`7 / 7` files verified) | `Get-FileHash -Algorithm SHA256` on `models/aegis_live_v4/*` and `models/aegis_live_v6_3/*` | `CONFIRMED CORRECT` |

---

## Safety Footer

- Files modified = `0` (0 source or test files modified; audit report [BATCH26D1_TIMESTAMP_ORDERING_AUDIT.md](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/reports/BATCH26D1_TIMESTAMP_ORDERING_AUDIT.md) created).
- PCAP scans = `NO`.
- Dataset generation = `NO`.
- Training = `NO`.
- Inference = `NO` (`AEGISLiveV4Inference` mocked in runtime tests and synthetic probes).
- Frozen hashes changed = `0` (all 7 V4 and V6.3 SHA-256 hashes verified unchanged via `Get-FileHash -Algorithm SHA256`):
  - `models/aegis_live_v4/aegis_live_xgboost_baseline.pkl`: `C0E5AEE8E9E1726F12BFC19E98C440FE44FE278D1796D5866FE2A2488D458A56`
  - `models/aegis_live_v4/temperature_scaling.json`: `C3AC0662CA896D06D4B9F62035A637FE04C8E01C00C466790D2EC0F03E581F99`
  - `models/aegis_live_v4/ood_parameters.pkl`: `579063CFC4DDA3E396AB0D026198F14F8EF593EBF9910F290157AA0F669712C3`
  - `models/aegis_live_v4/abstention_config.json`: `7D49F05BEF592CB838EF69F813A1C297BECA9E0A007BBAD1AAD6F5F0F4AC397F`
  - `models/aegis_live_v6_3/config_b_model.pkl`: `09AFB03CBE74201E4E3768E367FB35C70D727BC41A5B9FFC6180C3305FDF1F84`
  - `models/aegis_live_v6_3/temperature_scaling_b.json`: `2CEDF27975B4C28ABA2BBF52B84B0A5C5C49CAAF1EF3FA861AC6660885AA3B6F`
  - `models/aegis_live_v6_3/abstention_config_b.json`: `E30BC4571C5045C3374AAE930145821F1277E1C9ABA56E4F419F7B10B39E2039`
- Automated blocking = `NO`.


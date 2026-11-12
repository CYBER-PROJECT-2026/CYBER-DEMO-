# BATCH 26D — AEGIS V4 Timestamp Ordering Implementation Report

**Date:** 2026-11-12  
**Status:** IMPLEMENTED AND VERIFIED (Deterministic Synthetic Suite)  
**Policy Reference:** [BATCH26C_TIMESTAMP_ORDERING_POLICY.md](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/reports/BATCH26C_TIMESTAMP_ORDERING_POLICY.md)

---

## 1. Architecture & Data Flow

Batch 26D implements the two-tier timestamp-ordering architecture approved in Batch 26C across both the live/replay runtime pipeline and the offline dataset builder, without modifying any V4 feature definitions, flow identity keys, packet length parsing, directionality logic, or frozen model artifacts:

```text
[Packet Stream Input]
   │
   ├── Live Capture / Default Replay (reorder_window_sec = 0.0)
   │      └── Zero buffering latency → immediate handoff
   │
   └── Offline PCAP Replay / Dataset Builder (reorder_window_sec = 0.25, max_packets = 10000)
          └── BoundedReorderBuffer (min-heap on (timestamp, arrival_sequence))
                 ├── Watermark = max_seen_timestamp - 0.25 s
                 ├── Emits entries <= watermark in stable arrival order
                 ├── Rejects late entries strictly earlier than last_emitted_ts
                 ├── Fails closed with ReorderBufferOverflowError if len(heap) > 10000
                 └── Drains remaining heap chronologically at EOF
   │
   ▼
[Strict Monotonic Processing Core]
   ├── MonotonicTimestampGuard (ts >= last_accepted_ts)
   │      ├── Accepted → updates NetworkTopologyTracker & FlowWindow
   │      └── Rejected (ts < last_accepted_ts) → telemetry + reliability metadata only
   │                                             (never enters tracker or FlowWindow)
   ├── NetworkTopologyTracker.add_packet (direct guard: ts >= last_ts, raises ValueError)
   └── FlowWindow.add_packet (direct guard: pkt.timestamp >= last_timestamp, raises ValueError)
```

---

## 2. Exact Files & Functions Changed

| File | Symbols Added / Updated | Responsibility |
| :--- | :--- | :--- |
| [packet_window.py](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/packet_window.py) | `ReorderBufferOverflowError`, `MonotonicTimestampGuard`, `BoundedReorderBuffer`, `FlowWindow.ooo_rejected_count`, `FlowWindow.affected_by_ooo_rejection`, `FlowWindow.mark_ooo_rejection()`, `FlowWindow.add_packet()` | Shared strict chronological guard, bounded min-heap reorder buffer, fail-closed overflow exception, flow reliability tracking, and direct out-of-order rejection inside `FlowWindow`. |
| [network_topology_features.py](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/network_topology_features.py) | `NetworkTopologyTracker.add_packet()` | Direct core enforcement (`ts >= self.last_ts`), raising `ValueError` before mutating `self.last_ts` or FIFO `packet_queue`. |
| [v4_runtime.py](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/v4_runtime.py) | `WindowProvenanceRecordV4` (`affected_by_ooo_rejection`, `ooo_rejected_count`), `AegisLiveV4Pipeline` (`reorder_window_sec=0.0`, `max_reorder_packets=10000`), `ingest_packet()`, `replay_packet_stream()`, `replay_pcap()`, `get_summary_statistics()` | Integrates `MonotonicTimestampGuard` inside `ingest_packet()`, records `out_of_order_timestamp` diagnostic telemetry, propagates reliability metadata to provenance records, exposes `timestamp_guard` summary stats, and enables bounded 250 ms reordering in `replay_pcap()`. |
| [build_v4_dataset.py](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/build_v4_dataset.py) | `LeakageSafeV4DatasetBuilder` (`reorder_window_sec=0.25`, `max_reorder_packets=10000`), `_iter_ordered_packets()`, `extract_packet_stream_windows()` | Reorders offline packet streams via `BoundedReorderBuffer` and enforces `MonotonicTimestampGuard` prior to partition quarantine transitions and V4 feature extraction. |
| [test_aegis_live_v4_timestamp_ordering.py](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/tests/test_aegis_live_v4_timestamp_ordering.py) | `TestV4TimestampOrderingBatch26D` (`test_01` – `test_15`) | 15 deterministic synthetic regression tests covering monotonic core, watermark buffering, overflow fail-closed behavior, telemetry isolation, live/offline modes, builder/runtime parity, and causal snapshot preservation (with `AEGISLiveV4Inference` patched so zero inference runs). |

---

## 3. Watermark & Bounded Reorder Buffer Implementation

`BoundedReorderBuffer` in [packet_window.py](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/packet_window.py) implements bounded out-of-order recovery for offline replay and dataset construction:

- **Stable Tie-Breaking Key:** Buffered entries are pushed onto a Python `heapq` min-heap keyed by `(timestamp, arrival_sequence, payload)`. Equal timestamps preserve exact original stream arrival order deterministically.
- **Sliding Watermark:** As each packet arrives, `max_seen_timestamp` updates monotonically and the emission watermark advances to `watermark_ts = max_seen_timestamp - lateness_sec` (default `lateness_sec = 0.25` s / 250 ms).
- **Ordered Emission:** Every heap top with `ts <= watermark_ts` is popped and emitted cleanly in chronological `(timestamp, arrival_sequence)` order, updating `last_emitted_ts`.
- **EOF Drain:** At end of stream, remaining buffered packets are drained strictly in min-heap order so no packets within the final 250 ms tail are lost.
- **Dual Defense-in-Depth:** After emission from the buffer (or directly in zero-buffer mode), packets pass through `MonotonicTimestampGuard`, and both `FlowWindow.add_packet()` and `NetworkTopologyTracker.add_packet()` enforce `ts >= last_ts` directly so bypass callers cannot inject timestamp regressions either.

---

## 4. Overflow & Late-Packet Behavior

- **Bounded Memory & Fail-Closed Overflow:** `BoundedReorderBuffer` enforces `max_packets = 10000` (verified at boundary: 10,000 buffered packets succeed in `test_08`, and exceeding 10,000 triggers `ReorderBufferOverflowError` in `test_09`). Both offline replay and dataset builder fail closed rather than silently dropping or allocating unbounded memory.
- **Late Packets Behind Emitted Watermark:** If an offline packet arrives with a timestamp earlier than `last_emitted_ts` (e.g., regression > 250 ms or delayed behind the already-emitted watermark), `BoundedReorderBuffer.push()` rejects it (`earlier_than_last_emitted`), increments `rejected_late_count`, appends to `rejected_events` diagnostic telemetry, and does not insert it into the heap (`test_05`, `test_06`).
- **Live & Core Monotonic Rejection:** When `MonotonicTimestampGuard` rejects a packet (`ts < last_accepted_ts`):
  - Accepted and rejected counts plus previous/rejected timestamps are recorded in diagnostic telemetry (`out_of_order_timestamp` and `timestamp_guard` summary stats).
  - The active `FlowWindow` (if present for that flow key) has `mark_ooo_rejection()` called so `affected_by_ooo_rejection` and `ooo_rejected_count` propagate into `WindowProvenanceRecordV4` metadata only.
  - The rejected packet **never** mutates the canonical 29 V4 features, `FlowWindow` packet list/duration/IAT stats, or `NetworkTopologyTracker` FIFO state (`test_10`, `test_11`).
  - No silent clamping or hidden core reordering is performed (`test_12`).

---

## 5. Live vs. Offline Mode Distinction

| Execution Path | `reorder_window_sec` | `max_reorder_packets` | Behavior |
| :--- | :---: | :---: | :--- |
| Live Runtime (`AegisLiveV4Pipeline` / `ingest_packet`) | `0.0` (default) | `10000` | Zero buffering latency; immediate chronological enforcement via `MonotonicTimestampGuard`. |
| Runtime Packet Stream Replay (`replay_packet_stream`) | `0.0` (default, configurable) | `10000` | Zero buffering by default; callers can supply `reorder_window_sec=0.25` when replaying jittered streams. |
| Offline PCAP Replay (`replay_pcap`) | `0.25` (default) | `10000` | Bounded 250 ms min-heap reordering before `ingest_packet()`, draining cleanly at EOF. |
| Offline Dataset Builder (`LeakageSafeV4DatasetBuilder`) | `0.25` (default) | `10000` | Bounded 250 ms min-heap reordering inside `_iter_ordered_packets()` before partition quarantine and feature extraction. |

---

## 6. Regression Test Results

Executed after the final edit via `python -m pytest` (with `AEGISLiveV4Inference` mocked in the Batch 26D and temporal-causality tests so zero frozen model inference ran):

```text
tests/test_aegis_live_v4_timestamp_ordering.py   (15 passed)
tests/test_aegis_live_v4_temporal_causality.py   (18 passed)
tests/test_aegis_live_network_topology.py        ( 8 passed)
tests/test_aegis_live_v4_dataset.py              ( 8 passed)
tests/test_aegis_live_extractor.py               ( 6 passed)
tests/test_aegis_live_iat_regression.py          ( 8 passed)
----------------------------------------------------------------
Total: 63 passed, 0 failed, 0 skipped in 28.01s
```

All 15 new Batch 26D tests (`test_01_chronological_packets` through `test_15_temporal_snapshot_causality_preserved_after_reordering`) passed after the final test fix in `test_06`.

---

## 7. Builder / Runtime Parity

- Verified in `test_14_builder_runtime_parity_on_jittered_synthetic_stream` that when given an identical synthetic jittered stream (within 250 ms lateness), `LeakageSafeV4DatasetBuilder(reorder_window_sec=0.25)` and `AegisLiveV4Pipeline.replay_packet_stream(..., reorder_window_sec=0.25)` produce identical ordered feature rows across all 29 canonical V4 features.
- Verified in `test_15_temporal_snapshot_causality_preserved_after_reordering` that per-window causal topology snapshots captured in Batch 26B remain isolated and accurate after bounded 250 ms reordering.

---

## 8. Feature Compatibility Impact

- **Canonical 29-Feature Schema Preserved:** The 29 canonical V4 feature names, ordering, math definitions, and frozen model weights remain untouched. Out-of-order telemetry (`affected_by_ooo_rejection`, `ooo_rejected_count`, `timestamp_guard` counters) is strictly diagnostic/provenance metadata and is excluded from the 29-feature inference vector.
- **Jittered-Stream Drift Between Live and Offline Modes:** Because live capture (`reorder_window_sec=0.0`) rejects any packet with `ts < last_accepted_ts` immediately whereas offline replay/building (`reorder_window_sec=0.25`) recovers packets reordered by up to 250 ms, identical jittered packet streams can emit slightly different feature vectors (and packet counts/IATs) between unbuffered live mode and buffered offline mode (`test_03_mild_50ms_jitter_offline_vs_live`). On strictly chronological streams, live and offline outputs are identical (`test_01`).

---

## 9. Remaining Limitations (Deferred to Subsequent Phases)

Per the Batch 26A remediation roadmap and strict Batch 26D scope boundaries, the following audited issues remain intentionally untouched in this batch:

1. **Flow Identity (Phase 3):** Symmetric 3-tuple `(min(src, dst), max(src, dst), proto)` merging multiple ports/sessions between the same host pair and protocol.
2. **Packet Length Parsing (Phase 4):** `len(buf)` raw captured frame length rather than IP header total length / payload conventions.
3. **Directionality Assumption (Phase 5):** `_is_local(dst_ip)` heuristic and asymmetric upstream/downstream attribution assumptions.
4. **Live Capture Hardware Verification:** Timestamp ordering was verified on deterministic synthetic streams only; no Edge-IIoTset or live NIC PCAP scans were executed.

---

## 10. Recommendation for Batch 26E

Proceed to an independent Batch 26D code review / verification pass (analogous to Batch 26B.1) or directly to **Phase 3 (Flow Identity Remediation)** planning/implementation once Batch 26D changes are signed off, while keeping the V4 frozen model artifacts immutable until all planned V4 compatibility phases are complete.

---

## Safety Footer

- **Model Training / Fine-Tuning:** `NO`
- **Frozen Model Inference:** `NO` (all Batch 26D & temporal-causality runtime tests patched `AEGISLiveV4Inference`)
- **PCAP Scans:** `NO`
- **Dataset Generation:** `NO`
- **Automated Blocking:** `NO`
- **Frozen Model SHA-256 Hashes Changed:** `0` (verified unchanged after final edit via `Get-FileHash -Algorithm SHA256`):
  - `models/aegis_live_v4/aegis_live_xgboost_baseline.pkl`: `C0E5AEE8E9E1726F12BFC19E98C440FE44FE278D1796D5866FE2A2488D458A56`
  - `models/aegis_live_v4/temperature_scaling.json`: `C3AC0662CA896D06D4B9F62035A637FE04C8E01C00C466790D2EC0F03E581F99`
  - `models/aegis_live_v4/ood_parameters.pkl`: `579063CFC4DDA3E396AB0D026198F14F8EF593EBF9910F290157AA0F669712C3`
  - `models/aegis_live_v4/abstention_config.json`: `7D49F05BEF592CB838EF69F813A1C297BECA9E0A007BBAD1AAD6F5F0F4AC397F`
  - `models/aegis_live_v6_3/config_b_model.pkl`: `09AFB03CBE74201E4E3768E367FB35C70D727BC41A5B9FFC6180C3305FDF1F84`
  - `models/aegis_live_v6_3/temperature_scaling_b.json`: `2CEDF27975B4C28ABA2BBF52B84B0A5C5C49CAAF1EF3FA861AC6660885AA3B6F`
  - `models/aegis_live_v6_3/abstention_config_b.json`: `E30BC4571C5045C3374AAE930145821F1277E1C9ABA56E4F419F7B10B39E2039`

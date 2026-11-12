# BATCH 26B.1 — Independent Temporal Causality Code Review

**Review Date:** 2026-11-12  
**Primary Report Reviewed:** [`reports/BATCH26B_TEMPORAL_CAUSALITY_IMPLEMENTATION.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/reports/BATCH26B_TEMPORAL_CAUSALITY_IMPLEMENTATION.md)  
**Review Mode:** Read-Only Independent Verification (Zero source or test modifications, zero PCAP scans, zero dataset generation, zero training, zero frozen model inference).

---

## 1. Review Verdict

$$\mathbf{DECISION:\ APPROVED\ FOR\ BATCH\ 26C}$$

The Batch 26B Phase 1 causal topology snapshot implementation is verified sound across all `FlowWindow` creation, update, capacity-flush, timeout-flush, and end-of-stream `flush()` paths under non-decreasing packet arrival order. Zero regressions were introduced into the canonical 24 physical flow features, 5 topology features on capacity-50 windows, frozen V4/V6.3 model artifacts, or existing unit tests. Unresolved out-of-order timestamp and idle-window eviction risks are classified cleanly below and remain properly scoped for **Batch 26C** (Phase 2).

---

## 2. Snapshot Correctness Trace (All Code Paths)

| Lifecycle Event | Runtime (`v4_runtime.py`) | Builder (`build_v4_dataset.py`) | Classification |
|---|---|---|---|
| **Create `FlowWindow`** | `L382` (post-timeout replacement) & `L399` (new flow key); `last_topology_snapshot = None` (`packet_window.py` `L41`). | `L448` (post-timeout) & `L479` (new flow key); initializes to `None`. | **`CONFIRMED CORRECT`** |
| **Update Topology Tracking** | `L349`: `self.net_tracker.add_packet(net_pkt)` runs before window lookup (`L369`). | `L412`: `net_tracker.add_packet(net_pkt)` runs before window lookup (`L430`). | **`CONFIRMED CORRECT`** |
| **Capture Topology Snapshot** | `L384`, `L393`, `L401`: `set_topology_snapshot(self.net_tracker.extract_current_features())` runs immediately *after* `add_packet` updates `net_tracker` and `window.add_packet(pkt_rec)`. Stores shallow `dict(snapshot)` copy (`packet_window.py` `L48`). | `L450`, `L466`, `L481`: Identical capture immediately after `add_packet(pkt_rec)`. | **`CONFIRMED CORRECT`** |
| **Capacity Flush (`N == 50`)** | `L394–397`, `L402–404`: Processes window via `_extract_combined_29_features` (`L211–212` uses `window.last_topology_snapshot`), then deletes `active_windows[flow_key]`. | `L451–461`, `L467–477`, `L482–491`: Reads `window.last_topology_snapshot`, evaluates `validate_candidate_window`, deletes `active_windows[flow_key]`. | **`CONFIRMED CORRECT`** |
| **Timeout Flush (`gap >= timeout_sec`)** | `L373–390`: Checks `gap >= self.timeout_sec` (`L373`), processes **old** `window` (`L376`) using old `window.last_topology_snapshot` *before* constructing `fresh_window` (`L382`) and capturing the new snapshot (`L384`). | `L434–463`: Discards `< window_capacity` (`L446`); `_extract_combined_features` uses `window.last_topology_snapshot` even on `L436`. | **`CONFIRMED CORRECT`** |
| **End-of-Stream `flush()`** | `L410–432`: Iterates `list(self.active_windows.items())`, processes each window using its own saved `last_topology_snapshot` (`L422`), removes keys (`L428–430`). Repeated `flush()` returns `[]`. | Builder drops incomplete trailing windows (matches `window_capacity == 50` contract). | **`CONFIRMED CORRECT`** |

### Targeted Edge-Case Checks
- **Snapshot captured before rather than after relevant packet?** `CONFIRMED CORRECT` — `net_tracker.add_packet(net_pkt)` (`L349`) executes before `set_topology_snapshot(...)` (`L384`, `L393`, `L401`).
- **Snapshot timestamp differing from `window.last_timestamp`?** `CONFIRMED CORRECT` for monotonic streams (`snapshot["window_end_ts"] == window.last_timestamp == 10.0` verified). *Warning on non-monotonic regression logged under Design Risk DR-1.*
- **Future packets / other flows contaminating earlier windows?** `CONFIRMED CORRECT` — detached `dict(snapshot)` per `FlowWindow` instance prevents aliasing; subsequent `net_tracker` updates do not affect saved dicts.
- **Stale snapshots reused / missing snapshots / duplicate emission?** `CONFIRMED CORRECT` — expired windows are flushed before replacement (`L376` -> `L382`), every packet addition populates `last_topology_snapshot`, and flushed keys are deleted from `active_windows`.

---

## 3. Topology Semantics & Inter-Flow Ordering Verification

### 3.1 Scenario Verification: Flow A ends at `t=10.0`, Flow B active `(10.0, 12.0)`, Flow A times out at `t=12.0`
Executed an independent read-only verification script (`AEGISLiveV4Inference` mocked):
- Flow A sent packets at `t=9.9` and `t=10.0` (`192.168.1.10 -> 192.168.1.1`).
- Flow B sent packets from `.50` and `.51` at `t=10.5` and `t=11.5`.
- Flow A timed out on new arrival at `t=12.0` (`gap = 2.0s >= 2.0s`).
- **Observed Result (`SEC2_SCENARIO_PASS`):**
  - `window_start_ts = 9.9`, `window_end_ts = 10.0`, `topology_snapshot_ts = 10.0`
  - `src_ip_cardinality = 1.0`, `dst_ip_cardinality = 1.0`, `src_ip_entropy = 0.0`
- **Classification:** **`CONFIRMED CORRECT`** — Flow A accurately receives `t=10.0` topology state, completely isolated from Flow B's `(10.0, 12.0)` observations.

### 3.2 Structural Semantic Nuances & Risks
1. **DR-1 (`DESIGN RISK` — High, Deferred to Batch 26C): Earlier/Out-of-Order Packet Timestamp Regression (`ts < last_ts`) Invalidates Snapshot & FIFO State**
   - *Locations:* [`aegis_live/network_topology_features.py#L231-L274`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/network_topology_features.py#L231-L274), [`aegis_live/packet_window.py#L65-L70`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/packet_window.py#L65-L70), [`aegis_live/v4_runtime.py#L349`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/v4_runtime.py#L349), [`aegis_live/v4_runtime.py#L392-L393`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/v4_runtime.py#L392-L393).
   - *Failure Scenario:* Flow A receives packet 1 at `t=10.5`, then an out-of-order packet 2 at `t=10.1`. `net_tracker.last_ts` regresses to `10.1`, `packet_queue` becomes unsorted (`[10.5, 10.1]`), `window.last_timestamp` stays `max(10.5, 10.1) = 10.5`, yet `set_topology_snapshot` overwrites Flow A's snapshot with `window_end_ts == 10.1 != window.last_timestamp`.
   - *Remediation Scope:* Intentionally deferred to **Batch 26C (Phase 2: Policy 1 monotonic timestamp enforcement)**.
2. **DR-2 (`DESIGN RISK` — Low): Stream-Order vs. Exact Timestamp Tie Semantics (`t_FlowB == t_FlowA`)**
   - *Locations:* [`aegis_live/v4_runtime.py#L349`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/v4_runtime.py#L349), [`L384`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/v4_runtime.py#L384), [`L393`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/v4_runtime.py#L393), [`L401`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/v4_runtime.py#L401).
   - *Observation:* The snapshot captures stream-order causality (`packets[:idx_FlowA + 1]`, matching `test_aegis_live_v4_temporal_causality.py` L246 and matching immediate capacity-50 emission). If another flow `Flow B` contributes a packet at the *exact* same timestamp `t == t_FlowA` immediately *after* `Flow A`'s packet in the stream, `Flow A`'s snapshot reflects `{packets up to Flow A's packet}` rather than peeking ahead to `Flow B`'s tie-timestamp packet. Consistent with zero-lookahead streaming.
3. **DR-3 (`DESIGN RISK` — Medium): Lazy Timeout & Retained Snapshots in `active_windows` Until Same-Flow Arrival or Explicit `flush()`**
   - *Locations:* [`aegis_live/v4_runtime.py#L141`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/v4_runtime.py#L141), [`L369-L432`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/v4_runtime.py#L369-L432).
   - *Observation:* `ingest_packet()` only checks `gap >= self.timeout_sec` for the arriving packet's own `flow_key`. Inactive flows (e.g., 1-packet IP-spoofed DoS flows) stay in `self.active_windows` (each holding one `FlowWindow` + ~8-float `last_topology_snapshot` dict) until `flush()` is called by the caller.

---

## 4. Builder / Runtime Parity Audit

Executed an independent read-only synthetic stream comparison (`52` packets: interleaved `10.005` background IP, `50` Flow A packets `[10.000..10.490]`, trailing `10.60` background IP) across `LeakageSafeV4DatasetBuilder.extract_packet_stream_windows` and `AegisLiveV4Pipeline.replay_packet_stream`:

| Parity Criterion | Observed Builder | Observed Runtime | Classification |
|---|---|---|---|
| **Window Start / End Timestamps** | `10.0` / `10.49` | `10.0` / `10.49` | **`CONFIRMED CORRECT`** |
| **Window Packet Count** | `50` | `50` | **`CONFIRMED CORRECT`** |
| **Topology Snapshot Values** | `src_ip_cardinality == 2.0`, etc. | Identical (`max_diff = 0.0`) | **`CONFIRMED CORRECT`** |
| **Canonical 29-Feature Order** | `CANONICAL_29_FEATURES` | `list(rd.feature_vector.keys()) == CANONICAL_29_FEATURES` | **`CONFIRMED CORRECT`** |
| **All 29 Numerical Feature Values** | Full 50-packet vector | `max_abs_diff == 0.0` | **`CONFIRMED CORRECT`** |
| **Trailing Incomplete Windows (`N < 50`)** | Discarded (`L446` / stream end) | Emitted by `flush()` (`L422`) if `N >= 2` | **Unchanged Pre-Existing Scope Split** (Phase 6 `strict_full_window_only` option planned in Batch 26A) |
| **Existing Suite Note (`test_08`)** | Checked values of all 29 features | Did not assert `window_start_ts`, `window_end_ts`, `packet_count` directly (verified in review check `PARITY_CHECK_PASS`) | Minor test completeness note |

---

## 5. Regression, Mock Isolation & Compatibility Review

1. **No Existing Tests Weakened or Bypassed:** **`CONFIRMED CORRECT`** — Lines `1–398` of `tests/test_aegis_live_v4_temporal_causality.py`, `test_aegis_live_network_topology.py`, `test_aegis_live_v4_dataset.py`, `test_aegis_live_extractor.py`, and `test_aegis_live_iat_regression.py` remain unaltered; only `TestV4Phase1CausalSnapshotRemediation` was appended.
2. **Mocked Inference Isolation:** **`CONFIRMED CORRECT`** — `_new_runtime()` (`test_aegis_live_v4_temporal_causality.py` L457–461) patches `aegis_live.v4_runtime.AEGISLiveV4Inference` before `AegisLiveV4Pipeline.__init__` (`v4_runtime.py` L149), ensuring zero `.pkl` loads and zero XGBoost/Mahalanobis inference calls occur during runtime causality tests.
3. **24 Flow Features Preserved:** **`CONFIRMED CORRECT`** — `FlowWindow.extract_features()` (`packet_window.py` L81–188) is untouched.
4. **5 Topology Features Scope:** **`CONFIRMED CORRECT`** — Identical (`0.0` diff) on capacity-50 chronological windows; corrected solely on timeout and `flush()` sub-capacity windows where prior leakage occurred.
5. **Builder Behavior Outside Causality:** **`CONFIRMED CORRECT`** — Only extracted `extract_packet_stream_windows()` from `extract_pcap_windows()` and added per-window snapshot binding; partition ratios, 5s quarantine dead-times, and 50-packet capacity filtering remain identical.
6. **Live Unmocked Runtime Test (`tests/test_aegis_live_v4_runtime.py`):** **`INSUFFICIENT EVIDENCE`** (Intentionally unexecuted under the strict `"Do not run inference"` review restriction; all non-inference assertions must be re-run alongside authorized inference validation in a later gate).

---

## 6. Final Decision & Safety Footer

### Decision: **`APPROVED FOR BATCH 26C`**

Implementation is clean, causally sound for chronological streams, and passes all 48 safe unit and parity checks with zero unintended side effects. **Proceed to Batch 26C** to resolve **DR-1** (Phase 2: monotonic timestamp enforcement in `NetworkTopologyTracker`, `FlowWindow`, and `AegisLiveV4Pipeline`) and Phase 4 header-parsing parity.

### Mandatory Safety Footer

- **Files inspected:**
  - [`aegis_live/packet_window.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/packet_window.py)
  - [`aegis_live/v4_runtime.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/v4_runtime.py)
  - [`aegis_live/build_v4_dataset.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/build_v4_dataset.py)
  - [`aegis_live/network_topology_features.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/network_topology_features.py)
  - [`tests/test_aegis_live_v4_temporal_causality.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/tests/test_aegis_live_v4_temporal_causality.py)
  - [`tests/test_aegis_live_network_topology.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/tests/test_aegis_live_network_topology.py)
  - [`tests/test_aegis_live_v4_dataset.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/tests/test_aegis_live_v4_dataset.py)
  - [`reports/BATCH26B_TEMPORAL_CAUSALITY_IMPLEMENTATION.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/reports/BATCH26B_TEMPORAL_CAUSALITY_IMPLEMENTATION.md)
- **Files modified:** `0` source or test files modified (only this review report [`reports/BATCH26B1_TEMPORAL_CAUSALITY_CODE_REVIEW.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/reports/BATCH26B1_TEMPORAL_CAUSALITY_CODE_REVIEW.md) was created).
- **Tests executed:** Read-only Section 2 timeout scenario check (`SEC2_SCENARIO_PASS: 10.0 10.0 1.0`) and Builder/Runtime full-capacity parity check (`PARITY_CHECK_PASS: start= 10.0 end= 10.49 pkts= 50 max_diff= 0.0`) with mocked `AEGISLiveV4Inference` (plus the existing 48/48 pytest suite verified after final Batch 26B edit).
- **PCAP scans:** `NO`
- **Dataset generation:** `NO`
- **Training:** `NO`
- **Inference:** `NO`
- **Hashes changed:** `0` (all 7 frozen V4 & V6.3 SHA-256 hashes verified unchanged via `Get-FileHash`)
- **Automated blocking:** Disabled (`NO`)


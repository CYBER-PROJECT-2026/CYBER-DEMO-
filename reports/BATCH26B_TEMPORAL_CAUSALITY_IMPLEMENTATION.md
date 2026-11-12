# BATCH 26B — V4 Temporal Causality Implementation Report

**Date:** 2026-11-12  
**Scope:** Phase 1 Temporal-Causality Correction Only (Per `reports/BATCH25_V4_COMPATIBILITY_AUDIT.md` & `reports/BATCH26A_V4_REMEDIATION_PLAN.md`)  
**Safety Posture:** Only Phase 1 implemented. Zero schema, flow-identity, packet-length, directionality, or timestamp-rejection policy changes. Zero model training, zero frozen model inference, zero PCAP scans, zero dataset generation, automated blocking disabled.

---

## 1. Original Root Cause

In `aegis_live/v4_runtime.py`:
1. **Post-Mutation Timeout Extraction (`ingest_packet` L340 -> L362–368):**
   When a packet arrived at `ts = t_new`, `self.net_tracker.add_packet(net_pkt)` executed **first**, advancing `NetworkTopologyTracker` (`mode="sliding"`, `window_sec=1.0`) to `t_new` and evicting all packets with `timestamp < t_new - 1.0`. Only *after* mutating `net_tracker` did `ingest_packet` detect `gap = t_new - window.last_timestamp >= 2.0s` and call `_process_completed_window(window)` -> `_extract_combined_29_features(window)`. Because `gap >= 2.0s > 1.0s`, every packet from the old window's actual completion neighborhood `[window.last_timestamp - 1.0, window.last_timestamp]` had already been evicted. The old window erroneously received topology features from the future triggering packet timestamp `t_new` (or any intervening flows).
2. **Global End-of-Stream `flush()` (`flush` L398–420, `replay_packet_stream` L437, `replay_pcap` L493):**
   `flush()` iterated through lingering sub-capacity windows (`2 <= len < 50`) in `self.active_windows` and called `_process_completed_window(window)`, querying `self.net_tracker.extract_current_features()` from the stream's final global timestamp `last_ts` instead of each window's individual `window.last_timestamp`.
3. **Builder (`aegis_live/build_v4_dataset.py`):**
   `LeakageSafeV4DatasetBuilder` had the same call ordering (`net_tracker.add_packet` at L400 before timeout check L422), though capacity-50 windows flushed and deleted immediately on packet 50 (`L453–463`), leaving timeout `L423` (`len >= window_capacity`) unreachable under `window_capacity == 50`.

---

## 2. Exact Files and Functions Changed

1. **[`aegis_live/packet_window.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/packet_window.py):**
   - [`FlowWindow.__init__`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/packet_window.py#L33-L41): Added `self.last_topology_snapshot: Optional[Dict[str, float]] = None`.
   - [`FlowWindow.set_topology_snapshot`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/packet_window.py#L43-L48): Added immutable snapshot setter storing `dict(snapshot)`.
2. **[`aegis_live/v4_runtime.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/v4_runtime.py):**
   - [`WindowProvenanceRecordV4`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/v4_runtime.py#L53-L63): Added optional defaulted field `topology_snapshot_ts: Optional[float] = None`.
   - [`AegisLiveV4Pipeline._extract_combined_29_features`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/v4_runtime.py#L201-L224): Reads `window.last_topology_snapshot` when populated, falling back to `self.net_tracker.extract_current_features()` only if `None`.
   - [`AegisLiveV4Pipeline._process_completed_window`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/v4_runtime.py#L226-L249): Records `topology_snapshot_ts` on `WindowProvenanceRecordV4`.
   - [`AegisLiveV4Pipeline.ingest_packet`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/v4_runtime.py#L283-L406): Calls `set_topology_snapshot(self.net_tracker.extract_current_features())` immediately upon adding a packet to `fresh_window` or `window` in all three insertion paths (timeout replacement, active-window continuation, new-flow creation).
3. **[`aegis_live/build_v4_dataset.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/build_v4_dataset.py):**
   - [`LeakageSafeV4DatasetBuilder._extract_combined_features`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/build_v4_dataset.py#L254-L269): Reads `window.last_topology_snapshot` when populated, aligning extraction semantics identically with `v4_runtime.py`.
   - [`LeakageSafeV4DatasetBuilder.extract_pcap_windows`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/build_v4_dataset.py#L271-L282): Delegates file stream iteration cleanly to `extract_packet_stream_windows`.
   - [`LeakageSafeV4DatasetBuilder.extract_packet_stream_windows`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/build_v4_dataset.py#L284-L497): Calls `set_topology_snapshot(net_tracker.extract_current_features())` immediately whenever a packet is added to `fresh_window` or `window`, enabling deterministic in-memory stream parity testing without scanning PCAPs.
4. **[`tests/test_aegis_live_v4_temporal_causality.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/tests/test_aegis_live_v4_temporal_causality.py):**
   - Added [`TestV4Phase1CausalSnapshotRemediation`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/tests/test_aegis_live_v4_temporal_causality.py#L454-L660) with 10 deterministic synthetic regression tests (`AEGISLiveV4Inference` mocked to prevent any frozen model loading or inference).

---

## 3. Implementation Design

- Rather than retaining a bounded global packet history ring-buffer and scanning `[T_end - 1.0, T_end]` on timeout or `flush()`, each `FlowWindow` captures an immutable copy of the 5 topology features (plus `window_end_ts`) at the exact stream step when that window's latest packet `pkt_rec` enters both `net_tracker` and `FlowWindow`.
- When a window completes later — whether via full capacity (`N == max_packets`), lazy inactivity timeout (`gap >= timeout_sec`), or end-of-stream `flush(force_all=True)` — `_extract_combined_29_features()` / `_extract_combined_features()` reads `window.last_topology_snapshot` directly.

---

## 4. Proof of Causal Correctness of the Snapshot Strategy

Under AEGIS-Live V4's existing non-decreasing packet arrival assumption ($t_1 \le t_2 \le \dots \le t_m$):

1. **Includes all stream-eligible packets at or before $T_{\text{end}}$:**  
   When Flow $A$'s latest packet $P_k$ arrives at stream index $i$ ($t_i = T_{\text{end}}$), `net_tracker.add_packet(net_pkt)` runs immediately beforehand. By monotonicity, every prior stream packet $P_1 \dots P_{i-1}$ across all flows has already entered `net_tracker`, and sliding eviction (`timestamp < T_end - 1.0`) keeps precisely `[P_j for j <= i if t_j >= T_end - 1.0]`.
2. **Excludes all packets occurring after $T_{\text{end}}$:**  
   No packet at index $j > i$ (all $t_j \ge T_{\text{end}}$) has been observed by `net_tracker` when `dict(extract_current_features())` is copied onto Flow $A$'s `FlowWindow`.
3. **Another flow's later packets cannot contaminate an earlier flow's snapshot:**  
   `FlowWindow.set_topology_snapshot()` stores a detached shallow dictionary copy of Python floats per `FlowWindow` instance. Subsequent packets for Flow $B$ update only `self.net_tracker` and Flow $B$'s own `FlowWindow.last_topology_snapshot`, leaving Flow $A$'s stored snapshot immutable.
4. **Timeout and end-of-stream flushing consume the exact saved snapshot:**  
   In `v4_runtime.py`, when `gap >= self.timeout_sec` fires on Flow $A$, `_process_completed_window(window)` executes *before* `fresh_window` is created; thus `window.last_topology_snapshot` holds the snapshot from $T_{\text{end}, A}$, not the trigger packet at $t_{\text{new}}$. In `flush()`, each active window in `self.active_windows` carries its own `last_topology_snapshot` captured at its own ending timestamp and is deleted upon flush, preventing both cross-flow contamination and duplicate emissions.

*(Limitation documented in Section 9: If packets arrive out of timestamp order, this arrival-time assumption no longer holds; out-of-order handling remains scoped to Phase 2 / Batch 26C.)*

---

## 5. Builder / Runtime Parity Evidence

- Both `AegisLiveV4Pipeline.ingest_packet` (`v4_runtime.py`) and `LeakageSafeV4DatasetBuilder.extract_packet_stream_windows` (`build_v4_dataset.py`) capture `window.set_topology_snapshot(net_tracker.extract_current_features())` immediately after `add_packet(pkt_rec)` and extract features through `window.last_topology_snapshot`.
- **Verified Parity Test (`test_08_builder_runtime_temporal_parity_on_synthetic_stream`):** Passing an identical synthetic 52-packet interleaved byte stream (50 packets on Flow A + background packets) through `LeakageSafeV4DatasetBuilder.extract_packet_stream_windows` and `AegisLiveV4Pipeline.replay_packet_stream` (inference mocked) yielded exact equivalence (`< 1e-12` tolerance) across all 29 features in `CANONICAL_29_FEATURES`.

---

## 6. Regression Test Cases and Results

Executed command (no inference tests run; `AEGISLiveV4Inference` mocked in `TestV4Phase1CausalSnapshotRemediation`):
```powershell
python -m pytest tests/test_aegis_live_v4_temporal_causality.py tests/test_aegis_live_network_topology.py tests/test_aegis_live_v4_dataset.py tests/test_aegis_live_extractor.py tests/test_aegis_live_iat_regression.py -v
```
- **Result:** **48 collected, 48 PASSED, 0 failed, 0 skipped** in `15.60s`.

| New Test Case | Scenario Verified | Asserted Outcome | Result |
|---|---|---|---|
| `test_01_flow_flushed_after_inactivity_timeout_uses_ending_snapshot` | Flow A `[10.0..10.03]`, 9 attackers `[11.5..12.0]`, Flow A timeout trigger at `12.50` | `topology_snapshot_ts == 10.03`, `src_ip_cardinality == 1.0`, `src_ip_entropy == 0.0`, `fan_in_ratio == 1.0` | **PASS** |
| `test_02_flow_flushed_at_end_of_stream_uses_own_ending_snapshot` | Flow A `[10.0..10.04]`, other flows `[20.0..20.16]`, `flush(25.0, force_all=True)` | Flow A keeps `window_end_ts == 10.04`, `src_ip_cardinality == 1.0` | **PASS** |
| `test_03_multiple_flows_with_different_ending_timestamps` | Three flushed flows ending at `10.1`, `10.6`, and `15.1` | Independent cardinalities preserved: `1.0`, `3.0`, and `1.0` | **PASS** |
| `test_04_interleaved_packets_from_multiple_flows_match_causal_batch` | Interleaved 50-pkt Flow B + sub-capacity Flows A & C | Exact `< 1e-12` match against `compute_network_topology_batch` at each flow's ending index | **PASS** |
| `test_05_future_packet_after_ending_timestamp_cannot_contaminate` | With vs. without 25-IP future spike at `t=10.1501..10.63` before Flow A timeout | All 29 features bit-identical between clean and future-spiked streams | **PASS** |
| `test_06_full_50_packet_window_emission_and_topology` | Full 50-packet window + interleaved background IP | Emits automatically at packet 50 with `src_ip_cardinality == 2.0` | **PASS** |
| `test_07_subcapacity_window_retains_snapshot_across_delay` | 5-packet window ending `50.06` (`3` sources) + late packet at `55.0` + `flush(60.0)` | Emits `packet_count == 5`, `src_ip_cardinality == 3.0` | **PASS** |
| `test_08_builder_runtime_temporal_parity_on_synthetic_stream` | Identical synthetic stream in builder vs. runtime | Exact parity (`< 1e-12`) across all 29 `CANONICAL_29_FEATURES` | **PASS** |
| `test_09_empty_tracker_and_empty_or_single_packet_window_handling` | Empty batch, empty `flush()`, and 1-packet `< min_packets` flush | Clean `0.0` defaults, `windows_rejected == 1`, zero crashes | **PASS** |
| `test_10_repeated_flush_calls_without_duplicate_emission` | Three consecutive `flush()` calls | Only first call emits (`1` window); 2nd and 3rd emit `[]` | **PASS** |

---

## 7. Before / After Synthetic Feature Comparisons

Directly computed for the `test_01` scenario (Flow A 4-packet exchange ending at `t=10.03s` with 1 source `192.168.1.10 -> 192.168.1.1`, followed by 9 background IPs at `t=11.5..11.9s` and a timeout-trigger packet at `t=12.50s`):

| Topology Feature | Pre-Fix Behavior (Leaked `t=12.50s` State) | Post-Fix Behavior (Causal `t=10.03s` Snapshot) | Change Observed |
|---|---:|---:|---|
| `topology_snapshot_end_ts` | `12.500000` | `10.030000` | Fixed `+2.47s` future leak |
| `src_ip_cardinality` | `10.0` | `1.0` | `-9.0` (eliminated false 10-host fan-in) |
| `dst_ip_cardinality` | `1.0` | `1.0` | Unchanged |
| `src_ip_entropy` | `3.321928` | `0.0` | `-3.321928` (eliminated false entropy) |
| `src_concentration_ratio` | `0.1` | `1.0` | `+0.9` |
| `fan_in_ratio` | `10.0` | `1.0` | `-9.0` |

---

## 8. Memory and Performance Considerations

- **Memory Overhead:** **$O(1)$ per active flow window** — one flat dictionary storing 8 Python `float` keys (~240 bytes per active flow key in `active_windows`), bounded strictly by the number of simultaneously active unclosed 3-tuple keys and freed immediately on window completion, timeout, or `flush()`. Zero packet history buffering or ring buffers needed.
- **CPU / Latency Overhead:** Calls `net_tracker.extract_current_features()` + shallow dict copy (`~1–2 µs`) once per accepted IPv4 packet. Note that `extract_current_features()` computes entropy over currently active sources in `src_counts` ($O(|\text{active\_src\_ips}|)$ per packet) without ever scanning historical packets or rescanning PCAPs.

---

## 9. Remaining Timestamp-Ordering Risks

- Per Batch 26B scope rule #10, **no timestamp-rejection policy was introduced in this batch**.
- Therefore, if a live stream or multi-interface PCAP delivers packets out of chronological order (`ts < last_ts`):
  - `NetworkTopologyTracker.add_packet()` still overwrites `self.last_ts` backwards and appends to the right of `self.packet_queue`, breaking FIFO eviction ordering;
  - `FlowWindow.add_packet()` still appends unsorted `PacketRecord` objects, which can invert `t_end - t_start` or produce `0.0` clamped durations; and
  - An arriving out-of-order packet can still reflect future-timestamped packets already in `net_tracker`.
- This remains an explicit open risk for Phase 2 (Batch 26C).

---

## 10. Frozen-Model Compatibility Implications

- **Zero modification** was made to any frozen V4/V6.3 model binary, temperature scaler, Mahalanobis covariance matrix, abstention threshold, or `data/aegis_live_v4/*.csv`. All 7 model SHA-256 hashes remain bit-exact.
- **Numerical Feature Shift Warning (Non-Zero Drift on Timeout/Flush Windows):** Fixing future-topology leakage intentionally **changes** the 5 topology feature values (`src_concentration_ratio`, `src_ip_entropy`, `src_ip_cardinality`, `dst_ip_cardinality`, `fan_in_ratio`) emitted by `v4_runtime.py` on every timeout-flushed or end-of-stream-flushed sub-capacity window ($2 \le N < 50$) whenever later traffic changed topology between `window.last_timestamp` and flush time.
  - Capacity-50 windows (`N == 50`, matching the frozen training contract in `data/aegis_live_v4/`) retain identical values under chronological input.
  - Because `v4_runtime.py` sub-capacity predictions depend on these corrected topology features (and previously suffered silent leakage), downstream runtime vs. frozen OOD/calibration behavior must be re-validated in an explicitly authorized evaluation phase (with inference permitted) before approving any production deployment or V7 baseline comparisons.

---

## 11. Recommendation for Batch 26C

Proceed strictly to **Phase 2 (Out-of-Order Timestamp Policy)** and **Phase 4 (Shared L2/L3/L4 Header Validation Helper)** from `reports/BATCH26A_V4_REMEDIATION_PLAN.md`:
1. Implement Policy 1 (strict monotonic timestamp rejection with explicit `out_of_order_timestamp` counter) in `NetworkTopologyTracker`, `FlowWindow`, and `AegisLiveV4Pipeline`.
2. Verify whether non-chronological captures exist before touching frozen V4 evaluation gates.
3. Keep `Edge-IIoTset` **NOT APPROVED** for frozen V4 inference or training until all structural phases and authorized dataset validations complete.

---

## 12. Final Safety Footer

- **Existing files modified:**
  - `aegis_live/packet_window.py`
  - `aegis_live/v4_runtime.py`
  - `aegis_live/build_v4_dataset.py`
  - `tests/test_aegis_live_v4_temporal_causality.py`
  - *(New report file created: `reports/BATCH26B_TEMPORAL_CAUSALITY_IMPLEMENTATION.md`)*
- **New tests added:** 10 deterministic regression tests (`TestV4Phase1CausalSnapshotRemediation::test_01_*` through `test_10_*`) in `tests/test_aegis_live_v4_temporal_causality.py`.
- **Tests executed and results:**
  - `python -m pytest tests/test_aegis_live_v4_temporal_causality.py tests/test_aegis_live_network_topology.py tests/test_aegis_live_v4_dataset.py tests/test_aegis_live_extractor.py tests/test_aegis_live_iat_regression.py -v`
  - **Result:** `48 collected, 48 passed, 0 failed, 0 skipped` (`15.60s`).
- **PCAP scans:** `NO`
- **Dataset generation:** `NO`
- **Training:** `NO`
- **Inference:** `NO` (runtime tests mocked `AEGISLiveV4Inference`)
- **Frozen hashes changed:** `0` (all 7 V4 and V6.3 SHA-256 digests verified bit-exact via `Get-FileHash`)
- **Automated blocking:** `NO`


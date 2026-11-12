# BATCH 26C — V4 Timestamp Ordering Policy Investigation

**Date:** 2026-11-12  
**Mode:** Investigation & Planning Only (Zero source/test modifications, zero PCAP scans, zero dataset generation, zero model inference, zero training).  
**Context:** Resolves the timestamp-ordering risk identified in `reports/BATCH25_V4_COMPATIBILITY_AUDIT.md`, `reports/BATCH26A_V4_REMEDIATION_PLAN.md`, `reports/BATCH26B_TEMPORAL_CAUSALITY_IMPLEMENTATION.md`, and `reports/BATCH26B1_TEMPORAL_CAUSALITY_CODE_REVIEW.md` (**DR-1**).

---

## 1. Executive Summary

- **`[CONFIRMED FACT]` Current Code Vulnerability:** Neither `FlowWindow` (`aegis_live/packet_window.py`), `NetworkTopologyTracker` (`aegis_live/network_topology_features.py`), `AegisLiveV4Pipeline` (`aegis_live/v4_runtime.py`), nor `LeakageSafeV4DatasetBuilder` (`aegis_live/build_v4_dataset.py`) validates non-decreasing packet timestamps.
- **`[EXPERIMENTAL RESULT]` Measured Impact on Out-of-Order Synthetic Streams:**
  1. **IAT & Duration Distortion:** A `-5.2s` timestamp regression (`[10.0, 10.1, 10.2, 5.0, 10.3]`) inflates `iat_mean_ms` from **`100.0 ms`** to **`1375.0 ms`** (`13.75x`) due to `max(dt, 0.0)` clamping on negative IATs paired with the huge recovery jump (`+5.3s`). A slight `-0.05s` regression inflates rate from `13.33 pps` to `16.67 pps` and distorts `iat_mean_ms` from `75.0 ms` (chronological) to `87.5 ms`.
  2. **Trapped FIFO Eviction:** In Scenario 7, a late packet at `t=5.0` inserted behind `t=10.2` in `NetworkTopologyTracker.packet_queue` cannot be popped by `while packet_queue[0].timestamp < cutoff_ts` (`10.2 >= 10.20`), inflating `src_ip_cardinality` from **`1.0`** to **`2.0`** until `t=10.2` itself expires.
  3. **Snapshot Timestamp Inversion:** In Scenario 8, a `-4.3s` regression resets `net_tracker.last_ts` backward to `6.1` (`window_start_ts = 10.0 > window_end_ts = 6.1`) and contaminates downstream causal snapshots.
- **`[RECOMMENDATION]` Dual-Surface Timestamp Strategy Preserving Frozen V4 Parity:**
  - **Live Network Capture:** **Policy A (Strict Monotonic Rejection + Explicit Out-of-Order Telemetry & Flow Taint Flag)** as zero-latency default (`ts < global_last_ts` rejected with counter + `contains_ooo_drops` provenance flag).
  - **Offline PCAP Replay & Builder:** **Policy B (Bounded Min-Heap Reordering with small lateness watermark $W \le 0.25\text{ s}$)** to recover harmless NIC/multi-interface clock jitter (`<= 250 ms`) without losing packets, while **rejecting/quarantining packets beyond watermark ($t < \text{max\_ts} - W$)** identically to Policy A for severe regressions.

---

## 2. Current Implementation Behavior & Root Causes (`[CONFIRMED FACT]`)

Trace of every timestamp-sensitive mechanism across the four inspected modules:

| Component / Function | Line Ref | Current Behavior | Failure Mode Under Out-of-Order Input (`ts < last_ts`) |
|---|---|---|---|
| `FlowWindow.add_packet` | [`packet_window.py#L60-L70`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/packet_window.py#L60-L70) | Checks `(pkt.timestamp - self.last_timestamp) >= timeout_sec` (`L60`), updates `self.last_timestamp = max(self.last_timestamp, pkt.timestamp)` (`L69`), appends unsorted `self.packets.append(pkt)` (`L70`). | Negative gap never triggers timeout (`L60`). Unsorted `pkt` enters `packets` at the end while `last_timestamp` stays at the high-water mark (`L69`). |
| `FlowWindow.extract_features` (Duration) | [`packet_window.py#L94-L106`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/packet_window.py#L94-L106) | Computes `t_start = packets[0].timestamp`, `t_end = packets[-1].timestamp` (`L94-95`), `duration_sec = max(t_end - t_start, 0.0)` (`L96`), `safe_duration_sec = max(duration_sec, 1e-6)` (`L100`). | If an out-of-order packet lands at index `-1` earlier than `packets[0]`, `t_end < t_start` clamps `duration_sec = 0.0`, producing `pkt_rate_pps = N * 1e6` (`1e6–5e7 pps`). |
| `FlowWindow.extract_features` (IAT) | [`packet_window.py#L114-L122`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/packet_window.py#L114-L122) | Computes consecutive differences `packets[i].timestamp - packets[i-1].timestamp` clamped via `max(dt, 0.0) * 1000.0` (`L117`). | Clamps negative delta to `0.0` instead of ordering timestamps, while the next forward packet produces an artificially doubled/massive delta, biasing `iat_mean_ms` and `iat_std_ms` upward. |
| `NetworkTopologyTracker.add_packet` | [`network_topology_features.py#L231-L274`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/network_topology_features.py#L231-L274) | Overwrites `self.last_ts = ts` unconditionally (`L232`). Appends `packet` to back of `self.packet_queue` (`L264`). Pops from front `while packet_queue[0].timestamp < ts - window_sec` (`L268-271`). Sets `window_start_ts = packet_queue[0].timestamp` (`L273`). | **1)** `last_ts` moves backwards (`window_start_ts > window_end_ts`). **2)** `packet_queue` becomes unsorted, **trapping** old late packets behind newer head elements (`Scenario 7` proves failed eviction!). **3)** `cutoff_ts` regresses, stopping normal expirations. |
| `AegisLiveV4Pipeline.ingest_packet` (Timeout & Snapshot) | [`v4_runtime.py#L349, L371-L406`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/v4_runtime.py#L349) | Updates `net_tracker.add_packet(net_pkt)` (`L349`). Computes `gap = pkt_rec.timestamp - window.last_timestamp` (`L371`). Stores `set_topology_snapshot(...)` (`L384/L393/L401`). | **1)** A late packet arriving *after* a flow already timed out (`L373`) has negative `gap` against `fresh_window`, contaminating the new post-timeout window (`Scenario 6`). **2)** Late packet overwrites `last_topology_snapshot` with `snapshot["window_end_ts"] < window.last_timestamp`. |
| `AegisLiveV4Pipeline.replay_*` & `flush` | [`v4_runtime.py#L442, L494`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/v4_runtime.py#L442) | Tracks `last_ts = float(ts)` blindly and passes `last_ts + 5.0` to `flush()`. | If final packet regressed significantly (`> 5s` behind another window), `current_time` passed to `flush()` could regress (`force_all=True` mitigates end-of-stream, non-forced periodic calls would fail `gap >= 2.0`). |
| `LeakageSafeV4DatasetBuilder` | [`build_v4_dataset.py#L330-L360, L412-L493`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/build_v4_dataset.py#L330) | Identical `net_tracker` / `FlowWindow` logic; quarantine exits when `ts >= quarantine_end`. | If an unsorted PCAP exits quarantine on packet `k` and packet `k+1` regresses `ts < quarantine_end`, packet `k+1` leaks straight into validation/test without quarantine! *(Existing CICIoT2023 `80k` dataset verified free of such inversions by `test_all_80k_v4_rows_temporal_invariants`.)* |

---

## 3. Three-Policy Comparison Analysis

### Policy A — Strict Monotonic Rejection (`ts < global_last_ts`)
- **Mechanics:** Maintain `global_last_ts` across ingestion (or reject when `ts < global_last_ts`). Any packet earlier than the highest seen timestamp is rejected immediately with `packet_rejection_reasons["out_of_order_timestamp"]` before touching `net_tracker` or `FlowWindow`.
- **Packet Loss:** Drops *every* slightly delayed packet (`-1 ms` jitter loses the packet!). Look at Scenario 8: rejecting `10.12` on Flow B after Flow A `10.20` distorted Flow B's `iat_mean_ms` from `150 ms` to `300 ms`.
- **Real-Time Capture Impact:** Zero buffering latency (`0 ms`), constant $O(1)$ memory, strictly causal, safe against clock-skew attacks or late spoofed packets. Kernel/libpcap timestamps on a single NIC are generally monotonic, though multi-queue AF_PACKET/RSS or NTP step adjustments can emit occasional sub-millisecond regressions.
- **PCAP Replay & Edge-IIoTset Impact:** Safe against corrupted multi-second regressions, **but** if Edge-IIoTset sensor PCAPs contain multi-interface merge jitter (e.g., 10–100 ms skew between devices), Policy A would drop valid benign packets and alter IATs/rates artificially!
- **Required Telemetry:** `out_of_order_packets_rejected`, `max_ts_regression_sec`, plus per-window `ooo_drops_adjacent` count so analysts know a flow lost late packets.

### Policy B — Bounded Reordering (Sliding Watermark Min-Heap, bound $W$)
- **Mechanics:** Buffer incoming packets in a min-heap keyed by `(ts, seq_id, buf)`. Advance `high_water_ts = max(high_water_ts, ts)` and `watermark = high_water_ts - W`. Pop and process all packets with `ts <= watermark` in **strictly non-decreasing timestamp order**. Any incoming packet with `ts < watermark` is too late and rejected explicitly (`late_beyond_watermark`). On `flush()` / end-of-stream, drain remaining heap in sorted order.
- **Buffer Memory:** $O(\text{PPS}_{\max} \times W)$ packets in heap (e.g., at $50,000\text{ pps}$ SYN flood and $W = 0.25\text{ s}$, up to ~12,500 frames / ~1–15 MB unless bounded by `max_buffer_packets` cap!). Must bound heap capacity deterministically (e.g., `max_buffer_packets = 10000`, advancing watermark early on cap pressure).
- **Processing Latency:** Adds **watermark lag up to $W$** (plus idle delay if traffic completely stops until a heartbeat/flush advances `high_water_ts`!). Unsuitable for $< 1\text{ ms}$ live inference claims unless $W$ is tiny ($10\text{–}50\text{ ms}$) or skipped in live capture; **ideal for offline PCAP replay** where end-of-stream flush drains the entire buffer with zero effective delay!
- **Timeout & Snapshot Causality:** Because packets exit the heap in sorted chronological order *before* entering `net_tracker` and `FlowWindow`, **100% of FIFO eviction invariants, IAT deltas, timeout checks, and per-window topology snapshots remain causally exact!** Even tie timestamps work cleanly via `(ts, seq_id)` deterministic tie-breaking.
- **Builder/Runtime Parity:** Using identical `(ts, seq_id)` bounded heap sorting in offline PCAP replay and `LeakageSafeV4DatasetBuilder` ensures identical chronological stream inputs and identical 29-feature outputs.

### Policy C — Preserve Arrival Order with Explicit Late-Packet Isolation
- **Mechanics:** Keep stream order for incoming packets, detect `ts < last_ts`, isolate late packets from corrupting FIFO eviction / IATs, and flag affected windows as unreliable (`contains_late_packets = True`).
- **Critical Architectural Dilemma (`[CONFIRMED FACT]`):**
  1. *Global isolation (`ts < global_last_ts` excluded from `FlowWindow` and `net_tracker`, flagging flows):* Drops the exact same packets as Policy A (`Scenario 2`, `Scenario 3`, `Scenario 8` show identical feature values to Policy A!), gaining zero packet recovery over Policy A.
  2. *Local per-flow acceptance + topology skip (accept Flow B `10.12` because `10.12 > FlowB.last_ts 10.10`, skip `net_tracker` because `10.12 < global_last_ts 10.20`):* Silently undercounts `net_tracker` (`src_ip_cardinality`, `src_concentration_ratio`), diverging from offline batch reference topology!
  3. *Accept late packet into `FlowWindow` and re-sort packets only inside `extract_features()`:* Fails whenever timeout already flushed the earlier window (`Scenario 6`) and still leaves `net_tracker` corrupted or divergent.
- **Verdict on Policy C:** Cannot simultaneously preserve valid multi-flow topology, preserve IAT accuracy, and retain out-of-order packets. Its useful idea — *explicit provenance taint flags when out-of-order events touch a flow* — is strictly better implemented as metadata telemetry alongside Policy A / Policy B!

---

## 4. Deterministic Synthetic Experiments (`[EXPERIMENTAL RESULT]`)

Executed via isolated simulation harness (`AEGISLiveV4Inference` mocked, zero repo modifications, zero PCAP scans). Parameters: `timeout_sec = 2.0`, `net_window_sec = 1.0`, Policy B `lateness_bound W = 0.25 s`.

| Scenario | Current Baseline | Policy A (Strict Reject) | Policy B (Bounded $W=0.25\text{ s}$) | Policy C (Global Isolate + Taint) |
|---|---|---|---|---|
| **1. Chronological** (`5` pkts `[10.0..10.4]`) | `acc=5 rej=0 wins=1`<br>`dur=400 iat=100 pps=12.5`<br>`card=1.0 snap=10.4` | **Identical** | **Identical** | **Identical** (`flagged=0`) |
| **2. Slightly Late** (`-0.05s`: `[10.0, 10.1, 10.2, 10.15, 10.3]`) | `acc=5 rej=0 wins=1`<br>`iat=87.5` *(corrupted)* `pps=16.67` | `acc=4 rej=1 wins=1`<br>`iat=100.0 pps=13.33`<br>*(drops `10.15`)* | `acc=5 rej=0 wins=1`<br>**`iat=75.0 pps=16.67`** *(exact sorted recovery!)* | `acc=4 rej=1 wins=1`<br>`iat=100.0 flagged=1` |
| **3. Multiple Late** (`[10.0, 10.2, 10.05, 10.3, 10.15, 10.4]`) | `acc=6 rej=0 wins=1`<br>`iat=140.0` *(corrupted)* `pps=15.0` | `acc=4 rej=2 wins=1`<br>`iat=133.33 pps=10.0`<br>*(drops 2 pkts)* | `acc=6 rej=0 wins=1`<br>**`iat=80.0 pps=15.0`** *(exact sorted `[10.0..10.4]`)* | `acc=4 rej=2 wins=1`<br>`iat=133.33 flagged=1` |
| **4. Large Regression** (`-5.2s`: `[10.0, 10.1, 10.2, 5.0, 10.3]`) | `acc=5 rej=0 wins=1`<br>**`iat=1375.0 ms`** *(13.75x spike!)* | `acc=4 rej=1 wins=1`<br>**`iat=100.0 pps=13.33`** | `acc=4 rej=1 wins=1`<br>**`iat=100.0 pps=13.33`** *(beyond `W=0.25s`, rejected)* | `acc=4 rej=1 wins=1`<br>`iat=100.0 flagged=1` |
| **5. Equal Timestamps** (`[10.0, 10.0, 10.0(B), 10.0]`) | `acc=4 rej=0 wins=1`<br>`dur=0.0 iat=0.0 pps=3e6 card=2.0` | **Identical** | **Identical** | **Identical** (`flagged=0`) |
| **6. Late After Timeout** (`[10.0, 10.1, 12.5, 10.05, 12.6]`) | `acc=5 rej=0 wins=2`<br>W2 **`iat=1275.0` `pps=30.0`** *(polluted!)* | `acc=4 rej=1 wins=2`<br>W2 **`iat=100.0 pps=20.0`** | `acc=4 rej=1 wins=2`<br>W2 **`iat=100.0 pps=20.0`** | `acc=4 rej=1 wins=2`<br>W2 `iat=100.0 flagged=2` |
| **7. Late Affecting FIFO** (`10.0, 10.1, 10.2(B), 5.0(C), 11.15(B), 11.20(B)`) | `acc=6 rej=0 wins=2`<br>W2 **`src_card=2.0`** *(C trapped in FIFO!)* | `acc=5 rej=1 wins=2`<br>W2 **`src_card=1.0`** *(clean)* | `acc=5 rej=1 wins=2`<br>W2 **`src_card=1.0`** *(clean)* | `acc=5 rej=1 wins=2`<br>W2 **`src_card=1.0`** *(clean)* |
| **8. Interleaved Regressions** (B lags `-0.08s`, C lags `-4.3s`) | `acc=8 rej=0 wins=3`<br>W2 `card=3.0`, W3 **`snap=6.1`** *(regressed)* | `acc=5 rej=3 wins=2`<br>B **`iat=300.0` `pps=6.67`** *(drops B pkt)* | `acc=6 rej=2 wins=2`<br>B **`iat=150.0 pps=10.0`**, C rejected | `acc=5 rej=3 wins=2`<br>B `iat=300.0` `flagged=1` |

**Key Experimental Takeaways:**
- `CURRENT` fails severely in Scenarios 2, 3, 4 (`1375.0 ms` IAT), 6 (`1275.0 ms` IAT post-timeout), 7 (`src_card=2.0` FIFO trap), and 8 (`snap_ts=6.1` backward jump).
- `POLICY A` protects all invariants with zero memory/latency cost, but loses slightly jittered packets in Scenarios 2, 3, and 8.
- `POLICY B` (`W=0.25s`) achieves **100% packet recovery + exact sorted IAT/rate/topology** on mild jitter (`S2`, `S3`, `S8`) while rejecting multi-second anomalies (`S4`, `S6`, `S7`, `S8` Flow C) identically to Policy A.

---

## 5. Recommended Policy (`[RECOMMENDATION]`)

Adopt **Configurable Unified Ingestion Ordering** so feature extraction guarantees strictly non-decreasing timestamps regardless of ingestion mode:

1. **Live Network Capture (`AegisLiveV4Pipeline.ingest_packet` default):** **Policy A (`reorder_window_sec = 0.0`)**
   - *Justification:* Single-interface live kernel capture timestamps are normally monotonic; $0\text{ ms}$ added latency preserves real-time `<1 ms` inference guarantees and prevents attacker-controlled timestamp spoofing memory exhaustion. Reject regressions with explicit telemetry `packet_rejection_reasons["out_of_order_timestamp"]` and mark active affected flows (`has_ooo_drop = True`).
2. **Offline PCAP Replay & Dataset Builder (`replay_pcap` / `LeakageSafeV4DatasetBuilder`):** **Policy B (`reorder_window_sec = 0.25`, bounded `max_reorder_buffer = 10000`)**
   - *Justification:* Authentic multi-device IoT PCAPs (such as Edge-IIoTset sensor/gateway captures) may exhibit minor OS buffer or multi-interface merge jitter ($\le 250\text{ ms}$) without malice. Offline replay tolerates a `250 ms` watermark heap (`drain()` on EOF adds `0` user wait!) while strictly rejecting `> 250 ms` out-of-order regressions cleanly via the exact same Policy A guard inside `ingest_packet()`.
   - *Note:* Always verify builder and replay share identical `reorder_window_sec` when auditing dataset/runtime parity on jittered captures.
3. **Frozen V4 Compatibility Guarantee:** Both `Policy A` (`W=0.0`) and `Policy B` (`W=0.25`) produce **byte-identical output on chronological streams** (`Scenario 1`, `Scenario 5`, and CICIoT2023 V4's 80,000 verified rows) and protect frozen V4 OOD/XGBoost from corrupted `1375 ms` IAT artifacts and trapped FIFO cardinalities.

---

## 6. Exact Future Code Changes (Batch 26D Implementation Spec — Do Not Implement Now)

1. **[`aegis_live/network_topology_features.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/network_topology_features.py):**
   - `NetworkTopologyTracker.add_packet`: Add strict guard before `self.last_ts = ts` (`L232`):
     ```python
     if self.last_ts is not None and ts < self.last_ts:
         raise ValueError(f"Out-of-order packet timestamp {ts:.6f} < last_ts {self.last_ts:.6f}")
     ```
2. **[`aegis_live/packet_window.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/packet_window.py):**
   - `FlowWindow.__init__`: Add `self.ooo_packets_dropped_adjacent: int = 0` (provenance metadata only).
   - `FlowWindow.add_packet`: Enforce `if self.last_timestamp is not None and pkt.timestamp < self.last_timestamp: raise ValueError(...)` before `L60`.
3. **[`aegis_live/v4_runtime.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/v4_runtime.py):**
   - Track `self.global_last_ts: Optional[float] = None` in `AegisLiveV4Pipeline.__init__` and `reset()`.
   - At top of `ingest_packet` (`L291`): If `self.global_last_ts is not None and ts < self.global_last_ts`: increment `self._record_packet_rejection("out_of_order_timestamp")`, flag affected `flow_key` (`ooo_packets_dropped_adjacent += 1` if in `active_windows`), and `return []`. Update `self.global_last_ts = ts` once accepted.
   - Add bounded min-heap reorder wrapper `BoundedReorderBuffer(lateness_sec: float = 0.25, max_packets: int = 10000)` used optionally in `replay_pcap(pcap_path, reorder_window_sec: float = 0.25)` and `replay_packet_stream(..., reorder_window_sec: float = 0.0)` so live capture defaults `0.0` (Policy A) and offline PCAPs can use `0.25` (Policy B) cleanly before feeding monotonic packets into `ingest_packet()`.
   - Track `last_ts = max(last_ts, float(ts))` in `replay_*` before `flush()`.
4. **[`aegis_live/build_v4_dataset.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/build_v4_dataset.py):**
   - Wrap incoming `(ts, buf)` stream in identical `BoundedReorderBuffer` (or strict monotonic reject `out_of_order_timestamp` counter in `exclusion_counts[pcap_file]`) BEFORE quarantine transitions (`L330–360`) and `net_tracker.add_packet` (`L412`) so out-of-order packets cannot bypass train/val/test quarantine boundaries.

---

## 7. Required Regression Tests (For Future Implementation Phase)

1. `test_topology_tracker_rejects_timestamp_regression` — Assert `ValueError` and uncorrupted `packet_queue`/`last_ts`/`src_ip_cardinality` on `ts < last_ts`.
2. `test_flow_window_rejects_timestamp_regression` — Assert `ValueError` and preserved `packets` / IAT invariants on `pkt.timestamp < last_timestamp`.
3. `test_runtime_policy_a_rejects_regressions_and_prevents_fifo_trap` — Run Scenario 4 & Scenario 7 through `AegisLiveV4Pipeline` (`AEGISLiveV4Inference` mocked), asserting exact rejection counts, clean `iat_mean_ms`, `src_ip_cardinality == 1.0`, and `window_start_ts <= window_end_ts`.
4. `test_runtime_policy_a_late_packet_after_timeout_does_not_corrupt_new_window` — Run Scenario 6 through runtime, asserting old & new window IAT/rate parity.
5. `test_bounded_reorder_buffer_recovers_mild_jitter_and_rejects_beyond_watermark` — Run Scenarios 2, 3, 4, and 8 with `BoundedReorderBuffer(lateness_sec=0.25)` proving mild jitter is sorted chronologically (`iat=75.0 ms`, `80.0 ms`, `150.0 ms`) and `-4.3s` / `-5.2s` regressions are rejected with counter.
6. `test_builder_and_runtime_equal_timestamps_and_quarantine_guard` — Verify Scenario 5 equal timestamps accepted monotonically (`seq_id` order) and out-of-order regression cannot bypass `build_v4_dataset.py` quarantine boundary.

---

## 8. Compatibility Risks & Unresolved Questions

### Compatibility Risks
- **`[CONFIRMED FACT]` Zero Risk to Frozen V4 on Chronological Streams:** Scenarios 1 & 5 and all 80,000 existing CICIoT2023 V4 rows satisfy non-decreasing timestamps (`test_all_80k_v4_rows_temporal_invariants`).
- **`[DESIGN RISK]` Packet Loss Sensitivity under Policy A on Jittered PCAPs:** If `replay_pcap` is run with `W=0.0` on a multi-interface capture with `-10ms` skew between flows, legitimate packets will be dropped, slightly lowering `pkt_rate_pps` and altering `iat_mean_ms`. Using Policy B (`W=0.25s`) on offline PCAPs mitigates this completely.
- **`[DESIGN RISK]` Bounded Heap Cap Tradeoff under Floods:** If a 100k pps attack hits a `max_packets = 10000` heap cap with `W=0.25s`, forcing early watermark eviction drops packets lagging $> 100\text{ ms}$; telemetry must record `reorder_buffer_forced_evictions`.

### Explicit Unresolved Questions (`[INSUFFICIENT EVIDENCE]`)
1. **Do authentic Edge-IIoTset PCAPs contain out-of-order timestamps, and what is their maximum regression magnitude ($\Delta t$)?** *Cannot be answered without authorized acquisition and inspection of Edge-IIoTset PCAPs (prohibited in Batch 26C).*
2. **Do any existing CICIoT2023 PCAPs (from `BenignTraffic.pcap`, `DDoS-SYN_Flood.pcap`, etc.) have minor regressions filtered/masked before reaching `validation.csv`/`test.csv`?** *Cannot be answered without PCAP scans (prohibited in Batch 26C).*
3. **Should Scenario 5 (`equal timestamps` yielding `duration = 0.0 -> pps = N * 1e6 > 1e7`) be bounded or left untouched for frozen V4 parity?** *Tied to Phase 6 schema bounds reconciliation (`BATCH26A_V4_REMEDIATION_PLAN.md`).*

---

## 9. Final Safety Footer

- **Files inspected:** [`aegis_live/packet_window.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/packet_window.py), [`aegis_live/network_topology_features.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/network_topology_features.py), [`aegis_live/v4_runtime.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/v4_runtime.py), [`aegis_live/build_v4_dataset.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/build_v4_dataset.py), `tests/test_aegis_live_*.py`, and Batch 25/26A/26B/26B.1 reports.
- **Files modified:** `0` source or test files modified (only this investigation deliverable [`reports/BATCH26C_TIMESTAMP_ORDERING_POLICY.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/reports/BATCH26C_TIMESTAMP_ORDERING_POLICY.md) and conversation scratch simulator outside repo).
- **PCAP scans:** `NO`
- **Dataset generation:** `NO`
- **Training:** `NO`
- **Inference:** `NO` (`AEGISLiveV4Inference` mocked in synthetic policy benchmark)
- **Frozen hashes changed:** `0` (all 7 frozen V4 & V6.3 SHA-256 hashes unchanged)
- **Automated blocking:** `NO`


# BATCH 26A — AEGIS V4 Compatibility Remediation Plan

**Date:** 2026-11-12  
**Mode:** Planning-Only Task (Zero Source, Test, Dataset, or Model Modifications)  
**Baseline Reference:** [`reports/BATCH25_V4_COMPATIBILITY_AUDIT.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/reports/BATCH25_V4_COMPATIBILITY_AUDIT.md)  
**Production Freeze:** V4 Authoritative Model (29 Features) & V6.3 Staging Artifacts Frozen; Edge-IIoTset remains **NOT APPROVED** for direct V4 inference or training.

---

## 1. Executive Summary

This remediation plan translates the verified findings of the Batch 25 audit into a **six-phase, compatibility-preserving engineering roadmap** for a future implementation phase (**Batch 26B**).

**Core Design Principle:** Frozen V4 (`models/aegis_live_v4/aegis_live_xgboost_baseline.pkl`, temperature calibration, Mahalanobis OOD parameters, and abstention thresholds) was trained strictly on:
- **Exact 50-packet** windows (`WINDOW_CAPACITY = 50`),
- **3-tuple** `(sorted_src_ip, sorted_dst_ip, ip_proto)` FlowWindow grouping,
- **Raw Ethernet frame buffer length** (`len(buf)`), and
- **Destination-subnet directionality** (`is_inbound = _is_local(dst_ip)` against `192.168.0.0/16`).

Consequently, remediation must **strictly separate** two classes of action:
1. **Safe V4 Bug Fixes (Allowed in Batch 26B without invalidating frozen V4):** Fixing future-topology leakage on timeout/flush windows (Phase 1), enforcing strict monotonic timestamp policy (Phase 2), unifying builder/runtime malformed-header parsing and diagnostic telemetry (Phase 4), adding non-mutating directionality degeneracy diagnostics (Phase 5), and enabling reconciled feature contract validation + sub-capacity provenance flags (Phase 6).
2. **Distribution-Breaking Structural Changes (Strictly Prohibited for Frozen V4 — Deferred to V7):** Switching FlowWindow keys to 5-tuple, switching `len(buf)` to `ip.len` or payload length, altering destination-subnet directionality math, or retraining/recalibrating thresholds.

---

## 2. Batch 25 Finding-by-Finding Assessment

| Batch 25 ID | Classification | Summary | Remediation Verdict for Batch 26B |
|---|---|---|---|
| **D1** | `CONFIRMED DEFECT` (Critical) | Timeout & `flush()` future-topology leakage in `v4_runtime.py` | **FIX IN PHASE 1** — Snapshot per-window topology at each window's `last_timestamp`. |
| **D2** | `CONFIRMED DEFECT` (High) | Out-of-order timestamp state corruption in `FlowWindow` & `NetworkTopologyTracker` | **FIX IN PHASE 2** — Enforce Policy 1 (strict monotonic rejection + explicit telemetry). |
| **D3** | `CONFIRMED DEFECT` (High) | Builder (`N == 50`) vs. runtime (`2 <= N <= 50`) window-size mismatch | **MITIGATE IN PHASE 6** — Tag sub-capacity provenance (`is_subcapacity_window`) & add optional `strict_full_window_only` parity mode; do not alter V4 training contract. |
| **D4** | `CONFIRMED DEFECT` (Medium) | Builder omits `ihl < 20`, `len < 14+ihl`, and TCP/UDP truncation checks | **FIX IN PHASE 4** — Unify strict L3/L4 validation across runtime and builder via shared helper. |
| **D5** | `CONFIRMED DEFECT` (Low) | Unreachable dead branch at `build_v4_dataset.py` L423 | **CLEAN UP IN PHASE 4** — Remove dead capacity-50 acceptance inside `gap >= timeout_sec`. |
| **M1** | `DOCUMENTATION MISMATCH` (Med) | Docs claim "5-tuple flow window" while code uses 3-tuple `(ip_a, ip_b, proto)` | **FIX DOCS IN PHASE 3** — Update docstrings to "3-tuple host-pair + IP-protocol window". |
| **M2** | `DOCUMENTATION MISMATCH` (Low) | `v4_runtime.py` L205–206 topology causality claim violated on timeout/flush | **RESOLVED BY PHASE 1** — Snapshot fix makes the docstring contract strictly true. |
| **M3** | `DOCUMENTATION MISMATCH` (Low) | `FEATURE_DEFINITIONS` bounds (`20`, `1e7`) vs. `validate_candidate_window` (`14`) vs. `packet_window` (`5e7`), plus unused `validate_29_features()` imports | **RECONCILE IN PHASE 6** — Align schema bounds and invoke `validate_29_features()` cleanly. |
| **R1** | `DESIGN RISK` (High) | 3-tuple merges multi-port TCP/UDP services between same two hosts | **DEFER TO V7** — Retain 3-tuple for V4 CICIoT2023 compatibility; record multi-port contention in window telemetry (`distinct_port_pairs_seen`). |
| **R2** | `DESIGN RISK` (High) | `is_inbound = _is_local(dst_ip)` collapses East-West LAN and unregistered subnets | **MITIGATE IN PHASE 5** — Preserve V4 math; add explicit 4-category direction diagnostics + degeneracy flags. |
| **R3** | `DESIGN RISK` (Medium) | `len(buf)` sensitivity to NIC padding & `0x8100` VLAN frame drops | **MITIGATE IN PHASE 4** — Preserve `len(buf)` for V4 parity; add explicit `vlan_8021q_dropped` and `ipv4_fragment_detected` telemetry counters. |
| **R4** | `DESIGN RISK` (Medium) | Single active window per `(ip_a, ip_b, proto)` under multi-port multiplexing | **DEFER TO V7** — Linked directly to R1 (3-tuple V4 contract). |
| **N1–N3** | `NO ISSUE FOUND` | 29-feature order, streaming-vs-batch math on monotonic 50-pkt windows | **PRESERVE** — Verify zero regression via existing 38 unit tests. |
| **E1** | `INSUFFICIENT EVIDENCE` | Whether `aegis_live/v4_inference.py` calls `validate_29_features()` or casts `float32` | **VERIFY PRIOR TO PHASE 6** — Scope-bounded outside Batch 25/26A (`v4_inference.py` uninspected). |
| **E2** | `INSUFFICIENT EVIDENCE` | Exact padding, VLAN tags, subnets, and timestamp order inside original CICIoT2023 & unacquired Edge-IIoTset PCAPs | **VERIFY ON INGESTION** — No PCAP scans in Batch 26A; telemetry counters from Phase 4/5 will surface this automatically when PCAPs are replayed. |

---

## 3. Root Cause and Exact Code Locations

### 3.1 Temporal Causality Leakage (Issue A / D1)
- **Root Cause 1 — Post-Mutation Timeout Check (`aegis_live/v4_runtime.py` L331–368):**
  * At `L340`, `self.net_tracker.add_packet(net_pkt)` ingests the arriving packet at `ts = t_new` into `NetworkTopologyTracker` (`mode="sliding"`, `window_sec=1.0`).
  * Inside `NetworkTopologyTracker.add_packet()` (`aegis_live/network_topology_features.py` L263–274), `cutoff_ts = ts - 1.0` evicts every packet older than `[t_new - 1.0, t_new]`.
  * At `L362–368`, `AegisLiveV4Pipeline.ingest_packet()` checks `if gap >= self.timeout_sec:` (`2.0s`). If an existing window with `2 <= len < 50` ended at `t_old_end <= t_new - 2.0`, `L367` calls `self._process_completed_window(window)` -> `_extract_combined_29_features(window)` (`L200–218`) -> `self.net_tracker.extract_current_features()` (`L209`). Because `t_new - t_old_end >= 2.0 > 1.0`, **zero packets from `[t_old_end - 1.0, t_old_end]` remain in `net_tracker`**. The old window inherits topology features at `t_new`.
- **Root Cause 2 — End-of-Stream `flush()` Uses Global `last_ts` (`aegis_live/v4_runtime.py` L398–420, L437, L493):**
  * `flush()` iterates over `self.active_windows` and calls `self._process_completed_window(window)` (`L410`) for every remaining window with `len >= min_packets`. Every flushed window reads `self.net_tracker.extract_current_features()`, which reflects `last_ts` (end of stream), regardless of each window's actual `window.last_timestamp`.
- **Note on Inter-Flow Clock Advance:** Even reordering L340 *after* L368 would **not** fix Flow A if Flow B sent packets at `t=10.5..11.5` after Flow A stopped at `t=10.0` (before Flow A timed out at `t=12.5` or `flush()`). At `t=12.5`, `self.net_tracker` is already at `t=11.5`, not `t=10.0`. Only per-window causal snapshots solve both inter-flow advancement and end-of-stream flush!

### 3.2 Out-of-Order Timestamps (Issue B / D2)
- `aegis_live/network_topology_features.py` L231–232 unconditionally sets `self.last_ts = ts`, and L264 appends `packet` to the right of `self.packet_queue` even if `ts < self.last_ts`. L269–271 (`while self.packet_queue[0].timestamp < cutoff_ts:`) and L273 (`window_start_ts = self.packet_queue[0].timestamp`) break FIFO monotonicity when queue elements are unsorted.
- `aegis_live/packet_window.py` L52–62 computes `gap = pkt.timestamp - self.last_timestamp` (negative on regression), keeps `self.last_timestamp = max(...)` at L61, yet appends unsorted `pkt` at L62. `extract_features()` L86–87 (`packets[0].timestamp`, `packets[-1].timestamp`) and L107–109 (`max(dt, 0.0)`) then read unsorted endpoints and clamp negative deltas.

### 3.3 Flow Identity & Parsing Parity (Issues C & D)
- **3-Tuple Identity:** `aegis_live/build_v4_dataset.py` L412–414 and `aegis_live/v4_runtime.py` L355–356 use `(ip_pair[0], ip_pair[1], ip_proto)`, contradicting docstrings at `build_v4_dataset.py` L18/L649, `network_topology_features.py` L7, and `v4_runtime.py` L119.
- **Parser Divergence:** `build_v4_dataset.py` L377–389 omits `ihl < 20`, `len(buf) < 14 + ihl` (present in `v4_runtime.py` L299–302), `truncated_tcp_header` (`v4_runtime.py` L315–321), and `truncated_udp_header` (`v4_runtime.py` L323–328).
- **Packet Length & Encapsulation:** Both use `len(buf)` (`build_v4_dataset.py` L405, `v4_runtime.py` L346), reject non-`0x0800` Ethernet frames (L361 vs L290), and do not inspect IPv4 `frag_offset`.

### 3.4 Directionality & Feature Contract (Issues E & F)
- **Directionality:** `build_v4_dataset.py` L77, L119–123, L402 and `v4_runtime.py` L125, L168–179, L343 compute `is_inbound = self._is_local(dst_ip)` with default `192.168.0.0/16`.
- **Feature Contract:** `build_v4_dataset.py` L130 enforces `len == 50`; `v4_runtime.py` L128 defaults `min_packets_to_extract = 2`. `packet_window.py` L92 allows `pkt_rate_pps = 5e7 > 1e7`. `validate_29_features()` (`feature_schema.py` L180–224) is imported unused at `v4_runtime.py` L47 and `build_v4_dataset.py` L49.

---

## 4. Recommended Technical Decisions with Alternatives

### Topic A: Causal Topology Snapshot Strategy
- **Recommended Decision:** **O(1) Per-Window Topology Snapshot Cache (`last_topology_snapshot`) on `FlowWindow` (or runtime window map).**
  * Whenever a packet `pkt_rec` is added to an existing or fresh `FlowWindow` (immediately after `self.net_tracker.add_packet(net_pkt)` updates `net_tracker` to `ts == window.last_timestamp`!), store the current 5-feature topology dictionary from `self.net_tracker.extract_current_features()` on that window (`window_snapshots[flow_key]`).
  * When a window completes at capacity (`len == 50`), the current `net_tracker` state *is* the snapshot at `t_50`.
  * When a window instead flushes later via `gap >= timeout_sec` or `flush()`, `_process_completed_window()` reads the window's saved `last_topology_snapshot` (captured at its own `window.last_timestamp`) instead of querying the advanced `self.net_tracker`.
  * **Handles Multiple Concurrent Windows:** If Flow A ends at `t=10.2s`, Flow B at `t=14.7s`, and Flow C at `t=25.0s`, each window holds its own exact `[T_end - 1.0, T_end]` snapshot independently, with **zero PCAP rescans** and **O(1) memory per active flow**!
- **Alternative Considered & Rejected:**
  * *Bounded Replay Event Ring Buffer (`deque` of packet tuples over last `max_idle_sec`):* Consumes $O(\text{PPS} \times \text{retention})$ memory during floods, degrades to $O(W)$ scans per flushed flow, and loses history if a flow sits idle longer than the buffer horizon before `flush(force_all=True)`.
  * *Reordering `add_packet` after timeout check only:* Fails whenever another flow (Flow B) advances `net_tracker` between Flow A's last packet and Flow A's timeout/flush.

### Topic B: Out-of-Order Timestamp Policy
- **Recommended Decision:** **Policy 1 — Strict Monotonic Timestamp Rejection + Explicit Telemetry.**
  * Enforce `if ts < self.last_ts:` inside `NetworkTopologyTracker.add_packet()` (raise `ValueError` or return explicit `(False, None)` with rejection state, configurable) and in `AegisLiveV4Pipeline.ingest_packet()` **before** mutating `net_tracker` or any `FlowWindow`: track `global_last_ts`; if `ts < global_last_ts`, reject the packet immediately, increment `packet_rejection_reasons["out_of_order_timestamp"]`, and return `[]`.
  * Similarly, in `FlowWindow.add_packet()`, raise `ValueError` if `pkt.timestamp < self.last_timestamp` (preventing direct callers from inverting `packets[0]` / `packets[-1]`).
  * For offline PCAPs generated by multi-interface merges with minor jitter, provide an explicit opt-in offline pre-sort check/warning outside live streaming rather than silently buffering or clamping in runtime.
- **Alternatives Considered & Rejected:**
  * *Policy 2 (Bounded Reorder Buffer / Watermark):* Adds buffering latency to real-time detection, complicates topology-vs-flow synchronization, and alters window closure order near capacity/timeout boundaries.
  * *Policy 3 (Timestamp Clamping `ts = max(ts, last_ts)`):* **Unsafe.** Fabricates `dt = 0.0` micro-bursts, triggers `safe_duration_sec = 1e-6` rate explosions (`5e7` pps), and corrupts `iat_mean_ms`/`iat_std_ms`.

### Topic C: Flow Identity Decision
- **Recommended Decision:** **Preserve Existing 3-Tuple `(sorted_src_ip, sorted_dst_ip, ip_proto)` for Frozen V4; Correct All Docstrings; Defer 5-Tuple to V7.**
  * *Evidence Basis:* `build_v4_dataset.py` (L412–414, L670) proves Frozen V4 (`train.csv`, `ood_parameters.pkl`, XGBoost splits) was trained on 3-tuple windows specifically because CICIoT2023 DoS generators randomize ephemeral `sport` per packet. Switching frozen V4 to bidirectional 5-tuple would shatter DoS flood windows into 1-packet orphans (zero capacity-50 windows) and invalidate OOD parameters.
  * Fix documentation (`DOCUMENTATION MISMATCH` M1) to accurately say **"3-tuple host-pair + IP-protocol window"**.
  * Add non-feature provenance telemetry (`unique_port_pairs_in_window` / `ports_seen`) so analysts/auditors can observe when a 3-tuple V4 window multiplexed multiple IoT services.

### Topic D: Packet Length & Parsing Parity
- **Recommended Decision:** **Preserve `len(buf)` (Raw Captured Frame Length) for Frozen V4; Unify Strict L3/L4 Header Validation.**
  * *Evidence Basis:* `build_v4_dataset.py` L405 and `v4_runtime.py` L346 identically use `len(buf)`. Frozen V4 training distributions (`distribution_audit.csv`, `ood_parameters.pkl`) are calibrated to `len(buf)`. Changing to `ip.len` (-14B shift) or transport payload length would break frozen V4 splits around tight control-packet sizes without retraining.
  * *Unrevealed Evidence (`INSUFFICIENT EVIDENCE`):* Whether original CICIoT2023 PCAPs had 60B NIC padding or unpadded host captures cannot be inspected under Batch 26A's no-PCAP-scan rule.
  * *Parser Parity Fix:* Create a shared parser helper enforcing identical checks in both `v4_runtime.py` and `build_v4_dataset.py`:
    - `len(buf) >= 14`, `eth_type == 0x0800` (with explicit telemetry counter if `eth_type == 0x8100` VLAN),
    - `len(buf) >= 34`, `ihl >= 20`, `len(buf) >= 14 + ihl`,
    - Reject non-initial IPv4 fragments (`frag_offset > 0` or record `ipv4_fragment_dropped`) or avoid parsing L4 ports/flags when `frag_offset > 0`,
    - Reject `truncated_tcp_header` (`len < 14 + ihl + 14`) and `truncated_udp_header` (`len < 14 + ihl + 4`).

### Topic E: Directionality
- **Recommended Decision:** **Preserve `_is_local(dst_ip)` Math for Frozen V4 Parity; Add Subnet & East-West Degeneracy Telemetry + Guardrails.**
  * *Evidence Basis:* `build_v4_dataset.py` L77, L402 trained Frozen V4 using `is_inbound = _is_local(dst_ip)` against `192.168.0.0/16`. Altering this formula (e.g., to initiator/responder or `src`-checked symmetry) shifts `inbound_pkt_ratio` / `outbound_pkt_ratio` away from the frozen training manifold.
  * *Recommended Telemetry Matrix for Every Window/Session:*
    - Track packet counts across all 4 endpoint cases:
      1. `local -> external` (`is_inbound = False`)
      2. `external -> local` (`is_inbound = True`)
      3. `local -> local` (`is_inbound = True` under current V4 math — **flag as `east_west_both_local`**)
      4. `external -> external` (`is_inbound = False` under current V4 math — **flag as `unregistered_or_external_both`**)
    - If a window or session has `100%` `local -> local` or `100%` `external -> external` (unconfigured subnet, e.g., `10.0.0.0/8` with default `192.168.0.0/16`), attach explicit provenance warning (`directionality_degenerate = True` / reason flag) so evaluators never mistake East-West or misconfigured subnet traffic for valid perimeter V4 directionality!

### Topic F: Feature Contract
- **Recommended Decision:** **Reconcile Schema Bounds, Wire `validate_29_features()` Cleanly, and Flag Sub-Capacity (`N < 50`) Provenance.**
  * Reconcile `FEATURE_DEFINITIONS` vs. `validate_candidate_window()` vs. `validate_29_features()`:
    - `pkt_size_min` minimum: IPv4+Ethernet minimum parsed packet is `34` bytes (`v4_runtime.py` L294 / `build_v4_dataset.py` L365); update documentation/bounds consistently (`>= 34` for parsed frames, or keep `>= 14`/`20` metadata clear).
    - `pkt_rate_pps` upper bound: On $N=50$ identical-timestamp packets (`duration_sec = 0`), `safe_duration_sec = 1e-6` yields `5e7`. Either keep `FEATURE_DEFINITIONS["pkt_rate_pps"]["max"] = 1e8` (matching `50 / 1e-6` + safety margin) or note clamped behavior explicitly.
  * **Where to call `validate_29_features()`:**
    - First inspect `aegis_live/v4_inference.py` at the start of Batch 26B to confirm whether it already casts to `float32` / validates.
    - Call `validate_29_features(feats_29)` (or a lightweight dict validator to avoid per-window Pandas DataFrame overhead if latency-critical, keeping identical rules!) inside `build_v4_dataset.py` and optionally in `v4_runtime.py` `_process_completed_window()` before inference, removing dead unused imports.
  * **Full-Capacity (`N=50`) vs. Sub-Capacity (`2 <= N <= 49`):**
    - Add `is_subcapacity: bool` (`packet_count < max_packets`) to `WindowProvenanceRecordV4` and optional `strict_full_window_only: bool = False` constructor flag on `AegisLiveV4Pipeline` so benchmark harnesses evaluating V4 parity can restrict strictly to `N == 50` (matching `build_v4_dataset.py`), while live mode retains `min_packets = 2` with explicit sub-capacity disclosure.

---

## 5. Six-Phase Remediation Plan

### Prioritized Remediation Summary Table

| Priority | Phase | Issue | Files / Functions | Proposed Change | Regression Test | Compatibility Risk |
|---|---|---|---|---|---|---|
| **P0 (Critical)** | **Phase 1: Temporal Correctness** | D1, M2: Timeout & `flush()` future topology leakage | `v4_runtime.py`: `FlowWindow` snapshot tracking in `ingest_packet`, `_extract_combined_29_features`, `_process_completed_window`, `flush` | Cache 5-feature topology snapshot per active window at `window.last_timestamp`; read cached snapshot on timeout/`flush()`. | `test_timeout_flush_uses_window_end_topology`, `test_end_of_stream_multi_window_flush_causality` | **NONE** (Preserves bit-exact 50-pkt output; fixes broken timeout/flush causality). |
| **P1 (High)** | **Phase 2: Timestamp Policy** | D2: Out-of-order corruption in FIFO & `FlowWindow` | `v4_runtime.py`: `ingest_packet`; `network_topology_features.py`: `NetworkTopologyTracker.add_packet`; `packet_window.py`: `FlowWindow.add_packet` | Enforce strict monotonic policy (`ts >= last_ts`); reject regressions with explicit counter instead of corrupting `deque`/IAT. | `test_topology_rejects_timestamp_regression`, `test_flow_window_rejects_regression`, `test_runtime_counts_out_of_order_rejection` | **NONE** on chronological PCAPs/streams. Explicit error on unsorted inputs. |
| **P1 (High)** | **Phase 3: Flow Identity Decision** | M1, R1: 3-tuple vs 5-tuple docs mismatch & multi-port visibility | `build_v4_dataset.py` L18, L649; `v4_runtime.py` L119; `network_topology_features.py` L7 | Keep 3-tuple for V4; fix docstrings ("3-tuple host-pair + IP-proto"); track `port_pairs_seen` telemetry on `WindowProvenanceRecordV4`. | `test_3tuple_flow_key_contract_preserved`, `test_multi_port_provenance_telemetry` | **NONE** (Explicitly protects frozen V4 from 5-tuple DoS breakage). |
| **P2 (Medium)** | **Phase 4: Parser & Packet-Length Parity** | D4, D5, R3: Missing `ihl`/L4 bounds in builder, VLAN/fragmentation telemetry | Shared Ethernet/IPv4/L4 parser helper in `packet_window.py` (or `v4_runtime.py`), called by `build_v4_dataset.py` & `v4_runtime.py` | Keep `len(buf)`; enforce identical `ihl >= 20`, `len >= 14+ihl`, TCP/UDP truncation & fragment offset handling; track VLAN `0x8100`; remove dead L423 code. | `test_builder_runtime_parser_parity`, `test_vlan_8021q_telemetry_counter`, `test_ipv4_fragment_safe_handling` | **LOW** (Identical on valid untagged TCP/UDP; prevents invalid IHL/truncated crash/garbage). |
| **P2 (Medium)** | **Phase 5: Directionality** | R2: East-West `(1,0)` and unconfigured `(0,1)` collapse | `v4_runtime.py`: `_is_local`, `ingest_packet`, `get_summary_statistics`; `FlowWindow` provenance | Keep `_is_local(dst_ip)` formula for V4; track 4-way matrix (`L->E`, `E->L`, `L->L`, `E->E`) & flag `directionality_degenerate`. | `test_directionality_diagnostics_both_local_and_neither_local` | **NONE** (Purely additive telemetry/provenance; math unchanged). |
| **P3 (Low-Med)** | **Phase 6: Feature Contract Validation** | D3, M3: Bounds reconciliation, unused `validate_29_features()`, `N<50` flag | `feature_schema.py`: `FEATURE_DEFINITIONS`, `validate_29_features`; `v4_runtime.py`: `_process_completed_window`; `build_v4_dataset.py`: `validate_candidate_window` | Align bounds (`pkt_rate_pps`, `pkt_size_min`); wire `validate_29_features()` (or dict equivalent); add `is_subcapacity` provenance & `strict_full_window_only` option. | `test_validate_29_features_bounds_consistency`, `test_subcapacity_provenance_flag`, `test_strict_full_window_only_mode` | **LOW** (Verify `v4_inference.py` compatibility first; default runtime behavior preserved). |

---

### Detailed Six-Phase Specifications

#### Phase 1: Temporal Correctness (P0)
- **Exact Functions to Modify:**
  - `aegis_live/v4_runtime.py`: `AegisLiveV4Pipeline.__init__`, `reset`, `_extract_combined_29_features`, `_process_completed_window`, `ingest_packet`, `flush`.
- **Expected Behavior:**
  - *Before:* Calling `net_tracker.add_packet` at `t_new` before timeout flush overwrites the `1.0s` sliding buffer, causing `window` (which ended at `t_old_end <= t_new - 2.0`) to receive future topology features from `t_new`. `flush()` applies final `last_ts` topology to all lingering windows.
  - *After:* Maintain `self.window_topology_snapshots: Dict[Tuple[str, str, int], Dict[str, float]]`. Every time a packet is added to `window`, immediately save `self.net_tracker.extract_current_features()`. On timeout flush or `flush()`, `_extract_combined_29_features` reads the saved snapshot for `flow_key` (asserting `snapshot["window_end_ts"] == window.last_timestamp`) instead of the current `net_tracker`.
- **Tests to Add:** Deterministic timeout test with intervening traffic burst from spoofed IPs; multi-window end-of-stream `flush()` test verifying Flow A (`t=10s`) and Flow B (`t=20s`) retain independent topology snapshots.
- **Acceptance Criteria:** Replaying packets `[0..T_k]` vs. `[0..T_end]` produces bit-identical 29-feature vectors for every window whose `window_end_ts <= T_k`, including timeout-flushed and end-of-stream flushed windows.
- **Frozen Model Compatibility Affected?** **NO** — Capacity-50 chronological windows are byte-identical; only buggy timeout/flush vectors become causally accurate.

#### Phase 2: Timestamp Policy (P1)
- **Exact Functions to Modify:**
  - `aegis_live/network_topology_features.py`: `NetworkTopologyTracker.add_packet`
  - `aegis_live/packet_window.py`: `FlowWindow.add_packet`
  - `aegis_live/v4_runtime.py`: `AegisLiveV4Pipeline.__init__`, `reset`, `ingest_packet`
- **Expected Behavior:**
  - *Before:* Out-of-order `ts < last_ts` silently corrupts `deque` order, regresses `last_ts`, and produces unsorted `FlowWindow.packets` with `0.0` clamped durations.
  - *After:* Track `self.global_last_ts` in `AegisLiveV4Pipeline`. If `ts < self.global_last_ts`, increment `packet_rejection_reasons["out_of_order_timestamp"]` and return `[]` cleanly without mutating `net_tracker` or `active_windows`. `NetworkTopologyTracker.add_packet` and `FlowWindow.add_packet` raise `ValueError` on `ts < last_ts` if called directly out of order.
- **Tests to Add:** Direct unit tests on `NetworkTopologyTracker`, `FlowWindow`, and `AegisLiveV4Pipeline` feeding `[100.0, 100.5, 99.9]` verifying rejection, intact FIFO state, and incremented counter.
- **Acceptance Criteria:** Zero state corruption on timestamp regression; 100% pass rate on all existing monotonic tests.
- **Frozen Model Compatibility Affected?** **NO** for monotonic captures.

#### Phase 3: Flow Identity Decision (P1)
- **Exact Functions / Locations to Modify:**
  - Docstrings only in `aegis_live/build_v4_dataset.py` (L5, L18, L649), `aegis_live/v4_runtime.py` (L119), `aegis_live/network_topology_features.py` (L7).
  - Telemetry metadata addition in `aegis_live/packet_window.py` (`FlowWindow.port_pairs_seen`) and `aegis_live/v4_runtime.py` (`WindowProvenanceRecordV4`).
- **Expected Behavior:**
  - *Before:* Code uses 3-tuple `(ip_a, ip_b, proto)` while docs falsely claim "5-tuple"; no count of distinct `(sport, dport)` conversations merged per window.
  - *After:* Docs accurately state "3-tuple host-pair + IP-protocol window"; provenance tracks how many distinct bidirectional 5-tuple connections merged inside the window (`merged_connection_count`), without touching the 29 numeric features.
- **Tests to Add:** Assert `flow_key` stays 3-tuple and `merged_connection_count == 2` when MQTT + HTTP share `(192.168.1.10, 192.168.1.1, 6)`.
- **Acceptance Criteria:** Zero change to 29-feature numeric values; clear audit visibility of multi-service multiplexing.
- **Frozen Model Compatibility Affected?** **NO** (Preserves V4 3-tuple feature distribution).

#### Phase 4: Parser & Packet-Length Parity (P2)
- **Exact Functions to Modify:**
  - Extract shared `parse_ipv4_ethernet_frame(buf)` helper (e.g. in `aegis_live/packet_window.py`), invoked by `aegis_live/v4_runtime.py` (`ingest_packet`) and `aegis_live/build_v4_dataset.py` (`extract_pcap_windows`).
  - Remove unreachable dead code in `aegis_live/build_v4_dataset.py` (L423–433).
- **Expected Behavior:**
  - *Before:* `v4_runtime.py` rejects `ihl < 20`, `len < 14 + ihl`, `truncated_tcp_header`, `truncated_udp_header`, whereas `build_v4_dataset.py` ingests them. Neither checks `frag_offset > 0` or distinguishes VLAN `0x8100`.
  - *After:* Identical L2/L3/L4 validation in both builder and runtime using `len(buf)`; non-initial IPv4 fragments (`frag_offset > 0`) do not read garbage `sport`/`dport`/`tcp_flags`; VLAN `0x8100` frames are recorded specifically under `non_ipv4_vlan_8021q` rejection reason.
- **Tests to Add:** Feed truncated TCP/UDP, malformed `ihl=12`, VLAN `0x8100`, and fragmented IPv4 (`frag_offset > 0`) to both helper paths.
- **Acceptance Criteria:** Builder and runtime emit identical rejection outcomes and identical `PacketRecord` / `NetworkTopologyPacket` objects for all synthetic frames.
- **Frozen Model Compatibility Affected?** **LOW / NONE** on valid unfragmented Ethernet II IPv4 frames.

#### Phase 5: Directionality Observability (P2)
- **Exact Functions to Modify:**
  - `aegis_live/v4_runtime.py`: `ingest_packet`, `_process_completed_window`, `WindowProvenanceRecordV4`, `get_summary_statistics`.
- **Expected Behavior:**
  - *Before:* East-West `local -> local` silently becomes `inbound=1.0, outbound=0.0` and unregistered `10.x -> 10.x` silently becomes `inbound=0.0, outbound=1.0` with no warning.
  - *After:* Exact same `is_inbound` boolean for V4 numeric features, plus explicit per-window and session counters (`both_local_pkts`, `neither_local_pkts`, `directionality_degenerate`).
- **Tests to Add:** Synthetic East-West and `10.0.0.0/8` tests confirming `directionality_degenerate == True` while keeping `inbound_pkt_ratio` parity.
- **Acceptance Criteria:** Zero change to 29 features; automatic detection of East-West / misconfigured-subnet capture mismatch.
- **Frozen Model Compatibility Affected?** **NO** (Telemetry only).

#### Phase 6: Feature Contract Validation (P3)
- **Exact Functions to Modify:**
  - First inspect `aegis_live/v4_inference.py` (outside Batch 25/26A scope) before touching inference handoff.
  - `aegis_live/feature_schema.py`: Reconcile `FEATURE_DEFINITIONS` (`pkt_rate_pps.max`, `pkt_size_min.min`) with parser reality; export lightweight dict-level `validate_29_features_dict()`.
  - `aegis_live/v4_runtime.py`: Add `strict_full_window_only: bool = False`, add `is_subcapacity` provenance flag, and remove unused `validate_29_features` import (or call it).
  - `aegis_live/build_v4_dataset.py`: Clean up unused import / align `validate_candidate_window()`.
- **Expected Behavior:**
  - *Before:* `validate_29_features` unused; `pkt_rate_pps` on zero-duration bursts reaches `5e7 > 1e7`; runtime emits $N < 50$ windows silently alongside $N = 50$.
  - *After:* Validated finite float contract on every emitted window; `is_subcapacity = (pkt_cnt < self.max_packets)` explicitly recorded; `strict_full_window_only=True` allows strict `N == 50` parity with offline builder.
- **Tests to Add:** Boundary checks on zero-duration burst bounds, sub-capacity flag check (`N=5` -> `is_subcapacity=True`), and `strict_full_window_only=True` behavior.
- **Acceptance Criteria:** Zero NaN/Inf, consistent bounds between schema and builder, full provenance on sub-capacity emissions.
- **Frozen Model Compatibility Affected?** **LOW** (verify against `v4_inference.py` and existing tests before enabling strict upper-bound rejections in live runtime).

---

## 6. Deterministic Regression Test Specifications

| Spec ID | Target Phase | Scenario & Inputs | Deterministic Assertions |
|---|---|---|---|
| **SPEC-1** | Phase 1 | Flow A sends 5 TCP packets at `t=10.0..10.4` while 1 topology source is active (`src_ip_cardinality=1.0`). Flow B sends packets from 10 distinct IPs at `t=11.0..11.5`. Flow A receives packet #6 at `t=13.0` (`gap = 2.6s >= 2.0s`). | Flushed Flow A window (`t_end=10.4`) MUST have `src_ip_cardinality == 1.0`, `window_end_ts == 10.4` (NOT `10.0`/`11.0` from Flow B or packet #6). |
| **SPEC-2** | Phase 1 | Flow A (`t=10.0..10.4`, 3 pkts, 2 sources) and Flow B (`t=20.0..20.4`, 3 pkts, 8 sources) remain incomplete; call `pipeline.flush(current_time=25.0, force_all=True)`. | Flow A detection emits `src_ip_cardinality == 2.0`; Flow B emits `src_ip_cardinality == 8.0`. |
| **SPEC-3** | Phase 2 | Feed packets at `t = [100.0, 100.5, 99.8]` into `AegisLiveV4Pipeline`. | Packet 3 is rejected cleanly; `pipeline.packet_rejection_reasons["out_of_order_timestamp"] == 1`; `net_tracker.last_ts == 100.5`; `FlowWindow` duration/IAT unaffected. |
| **SPEC-4** | Phase 2 | Call `NetworkTopologyTracker.add_packet` and `FlowWindow.add_packet` directly with `t=99.0` after `t=100.0`. | Both raise `ValueError` explicitly without mutating internal `deque` or `packets`. |
| **SPEC-5** | Phase 3 | Feed interleaved MQTT (`1883`) and HTTP (`80`) TCP packets between `192.168.1.10` and `192.168.1.1`. | Verify 3-tuple `flow_key == ('192.168.1.1', '192.168.1.10', 6)` (V4 preserved) AND `provenance.merged_connection_count == 2`. |
| **SPEC-6** | Phase 4 | Pass truncated TCP (`38B`), malformed IP `ihl=12`, VLAN `0x8100`, and fragmented IPv4 (`MF=0, offset=100`) to shared parser used by runtime and builder. | Truncated/malformed rejected identically in both; VLAN increments `non_ipv4_vlan_8021q`; fragment sets `sport=0, dport=0, tcp_flags=0` safely. |
| **SPEC-7** | Phase 5 | Run 50 bidirectional packets between `192.168.1.10 <-> 192.168.1.20` (both local) and separately `10.0.0.10 <-> 10.0.0.20` (neither local with default `192.168.0.0/16`). | Verify `inbound_pkt_ratio` remains `1.0` (both local) / `0.0` (neither local) for V4 parity, AND `directionality_degenerate == True` is surfaced in provenance/stats. |
| **SPEC-8** | Phase 6 | Run 5-packet burst at `t=10.0..10.1` + `flush()`. Test with `strict_full_window_only=False` vs `True`. | With `False`: emits window with `provenance.is_subcapacity == True`. With `True`: rejects with `windows_rejected["subcapacity_flushed_window"] == 1`. |

---

## 7. Frozen-Model Compatibility Risk Assessment

| Frozen Artifact / Invariant | Risk from Batch 26B Plan | Justification |
|---|---|---|
| `models/aegis_live_v4/aegis_live_xgboost_baseline.pkl` | **ZERO MUTATION / NO DRIFT** | Feature order (29), `len(buf)`, `_is_local(dst_ip)`, 3-tuple windowing, and capacity-50 semantics remain intact. |
| `models/aegis_live_v4/temperature_scaling.json` ($T=1.199787$) | **NO DRIFT** | Logit distribution on capacity-50 valid captures is unchanged. |
| `models/aegis_live_v4/ood_parameters.pkl` (Mahalanobis $\le 7.92$) | **IMPROVED ACCURACY (ZERO DRIFT)** | Fixing Phase 1 timeout/flush leakage stops bogus future topology shifts on sparse flows while keeping 50-pkt calibration inputs identical. |
| `models/aegis_live_v4/abstention_config.json` & V6.3 models | **ZERO MUTATION** | Thresholds and V6.3 staging files are completely untouched. |

---

## 8. Explicit List of Changes That MUST NOT Be Made in Batch 26B

1. **DO NOT** switch `FlowWindow` `flow_key` from 3-tuple `(ip_a, ip_b, ip_proto)` to 5-tuple for frozen V4 (breaks randomized-sport DoS detection and `ood_parameters.pkl`).
2. **DO NOT** replace `len(buf)` with `ip.len` (-14B shift) or TCP/UDP payload length for frozen V4.
3. **DO NOT** rewrite `is_inbound = _is_local(dst_ip)` formula into symmetrical or flow-origin directionality for frozen V4.
4. **DO NOT** implement timestamp clamping (`ts = max(ts, last_ts)`), which creates artificial zero-IAT spikes.
5. **DO NOT** drop or alter `min_packets_to_extract = 2` default silently without keeping backwards compatibility; only add explicit `is_subcapacity` flagging + opt-in `strict_full_window_only` mode.
6. **DO NOT** regenerate `data/aegis_live_v4/*.csv`, retrain, recalibrate, or adjust OOD thresholds ($7.92$ / $0.75$).
7. **DO NOT** approve Edge-IIoTset for direct V4 inference or training based solely on code fixes — structural domain shifts (LAN-internal directionality, multi-service 3-tuple multiplexing, and protocol novelty) still require a separate V7 architecture & authentic Edge-IIoTset PCAP validation.

---

## 9. Recommended Implementation Sequence for Batch 26B

1. **Pre-Step:** Inspect `aegis_live/v4_inference.py` (check how `predict_sample()` handles `float32` / `validate_29_features()`).
2. **Step 1 (Phase 1):** Implement per-window causal `last_topology_snapshot` caching in `v4_runtime.py` + `SPEC-1` & `SPEC-2` unit tests. Run existing safe causality tests (`test_aegis_live_v4_temporal_causality.py`).
3. **Step 2 (Phase 2):** Implement strict monotonic timestamp policy across `NetworkTopologyTracker`, `FlowWindow`, and `v4_runtime.py` + `SPEC-3` & `SPEC-4` tests.
4. **Step 3 (Phase 3):** Update misleading "5-tuple" docstrings to "3-tuple host-pair + IP-protocol" and add `merged_connection_count` telemetry + `SPEC-5`.
5. **Step 4 (Phase 4):** Unify Ethernet/IPv4/L4 frame parser validation between `v4_runtime.py` and `build_v4_dataset.py` (keeping `len(buf)`), remove dead code at `build_v4_dataset.py` L423, add VLAN/fragment telemetry + `SPEC-6`.
6. **Step 5 (Phase 5 & 6):** Add directionality degeneracy diagnostics (`SPEC-7`), reconcile `FEATURE_DEFINITIONS` bounds, wire `validate_29_features()` / dict validator cleanly, and add `is_subcapacity` + `strict_full_window_only` mode (`SPEC-8`).
7. **Step 6 (Verification Gate):** Execute all unit tests (including `test_aegis_live_v4_runtime.py` only when authorized in implementation phase) and verify all 7 frozen V4/V6.3 SHA-256 hashes remain bit-exact.

---

## 10. Audit & Compliance Footer

- **Files inspected:**
  - `aegis_live/feature_schema.py`
  - `aegis_live/packet_window.py`
  - `aegis_live/network_topology_features.py`
  - `aegis_live/v4_runtime.py`
  - `aegis_live/build_v4_dataset.py`
  - `reports/BATCH25_V4_COMPATIBILITY_AUDIT.md`
  - Relevant V4 tests: `tests/test_aegis_live_extractor.py`, `tests/test_aegis_live_iat_regression.py`, `tests/test_aegis_live_network_topology.py`, `tests/test_aegis_live_v4_temporal_causality.py`, `tests/test_aegis_live_v4_dataset.py`, `tests/test_aegis_live_v4_runtime.py`, `tests/test_aegis_live_v4_model.py`, `tests/test_aegis_live_v4_ood.py`, `tests/test_aegis_live_v4_calibration.py`
- **Files modified:** **0** existing source, test, dataset, or model files modified (only this planning deliverable `reports/BATCH26A_V4_REMEDIATION_PLAN.md` was created).
- **Tests executed:** **NONE** in Batch 26A (planning-only phase; 38 safe unit tests passed in preceding Batch 25 with zero modifications since).
- **PCAP scans:** **NO**
- **Dataset generation:** **NO**
- **Training:** **NO**
- **Inference:** **NO**
- **Frozen hashes changed:** **0**
- **Automated blocking:** **NO**


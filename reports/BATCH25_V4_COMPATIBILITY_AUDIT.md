# AEGIS-IoT — Batch 25: V4 Compatibility Audit Report

**Audit Date:** 2026-11-12  
**Audit Scope:** Read-only code & test audit of V4 29-feature schema, flow windowing, causal network topology tracking, live runtime pipeline, and V4 dataset builder (`aegis_live/`).  
**Context:** Completed 24 batches of Edge-IIoTset forensic investigation; Edge-IIoTset is **NOT YET APPROVED** for direct V4 model inference or training.  
**Safety Posture:** Read-only audit. Zero source modifications, zero model training, zero frozen model inference, zero changes to frozen hashes or SQLite production records, automated blocking disabled.

---

## 1. Executive Summary

An evidence-based source audit of the five core V4 feature and runtime modules confirms that **direct evaluation or training of frozen V4 models on new heterogeneous IoT datasets (such as Edge-IIoTset) faces critical compatibility barriers and verified code defects**:

1. **Flow Identity (`DOCUMENTATION MISMATCH`, `DESIGN RISK`):** Both [`build_v4_dataset.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/build_v4_dataset.py#L412-L414) and [`v4_runtime.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/v4_runtime.py#L355-L356) key `FlowWindow` by a **3-tuple** `(ip_a, ip_b, ip_proto)`, omitting `sport` and `dport` despite repeated docstring references to "5-tuple flow windows". All concurrent/sequential TCP or UDP connections between the same two hosts (e.g., MQTT + HTTP + SSH between a sensor and edge broker) are merged into a single window, distorting timing, size, flag, and service-port features.
2. **Window Causality (`CONFIRMED DEFECT` — Critical):** Fixed-capacity (50-packet) windows under strictly monotonic timestamps are causal, **but** [`v4_runtime.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/v4_runtime.py#L331-L368) mutates `self.net_tracker` with the newly arrived packet ($t_{\text{new}} \ge t_{\text{old\_end}} + 2.0\text{ s}$) **before** checking flow inactivity timeout (`gap >= 2.0s`). Consequently, every timeout-flushed or end-of-stream `flush()` window receives **future topology features** from $t_{\text{new}}$ (or `last_ts`) rather than `window.last_timestamp`. Additionally, both [`FlowWindow`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/packet_window.py#L52-L62) and [`NetworkTopologyTracker`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/network_topology_features.py#L231-L274) lack monotonicity guards and corrupt state on out-of-order timestamps.
3. **Directionality (`DESIGN RISK` — High):** Directionality uses `is_inbound = self._is_local(dst_ip)` against a default `192.168.0.0/16` subnet, ignoring `src_ip`. Both-local (East-West LAN) flows collapse to `inbound_pkt_ratio = 1.0, outbound_pkt_ratio = 0.0` in both directions; neither-local flows (e.g., `10.0.0.0/8` under default config) collapse to `0.0 / 1.0`. Feature validity depends heavily on perimeter tap placement.
4. **Packet Length (`DESIGN RISK` — Medium):** Both builder and runtime consistently use raw Ethernet frame buffer length `len(buf)` (ignoring `ip.len`) and reject non-`0x0800` Ethernet frames (including 802.1Q VLAN tags). NIC 60-byte L2 padding vs. pre-NIC 54-byte unpadded frames shifts size distributions around control-packet boundaries.
5. **Feature Parity (`CONFIRMED DEFECT` — High):** Canonical 29-feature order and core math match between builder and runtime on 50-packet windows, **but** runtime emits $2 \le N \le 49$ sub-capacity windows on timeout/flush (where `iat_std_ms == 0.0` for $N=2$ and ratios quantize coarsely), builder lacks IP header length (`ihl`) and truncated L4 header rejections present in runtime, and neither file invokes the imported `validate_29_features()` function.

---

## 2. Files Inspected

Strict Phase 1 file scope observed:

| Category | File Path | Lines |
|---|---|---:|
| Schema | [`aegis_live/feature_schema.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/feature_schema.py) | 274 |
| Flow Window | [`aegis_live/packet_window.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/packet_window.py) | 180 |
| Topology Tracker | [`aegis_live/network_topology_features.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/network_topology_features.py) | 306 |
| Live V4 Runtime | [`aegis_live/v4_runtime.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/v4_runtime.py) | 537 |
| V4 Dataset Builder | [`aegis_live/build_v4_dataset.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/build_v4_dataset.py) | 749 |
| Related Safe Tests (Executed) | [`tests/test_aegis_live_extractor.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/tests/test_aegis_live_extractor.py) | 130 |
| Related Safe Tests (Executed) | [`tests/test_aegis_live_iat_regression.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/tests/test_aegis_live_iat_regression.py) | 244 |
| Related Safe Tests (Executed) | [`tests/test_aegis_live_network_topology.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/tests/test_aegis_live_network_topology.py) | 212 |
| Related Safe Tests (Executed) | [`tests/test_aegis_live_v4_temporal_causality.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/tests/test_aegis_live_v4_temporal_causality.py) | 398 |
| Related Safe Tests (Executed) | [`tests/test_aegis_live_v4_dataset.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/tests/test_aegis_live_v4_dataset.py) | 198 |
| Related Inference Test (Inspected Only) | [`tests/test_aegis_live_v4_runtime.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/tests/test_aegis_live_v4_runtime.py) | 362 |

*(Note: `test_aegis_live_v4_runtime.py` was inspected read-only and NOT executed because it instantiates `AEGISLiveV4Inference` and runs frozen model inference.)*

---

## 3. Exact Source Evidence with Line Numbers

### 3.1 Flow Identity
- **Builder Flow Key (`aegis_live/build_v4_dataset.py` L412–414):**
  ```python
  ip_pair = tuple(sorted([src_ip, dst_ip]))
  flow_key = (ip_pair[0], ip_pair[1], ip_proto)
  flow_id_str = f"{ip_pair[0]}<->{ip_pair[1]}:{ip_proto}"
  ```
- **Runtime Flow Key (`aegis_live/v4_runtime.py` L355–356, L222):**
  ```python
  ip_pair = tuple(sorted([src_ip, dst_ip]))
  flow_key = (ip_pair[0], ip_pair[1], ip_proto)
  ...
  flow_id = f"{window.flow_key[0]} <-> {window.flow_key[1]} proto={window.flow_key[2]}"
  ```
- **Topology 5-Tuple Flow Key (`aegis_live/network_topology_features.py` L54–58):**
  ```python
  @property
  def canonical_flow_key(self) -> Tuple[str, str, int, int, int]:
      """Canonical bidirectional 5-tuple key."""
      if (self.src_ip, self.src_port) <= (self.dst_ip, self.dst_port):
          return (self.src_ip, self.dst_ip, self.src_port, self.dst_port, self.protocol)
      return (self.dst_ip, self.src_ip, self.dst_port, self.src_port, self.protocol)
  ```
- **Conflicting "5-tuple" Documentation:**
  - `aegis_live/build_v4_dataset.py` L18: `"Topology features are NOT calculated inside individual 5-tuple flow windows"`
  - `aegis_live/build_v4_dataset.py` L649: `"existing per-5-tuple flow window (50 pkts)"`
  - `aegis_live/network_topology_features.py` L7: `"(as opposed to isolated 5-tuple flow windows)"`
  - `aegis_live/v4_runtime.py` L119: `"Per-flow 50-packet sliding windows (with 2.0s inactivity timeout)"` and L242–249 tracks `window.ports_seen` across multiple ports inside a single `FlowWindow`.

### 3.2 Window Causality & Timestamp Ordering
- **Topology Update Precedes Timeout Flush (`aegis_live/v4_runtime.py` L331–368):**
  - L340: `self.net_tracker.add_packet(net_pkt)` runs **first**, ingesting the packet at $t_{\text{new}}$ and evicting packets `< ts - 1.0` (`network_topology_features.py` L268–271).
  - L362–368: `if gap >= self.timeout_sec:` then checks `if len(window.packets) >= self.min_packets:` and calls `self._process_completed_window(window)` (L367), which calls `self.net_tracker.extract_current_features()` (L209) — reading state at $t_{\text{new}}$ instead of `window.last_timestamp`!
  - Same pattern in `aegis_live/build_v4_dataset.py` L400 -> L422–424 (though L423 `len(window.packets) >= self.window_capacity` is unreachable dead code because capacity-50 windows already flushed at L453–463).
- **End-of-Stream `flush()` Uses Final `net_tracker` State (`aegis_live/v4_runtime.py` L398–420, L437, L493):**
  - L493: `self.flush(current_time=last_ts + 5.0, force_all=True)` flushes all remaining windows with `>= 2` packets, calling `_process_completed_window(window)` (L410) against `self.net_tracker` at `last_ts`, not each window's `window.last_timestamp`.
- **Contradicted Causality Guarantee (`aegis_live/v4_runtime.py` L205–206):**
  - `"5 features from NetworkTopologyTracker represent packets with t <= window.last_timestamp and t >= window.last_timestamp - 1.0s."`
- **Lack of Timestamp Monotonicity Enforcement:**
  - `aegis_live/network_topology_features.py` L186 states `"packets arrive in non-decreasing timestamp order"`, but `add_packet` (L231–274) has no `ts < self.last_ts` check; L232 unconditionally sets `self.last_ts = ts`, and L264 appends unconditionally to the back of `self.packet_queue` before `while self.packet_queue[0].timestamp < cutoff_ts:` (L269–271) and `self.window_start_ts = self.packet_queue[0].timestamp` (L273).
  - `aegis_live/packet_window.py` L52–62 computes `gap = pkt.timestamp - self.last_timestamp` (negative on regression, so never triggers timeout), updates `self.last_timestamp = max(...)` (L61), yet appends unsorted `self.packets.append(pkt)` (L62). `extract_features()` then uses `self.packets[0].timestamp` and `self.packets[-1].timestamp` (L86–88) and clamps `max(dt, 0.0)` (L109).

### 3.3 Directionality
- **Builder (`aegis_live/build_v4_dataset.py` L77, L108, L119–123, L402):**
  - `LOCAL_SUBNET = "192.168.0.0/16"`, `_is_local` checks `ipaddress.ip_address(ip_str) in self.local_network`, and L402 sets `is_inbound = self._is_local(dst_ip)`.
- **Runtime (`aegis_live/v4_runtime.py` L125, L168–190, L343):**
  - Default `local_subnet: str = "192.168.0.0/16"` (supports supplementary subnets via `add_local_subnet()`), and L343 sets `is_inbound = self._is_local(dst_ip)`.
- **Window Ratio Math (`aegis_live/packet_window.py` L150–153):**
  - `inbound_cnt = sum(1 for p in self.packets if p.is_inbound)`, `outbound_cnt = N - inbound_cnt`.

### 3.4 Packet Length & Framing
- **Runtime Parsing (`aegis_live/v4_runtime.py` L285–329, L338, L346):**
  - Requires `len(buf) >= 14`, `eth_type == 0x0800` at bytes 12–13, `len(buf) >= 34`, `ihl >= 20`, `len(buf) >= 14 + ihl`, `len(buf) >= l4_offset + 14` (TCP) or `+ 4` (UDP). Sets `pkt_len=len(buf)`.
- **Builder Parsing (`aegis_live/build_v4_dataset.py` L356–389, L398, L405):**
  - Requires `len(buf) >= 14`, `eth_type == 0x0800`, `len(buf) >= 34`. Sets `pkt_len=len(buf)`. Does **not** check `ihl < 20`, `len(buf) < 14 + ihl`, or truncated TCP/UDP headers.
- **Schema vs. Builder Size Bounds:**
  - `aegis_live/feature_schema.py` L47–50 defines `pkt_size_mean/min/max` `min: 20.0`.
  - `aegis_live/build_v4_dataset.py` L160 enforces `if min_sz < 14 or max_sz > 65535`.

### 3.5 Feature Parity, Bounds, and Schema Validation
- **29-Feature Order (`aegis_live/feature_schema.py` L14–39, L169–177):**
  - 24 `CANONICAL_24_FEATURES` (`duration_ms` .. `outbound_pkt_ratio`) + 5 `V4_TOPOLOGY_FEATURES` (`src_concentration_ratio`, `src_ip_entropy`, `src_ip_cardinality`, `dst_ip_cardinality`, `fan_in_ratio`).
- **Rate Clamping Overflow (`aegis_live/packet_window.py` L92–93 vs. `feature_schema.py` L43):**
  - `safe_duration_sec = max(duration_sec, 1e-6)` allows `pkt_rate_pps` up to $50 / 10^{-6} = 5 \times 10^7\text{ pps}$ on 50-packet zero-duration bursts, exceeding `FEATURE_DEFINITIONS["pkt_rate_pps"]["max"] = 1e7`.
- **Unused Validator Imports:**
  - `validate_29_features` is imported at `aegis_live/v4_runtime.py` L47 and `aegis_live/build_v4_dataset.py` L49, and is never called in either file.
  - `validate_29_features()` (`feature_schema.py` L180–224) casts to `np.float32`, checks NaN/Inf, ratio range `[-1e-4, 1.0001]`, and non-negativity of the 5 topology features, but skips `FEATURE_DEFINITIONS` bounds (`pkt_rate_pps`, `ip_ttl_mean >= 1.0`, `pkt_size_min >= 20.0`) and `inbound_pkt_ratio + outbound_pkt_ratio == 1.0`.

---

## 4. Current Behavior vs. Intended Behavior

| Dimension | Intended Behavior | Current Observed Behavior | Gap Summary |
|---|---|---|---|
| **Flow Identity** | Isolated bidirectional 5-tuple flow windows (`(ip1, ip2, p1, p2, proto)`), matching docstrings. | Host-pair + IP-proto **3-tuple** `(ip_a, ip_b, proto)` omitting ports. | Concurrent TCP/UDP services between the same two hosts merge into one window. |
| **Window Causality (Timeout/Flush)** | Topology features for any completed window reflect `[T_end - 1.0s, T_end]` where `T_end = window.last_timestamp`. | `net_tracker.add_packet` mutates topology state to `t_new` (or `last_ts`) *before* timeout/flush window extraction. | Timeout-flushed ($2 \le N < 50$) and `flush()` windows leak future topology state (`t > window.last_timestamp`). |
| **Timestamp Ordering** | Monotonically non-decreasing timestamps enforced or out-of-order frames handled cleanly. | Silent state corruption on `ts < last_ts` in both `FlowWindow` and `NetworkTopologyTracker`. | FIFO deque order breaks; `last_ts` regresses; `t_end < t_start` yields `duration=0.0` & rate spikes. |
| **Directionality** | Accurate client/server or ingress/egress separation invariant to tap position. | `is_inbound = _is_local(dst_ip)` with `192.168.0.0/16` default, ignoring `src_ip`. | Both-local LAN traffic collapses to `(1.0, 0.0)`; unregistered subnets collapse to `(0.0, 1.0)`. |
| **Packet Length** | Consistent packet-size representation across capture environments. | Raw `len(buf)` including L2 header & NIC padding; `0x8100` VLAN frames dropped. | 54B unpadded host capture vs. 60B NIC-padded frame shifts size features; VLAN traffic silently dropped. |
| **Window Size Parity** | Live runtime windows match the 50-packet distribution frozen V4 trained on. | Builder requires `N == 50`; runtime defaults to `min_packets = 2`. | Timeout/flush emits $2 \le N < 49$ windows with `iat_std_ms=0.0` (at $N=2$) and quantized ratios. |

---

## 5. Audit Findings by Classification

### 5.1 `CONFIRMED DEFECT`
1. **D1 (`CONFIRMED DEFECT` — Critical): Future-Topology Leakage on Timeout & End-of-Stream `flush()` in `v4_runtime.py`** (`v4_runtime.py` L340 -> L364–368, L406–415, L437, L493). Updating `self.net_tracker` with the triggering future packet at $t_{\text{new}} \ge t_{\text{old\_end}} + 2.0\text{ s}$ evicts the old window's 1.0 s neighborhood before `_process_completed_window(window)` reads `net_tracker`. Similarly, `flush()` evaluates stale incomplete windows against `net_tracker` at `last_ts`.
2. **D2 (`CONFIRMED DEFECT` — High): Out-of-Order Timestamp State Corruption** (`network_topology_features.py` L231–274; `packet_window.py` L52–62, L86–88, L107–109). No guard rejects or reorders packets with `ts < last_ts`. FIFO eviction in `NetworkTopologyTracker` breaks when an older packet is appended to the right of `packet_queue`, and `FlowWindow.extract_features()` computes `t_end - t_start` from `packets[-1]` minus `packets[0]` without sorting.
3. **D3 (`CONFIRMED DEFECT` — High): Builder vs. Runtime Window-Size Mismatch (`N == 50` vs. `2 <= N <= 50`)** (`build_v4_dataset.py` L130 vs. `v4_runtime.py` L128, L366, L409). Frozen V4 was trained and evaluated strictly on 50-packet windows (verified across all 80,000 rows), whereas `v4_runtime.py` emits partial windows ($2 \le N \le 49$) on timeout/flush, causing systematic variance/quantization shift (`iat_std_ms == 0.0` at $N=2$, extreme rate inflation on tight micro-bursts).
4. **D4 (`CONFIRMED DEFECT` — Medium): Builder vs. Runtime Header Validation Divergence** (`build_v4_dataset.py` L377–389 vs. `v4_runtime.py` L299–328). Runtime rejects `ihl < 20`, `len(buf) < 14 + ihl`, `truncated_tcp_header`, and `truncated_udp_header`; builder does not, silently feeding malformed/truncated packets with `sport=0, dport=0, tcp_flags=0` into both `net_tracker` and `FlowWindow`.
5. **D5 (`CONFIRMED DEFECT` — Low):** Dead timeout-acceptance branch in `build_v4_dataset.py` L423 (`if gap >= self.timeout_sec: if len(window.packets) >= self.window_capacity:`). Because capacity-50 windows are immediately processed and deleted at L453–463, this condition can never be true.

### 5.2 `DOCUMENTATION MISMATCH`
1. **M1 (`DOCUMENTATION MISMATCH` — Medium):** `build_v4_dataset.py` (L18, L649), `network_topology_features.py` (L7), and `v4_runtime.py` (L119) describe per-flow windows as **5-tuple** windows, whereas the implemented `FlowWindow` key is a **3-tuple** `(ip_a, ip_b, ip_proto)`.
2. **M2 (`DOCUMENTATION MISMATCH` — Low):** `v4_runtime.py` L205–206 guarantees topology features reflect `[window.last_timestamp - 1.0s, window.last_timestamp]`, which fails for timeout-flushed and `flush()`-emitted windows due to Defect D1.
3. **M3 (`DOCUMENTATION MISMATCH` — Low):** `feature_schema.py` L43 & L49 list `pkt_rate_pps.max = 1e7` and `pkt_size_min.min = 20.0`, while `packet_window.py` L92 allows `50 * 1e6 = 5e7` and `build_v4_dataset.py` L160 checks `< 14` (`validate_29_features()` checks neither). Both `v4_runtime.py` L47 and `build_v4_dataset.py` L49 import `validate_29_features` without calling it.

### 5.3 `DESIGN RISK`
1. **R1 (`DESIGN RISK` — High): Multi-Service Flow Merging under 3-Tuple Identity** (`build_v4_dataset.py` L413; `v4_runtime.py` L356). In realistic IoT environments (like Edge-IIoTset where multiple sensors/clients share a gateway IP or broker IP), 3-tuple keying merges concurrent connections on different ports (e.g., MQTT + HTTP + SSH), corrupting IAT, duration, packet sizes, TCP flag ratios, and single `service_port` attribution. *(Cannot be unilaterally changed to 5-tuple without re-evaluating V4 because CICIoT2023 DoS generators cycled ephemeral source ports per packet.)*
2. **R2 (`DESIGN RISK` — High): Deployment-Dependent Directionality Collapse (`is_inbound = _is_local(dst_ip)`)** (`build_v4_dataset.py` L77, L402; `v4_runtime.py` L125, L343). Even with `local_subnet` configured to match a testbed, LAN-internal (both-local) traffic collapses to `(1.0, 0.0)`, while misconfigured non-`192.168.0.0/16` deployments collapse to `(0.0, 1.0)`, shifting `inbound_pkt_ratio` / `outbound_pkt_ratio` relative to perimeter-captured CICIoT2023 training data.
3. **R3 (`DESIGN RISK` — Medium): L2 Frame Length & VLAN Encapsulation Sensitivity** (`v4_runtime.py` L290, L346; `build_v4_dataset.py` L361, L405). Raw `len(buf)` includes Ethernet L2 60-byte minimum-frame padding (shifting 54B host-side ACKs vs. 60B wire-side ACKs) and potential FCS bytes, while 802.1Q VLAN-tagged (`0x8100`) and non-Ethernet PCAP link types are dropped.
4. **R4 (`DESIGN RISK` — Medium):** `FlowWindow` in [`v4_runtime.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/v4_runtime.py#L140) holds only one active window per `(ip_a, ip_b, proto)` key with FIFO timeout; high-rate multi-port spoofing from/to the same IP pair cannot be separated.

### 5.4 `NO ISSUE FOUND`
1. **N1 (`NO ISSUE FOUND`):** Canonical 29-feature order (`CANONICAL_29_FEATURES` = `CANONICAL_24_FEATURES` + `V4_TOPOLOGY_FEATURES`) is identical across `feature_schema.py`, `build_v4_dataset.py`, and `v4_runtime.py`.
2. **N2 (`NO ISSUE FOUND`):** Streaming vs. offline batch numerical equivalence in `NetworkTopologyTracker` (sliding and tumbling) for monotonic packet streams, plus causal non-interference for capacity-50 windows (`N == 50`).
3. **N3 (`NO ISSUE FOUND`):** Shared `FlowWindow.extract_features()` and `compute_network_topology_from_counts()` math parity on valid, chronological 50-packet untagged Ethernet II IPv4 windows.

### 5.5 `INSUFFICIENT EVIDENCE`
1. **E1 (`INSUFFICIENT EVIDENCE`):** Whether `aegis_live/v4_inference.py` (`AEGISLiveV4Inference.predict_sample()`) invokes `validate_29_features()`, enforces `np.float32` casting, or clips out-of-range inputs prior to XGBoost/Mahalanobis scoring (`v4_inference.py` is outside the 5-file Phase 1 inspection list).
2. **E2 (`INSUFFICIENT EVIDENCE`):** Actual IP subnet boundaries, VLAN tagging (`0x8100`), link-layer header types, L2 padding uniformity, timestamp monotonicity, and port multiplexing rates inside the unacquired Edge-IIoTset PCAP archive (cannot be measured until the archive is acquired).

---

## 6. Builder / Runtime Parity Table

| Property / Step | `build_v4_dataset.py` (V4 Offline Builder) | `v4_runtime.py` (V4 Live Pipeline) | Parity Verdict |
|---|---|---|---|
| **Canonical Feature Order** | `CANONICAL_29_FEATURES` (24 + 5) | `CANONICAL_29_FEATURES` (24 + 5) | **PARITY MATCH** (`NO ISSUE FOUND`) |
| **Flow Identity Key** | `(sorted_ip0, sorted_ip1, ip_proto)` (3-tuple) | `(sorted_ip0, sorted_ip1, ip_proto)` (3-tuple) | **PARITY MATCH** (`DOCUMENTATION MISMATCH` vs. 5-tuple docs) |
| **Ethernet Parsing** | `len >= 14`, `eth_type == 0x0800`, `len >= 34` | `len >= 14`, `eth_type == 0x0800`, `len >= 34` | **PARITY MATCH** (both drop `0x8100` VLANs) |
| **IP `ihl` & Truncated L4 Rejection** | **No** `ihl < 20`, `len < 14+ihl`, or L4 length rejection | **Yes** — rejects `invalid_ip_header_length`, `truncated_tcp/udp_header` | **DIVERGENCE** (`CONFIRMED DEFECT` D4) |
| **Packet Length Field** | `len(buf)` (raw L2 Ethernet frame) | `len(buf)` (raw L2 Ethernet frame) | **PARITY MATCH** (`DESIGN RISK` R3 on L2 padding) |
| **Directionality Rule** | `is_inbound = _is_local(dst_ip)` (`192.168.0.0/16`) | `is_inbound = _is_local(dst_ip)` (`192.168.0.0/16` + `add_local_subnet`) | **PARTIAL MATCH** (`DESIGN RISK` R2) |
| **`PacketRecord` Ports (`sport`/`dport`)** | Not passed (defaults to `0`) | Passed (`sport`, `dport`) for `ports_seen` | **DIVERGENCE** (does not affect 29 numeric features) |
| **Window Emission Size (`N`)** | Strictly `N == 50` (drops `< 50`) | `min_packets_to_extract = 2` (emits `2 <= N <= 50`) | **DIVERGENCE** (`CONFIRMED DEFECT` D3) |
| **Timeout Topology Snapshot Timing** | `L423` unreachable dead code (`N < 50` discarded) | `net_tracker` updated at `t_new` *before* old window flushes | **DIVERGENCE** (`CONFIRMED DEFECT` D1) |
| **End-of-Stream `flush()` Emission** | Partial windows discarded | Partial windows (`N >= 2`) emitted against final `net_tracker` | **DIVERGENCE** (`CONFIRMED DEFECT` D1, D3) |
| **`validate_29_features()` Call** | Unused import; uses `validate_candidate_window()` | Unused import; passes dict to `v4_inference` | **DIVERGENCE** (`M3` / `E1`) |
| **Numeric Dtype Emitted** | Python `float` (`float64`) saved to CSV | Python `float` (`float64`) in `DetectionResultV4` | **PARITY MATCH** (casts to `float32` only in unused schema validator) |

---

## 7. Risk Severity & Remediation Priority

| Priority | ID | Classification | Summary | Impact on Frozen V4 / Edge-IIoTset Readiness |
|---|---|---|---|---|
| **P0 (Critical)** | **D1** | `CONFIRMED DEFECT` | Timeout & `flush()` future-topology leakage in `v4_runtime.py` | Timeout-flushed flows ($2 \le N < 50$) receive future topology context instead of `window.last_timestamp`, invalidating live/PCAP evaluation of sparse IoT traffic. |
| **P1 (High)** | **D3** | `CONFIRMED DEFECT` | Builder `N == 50` vs. runtime `min_packets = 2` window-size mismatch | Partial IoT windows ($2 \le N \le 49$) exhibit zero `iat_std_ms` (at $N=2$), quantized ratios, and rate inflation, triggering false OOD/class predictions under frozen V4 thresholds. |
| **P1 (High)** | **D2** | `CONFIRMED DEFECT` | No timestamp regression guard in `FlowWindow` & `NetworkTopologyTracker` | Out-of-order packets corrupt FIFO eviction, invert window duration, and produce spurious rate spikes. |
| **P1 (High)** | **R1 / M1** | `DESIGN RISK` / `DOC MISMATCH` | 3-tuple `(ip_a, ip_b, proto)` merges multi-port TCP/UDP connections | Interleaves separate MQTT/HTTP/SSH/CoAP flows between gateway and sensor; cannot be altered in-place without breaking frozen V4 CICIoT2023 feature parity. |
| **P1 (High)** | **R2** | `DESIGN RISK` | `is_inbound = _is_local(dst_ip)` collapses East-West & unregistered subnets | LAN-internal Edge-IIoTset traffic and `10.x`/`172.16.x` networks experience `(1,0)` or `(0,1)` collapse, invalidating perimeter-trained directional features. |
| **P2 (Medium)** | **D4** | `CONFIRMED DEFECT` | Builder lacks `ihl` and truncated TCP/UDP rejection present in runtime | Edge-case malformed packets are accepted offline by builder but dropped at runtime. |
| **P2 (Medium)** | **R3** | `DESIGN RISK` | Raw `len(buf)` sensitivity to L2 NIC padding & VLAN (`0x8100`) drops | Pre-NIC vs. post-NIC padding (54B vs. 60B) shifts tight XGBoost splits; VLAN captures fail silently. |
| **P3 (Low)** | **M2 / M3 / D5** | `DOC MISMATCH` / `CONFIRMED DEFECT` | Schema bounds mismatch (`14` vs `20`, `5e7` vs `1e7`), unused validator imports, dead code at L423 | Confusing audit surface and missed validation on synthetic/extreme vectors. |

---

## 8. Recommended Unit Tests (Not Implemented)

Per audit rules, the following tests are recommended for a future approved remediation phase (do **not** implement during Batch 25):

1. **`test_runtime_timeout_flush_uses_window_last_timestamp_topology`:** Ingest a 5-packet burst at `t=10.0..10.4` with 1 source, advance `net_tracker` via another flow or a trigger packet at `t=13.0` from 50 new spoofed sources, verify the timed-out window's 5 topology features match the snapshot at `t=10.4` (cardinality `1.0`), NOT `t=13.0` (cardinality `50.0`).
2. **`test_runtime_end_of_stream_flush_causality`:** Ingest two independent flows ending at `t=10.0` (under high cardinality) and `t=20.0` (single source), call `flush()`, and assert the `t=10.0` window preserves its own `t=10.0` topology state rather than inheriting `t=20.0`.
3. **`test_out_of_order_timestamp_regression_handling`:** Feed packets with `ts = [100.0, 100.5, 99.8]` into `FlowWindow` and `NetworkTopologyTracker` and assert deterministic rejection or chronological sorting with zero FIFO corruption and `duration_ms >= 0`.
4. **`test_concurrent_multi_port_tcp_isolation_behavior`:** Feed interleaved MQTT (`1883`) and HTTP (`80`) packets between `192.168.1.10` and `192.168.1.1` to explicitly document/verify whether 3-tuple merging is intended (V4 legacy compatibility) or isolated (future V7 5-tuple contract).
5. **`test_directionality_east_west_and_non_192_168_subnets`:** Assert explicit behavior on both-local (`192.168.1.10 -> 192.168.1.1`) and non-`192.168` (`10.0.0.10 -> 10.0.0.1`) packets so subnet/tap sensitivity is regression-tracked.
6. **`test_builder_runtime_malformed_header_parity`:** Feed a truncated TCP header (`len(buf) == 40`, IP header `20B` + TCP `6B`) and invalid `ihl = 0` frame to both builder frame-parser and runtime `ingest_packet()`, verifying identical rejection counts.
7. **`test_vlan_8021q_and_ethernet_padding_bounds`:** Feed 802.1Q (`0x8100`) frames and 54B vs. 60B padded TCP ACK frames to assert deterministic handling and explicit rejection telemetry.

---

## 9. Files Requiring Modification in a Separately Approved Remediation Phase

If remediation is later approved (note: **any** change affecting feature values for frozen V4 requires separating V4 legacy compatibility from V7/V6.x fixes so frozen V4 calibration is not silently broken):

- **For Live Runtime Causality (`D1`, `D2`, `M2`) without altering 50-packet V4 semantics:**
  - [`aegis_live/v4_runtime.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/v4_runtime.py): Snapshot `net_tracker` or trigger timeout flush *before* mutating `net_tracker` with future packets, and handle `flush()` per-window topology timestamps cleanly; explicitly reject or log out-of-order `ts < last_ts`.
  - [`aegis_live/network_topology_features.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/network_topology_features.py): Enforce/reject `ts < self.last_ts` monotonicity in `add_packet()`.
  - [`aegis_live/packet_window.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/packet_window.py): Guard or reject `pkt.timestamp < self.last_timestamp` and fix `safe_duration_sec` / zero-duration upper bounds consistency.
- **For Builder/Runtime Consistency & Schema Enforcement (`D4`, `D5`, `M3`):**
  - [`aegis_live/build_v4_dataset.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/build_v4_dataset.py): Align `ihl` & truncated L4 checks with runtime, remove unreachable dead code at L423, and align docstring "5-tuple" -> "3-tuple host-pair + proto". *(Note: DO NOT overwrite existing frozen V4 CSVs/models.)*
  - [`aegis_live/feature_schema.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/feature_schema.py): Reconcile `FEATURE_DEFINITIONS` bounds (`14` vs `20`, `5e7` vs `1e7`) with `validate_29_features()`.
- **For Structural V4 Incompatibilities on Edge-IIoTset (`R1`, `R2`, `R3`, `D3`) — V7 / Future Architecture Only:**
  - Transitioning to true 5-tuple flow isolation (`R1`), tap-agnostic directionality (`R2`), IP `tot_len` / VLAN-aware framing (`R3`), and variable-window/sub-capacity calibration (`D3`) changes the feature distribution and therefore **cannot be retrofitted onto the frozen V4 model**; these must target a separately trained/calibrated V7 candidate after authentic Edge-IIoTset validation.

---

## 10. Final Recommendation

1. **Maintain the Edge-IIoTset Block for Frozen V4:** **Edge-IIoTset remains NOT APPROVED for direct frozen V4 inference or training.** Beyond the unacquired archive/ground-truth blockers from Batches 1–24, this code audit confirms structural mismatches (`D1` timeout causality bug in `v4_runtime.py`, `D3` `2..49` vs `50` window size mismatch, `R1` 3-tuple multi-service merging, `R2` perimeter-only `_is_local(dst_ip)` directionality collapse, and `R3` Ethernet padding / VLAN drops) that will skew features extracted from multi-service LAN-internal Edge-IIoTset captures.
2. **Preserve Frozen V4 Scope:** Do **not** alter frozen V4 feature math (3-tuple keying or `len(buf)`) in place unless strictly bug-fixing causality/bounds (`D1`, `D2`, `D4`) while verifying 100% parity on existing 50-packet V4 test vectors.
3. **Target Future Architecture (V7 Research Track):** Address 5-tuple isolation, East-West directionality, IP payload vs. L2 frame length, and out-of-order resilience in an isolated next-generation pipeline once authentic Edge-IIoTset PCAPs are acquired and Zeek-aligned.

---

## 11. Audit Compliance & Verification Footer

- **Files Inspected:**
  - `aegis_live/feature_schema.py`
  - `aegis_live/packet_window.py`
  - `aegis_live/network_topology_features.py`
  - `aegis_live/v4_runtime.py`
  - `aegis_live/build_v4_dataset.py`
  - `tests/test_aegis_live_extractor.py`, `tests/test_aegis_live_iat_regression.py`, `tests/test_aegis_live_network_topology.py`, `tests/test_aegis_live_v4_temporal_causality.py`, `tests/test_aegis_live_v4_dataset.py`, `tests/test_aegis_live_v4_runtime.py` (inspected only)
- **Tests Executed and Results:**
  - Command: `python -m pytest tests/test_aegis_live_extractor.py tests/test_aegis_live_iat_regression.py tests/test_aegis_live_network_topology.py tests/test_aegis_live_v4_temporal_causality.py tests/test_aegis_live_v4_dataset.py -v`
  - Result: **38 collected, 38 PASSED, 0 failed, 0 skipped** in `16.93s` (no model inference executed).
- **Files Modified:** **0 existing source, test, dataset, or model files modified** (only this new audit report `reports/BATCH25_V4_COMPATIBILITY_AUDIT.md` was created).
- **Frozen Model Hashes Modified:** **0 (Verified via `Get-FileHash -Algorithm SHA256` across all 7 V4 and V6.3 frozen artifacts):**
  - `models/aegis_live_v4/aegis_live_xgboost_baseline.pkl`: `C0E5AEE8E9E1726F12BFC19E98C440FE44FE278D1796D5866FE2A2488D458A56` (Unchanged)
  - `models/aegis_live_v4/temperature_scaling.json`: `C3AC0662CA896D06D4B9F62035A637FE04C8E01C00C466790D2EC0F03E581F99` (Unchanged)
  - `models/aegis_live_v4/ood_parameters.pkl`: `579063CFC4DDA3E396AB0D026198F14F8EF593EBF9910F290157AA0F669712C3` (Unchanged)
  - `models/aegis_live_v4/abstention_config.json`: `7D49F05BEF592CB838EF69F813A1C297BECA9E0A007BBAD1AAD6F5F0F4AC397F` (Unchanged)
  - `models/aegis_live_v6_3/config_b_model.pkl`: `09AFB03CBE74201E4E3768E367FB35C70D727BC41A5B9FFC6180C3305FDF1F84` (Unchanged)
  - `models/aegis_live_v6_3/temperature_scaling_b.json`: `2CEDF27975B4C28ABA2BBF52B84B0A5C5C49CAAF1EF3FA861AC6660885AA3B6F` (Unchanged)
  - `models/aegis_live_v6_3/abstention_config_b.json`: `E30BC4571C5045C3374AAE930145821F1277E1C9ABA56E4F419F7B10B39E2039` (Unchanged)
- **Training Performed:** **NO**
- **Inference Performed:** **NO**
- **Automated Blocking Enabled:** **NO**


# AEGIS Batch 28 — End-to-End V4 Compatibility Validation

**Date:** 2026-11-12  
**Overall decision:** **CONDITIONALLY COMPATIBLE**  
**Frozen V4 ready for controlled evaluation now:** **NO — INSUFFICIENT EVIDENCE until malformed-packet ordering and sub-capacity policy are resolved or explicitly excluded by the evaluation protocol.**

## 1. Executive summary

The corrected builder and runtime preserve the frozen 29-feature schema, symmetric bidirectional 3-tuple window identity, raw Ethernet `len(buf)` packet sizes, destination-local inbound formula, and causal 1-second packet-global topology snapshots. On deterministic valid synthetic streams, their full 50-packet output matched exactly across all 29 features for interleaved TCP and UDP traffic with equal timestamps and mild reordering: every numeric delta was `0.0`.

Two limitations prevent an unconditional readiness decision:

1. In bounded offline mode, full IPv4/L4 validation occurs **after** the candidate frame can enter the timestamp reorder buffer. A malformed packet can advance its reorder watermark and cause a later valid packet to be rejected. A synthetic reproduction confirmed this in both builder and runtime. Live runtime direct-ingest validates before its strict timestamp guard, so offline and live outcomes differ.
2. The training builder emits full 50-packet windows only and drops timed-out sub-capacity windows; it does not flush active partial windows at EOF. Runtime can emit windows of 2–49 packets on timeout or EOF. Those runtime features have no corresponding frozen training-window construction evidence.

Thus, valid full-window feature parity is confirmed, but malformed-input ordering, partial windows, and real-data distribution equivalence are not cleared. No classifier was loaded or run.

## 2. Original frozen training contract

### Evidence and confidence

The checked-in V4 dataset-generation source (`git show HEAD:aegis_live/build_v4_dataset.py`), `aegis_live/train_v4_baseline.py`, `models/aegis_live_v4/training_metadata.json`, the V4 manifest, and the supplied Batch 25–27 reports support the contract below. The metadata records a 29-feature raw-value schema, 56,000/12,000/12,000 train/validation/test rows, 1.0-second topology observation window, and no scaler or synthetic training generation. It does **not** bind the trained model to a dataset-builder source hash or record parser rejection counts; historical parser behavior is therefore supported by the checked-in source but not independently tied to the exact training execution.

### Confirmed contract

- **Features:** 24 flow features followed by five topology features. The exact order is listed in Section 4.
- **Identity:** symmetric bidirectional `(min(src_ip, dst_ip), max(src_ip, dst_ip), ip_proto)`. Ports are excluded from window identity. A separate bidirectional 5-tuple is used inside topology accounting for `unique_flow_count`, which is excluded from the canonical 29.
- **Windows:** non-overlapping capacity windows of 50 packets. Reaching 50 completes the window; subsequent packets begin another window for that key.
- **Timeout and EOF:** on a gap `>= 2.0s`, the builder accepts an old window only if it is full; otherwise it records `flushed_subcapacity_window`. The checked-in training builder does not flush active windows at end of stream. These behaviors define the available training-generation evidence.
- **Packet length:** raw Ethernet buffer length `len(buf)`, including the Ethernet header and any captured Ethernet padding; IP total length does not define the size feature.
- **Direction:** `is_inbound = _is_local(dst_ip)` with default `192.168.0.0/16`; the source address does not affect the canonical inbound flag.
- **Topology:** packet-global 1.0-second sliding window, updated causally in capture order. It aggregates source and destination addresses across active traffic: source concentration is the maximum source packet share, entropy is Shannon entropy over source packet shares, cardinalities are distinct source/destination counts, and `fan_in_ratio = distinct_sources / distinct_destinations`. A separate 5-tuple count is maintained but excluded from V4 features.
- **Timestamp assumption:** the checked-in training builder consumes capture timestamps in stream order and contains no historical bounded reorder buffer. The training metadata does not explicitly record the exact timestamp policy used in the training run. Current 250 ms offline reordering is therefore a later hardening behavior, not proven original training behavior.
- **Original parsing:** the checked-in generator accepts Ethernet II IPv4 only, checks minimum Ethernet and IPv4-prefix lengths, then reads fields by offsets. It does not validate IPv4 version/IHL, IP total length, fragment offsets, or complete transport header length. Its TCP and UDP field reads use shorter buffer-presence checks than the corrected parser. VLAN-tagged/non-IPv4 frames are excluded. Exact malformed/truncated packet counts in the training captures are unavailable without prohibited capture scans.

## 3. Corrected runtime and builder contract

- Both paths call the shared `parse_ipv4_ethernet_frame()` for IPv4 version, IHL, total length, captured payload bounds, and TCP/UDP header checks. VLAN and other non-IPv4 Ethernet frames are rejected with reason counters. Padding is accepted and `pkt_len` remains `len(buf)`.
- Non-initial IPv4 fragments are accepted without interpreting fragment payload as ports or flags; ports/flags are zero. Initial fragments must contain a complete relevant transport header to be accepted. The old generator had no equivalent fragment-aware behavior, so malformed fragment-related feature deltas are possible.
- Both paths retain the same 3-tuple key and `_is_local(dst_ip)` canonical direction formula. Direction classes are diagnostics only.
- The topology tracker uses a 1-second sliding FIFO and removes packets strictly older than `current_ts - 1.0`; equality at the cutoff remains in-window. Each flow window stores the topology feature snapshot from its latest accepted packet, preventing later packets from contaminating timeout/EOF output.
- Timestamp guards reject non-finite and regressing timestamps. Offline mode uses bounded 250 ms reordering; zero-buffer live mode preserves arrival order and rejects regressions. Equal timestamps are accepted in stable arrival order.
- Builder output is restricted to full 50-packet windows. Runtime defaults to `min_packets_to_extract=2` and can emit 2–49 packet windows on timeout/EOF. This is an explicit, material train/runtime window-population difference.
- Canonical feature values were finite Python numeric values for the synthetic full-window comparison. Their mapping order is the canonical 29. `validate_29_features()` casts to float32, but current runtime/builder extraction does not call that utility; the builder performs its own finite/range checks.

### Malformed packet ordering blocker

The replay/builder pre-reorder filter only checks Ethernet framing, EtherType, and the minimum IPv4 prefix. Full IPv4/L4 validation runs downstream, after the reorder buffer/ordering guard can already observe candidate timestamps.

Synthetic evidence with lateness `0.25s`: stream `(valid@10.0, invalid-total-length@10.4, valid@10.1, valid@10.5)` accepted 2 packets and rejected the later valid packet as out-of-order; removing the invalid packet accepted all 3 valid packets. Runtime counters with the invalid packet were `out_of_order_timestamp: 1`, `invalid_ipv4_total_length: 1`; without it there were no rejections. Builder counters likewise showed one `out_of_order_timestamp` and one `invalid_ipv4_total_length`, versus none without the malformed frame. Invalid packets do not reach flow/topology aggregation, but they can mutate ordering state and alter which valid packets reach it. This fails the strict interpretation of “invalid packets cannot mutate state.”

## 4. Canonical 29-feature parity table

The order below is the training metadata order. For valid synthetic full-capacity streams, **runtime minus builder was exactly `0.0` for every feature**, for both TCP (6) and UDP (17).

| # | Feature | TCP delta | UDP delta |
|---:|---|---:|---:|
| 1 | `duration_ms` | 0.0 | 0.0 |
| 2 | `pkt_rate_pps` | 0.0 | 0.0 |
| 3 | `byte_rate_bps` | 0.0 | 0.0 |
| 4 | `iat_mean_ms` | 0.0 | 0.0 |
| 5 | `iat_std_ms` | 0.0 | 0.0 |
| 6 | `pkt_size_mean` | 0.0 | 0.0 |
| 7 | `pkt_size_std` | 0.0 | 0.0 |
| 8 | `pkt_size_min` | 0.0 | 0.0 |
| 9 | `pkt_size_max` | 0.0 | 0.0 |
| 10 | `syn_ratio` | 0.0 | 0.0 |
| 11 | `ack_ratio` | 0.0 | 0.0 |
| 12 | `rst_ratio` | 0.0 | 0.0 |
| 13 | `fin_ratio` | 0.0 | 0.0 |
| 14 | `urg_ratio` | 0.0 | 0.0 |
| 15 | `psh_ratio` | 0.0 | 0.0 |
| 16 | `ece_ratio` | 0.0 | 0.0 |
| 17 | `cwr_ratio` | 0.0 | 0.0 |
| 18 | `ip_ttl_mean` | 0.0 | 0.0 |
| 19 | `ip_ttl_std` | 0.0 | 0.0 |
| 20 | `tcp_ratio` | 0.0 | 0.0 |
| 21 | `udp_ratio` | 0.0 | 0.0 |
| 22 | `icmp_ratio` | 0.0 | 0.0 |
| 23 | `inbound_pkt_ratio` | 0.0 | 0.0 |
| 24 | `outbound_pkt_ratio` | 0.0 | 0.0 |
| 25 | `src_concentration_ratio` | 0.0 | 0.0 |
| 26 | `src_ip_entropy` | 0.0 | 0.0 |
| 27 | `src_ip_cardinality` | 0.0 | 0.0 |
| 28 | `dst_ip_cardinality` | 0.0 | 0.0 |
| 29 | `fan_in_ratio` | 0.0 | 0.0 |

## 5. Synthetic end-to-end results

- A deterministic in-memory stream contained 50 interleaved TCP packets and 50 interleaved UDP packets with equal timestamps, plus two other flows. Two packet timestamps were mildly reordered within the configured 250 ms window. It also contained a three-packet flow that timed out and a four-packet flow left at EOF.
- **Full windows:** builder emitted 2; runtime emitted the same 2 full windows. Both had 50 packets. Maximum absolute difference across the 29 features was `0.0` for each protocol. All output feature names were in canonical order, finite, and numeric.
- **Partial windows:** runtime emitted packet counts 3 (timeout) and 4 (EOF). Builder recorded one timed-out sub-capacity flow and did not emit either partial; its EOF active windows are not drained. They are intentionally omitted from the 29-feature delta table because no builder reference row exists.
- **Directionality:** the stream exercised external-to-local, local-to-local, and external-to-external traffic. Builder and runtime counts matched: 50, 54, and 4 respectively (local-to-external: 0). Direction diagnostics did not change the inbound/outbound canonical feature values.
- **Parser regressions:** deterministic checks covered raw frame length with padding, invalid IHL and total length, truncated IPv4 payload, truncated TCP/UDP headers, VLAN rejection, and non-initial fragment handling. Invalid runtime packets did not mutate flow/topology aggregation or the direct-ingest timestamp guard. The offline reorder-state exception is detailed above.
- **Causality:** selected existing tests for timeout/EOF snapshots, interleaved topology, full windows, sub-capacity snapshots, and builder/runtime parity passed.
- These tests establish implementation parity for tested synthetic valid traffic only. They do **not** establish compatibility of real feature distributions or model performance.

## 6. Training-distribution compatibility assessment

### Exact semantic matches

- Feature names/order and 24+5 composition.
- Symmetric 3-tuple host-pair + protocol grouping; ports excluded.
- 50-packet capacity and non-overlapping full-window boundaries.
- Raw Ethernet `len(buf)` size semantics, including captured padding.
- `_is_local(dst_ip)` inbound definition.
- 1-second causal packet-global topology calculations and the five selected topology features.

### Intentional corrections and expected feature-distribution changes

- Full IPv4 and transport validation now rejects malformed/truncated data the original generator could admit by loose offset checks. Exact frequencies cannot be measured without the original captures.
- Fragment-aware handling avoids interpreting non-initial fragment bytes as TCP/UDP ports or flags. The older builder could read those offsets when enough captured bytes were present; TCP flag and topology-only 5-tuple counts can therefore differ on fragments. Only the TCP flag effects can reach canonical V4 features; topology-only flow count is excluded.
- 250 ms offline reordering and strict timestamp rejection are later hardening. On clean chronological captures they should not change packet order; on jitter/regressions they can change packet inclusion and therefore IAT, rates, ratios, and topology.
- Runtime's 2–49 packet timeout/EOF windows are outside the full-50 training contract. No distribution claim is made for them.

### Non-reproducible evidence

The training metadata does not record source commit/hash for the builder, timestamp reorder settings, invalid/truncated packet rejection counts, fragment counts, or per-window packet-count histogram. Without capture scans or a provenance artifact binding exact source and packet rejection data, those details of the training population cannot be proven. Synthetic parity cannot substitute for those statistics.

## 7. Model integrity

All seven frozen artifact hashes were recalculated and matched the Batch 25 recorded SHA-256 values. No artifact was loaded for inference or modified.

| Artifact | SHA-256 | Status |
|---|---|---|
| `models/aegis_live_v4/aegis_live_xgboost_baseline.pkl` | `C0E5AEE8E9E1726F12BFC19E98C440FE44FE278D1796D5866FE2A2488D458A56` | unchanged |
| `models/aegis_live_v4/temperature_scaling.json` | `C3AC0662CA896D06D4B9F62035A637FE04C8E01C00C466790D2EC0F03E581F99` | unchanged |
| `models/aegis_live_v4/ood_parameters.pkl` | `579063CFC4DDA3E396AB0D026198F14F8EF593EBF9910F290157AA0F669712C3` | unchanged |
| `models/aegis_live_v4/abstention_config.json` | `7D49F05BEF592CB838EF69F813A1C297BECA9E0A007BBAD1AAD6F5F0F4AC397F` | unchanged |
| `models/aegis_live_v6_3/config_b_model.pkl` | `09AFB03CBE74201E4E3768E367FB35C70D727BC41A5B9FFC6180C3305FDF1F84` | unchanged |
| `models/aegis_live_v6_3/temperature_scaling_b.json` | `2CEDF27975B4C28ABA2BBF52B84B0A5C5C49CAAF1EF3FA861AC6660885AA3B6F` | unchanged |
| `models/aegis_live_v6_3/abstention_config_b.json` | `E30BC4571C5045C3374AAE930145821F1277E1C9ABA56E4F419F7B10B39E2039` | unchanged |

## 8. Remaining blockers and risks

1. Validate candidate frames fully before any reorder buffer or timestamp guard can observe them; add a regression that proves an invalid future-dated frame cannot cause a valid in-window packet to be dropped.
2. Define the evaluation policy for runtime 2–49 packet windows versus the frozen training builder's full-50-only rows. Preserve packet-count provenance and do not mix partial windows into full-window compatibility claims.
3. Recover or record the exact builder source/revision and training-time parser/timestamp settings if available; malformed and fragment population effects remain unquantified.
4. Keep directionality interpretation constrained to configured subnet locality; it is not client/server role inference.

## 9. Decision and Batch 29 recommendation

**Compatibility classification: CONDITIONALLY COMPATIBLE.** The corrected pipeline preserves the core V4 contract and gives exact builder/runtime parity on valid synthetic full windows. However, it is **not READY FOR CONTROLLED EVALUATION yet** because malformed candidates can affect offline ordering and sub-capacity behavior is outside the training window contract. No distribution equivalence or performance compatibility is established.

**Batch 29 recommendation:** address pre-order full-frame validation and explicitly select/document a full-window-only or separately stratified partial-window evaluation protocol. Then validate against a permitted, non-sealed evaluation source with recorded parser rejection and packet-count statistics before making any frozen-model evaluation claim. Do not alter the frozen model, calibration, OOD thresholds, or feature schema.

---

**Files modified:** `reports/BATCH28_END_TO_END_V4_COMPATIBILITY.md` only  
**Tests executed:** 13 targeted tests passed (parser/directionality, timestamp ordering/jitter, causal snapshots, full/sub-capacity windows, synthetic builder/runtime parity, UDP extractor); 3 upstream SHAP deprecation warnings. Additional in-memory parity and malformed-order reproductions completed.  
**PCAP scans:** NO  
**Dataset generation:** NO  
**Training:** NO  
**Inference:** NO  
**Frozen hashes changed:** 0  
**Automated blocking:** NO

# BATCH 26E — Workstream B: Flow Identity Contract Investigation (Read-Only)

**Date:** 2026-11-12  
**Mode:** Read-Only Evidence Audit (0 files modified, 0 PCAP scans, 0 training/inference runs, 0 Partition C / Kitsune holdout accesses)  
**Comparison:** Current `(sorted_src_ip, sorted_dst_ip, ip_proto)` 3-tuple vs. Bidirectional Normalized 5-tuple (`(ip_a, ip_b, port_a, port_b, ip_proto)`).

---

## Executive Summary

- **`CONFIRMED FACT`:** Frozen V4 (`models/aegis_live_v4/*`, `data/aegis_live_v4/{train,validation,test}.csv`) was built, trained, calibrated (`T = 1.199787`), and OOD-fitted strictly on **3-tuple** `FlowWindow` grouping `(min(src_ip, dst_ip), max(src_ip, dst_ip), ip_proto)` with `WINDOW_CAPACITY = 50` — **not** 5-tuple.
- **`DOCUMENTATION MISMATCH`:** Docstrings and construction reports repeatedly claim `"5-tuple flow windows"` while every executable dataset builder, runtime pipeline, and training pipeline uses 3-tuple grouping and writes portless `ip0<->ip1:proto` identifiers into `train.csv` / `validation.csv` / `test.csv`.
- **`CONFIRMED FACT`:** Source and destination ports were never used in training window grouping or any of the `CANONICAL_29_FEATURES`. `NetworkTopologyPacket.canonical_flow_key` computes a 5-tuple only for `unique_flow_count`, which is explicitly excluded from `CANONICAL_29_FEATURES`.
- **`DESIGN RISK`:** Introducing bidirectional 5-tuple grouping in-place would invalidate the frozen V4 feature distribution and break CICIoT2023 randomized-`sport` DoS window construction (`192.168.137.182<->192.168.137.187:6` would shatter into 1-packet flows). Frozen V4 **must retain 3-tuple identity**, while 5-tuple service isolation belongs in a newly trained successor model generation.

---

## 1. Exact Flow Identity Used to Generate Frozen V4 Training Samples

- **Classification:** **`CONFIRMED FACT`** (3-tuple execution) + **`DOCUMENTATION MISMATCH`** (misleading "5-tuple" prose)
- **Code Evidence:**
  - [`build_v4_dataset.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/build_v4_dataset.py#L537-L539): `ip_pair = tuple(sorted([src_ip, dst_ip]))`, `flow_key = (ip_pair[0], ip_pair[1], ip_proto)`, `flow_id_str = f"{ip_pair[0]}<->{ip_pair[1]}:{ip_proto}"`, persisted to the CSV `flow_id` column.
  - [`packet_window.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/packet_window.py#L238): `FlowWindow.__init__(self, flow_key: Tuple[str, str, int], max_packets: int = 50, timeout_sec: float = 2.0)`.
- **Frozen Dataset Verification ([`train.csv#L1-L6`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/data/aegis_live_v4/train.csv#L1-L6)):**
  - Directly inspected rows confirm portless 3-tuple `flow_id` strings (`192.168.137.14<->192.168.137.88:17`, `192.168.137.128<->8.8.4.4:17`, `13.249.178.34<->192.168.137.30:6`) with `packet_count == 50`.
- **Misleading "5-Tuple" Documentation (`DOCUMENTATION MISMATCH`):**
  - [`build_v4_dataset.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/build_v4_dataset.py#L18): `"Topology features are NOT calculated inside individual 5-tuple flow windows"`.
  - [`build_v4_dataset.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/build_v4_dataset.py#L779-L784) & [`aegis_live_v4_dataset_construction.md`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/reports/aegis_live_v4_dataset_construction.md#L12-L17): `"+--> existing per-5-tuple flow window (50 pkts)"`.
  - [`network_topology_features.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/network_topology_features.py#L7): `"(as opposed to isolated 5-tuple flow windows)."`.

---

## 2. Consistency Across Dataset Building, Training, and Runtime

- **Classification:** **`CONFIRMED FACT`**
- **Dataset Builder -> Manifest -> Model Chain:**
  - [`dataset_manifest.json`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/data/aegis_live_v4/dataset_manifest.json#L118-L122) binds `train.csv` (`378d981f...`), `validation.csv` (`f1a677eb...`), and `test.csv` (`bf1c7c37...`) SHA-256 digests directly to 3-tuple builder execution.
  - [`training_metadata.json`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/models/aegis_live_v4/training_metadata.json#L123-L128) binds frozen `aegis_live_xgboost_baseline.pkl` (`c0e5aee8...`) to those exact CSV SHA-256 hashes.
  - [`train_v4_baseline.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/train_v4_baseline.py#L188-L235), [`calibrate_v4.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/calibrate_v4.py#L258-L273), and [`audit_v4_ood.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/audit_v4_ood.py#L239-L280) train XGBoost, optimize `T = 1.199787`, and fit 29×29 class-conditional Mahalanobis matrices directly on `CANONICAL_29_FEATURES` from those CSVs (`train_v4_baseline.py#L229-L231` asserts `unique_flow_count` and `unique_5tuple_flow_count` are absent).
- **Runtime Pipeline Consistency:**
  - [`v4_runtime.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/v4_runtime.py#L403-L404) computes identical `flow_key = (ip_pair[0], ip_pair[1], ip_proto)` (`_mark_flow_ooo_from_bytes` uses the same key). (Only string formatting of provenance `flow_id` differs slightly: builder `"a<->b:p"` vs. runtime `"a <-> b proto=p"` in `_process_completed_window`.)
- **`NetworkTopologyPacket.canonical_flow_key` Interaction:**
  - [`network_topology_features.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/network_topology_features.py#L54-L58) constructs a bidirectional 5-tuple strictly to count `unique_flow_count` ([L127, L156](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/network_topology_features.py#L127)). Because `unique_flow_count` is excluded from `CANONICAL_29_FEATURES` ([`dataset_manifest.json#L71-L78`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/data/aegis_live_v4/dataset_manifest.json#L71-L78)), `canonical_flow_key` has zero effect on V4 features or window grouping.

---

## 3. Were Source and Destination Ports Used in Training Window Grouping?

- **Classification:** **`CONFIRMED FACT`** — **No.**
- In [`build_v4_dataset.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/build_v4_dataset.py#L500-L539), parsed `sport`/`dport` are passed only to `NetworkTopologyPacket` (where `flow_counts` is excluded from V4 features) and are not even passed into `PacketRecord` ([L528-L535](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/build_v4_dataset.py#L528-L535)).
- Runtime [`v4_runtime.py`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/v4_runtime.py#L433-L434) passes `sport`/`dport` to `PacketRecord` solely for advisory `ports_seen` / `service_port` provenance metadata, never for `flow_key` or `feats_29`.

---

## 4. Traffic Patterns Merged by the 3-Tuple

- **Classification:** **`CONFIRMED FACT`** (mechanism) + **`DESIGN RISK`** (semantic conflation)
- Merging `(min(src_ip, dst_ip), max(src_ip, dst_ip), ip_proto)` conflates:
  1. Concurrent/interleaved IoT services sharing host pairs & transport protocol (e.g., TCP MQTT `:1883` + HTTP `:80` + SSH `:22`).
  2. Parallel client sessions / NATed gateway sessions behind one IP using distinct ephemeral client ports toward the same server IP/protocol.
  3. Port-randomized DoS floods (`DoS-SYN_Flood.pcap` cycling ephemeral `sport` per packet) and vertical port scans from host `A` to `B`.
  4. Bidirectional packets & overlapping reverse connections on the same host pair and protocol.
- **Effect:** Combines packets across those connections inside [`FlowWindow.extract_features`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/packet_window.py#L303-L410), altering IATs, rates, packet-size statistics, TCP flag ratios, TTL statistics, and directional ratios.

---

## 5. Impact of Introducing Bidirectional Normalized 5-Tuple Identity

- **Classification:** **`CONFIRMED FACT`** (structural mechanics) + **`DESIGN RISK`** (operational trade-offs)
  1. **Severe Fragmentation of Port-Randomized DoS & Short Transactions:** CICIoT2023 DoS generators randomize ephemeral `sport` on every packet ([`aegis_live_v4_dataset_construction.md#L38`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/reports/aegis_live_v4_dataset_construction.md#L38)); 5-tuple grouping isolates them into single-packet keys, producing **0** capacity-50 DoS windows in dataset building and triggering 2.0s timeout / subcapacity rejections (`subcapacity_window_timeout`) at runtime.
  2. **Window Table Growth:** Under high-rate randomized-`sport` floods (12,000 pps cited in dataset docs with 2.0s timeout), `active_windows` expands from $O(|\text{host pairs} \times \text{proto}|)$ to up to tens of thousands of simultaneous 1-packet entries.
  3. **24 Physical Features Shift:** Concurrent services no longer mix IAT/size/flag distributions, altering statistical feature manifolds cleanly *per connection* while leaving single-window packet counts lower for short-lived IoT exchanges.
  4. **Topology Timing Shift:** `NetworkTopologyTracker` stays packet-global, but each window samples its `last_topology_snapshot` at its per-5-tuple `window_end_ts` rather than the merged host-pair `window_end_ts`.

---

## 6. Would Changing Flow Identity Invalidate the Frozen V4 Input Distribution?

- **Classification:** **`CONFIRMED FACT`** — **Yes.**
- Frozen `aegis_live_xgboost_baseline.pkl`, `temperature_scaling.json` (`T = 1.199787`), `ood_parameters.pkl` (fitted on all 56,000 rows of 3-tuple `train.csv` in [`audit_v4_ood.py#L243-L280`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/audit_v4_ood.py#L243-L280)), and `abstention_config.json` were calibrated strictly to 3-tuple, `N=50` windows.
- Switching runtime to 5-tuple breaks CICIoT2023 DoS representation completely, shifts benign/spoofing IAT, rate, packet-size, and flag distributions across [`CANONICAL_29_FEATURES`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/models/aegis_live_v4/training_metadata.json#L8-L38) (affecting top XGBoost split features), invalidates class-conditional Mahalanobis distances (`ood_parameters.pkl`), and violates builder-runtime parity against [`dataset_manifest.json`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/data/aegis_live_v4/dataset_manifest.json).

---

## 7. Recommendation: Preserve 3-Tuple Identity for Frozen V4

- **Classification:** **`CONFIRMED FACT`** (sufficient evidence to retain 3-tuple for V4) + **`DESIGN RISK`** (requires 5-tuple isolation in a retrained future model generation)
- **Conclusions:**
  1. **Preserve 3-tuple `(sorted_src_ip, sorted_dst_ip, ip_proto)` strictly for all Frozen V4 runtime and parity paths.** Unambiguous code, CSV (`flow_id`), SHA-256 manifest, calibration, and OOD evidence proves V4 was trained on 3-tuple grouping.
  2. **Remediate Documentation Mismatch:** Update misleading `"5-tuple"` docstrings/reports in a dedicated documentation/provenance phase to state `"3-tuple host-pair + IP-protocol window"` and expose non-feature multi-port contention telemetry (`merged_connection_count` / `ports_seen`) as recommended in [BATCH26A_V4_REMEDIATION_PLAN.md](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/reports/BATCH26A_V4_REMEDIATION_PLAN.md#L174-L183).
  3. **Defer Bidirectional 5-Tuple Grouping to a Newly Trained Successor Generation (V7+)** designed explicitly to handle both multi-service IoT connection isolation and port-randomized flood aggregation.

---

## Safety Footer

- **Files modified (Workstream B):** `0` (strictly read-only investigation; only deliverable `reports/BATCH26E_FLOW_IDENTITY_INVESTIGATION.md` created).
- **Tests executed and results:** Covered under Batch 26E consolidated suite (`68 passed, 0 failed, 0 skipped` in `29.76s`).
- **PCAP scans:** `NO`
- **Dataset generation:** `NO`
- **Training:** `NO`
- **Inference:** `NO`
- **Frozen hashes changed:** `0` (all 7 V4/V6.3 SHA-256 hashes verified unchanged via `Get-FileHash -Algorithm SHA256`)
- **Automated blocking:** `NO`

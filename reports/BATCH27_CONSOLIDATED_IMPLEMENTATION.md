# AEGIS-IoT — Batch 27 Consolidated Implementation

**Date:** 2026-11-12  
**Scope:** Shared packet parsing, directionality diagnostics, and frozen V4 contract documentation.  
**Safety:** No PCAP scans, dataset generation, training, inference, holdout access, automated blocking, or model/calibration/OOD edits.

## Implementation

- Builder and runtime now use `parse_ipv4_ethernet_frame()` from `aegis_live/packet_window.py`. Validation runs before timestamp, topology, or flow state changes. Rejection reasons are explicit for short Ethernet/IPv4 frames, VLAN/non-IPv4 frames, invalid IPv4 version/IHL/total length, truncated IPv4 payload, and truncated TCP/UDP headers.
- Ethernet padding is accepted; packet-size features and topology packet lengths preserve V4 `len(buf)` semantics. IPv4 fragment metadata is parsed; non-initial fragments do not attempt transport-header parsing and retain zero ports/flags.
- Added four diagnostic direction classes and count/proportion summaries. The canonical formula remains `is_inbound = _is_local(dst_ip)` and the canonical 29-feature list/order is unchanged.
- Documentation now states the frozen V4 window identity: symmetric bidirectional 3-tuple `(min(src_ip, dst_ip), max(src_ip, dst_ip), ip_proto)`, 50 packets, ports excluded from window identity. The topology tracker's separate 5-tuple use is identified as topology-only and excluded from canonical V4 features. Raw Ethernet packet length is retained.
- Training semantics were established from the supplied Batch 25/26E source evidence and dataset identity investigation; no completed-batch investigations were repeated.

## Exact files modified in Batch 27

- `aegis_live/packet_window.py`
- `aegis_live/v4_runtime.py`
- `aegis_live/build_v4_dataset.py`
- `aegis_live/network_topology_features.py` (documentation correction only)
- `reports/aegis_live_v4_dataset_construction.md`
- `tests/test_aegis_live_v4_parser_directionality.py` (new targeted synthetic regressions)
- `reports/BATCH27_CONSOLIDATED_IMPLEMENTATION.md` (this report)

## Verification

- `python -m py_compile` passed for the parser, runtime, builder, and new regression test.
- Targeted suite passed: **63 passed**, 3 upstream SHAP deprecation warnings. Covered parser/directionality regressions, timestamp ordering, temporal causality, IAT regression, extractor, and V4 dataset tests.
- Synthetic parser checks cover Ethernet padding, invalid IHL and total length, truncated IPv4 payload, TCP/UDP truncation, VLAN rejection, and non-initial fragments. A runtime regression confirms malformed packets leave timestamp, topology, and flow state unchanged.
- Shared parser invocation in builder and runtime establishes validation parity; both retain parsed `pkt_len == len(buf)`. Existing timestamp and causal snapshot safeguards passed in the targeted suite.
- `CANONICAL_29_FEATURES` remains 29 entries, including the same inbound/outbound positions. Direction diagnostics do not add or alter model features.

## Frozen artifact hashes

All seven hashes match the values recorded in the Batch 25 compatibility audit:

| Artifact | SHA-256 | Result |
|---|---|---|
| `models/aegis_live_v4/aegis_live_xgboost_baseline.pkl` | `C0E5AEE8E9E1726F12BFC19E98C440FE44FE278D1796D5866FE2A2488D458A56` | unchanged |
| `models/aegis_live_v4/temperature_scaling.json` | `C3AC0662CA896D06D4B9F62035A637FE04C8E01C00C466790D2EC0F03E581F99` | unchanged |
| `models/aegis_live_v4/ood_parameters.pkl` | `579063CFC4DDA3E396AB0D026198F14F8EF593EBF9910F290157AA0F669712C3` | unchanged |
| `models/aegis_live_v4/abstention_config.json` | `7D49F05BEF592CB838EF69F813A1C297BECA9E0A007BBAD1AAD6F5F0F4AC397F` | unchanged |
| `models/aegis_live_v6_3/config_b_model.pkl` | `09AFB03CBE74201E4E3768E367FB35C70D727BC41A5B9FFC6180C3305FDF1F84` | unchanged |
| `models/aegis_live_v6_3/temperature_scaling_b.json` | `2CEDF27975B4C28ABA2BBF52B84B0A5C5C49CAAF1EF3FA861AC6660885AA3B6F` | unchanged |
| `models/aegis_live_v6_3/abstention_config_b.json` | `E30BC4571C5045C3374AAE930145821F1277E1C9ABA56E4F419F7B10B39E2039` | unchanged |

## Compatibility risks and remaining issues

- Newly enforced full IPv4 and transport-length validation rejects malformed/truncated frames that the builder could previously admit and the runtime handled inconsistently. This is intentional parity hardening; frame-size features continue to include Ethernet padding as V4 did.
- VLAN-tagged frames remain explicitly rejected. Supporting VLAN decapsulation would be a separate behavior change and is not part of this implementation.
- Direction classes describe locality relative to the configured subnet(s); they do not infer client/server roles. The frozen inbound/outbound ratio can therefore retain its known dependence on tap placement.
- The added builder/runtime diagnostics are observational only and do not affect the frozen feature vector or flow identity.

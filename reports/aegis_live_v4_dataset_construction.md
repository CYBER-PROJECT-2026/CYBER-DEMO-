# AEGIS-Live v4 Dataset Construction Report

**Evaluation Contract**: TEMPORAL GENERALIZATION WITH PERSISTENT-FLOW IDENTITY  
**Target Schema**: Canonical 29 Features (24 Flow-Level Physical + 5 Network-Level Topology)  
**Total Dataset Size**: 80,000 sliding flow windows (56,000 Train, 12,000 Val, 12,000 Test)  
**Final Gate**: **`V4_DATASET_CONSTRUCTION = PASS`**  

---

## 1. Executive Summary & Architecture

AEGIS-Live v4 solves the topological information loss observed in v3. While v3 computed topology features inside isolated 5-tuple flow windows (causing `fan_in_ratio` to collapse to a constant 1.0), v4 implements a **dual extraction architecture**:

```
RAW PACKETS
    |
    +--> existing per-5-tuple flow window (50 pkts)
    |       |
    |       +--> 24 canonical physical features
    |
    +--> common network-level observation window (1.0s sliding)
            |
            +--> 5 validated network topology features
    |
    +--> combined 29-feature v4 row
```

### Feature Composition (29 Features):
- **Features 1..24**: Exact canonical 24 physical features (`duration_ms` through `outbound_pkt_ratio`).
- **Features 25..29**: 5 validated network-level topology features:
  1. `src_concentration_ratio`
  2. `src_ip_entropy`
  3. `src_ip_cardinality`
  4. `dst_ip_cardinality`
  5. `fan_in_ratio`

### Features Excluded with Proven Rationale:
- **`unique_flow_count`**: **EXCLUDED**. In the feasibility audit, DoS had median 11,196 flows while DDoS had 42. Investigation confirmed this is an artifact of the DoS attack generator cycling ephemeral source ports on every packet at 12,000 pps, not an indicator of distributed host topology.
- **`fan_out_ratio`**: **EXCLUDED**. Collinear inverse of `fan_in_ratio`.

---

## 2. Partition Population & Hashes

| Partition | Total Rows | DDoS | DoS | Spoofing | Benign | SHA-256 Hash |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 56,000 | 14,000 | 14,000 | 14,000 | 14,000 | `378d981f7871215487a2a46ef70e44107984df75e7317555eebe79f200fd9c63` |
| **Validation** | 12,000 | 3,000 | 3,000 | 3,000 | 3,000 | `f1a677ebd0bff3fdc5bc0f56d835062e8bd62a6ea0d68d86c4b7a779bcef4c78` |
| **Test** | 12,000 | 3,000 | 3,000 | 3,000 | 3,000 | `bf1c7c3783e79496dfee1a08bca5569d81c8dde07541acf6605cebf3158683f6` |
| **Total** | **80,000** | **20,000** | **20,000** | **20,000** | **20,000** | - |

---

## 3. Strict Temporal Quarantine Compliance

| PCAP Source | Train-Val Dead-Time Gap | Discarded Boundary Pkts (Train-Val) | Val-Test Dead-Time Gap | Discarded Boundary Pkts (Val-Test) |
| :--- | :--- | :--- | :--- | :--- |
| `DDoS-SYN_Flood.pcap` | 5.0010s (>= 5.0s) | 68,106 | 5.0114s (>= 5.0s) | 64,919 |
| `DoS-SYN_Flood.pcap` | 5.0001s (>= 5.0s) | 72,069 | 5.0018s (>= 5.0s) | 63,115 |
| `MITM-ArpSpoofing.pcap` | 5.0025s (>= 5.0s) | 3,083 | 5.0370s (>= 5.0s) | 1,202 |
| `BenignTraffic.pcap` | 5.0000s (>= 5.0s) | 56,140 | 5.0462s (>= 5.0s) | 499 |

---

## 4. Final Verdict

```
========================================================================================
  GATE VERDICT: V4_DATASET_CONSTRUCTION = PASS
========================================================================================
```

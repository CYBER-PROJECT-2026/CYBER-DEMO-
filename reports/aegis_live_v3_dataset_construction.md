# AEGIS-Live v3: Leakage-Safe Dataset Construction Report

## 1. Executive Summary & Audit Gate

| Audit Parameter | Specification / Result |
|---|---|
| **Audit Phase** | AEGIS-Live v3 Leakage-Safe Dataset Construction |
| **Evaluation Contract** | `TEMPORAL GENERALIZATION WITH PERSISTENT-FLOW IDENTITY` |
| **Audit Gate Decision** | **`V3_DATASET_CONSTRUCTION = PASS`** |
| **Feature Schema** | Canonical 27-Feature Schema (24 statistical + 3 validated topology features) |
| **Excluded Features** | `unique_5tuple_flow_count` (strictly rejected due to $KS=0.0361$ failure in feasibility audit) |
| **Source Data** | 4 authentic raw CICIoT2023 PCAP files totaling 7.81 GB (4,832,530 packets scanned) |
| **Extraction Engine** | `aegis_live/build_v3_dataset.py` |
| **AEGIS-Live v2 Invariant** | **100% FROZEN & UNTOUCHED** (Zero v2 files modified) |
| **v3 ML Model State** | **ZERO MODELS TRAINED** (Strictly dataset construction and audit only) |
| **Total Windows Extracted** | **80,000 balanced sliding flow-windows** (20,000 per class) |
| **Partitioning** | 56,000 Train (70%), 12,000 Validation (15%), 12,000 Test (15%) |
| **Temporal Quarantine** | Enforced 5.0-second state-machine raw timeline dead-time between partitions |
| **Cross-Partition Duplicates** | **0 exact duplicate 27-feature vectors across partitions** |

---

## 2. The 27-Feature Canonical Schema

The AEGIS-Live v3 schema preserves the exact 1..24 feature ordering of AEGIS-Live v2 and appends the 3 evidence-validated topology features:

| Index | Feature Name | Category | Mathematical Range | Physical Description |
|---|---|---|---|---|
| 1–24 | `CANONICAL_24_FEATURES` | Physical / Timing / Flags | Various | Preserved exactly from v2 baseline schema |
| 25 | `fan_in_ratio` | Topology | $[0.0, \infty)$ | Ratio of unique source IPv4 count to unique destination IPv4 count: $|\mathcal{S}_W| / |\mathcal{D}_W|$ |
| 26 | `src_concentration_ratio`| Topology | $[0.0, 1.0]$ | Fraction of packets originating from the dominant source IP: $\max C(s) / N_{\text{IP}}$ |
| 27 | `src_ip_cardinality_per_window`| Topology | $\mathbb{Z}_{\ge 1}$ | Distinct source IPv4 count in the window: $|\mathcal{S}_W|$ |

> [!NOTE]
> `unique_5tuple_flow_count` is **strictly excluded** from v3 in accordance with the feasibility audit findings. In volumetric SYN floods, both DDoS and DoS randomize source ports on every packet, saturating `unique_5tuple_flow_count` at 50 flows per 50 packets ($KS = 0.0361$, $96.04\%$ distribution overlap), providing zero discriminative signal for the flood classification boundary.

---

## 3. Dataset Construction & Temporal Quarantine Protocol

### 3.1 Partitioning & Balance

| Class Label | Source PCAP | Total Windows | Train (70%) | Validation (15%) | Test (15%) |
|---|---|---|---|---|---|
| **DDoS** | `DDoS-SYN_Flood.pcap` | 20,000 | 14,000 | 3,000 | 3,000 |
| **DoS** | `DoS-SYN_Flood.pcap` | 20,000 | 14,000 | 3,000 | 3,000 |
| **Spoofing** | `MITM-ArpSpoofing.pcap` | 20,000 | 14,000 | 3,000 | 3,000 |
| **Benign** | `BenignTraffic.pcap` | 20,000 | 14,000 | 3,000 | 3,000 |
| **TOTAL** | — | **80,000** | **56,000** | **12,000** | **12,000** |

### 3.2 Quarantine Dead-Time Audit Table

Every PCAP was subjected to an enforced $\ge 5.0\text{-second}$ state-machine dead-time between Train $\to$ Validation and Validation $\to$ Test:

| PCAP File | Label | Train End Ts | Val Start Ts | Train $\to$ Val Gap | Quarantined Pkts (Tr-Val) | Val End Ts | Test Start Ts | Val $\to$ Test Gap | Quarantined Pkts (Val-Te) |
|---|---|---|---|---|---|---|---|---|---|
| `DDoS-SYN_Flood.pcap` | DDoS | 1661537978.4838 | 1661537983.4848 | **5.0010s** | 68,106 | 1661537995.6709 | 1661538000.6823 | **5.0114s** | 64,919 |
| `DoS-SYN_Flood.pcap` | DoS | 1659447776.2928 | 1659447781.2932 | **5.0004s** | 72,021 | 1659447791.3806 | 1659447796.3815 | **5.0009s** | 63,115 |
| `MITM-ArpSpoofing.pcap` | Spoofing | 1666098731.1481 | 1666098736.1496 | **5.0015s** | 3,083 | 1666100891.6205 | 1666100896.6446 | **5.0241s** | 1,202 |
| `BenignTraffic.pcap` | Benign | 1665176981.6465 | 1665176986.6466 | **5.0001s** | 56,140 | 1665177887.4840 | 1665177892.5302 | **5.0462s** | 499 |

---

## 4. Architectural Analysis: Flow-Window Scope vs. Topology Visibility

An essential scientific observation emerges from comparing the v3 dataset extraction with the standalone raw-PCAP feasibility audit:

1. **Per-Flow Bidirectional Windowing Contract**:
   - In accordance with the prompt requirements, the dataset builder reused the exact validated v2 windowing contract (`flow_key = (min_ip, max_ip, proto)`).
   - Within an isolated flow window, the only participating endpoints are `min_ip` and `max_ip`.
   - Consequently:
     - When an attack stream is purely unidirectional (attacker $\to$ victim with no response), `src_ip_cardinality = 1.0`, `dst_ip_cardinality = 1.0`, and `fan_in_ratio = 1.0`.
     - When an attack stream is bidirectional (victim returns RST/ACK), `src_ip_cardinality = 2.0`, `dst_ip_cardinality = 2.0`, and `fan_in_ratio = 1.0`.
2. **Impact on Multi-Class Discrimination**:
   - In Benign traffic, communication is heavily bidirectional ($\text{cardinality} = 2.0$ for 97.8% of windows, $\text{src\_concentration} = 0.52$ median).
   - In DDoS flood traffic, communication is 98% unidirectional ($\text{cardinality} = 1.0$, $\text{src\_concentration} = 1.0$).
   - This provides strong separation between normal IoT sessions and unidirectional floods.
   - For distinguishing distributed botnets from single-source DoS within the same IP pair, the directional asymmetry and concentration variance offer complementary signals.

---

## 5. Artifact Checklist & Directory Structure

All v3 dataset files were generated in [`data/aegis_live_v3/`](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/data/aegis_live_v3/):

- `train.csv` (56,000 rows, SHA-256 verified)
- `validation.csv` (12,000 rows, SHA-256 verified)
- `test.csv` (12,000 rows, SHA-256 verified)
- `distribution_audit.csv` (Full 27-feature distribution breakdown across classes and partitions)
- `dataset_manifest.json` (Complete provenance metadata, PCAP hashes, quarantine records, exclusion logs)

```text
V3_DATASET_CONSTRUCTION = PASS
```

# AEGIS-Live v2 — Pre-Training Forensic Dataset Audit

**Document ID:** `AEGIS-LIVE-V2-PRETRAIN-AUDIT-20260920`  
**Date:** 2026-09-20  
**Target Repository:** `data/aegis_live_v2/`  
**Audited Files:**  
- `data/aegis_live_v2/train.csv` (16.91 MB, 56,000 rows)  
- `data/aegis_live_v2/validation.csv` (3.75 MB, 12,000 rows)  
- `data/aegis_live_v2/test.csv` (3.60 MB, 12,000 rows)  
- `data/aegis_live_v2/dataset_manifest.json`  
- `data/aegis_live_v2/distribution_audit.json`  
- `aegis_live/build_v2_dataset.py`  
- `reports/aegis_live_v2_dataset_construction.md`  

**AEGIS v1 State:** 100% FROZEN & UNTOUCHED (`models/aegis/*` untouched)  
**Schema Version:** Canonical 24-Feature Schema (`aegis_live/feature_schema.py`)  
**ML Model Training:** ZERO models trained (pre-training forensic dataset audit only)  

---

## 1. Executive Summary & Forensic Verdict

This audit conducted an independent, exhaustive forensic verification of the constructed AEGIS-Live v2 multi-class dataset prior to any model training, calibration, scaling, OOD fitting, or architecture changes.

In accordance with forensic audit mandates:
- **Zero models were trained.**
- **Zero scalers or calibrators were fitted.**
- **AEGIS v1 remained 100% frozen.**
- **The canonical 24-feature schema was not modified.**
- **PCAP files were preserved unchanged.**
- **Zero rows were silently altered or manipulated.**

### Audit Finding & Gate Decision:
While the dataset achieves **zero temporal lookahead leakage**, **zero overlapping windows**, **zero cross-partition exact feature duplicates**, **100% provenance integrity**, and **exact byte-identical reproduction**, the audit discovered that **canonical bidirectional flow leakage across partitions is NON-ZERO**:
- **37 canonical flows** appear in both Train and Validation.
- **43 canonical flows** appear in both Train and Test.
- **34 canonical flows** appear in both Validation and Test.

Because the pre-training audit gate strictly mandates:
$$\text{canonical flow leakage} = 0$$
as a non-negotiable passing condition, the formal verdict is:

```
========================================================================================
PRE-TRAINING FORENSIC DATASET AUDIT VERDICT
========================================================================================
PRETRAINING_DATASET_AUDIT = FAIL
========================================================================================
BLOCKING ISSUE:
Canonical bidirectional flow leakage across partitions is non-zero (37 flows cross
Train-Val, 43 flows cross Train-Test). This occurs because chronological temporal
segmentation of single-target attack PCAPs (DoS: 99.3% single flow, MITM: 94.3% single
flow) permits long-running flow endpoints to span from Train into Validation and Test.
========================================================================================
```

---

## 2. POINT 1 — Dataset Population & Sampling Audit

### a) Were all possible valid windows from each PCAP processed?
**No.** The raw PCAP captures contain millions of packets:
- `DDoS-SYN_Flood.pcap`: 26,597,011 packets (~530,000 potential 50-packet windows)
- `DoS-SYN_Flood.pcap`: 26,261,611 packets (~525,000 potential 50-packet windows)
- `MITM-ArpSpoofing.pcap`: 2,491,622 packets (~48,000 potential 50-packet windows)
- `BenignTraffic.pcap`: 3,664,164 packets (~70,000 potential 50-packet windows)

### b) What deterministic stopping/sampling rule was used?
A deterministic stream-accumulation stopping rule was used:
Extraction proceeded sequentially from packet 0 of each capture and terminated immediately upon accumulating exactly **20,000 valid 50-packet windows** per PCAP (`TARGET_WINDOWS_PER_CLASS = 20000`).

### c) How many candidate windows were available per PCAP before the cap?
- DDoS: ~530,000 windows available
- DoS: ~525,000 windows available
- MITM: ~48,000 windows available
- Benign: ~70,000 windows available

### d) Is the selected 20,000 simply the first 20,000 valid windows?
**Yes.** The 20,000 windows per PCAP are strictly the first 20,000 valid windows encountered in chronological packet arrival order.

### e) Is there any random selection?
**No.** Random selection was not used. Neither `random` nor `np.random` is imported or invoked anywhere in `aegis_live/build_v2_dataset.py`.

### f) If random selection exists, identify seed and mechanism:
**N/A.** Zero random selection exists; the process is 100% deterministic streaming prefix extraction.

### g) Confirmation that no labels were used to choose feature vectors:
**Confirmed.** All candidate windows that passed the physical/mathematical invariant filters were accepted in arrival order. Class labels were assigned 100% strictly from the source PCAP file provenance.

---

## 3. POINT 2 — Flow / Session Grouping Audit

### Forensic Inspection of Grouping Implementation:
In `aegis_live/build_v2_dataset.py`:
- `flow_id`: Defined as `f"{ip_pair[0]}<->{ip_pair[1]}:{ip_proto}"` where `ip_pair = tuple(sorted([src_ip, dst_ip]))`.
- `group_id`: Defined as `f"{class_label}_{current_partition}_blk{blk}"`.

> [!IMPORTANT]
> **Terminology Clarification:**  
> The construction report (`reports/aegis_live_v2_dataset_construction.md`) reported "28 train groups, 4 val groups, 8 test groups with 0 overlap."  
> **Correction:** These groups are **temporal block segments per class**, NOT canonical flow keys:
> - Train: 7 temporal blocks $\times$ 4 classes = 28 groups
> - Validation: 1 temporal block $\times$ 4 classes = 4 groups
> - Test: 2 temporal blocks $\times$ 4 classes = 8 groups
>
> While these temporal block groups do not overlap across partitions, the **underlying canonical bidirectional flow keys DO cross partitions**.

### Canonical Flow Intersection Across Partitions:

$$\text{Canonical Flow Key} = (\min(\text{src\_ip}, \text{dst\_ip}), \max(\text{src\_ip}, \text{dst\_ip}), \text{ip\_proto})$$

| Partition Comparison | Unique Canonical Flows | Overlapping Canonical Flows | Flow Overlap Percentage |
| :--- | :---: | :---: | :---: |
| **Train unique flows** | 460 | — | — |
| **Validation unique flows** | 97 | — | — |
| **Test unique flows** | 172 | — | — |
| **Train $\cap$ Validation** | — | **37** | 38.1% of Val flows |
| **Train $\cap$ Test** | — | **43** | 25.0% of Test flows |
| **Validation $\cap$ Test** | — | **34** | 35.1% of Val flows |
| **Train $\cap$ Validation $\cap$ Test** | — | **25** | 25.8% of Val flows |

### Per-Class Flow Overlap Breakdown:

| Class Label | Source PCAP | Train Flows | Val Flows | Test Flows | Train $\cap$ Val | Train $\cap$ Test | Val $\cap$ Test |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DDoS** | `DDoS-SYN_Flood.pcap` | 15 | 7 | 7 | **7** | **6** | **6** |
| **DoS** | `DoS-SYN_Flood.pcap` | 16 | 4 | 5 | **4** | **4** | **3** |
| **Spoofing** | `MITM-ArpSpoofing.pcap` | 252 | 73 | 121 | **16** | **19** | **18** |
| **Benign** | `BenignTraffic.pcap` | 177 | 13 | 39 | **10** | **14** | **7** |

### Why Flow Overlap Occurs:
In single-target attacks (DoS SYN flood and MITM ARP spoofing), the attack traffic is dominated by continuous communication between specific IP endpoints:
- In `DoS-SYN_Flood.pcap`, **99.3%** of packets belong to `192.168.137.182 <-> 192.168.137.187:6`.
- In `MITM-ArpSpoofing.pcap`, **94.3%** of packets belong to `192.168.137.162 <-> 205.251.251.216:6`.
- In `BenignTraffic.pcap`, real IoT devices communicate continuously or periodically with the local gateway `192.168.137.1`.

Because the dataset was partitioned **chronologically in time**, the same long-running network sessions continue to send packets across Train, Validation, and Test timelines.

---

## 4. POINT 3 — Temporal Split Audit

Every block timestamp range and row count was recomputed directly from `train.csv`, `validation.csv`, and `test.csv`:

### Per-PCAP Temporal Boundary Table:

| Source PCAP | Partition | Start Timestamp | End Timestamp | Duration (s) | Rows | Gap to Next Partition |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`DDoS-SYN_Flood.pcap`** | Train | 1661537880.5165 | 1661537978.4838 | 97.967s | 14,000 | +0.95 $\mu$s |
| | Validation | 1661537978.4838 | 1661537990.3158 | 11.832s | 3,000 | +1.19 $\mu$s |
| | Test | 1661537990.3158 | 1661538002.2852 | 11.969s | 3,000 | — |
| **`DoS-SYN_Flood.pcap`** | Train | 1659447720.7685 | 1659447776.2900 | 55.522s | 14,000 | +33.14 $\mu$s |
| | Validation | 1659447776.2900 | 1659447786.6052 | 10.315s | 3,000 | +20.03 $\mu$s |
| | Test | 1659447786.6053 | 1659447797.3049 | 10.700s | 3,000 | — |
| **`MITM-ArpSpoofing.pcap`** | Train | 1666038000.7444 | 1666098731.1481 | 60,730.404s | 14,000 | +3.10 $\mu$s |
| | Validation | 1666098731.1481 | 1666100862.3884 | 2,131.240s | 3,000 | +15.37 ms |
| | Test | 1666100862.4038 | 1666102899.7831 | 2,037.379s | 3,000 | — |
| **`BenignTraffic.pcap`** | Train | 1665162900.3504 | 1665176981.6465 | 14,081.296s | 14,000 | +25.03 $\mu$s |
| | Validation | 1665176981.6466 | 1665177235.7898 | 254.143s | 3,000 | +0.95 $\mu$s |
| | Test | 1665177235.7898 | 1665178907.4548 | 1,671.665s | 3,000 | — |

### Temporal Integrity Verification:
- $\text{Train End Time} < \text{Validation Start Time}$: **PASS** (strictly positive gap across all 4 PCAPs).
- $\text{Validation End Time} < \text{Test Start Time}$: **PASS** (strictly positive gap across all 4 PCAPs).
- Temporal interval overlap: **0 occurrences** (zero lookahead).
- Row sampling/randomization inside blocks: **0 occurrences** (pure chronological stream order).

### Temporal Block Distribution:
- **Train:** Exactly 7 blocks (Blocks 0–6) of exactly 2,000 rows each per class = 14,000 rows/class.
- **Validation:** Exactly 1 block (Block 7) of exactly 3,000 rows per class = 3,000 rows/class.
- **Test:** Exactly 2 blocks (Blocks 8–9) of exactly 1,500 rows each per class = 3,000 rows/class.

---

## 5. POINT 4 — 70/15/15 Claim Audit

The exact partition counts (56,000 / 12,000 / 12,000) were obtained through a deterministic streaming threshold rule:
1. Target windows per class was set to $N = 20,000$.
2. Target partition ratios:
   - $\text{Train} = 0.70 \times 20,000 = 14,000$ windows/class
   - $\text{Validation} = 0.15 \times 20,000 = 3,000$ windows/class
   - $\text{Test} = 0.15 \times 20,000 = 3,000$ windows/class
3. Transition rules:
   - When valid window count reaches 14,000, active flow buffer is flushed, boundary timestamp is recorded, and partition transitions to `validation`.
   - When valid window count reaches 17,000, active flow buffer is flushed, boundary timestamp is recorded, and partition transitions to `test`.
   - When valid window count reaches 20,000, PCAP reader terminates.
4. Total across 4 classes:
   - Train: $14,000 \times 4 = 56,000$ (70.0%)
   - Validation: $3,000 \times 4 = 12,000$ (15.0%)
   - Test: $3,000 \times 4 = 12,000$ (15.0%)
   - Total: $80,000$ (100.0%)

There is zero post-hoc row dropping, zero random subsampling, and zero unexplained row exclusion.

---

## 6. POINT 5 — Window Integrity Audit

Recomputed directly across all 80,000 rows in `train.csv`, `validation.csv`, and `test.csv`:

| Invariant / Check | Evaluated Criterion | Observed Count | Invariant Status |
| :--- | :--- | :---: | :---: |
| **Packet Count** | Window capacity == 50 packets | 80,000 / 80,000 (Min: 50, Max: 50) | **PASS** |
| **Internal Gap $\ge$ 2.0s** | Consecutive packet gap $\ge$ 2.0s inside window | 0 / 80,000 | **PASS** |
| **Feature Vector Size** | Exactly 24 canonical features present | 80,000 / 80,000 | **PASS** |
| **NaN Values** | Features containing NaN | 0 / 80,000 | **PASS** |
| **Inf Values** | Features containing $\pm\infty$ | 0 / 80,000 | **PASS** |
| **Negative Duration** | `duration_ms` < 0 | 0 / 80,000 | **PASS** |
| **Negative Rates** | `pkt_rate_pps` < 0 or `byte_rate_bps` < 0 | 0 / 80,000 | **PASS** |
| **Packet Size Validity** | $14 \le \text{min} \le \text{mean} \le \text{max} \le 65535$ | 0 violations | **PASS** |
| **Ratio Bounds** | All 13 ratio features $\in [0.0, 1.0]$ | 0 violations | **PASS** |
| **Directional Sum** | $|\text{inbound} + \text{outbound} - 1.0| \le 10^{-5}$ | 0 violations (Max diff: 0.00e+00) | **PASS** |
| **TTL Bounds** | $1.0 \le \text{ip\_ttl\_mean} \le 255.0$ | 0 violations | **PASS** |

---

## 7. POINT 6 — Duplicate Audit

Recomputed across 24-feature floating-point tuples:

### Cross-Partition Intersections:
- **Train $\cap$ Validation:** **0** duplicate feature vectors
- **Train $\cap$ Test:** **0** duplicate feature vectors
- **Validation $\cap$ Test:** **0** duplicate feature vectors

### Within-Partition Duplicates:
- **Train:** 1 duplicate feature vector (55,999 unique out of 56,000; two distinct windows during a fixed-rate SYN flood exhibited identical float statistics)
- **Validation:** 0 duplicate feature vectors (12,000 unique out of 12,000)
- **Test:** 0 duplicate feature vectors (12,000 unique out of 12,000)

### Provenance Uniqueness:
When evaluated on `(label, pcap_file, flow_id, temporal_block_id, window_start_ts, window_end_ts)`:
- Exactly **80,000 unique records** out of 80,000 (100.0% uniqueness).

---

## 8. POINT 7 — Adjacency / Temporal Correlation Audit

Audit of boundary transitions between partitions:

| PCAP File | Boundary Transition | Minimum Timestamp Gap | Overlapping Windows | Boundary Windows Same Flow? |
| :--- | :---: | :---: | :---: | :---: |
| **`DDoS-SYN_Flood.pcap`** | Train $\to$ Validation | **+0.0000009537s** (0.95 $\mu$s) | 0 | **True** (`192.168.137.30<->192.168.137.58:6`) |
| | Validation $\to$ Test | **+0.0000011921s** (1.19 $\mu$s) | 0 | **True** (`192.168.137.17<->192.168.137.30:6`) |
| **`DoS-SYN_Flood.pcap`** | Train $\to$ Validation | **+0.0000331402s** (33.1 $\mu$s) | 0 | **True** (`192.168.137.182<->192.168.137.187:6`) |
| | Validation $\to$ Test | **+0.0000200272s** (20.0 $\mu$s) | 0 | **True** (`192.168.137.182<->192.168.137.187:6`) |
| **`MITM-ArpSpoofing.pcap`** | Train $\to$ Validation | **+0.0000030994s** (3.10 $\mu$s) | 0 | **True** (`172.253.122.128<->192.168.137.150:6`) |
| | Validation $\to$ Test | **+0.0153691769s** (15.4 ms) | 0 | **True** (`192.168.137.48<->51.145.143.28:6`) |
| **`BenignTraffic.pcap`** | Train $\to$ Validation | **+0.0000250340s** (25.0 $\mu$s) | 0 | **True** (`192.168.137.207<->74.125.0.107:6`) |
| | Validation $\to$ Test | **+0.0000009537s** (0.95 $\mu$s) | 0 | **True** (`192.168.137.175<->34.249.233.178:6`) |

> [!WARNING]
> **Adjacency Correlation Finding:**  
> In all 8 boundary transitions, the last window of partition $A$ and the first window of partition $B$ belong to the **exact same physical flow**. Because the inter-window gap is on the order of microseconds (e.g., 33 $\mu$s in DoS), these boundary windows represent contiguous traffic bursts from the same flow session crossing the partition boundary.

---

## 9. POINT 8 — Label Provenance Audit

Cross-tabulation of `pcap_file` vs `label` across all 80,000 rows:

| PCAP Filename | DDoS Count | DoS Count | Spoofing Count | Benign Count | Total Rows | Provenance Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `DDoS-SYN_Flood.pcap` | **20,000** | 0 | 0 | 0 | 20,000 | **100% Provenance Match** |
| `DoS-SYN_Flood.pcap` | 0 | **20,000** | 0 | 0 | 20,000 | **100% Provenance Match** |
| `MITM-ArpSpoofing.pcap` | 0 | 0 | **20,000** | 0 | 20,000 | **100% Provenance Match** |
| `BenignTraffic.pcap` | 0 | 0 | 0 | **20,000** | 20,000 | **100% Provenance Match** |

- Zero single PCAP contains multiple assigned labels.
- Zero class labels were inferred from packet features or behavioral heuristics.

---

## 10. POINT 9 — Feature Distribution Audit

Detailed per-feature descriptive statistics across Train (56,000), Validation (12,000), and Test (12,000):

| Feature | Train Mean $\pm$ Std | Val Mean $\pm$ Std | Test Mean $\pm$ Std | Train Min – Max | NaN / Inf |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `duration_ms` | 969.62 $\pm$ 3316.98 | 298.53 $\pm$ 1976.32 | 844.60 $\pm$ 2815.01 | [0.06, 49491.02] | 0 / 0 |
| `pkt_rate_pps` | 18999.28 $\pm$ 46720.14 | 17405.69 $\pm$ 38585.45 | 14565.01 $\pm$ 35967.65 | [1.01, 819200.00] | 0 / 0 |
| `byte_rate_bps` | 1.91e6 $\pm$ 3.87e6 | 3.36e6 $\pm$ 4.98e6 | 1.10e6 $\pm$ 2.25e6 | [136.04, 5.47e7] | 0 / 0 |
| `iat_mean_ms` | 19.79 $\pm$ 67.69 | 6.09 $\pm$ 40.33 | 17.24 $\pm$ 57.45 | [0.00, 1010.02] | 0 / 0 |
| `iat_std_ms` | 31.13 $\pm$ 81.20 | 9.57 $\pm$ 37.79 | 29.69 $\pm$ 69.82 | [0.00, 742.46] | 0 / 0 |
| `pkt_size_mean` | 441.77 $\pm$ 622.69 | 569.01 $\pm$ 600.45 | 468.40 $\pm$ 604.31 | [45.96, 4874.20] | 0 / 0 |
| `pkt_size_std` | 371.33 $\pm$ 551.35 | 574.56 $\pm$ 688.17 | 473.48 $\pm$ 659.92 | [0.00, 5677.41] | 0 / 0 |
| `pkt_size_min` | 83.35 $\pm$ 158.88 | 72.62 $\pm$ 99.76 | 64.90 $\pm$ 56.97 | [44.00, 1514.00] | 0 / 0 |
| `pkt_size_max` | 1322.97 $\pm$ 2125.31 | 2236.49 $\pm$ 2922.14 | 1855.16 $\pm$ 2815.05 | [46.00, 37714.00] | 0 / 0 |
| `syn_ratio` | 0.4703 $\pm$ 0.4725 | 0.4648 $\pm$ 0.4824 | 0.4724 $\pm$ 0.4784 | [0.00, 1.00] | 0 / 0 |
| `ack_ratio` | 0.4923 $\pm$ 0.4792 | 0.5262 $\pm$ 0.4845 | 0.4950 $\pm$ 0.4829 | [0.00, 1.00] | 0 / 0 |
| `rst_ratio` | 0.0405 $\pm$ 0.1552 | 0.0370 $\pm$ 0.1458 | 0.0357 $\pm$ 0.1459 | [0.00, 1.00] | 0 / 0 |
| `fin_ratio` | 0.0134 $\pm$ 0.0499 | 0.0018 $\pm$ 0.0176 | 0.0080 $\pm$ 0.0379 | [0.00, 0.48] | 0 / 0 |
| `urg_ratio` | 0.0000 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | [0.00, 0.00] | 0 / 0 |
| `psh_ratio` | 0.1265 $\pm$ 0.2147 | 0.0539 $\pm$ 0.1220 | 0.0779 $\pm$ 0.1603 | [0.00, 1.00] | 0 / 0 |
| `ece_ratio` | 0.0001 $\pm$ 0.0032 | 0.0002 $\pm$ 0.0057 | 0.0003 $\pm$ 0.0063 | [0.00, 0.20] | 0 / 0 |
| `cwr_ratio` | 0.0001 $\pm$ 0.0019 | 0.0001 $\pm$ 0.0028 | 0.0002 $\pm$ 0.0033 | [0.00, 0.12] | 0 / 0 |
| `ip_ttl_mean` | 88.80 $\pm$ 45.90 | 80.56 $\pm$ 25.31 | 86.81 $\pm$ 36.00 | [4.00, 247.00] | 0 / 0 |
| `ip_ttl_std` | 21.54 $\pm$ 33.42 | 16.94 $\pm$ 23.40 | 19.50 $\pm$ 32.18 | [0.00, 92.93] | 0 / 0 |
| `tcp_ratio` | 0.9528 $\pm$ 0.2121 | 0.9898 $\pm$ 0.1007 | 0.9626 $\pm$ 0.1898 | [0.00, 1.00] | 0 / 0 |
| `udp_ratio` | 0.0463 $\pm$ 0.2101 | 0.0092 $\pm$ 0.0957 | 0.0361 $\pm$ 0.1865 | [0.00, 1.00] | 0 / 0 |
| `icmp_ratio` | 0.0009 $\pm$ 0.0305 | 0.0010 $\pm$ 0.0316 | 0.0013 $\pm$ 0.0365 | [0.00, 1.00] | 0 / 0 |
| `inbound_pkt_ratio` | 0.7565 $\pm$ 0.3040 | 0.7463 $\pm$ 0.2750 | 0.7595 $\pm$ 0.2744 | [0.00, 1.00] | 0 / 0 |
| `outbound_pkt_ratio` | 0.2435 $\pm$ 0.3040 | 0.2537 $\pm$ 0.2750 | 0.2405 $\pm$ 0.2744 | [0.00, 1.00] | 0 / 0 |

---

## 11. POINT 10 — Feature Variance Audit

Variance metrics computed on TRAIN ONLY (56,000 rows):

| Feature | Variance | Std | Mean | Coefficient of Variation (CV) | Variance Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `duration_ms` | 1.100e+07 | 3316.98 | 969.62 | 342.09% | Normal |
| `pkt_rate_pps` | 2.183e+09 | 46720.14 | 18999.28 | 245.90% | Normal |
| `byte_rate_bps` | 1.496e+13 | 3.868e+06 | 1.908e+06 | 202.71% | Normal |
| `iat_mean_ms` | 4582.40 | 67.69 | 19.79 | 342.09% | Normal |
| `iat_std_ms` | 6593.01 | 81.20 | 31.13 | 260.82% | Normal |
| `pkt_size_mean` | 3.877e+05 | 622.69 | 441.77 | 140.95% | Normal |
| `pkt_size_std` | 3.040e+05 | 551.35 | 371.33 | 148.48% | Normal |
| `pkt_size_min` | 25244.14 | 158.88 | 83.35 | 190.63% | Normal |
| `pkt_size_max` | 4.517e+06 | 2125.31 | 1322.97 | 160.65% | Normal |
| `syn_ratio` | 0.2232 | 0.4725 | 0.4703 | 100.47% | Normal |
| `ack_ratio` | 0.2296 | 0.4792 | 0.4923 | 97.34% | Normal |
| `rst_ratio` | 0.0241 | 0.1552 | 0.0405 | 382.88% | Normal |
| `fin_ratio` | 0.0025 | 0.0499 | 0.0134 | 371.29% | Normal |
| **`urg_ratio`** | **0.0000** | **0.0000** | **0.0000** | **Undefined (NaN)** | **ZERO VARIANCE** |
| `psh_ratio` | 0.0461 | 0.2147 | 0.1265 | 169.63% | Normal |
| `ece_ratio` | 1.004e-05 | 0.003168 | 0.000108 | 2927.85% | Active (94 non-zero rows) |
| `cwr_ratio` | 3.652e-06 | 0.001911 | 0.000074 | 2597.43% | Active (115 non-zero rows) |
| `ip_ttl_mean` | 2106.97 | 45.90 | 88.80 | 51.69% | Normal |
| `ip_ttl_std` | 1116.92 | 33.42 | 21.54 | 155.17% | Normal |
| `tcp_ratio` | 0.0450 | 0.2121 | 0.9528 | 22.26% | Normal |
| `udp_ratio` | 0.0441 | 0.2101 | 0.0463 | 454.02% | Normal |
| `icmp_ratio` | 0.0009 | 0.0305 | 0.0009 | 3280.16% | Normal |
| `inbound_pkt_ratio` | 0.0924 | 0.3040 | 0.7565 | 40.19% | Normal |
| `outbound_pkt_ratio` | 0.0924 | 0.3040 | 0.2435 | 124.85% | Normal |

- **Zero Variance:** Exactly 1 feature (`urg_ratio`). Zero packets had URG set.
- **Near-Zero Variance:** 0 features (all other features have $\sigma \ge 10^{-4}$).
- **`ece_ratio` & `cwr_ratio` Status:** Active non-zero variance confirmed (94 and 115 non-zero rows in Train).
- **Feature Preservation:** In accordance with audit instructions, `urg_ratio` is retained untouched.

---

## 12. POINT 11 — Correlation Audit

Pearson correlation was computed on **TRAIN ONLY** (56,000 rows):

### Pairs with $|r| > 0.95$ in Train:
1. **(`duration_ms`, `iat_mean_ms`)**: $r = +1.000000$  
   *Physical Origin:* In a 50-packet capacity window, $\text{iat\_mean} \equiv \frac{\text{duration}}{49}$.
2. **(`inbound_pkt_ratio`, `outbound_pkt_ratio`)**: $r = -1.000000$  
   *Physical Origin:* Directional ratio complement invariant: $\text{inbound} + \text{outbound} \equiv 1.0$.
3. **(`tcp_ratio`, `udp_ratio`)**: $r = -0.989632$  
   *Physical Origin:* Binary transport division in IoT networks where almost all non-TCP traffic is UDP.

### Validation & Test Correlations (Reported for Observation Only):
- **Validation:** 4 pairs ($|r|>0.95$): `(duration_ms, iat_mean_ms)` ($+1.00$), `(inbound, outbound)` ($-1.00$), `(pkt_size_std, pkt_size_max)` ($+0.954$), `(ece_ratio, cwr_ratio)` ($+0.962$).
- **Test:** 4 pairs ($|r|>0.95$): `(duration_ms, iat_mean_ms)` ($+1.00$), `(inbound, outbound)` ($-1.00$), `(tcp_ratio, udp_ratio)` ($-0.981$), `(ece_ratio, cwr_ratio)` ($+0.986$).

Neither validation nor test correlations were used for feature selection or pruning.

---

## 13. POINT 12 — Source PCAP Integrity Audit

All 4 source PCAPs were re-hashed via SHA-256 and compared against `data/aegis_live_v2/dataset_manifest.json`:

| PCAP Filename | File Size (Bytes) | Manifest Hash | Recomputed Hash | Integrity |
| :--- | :---: | :--- | :--- | :---: |
| `DDoS-SYN_Flood.pcap` | 2,048,000,063 | `5e8124a87e12483af05cb5b4f36dc095c2a979b08f2ee4e18918c26ad3b041b8` | `5e8124a87e12483af05cb5b4f36dc095c2a979b08f2ee4e18918c26ad3b041b8` | **MATCH** |
| `DoS-SYN_Flood.pcap` | 2,048,000,026 | `3f45569b3438b1ee5fa9105922a423789e4de176a66d550082f5f082f6546ebd` | `3f45569b3438b1ee5fa9105922a423789e4de176a66d550082f5f082f6546ebd` | **MATCH** |
| `MITM-ArpSpoofing.pcap` | 2,046,238,529 | `da097ba27f5690cf4fe47b916c825735af8a829074a71202624740bf4b7cc196` | `da097ba27f5690cf4fe47b916c825735af8a829074a71202624740bf4b7cc196` | **MATCH** |
| `BenignTraffic.pcap` | 2,048,000,100 | `549a89337f529412f2507f862d06394c95e5966b9c01aaa980511590657d1dc8` | `549a89337f529412f2507f862d06394c95e5966b9c01aaa980511590657d1dc8` | **MATCH** |

All source PCAPs have remained 100% byte-for-byte immutable since dataset construction.

---

## 14. POINT 13 — Reproduction Audit

The complete dataset extraction was re-executed from the raw PCAPs using `python aegis_live/build_v2_dataset.py --output-dir data/temp_audit_v2` targeting an isolated temporary directory.

Row-for-row and byte-level comparison results:

| File | Row Count Check | Labels Equal? | Provenance Equal? | Max Float Diff | Byte-Identical? | Exact File Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `train.csv` | 56,000 == 56,000 | **True** | **True** | **0.00e+00** | **TRUE** | 17,729,381 bytes |
| `validation.csv` | 12,000 == 12,000 | **True** | **True** | **0.00e+00** | **TRUE** | 3,932,166 bytes |
| `test.csv` | 12,000 == 12,000 | **True** | **True** | **0.00e+00** | **TRUE** | 3,777,628 bytes |

The pipeline demonstrates **100.0% deterministic, byte-identical reproducibility**.

---

## 15. POINT 14 — Final Forensic Gate Verdict & Summary

### Evaluation of Passing Criteria:

| # | Mandatory Condition | Audit Finding | Status |
| :-: | :--- | :--- | :---: |
| 1 | Sampling mechanism is fully explained | Deterministic streaming prefix of first 20k valid windows | **MET** |
| 2 | No unexplained random row split | 100% deterministic chronological block assignment | **MET** |
| 3 | **Canonical flow leakage = 0** | **FAILED: 37 flows cross Train-Val; 43 cross Train-Test** | **UNMET** |
| 4 | Temporal leakage = 0 | Strictly non-decreasing time; train end < val start < test start | **MET** |
| 5 | Exact cross-partition feature duplicate = 0 | 0 duplicate feature tuples cross partitions | **MET** |
| 6 | No overlapping windows cross partitions | 0 overlapping windows across all boundaries | **MET** |
| 7 | All four classes present in all three partitions | Exactly 14k / 3k / 3k per class across Train / Val / Test | **MET** |
| 8 | All 24 features valid | 0 NaN, 0 Inf, 0 negative values, valid size & ratio bounds | **MET** |
| 9 | No NaN/Inf | Confirmed 0 / 80,000 | **MET** |
| 10 | Label provenance verified | 100% provenance from PCAP file names | **MET** |
| 11 | PCAP hashes verified | All 4 SHA-256 hashes match manifest exactly | **MET** |
| 12 | Reproduction succeeds | Confirmed byte-identical regeneration from raw PCAPs | **MET** |
| 13 | 70/15/15 split is reproducible | Confirmed exact 56k / 12k / 12k split | **MET** |
| 14 | No hidden preprocessing has occurred | Raw physical metrics preserved; 0 scalers or normalizers fitted | **MET** |

### Final Verdict:
```
========================================================================================
PRETRAINING_DATASET_AUDIT = FAIL
========================================================================================
```

---

## Concise Audit Summary

1. **Dataset Size:** 80,000 windows (Train: 56,000, Validation: 12,000, Test: 12,000). Exactly 20,000 per class.
2. **Exact Sampling Rule:** Sequential streaming prefix from packet 0 until accumulating 20,000 valid 50-packet windows per PCAP. Zero random sampling.
3. **Exact Grouping Rule:** Temporal block segments per class (`group_id = f"{class}_{partition}_blk{id}"`).
4. **Exact Split Rule:** Windows 0–13,999 $\to$ Train (70%), Windows 14,000–16,999 $\to$ Validation (15%), Windows 17,000–19,999 $\to$ Test (15%), with in-flight window flushing and boundary timestamp quarantine at transitions.
5. **Flow Leakage Result:** **NON-ZERO (37 flows in Train $\cap$ Val, 43 flows in Train $\cap$ Test).** Caused by single-target PCAPs (DoS: 99.3% single flow, MITM: 94.3% single flow) spanning across chronological time partitions.
6. **Temporal Leakage Result:** **ZERO.** Positive gaps across all boundaries ($\Delta t > 0$). Zero future lookahead.
7. **Duplicate Result:** **ZERO cross-partition 24-feature duplicates.** 1 within-Train duplicate. 80,000 / 80,000 unique with provenance.
8. **Zero-Variance Features:** Exactly 1 (`urg_ratio`). `ece_ratio` and `cwr_ratio` have active non-zero variance.
9. **High-Correlation Pairs:** Exactly 3 pairs in Train with $|r| > 0.95$: `(duration_ms, iat_mean_ms)`, `(inbound, outbound)`, and `(tcp_ratio, udp_ratio)`.
10. **Reproducibility Result:** **100.0% BYTE-IDENTICAL.** Regenerated CSVs from raw PCAPs match persisted CSVs byte-for-byte (Max float difference: $0.00\times 10^0$).
11. **Final Verdict:** **`PRETRAINING_DATASET_AUDIT = FAIL`** (blocked strictly by the non-zero canonical flow leakage criterion).

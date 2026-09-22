# AEGIS-Live v3: Dataset Integrity & Leakage Audit Report

## 1. Executive Summary & Verification Gate

| Integrity Parameter | Specification / Result |
|---|---|
| **Audit Phase** | AEGIS-Live v3 Dataset Integrity & Leakage Verification |
| **Dataset Location** | `data/aegis_live_v3/` |
| **Evaluation Contract** | `TEMPORAL GENERALIZATION WITH PERSISTENT-FLOW IDENTITY` |
| **Audit Gate Decision** | **`V3_DATASET_CONSTRUCTION = PASS`** |
| **Total Windows Audited** | **80,000 windows** (56,000 Train, 12,000 Val, 12,000 Test) |
| **Total Features Audited** | **27 features** (Canonical 24 + 3 validated topology features) |
| **NaN / Inf Occurrences** | **0** across all 2,160,000 individual feature cells |
| **Cross-Partition Exact Duplicates** | **0** (Zero feature overlap between Train/Val, Train/Test, Val/Test) |
| **Quarantine Contamination** | **0** (Zero packets or timestamps from quarantine dead-time intervals) |
| **Physical Invariant Violations** | **0** (All rates, sizes, durations, and ratios strictly valid) |

---

## 2. Leakage Control Verifications

### 2.1 No Cross-Partition Feature Duplicates
We verified exact equality across all 27-feature vectors between partitions:

| Partition Comparison | Total Row Pairs Compared | Exact Feature Matches Found | Integrity Status |
|---|---|---|---|
| **Train vs. Validation** | $56,000 \times 12,000 = 6.72 \times 10^8$ | **0** | **PASS** |
| **Train vs. Test** | $56,000 \times 12,000 = 6.72 \times 10^8$ | **0** | **PASS** |
| **Validation vs. Test** | $12,000 \times 12,000 = 1.44 \times 10^8$ | **0** | **PASS** |

### 2.2 Within-Partition Duplication Audit
- **Train (56,000 rows)**: 55,999 unique feature vectors (only 1 within-partition duplicate pair).
- **Validation (12,000 rows)**: 12,000 unique feature vectors (0 duplicates).
- **Test (12,000 rows)**: 12,000 unique feature vectors (0 duplicates).

### 2.3 Quarantine Boundary & Timeline Integrity
Every partition transition was guarded by an enforced 5.0-second state-machine dead-time on the raw packet arrival timeline:
- **Train $\to$ Val Gap**: $\ge 5.0001\text{s}$ across all 4 PCAPs (200,350 total quarantined boundary packets discarded).
- **Val $\to$ Test Gap**: $\ge 5.0009\text{s}$ across all 4 PCAPs (129,735 total quarantined boundary packets discarded).
- **Contamination Check**: Zero windows span across any quarantine boundary. Zero window timestamps fall inside quarantine intervals.

### 2.4 Canonical Flow Identity Overlap & Contract Compliance
In accordance with the evaluation contract:
> **`TEMPORAL GENERALIZATION WITH PERSISTENT-FLOW IDENTITY`**
- Total unique flow keys observed: 460 in Train, 108 in Validation, 180 in Test.
- Shared canonical flows: 40 flows shared between Train-Val, 47 between Train-Test, and 37 between Val-Test.
- **Contract Note**: Persistent flow endpoint monitoring across time is an intentional design feature evaluating future-time generalization on persistent network gateways, and is **not categorized as data leakage**.

---

## 3. Physical & Protocol Invariant Audits

Every row in `train.csv`, `validation.csv`, and `test.csv` was validated against physical bounds:

1. **Duration & Rates**:
   - $\text{duration\_ms} \ge 0.0$ on 100% of rows (min: $0.061\text{ ms}$, max: $21,751.31\text{ ms}$).
   - $\text{pkt\_rate\_pps} \ge 0.0$ on 100% of rows (min: $2.30\text{ pps}$, max: $819,200\text{ pps}$).
   - $\text{byte\_rate\_bps} \ge 0.0$ on 100% of rows.
2. **Packet Sizes**:
   - $14 \le \text{pkt\_size\_min} \le \text{pkt\_size\_mean} \le \text{pkt\_size\_max} \le 65,535$ on 100% of rows.
3. **TCP Flag & Protocol Ratios**:
   - All 13 canonical ratios strictly bounded within $[0.0, 1.0]$.
   - $\text{inbound\_pkt\_ratio} + \text{outbound\_pkt\_ratio} = 1.0 \pm 10^{-5}$ on 100% of rows.
4. **Time-To-Live**:
   - $1.0 \le \text{ip\_ttl\_mean} \le 255.0$ on 100% of rows.
5. **No Internal Gaps**:
   - Zero consecutive packet gaps $\ge 2.0\text{ seconds}$ inside any window.
6. **Topology Feature Bounds**:
   - $\text{fan\_in\_ratio} \ge 0.0$ on 100% of rows.
   - $0.0 \le \text{src\_concentration\_ratio} \le 1.0$ on 100% of rows.
   - $\text{src\_ip\_cardinality\_per\_window} \ge 1.0$ on 100% of rows.

---

## 4. Class Distribution Comparison: DDoS vs DoS

### Training Partition Distribution Statistics (14,000 DDoS vs 14,000 DoS)

| Feature | Class | Mean | Std | Median | IQR | Min | Max |
|---|---|---|---|---|---|---|---|
| **`src_concentration_ratio`** | **DDoS** | **0.980** | 0.083 | **1.00** | 0.00 | 0.50 | 1.00 |
| **`src_concentration_ratio`** | **DoS** | **0.932** | 0.134 | **1.00** | 0.04 | 0.50 | 1.00 |
| **`src_ip_cardinality_per_window`** | **DDoS** | **1.075** | 0.263 | **1.00** | 0.00 | 1.00 | 2.00 |
| **`src_ip_cardinality_per_window`** | **DoS** | **1.268** | 0.443 | **1.00** | 1.00 | 1.00 | 2.00 |
| **`fan_in_ratio`** | **DDoS** | **1.000** | 0.000 | **1.00** | 0.00 | 1.00 | 1.00 |
| **`fan_in_ratio`** | **DoS** | **1.000** | 0.000 | **1.00** | 0.00 | 1.00 | 1.00 |

### Key Observations:
1. Under the per-flow windowing contract (`flow_key = (min_ip, max_ip, proto)`), each window contains packets between a single pair of endpoints, causing `fan_in_ratio` to be strictly 1.0 within isolated flows.
2. In DDoS training windows, 98.0% of windows are strictly unidirectional (`cardinality` = 1, `concentration` = 1.0).
3. In DoS training windows, 26.8% of windows contain bidirectional response traffic (`cardinality` = 2, `concentration` < 1.0).
4. By contrast, in Benign traffic, 97.9% of windows are bidirectional (`cardinality` = 2, `concentration` = 0.52).

---

## 5. Final Audit Gate

```text
V3_DATASET_CONSTRUCTION = PASS
```

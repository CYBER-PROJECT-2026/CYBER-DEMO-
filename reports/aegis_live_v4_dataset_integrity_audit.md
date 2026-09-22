# AEGIS-Live v4 Dataset Integrity & Leakage Audit Report

**Evaluation Contract**: TEMPORAL GENERALIZATION WITH PERSISTENT-FLOW IDENTITY  
**Verification Target**: Zero Cross-Partition Leakage, Zero Temporal Contamination, Mathematical Invariant Compliance  
**Final Gate**: **`V4_DATASET_CONSTRUCTION = PASS`**  

---

## 1. Temporal Quarantine & Zero Leakage Audit

1. **State-Machine Quarantine Dead-Times**:
   - Enforced $\ge 5.0$ seconds state-machine quarantine intervals between Train $\to$ Val and Val $\to$ Test.
   - All packets arriving during quarantine intervals were recorded as discarded and prevented from entering any flow window or network tracker.
2. **Network Topology Tracker Boundary Reset**:
   - At every partition transition, `NetworkTopologyTracker.reset()` cleared all sliding window queues, IP frequency tables, and packet counters.
   - Zero state or packet information from earlier temporal partitions carries across into validation or test partitions.
3. **Zero Cross-Partition Feature Duplication**:
   - Exact 29-feature tuple hash checking confirmed **0 identical feature vectors** across partitions.

---

## 2. Invariant Compliance Checklist

- **Strict 29-Feature Order**: Verified.
- **24 Canonical Physical Features**: Unchanged from v2/v3 baseline.
- **Physical Bounds**: All durations $\ge 0$, packet rates $\ge 0$, packet sizes in $[14, 65535]$.
- **Ratio Bounds**: All 14 ratios strictly within $[0.0, 1.0]$.
- **Network Topology Bounds**: `src_ip_cardinality` $\ge 1.0$, `dst_ip_cardinality` $\ge 1.0$, `fan_in_ratio` $\ge 0.0$, `src_ip_entropy` $\ge 0.0$.
- **No NaN / Inf**: 100% verified across all 2,320,000 feature cells.
- **Immutable Baselines Preserved**: `models/aegis_live_v2/` and `models/aegis_live_v3/` remain untouched.

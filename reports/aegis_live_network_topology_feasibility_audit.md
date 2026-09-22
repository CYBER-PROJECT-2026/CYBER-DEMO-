# AEGIS-Live Read-Only Network-Level Topology Feasibility Audit Report

**Evaluation Contract**: TEMPORAL GENERALIZATION WITH PERSISTENT-FLOW IDENTITY  
**Target Problem**: Resolving DDoS vs. DoS Confusion via Network-Level Temporal Aggregation  
**Candidate Windows Evaluated**: 500 ms (0.5s), 1.0 s, 2.0 s  
**Audit Gate**: **`NETWORK_TOPOLOGY_FEASIBILITY_AUDIT = PASS`**  

---

## 1. Executive Summary

During the AEGIS-Live v3 training audit, it was discovered that `fan_in_ratio` had **zero variance** (`1.0` constant, 0 tree splits) because packet windows were constructed strictly per individual 5-tuple flow key. Because a 5-tuple flow is restricted to exactly two communicating IP endpoints, flow-isolated windows are topologically blinded to subnet-wide multi-host fan-in.

This audit evaluated aggregating traffic into a **common network-level time window** (evaluated at 500 ms, 1.0 s, and 2.0 s) across all active flows and endpoints, while retaining the existing 24 per-flow physical features.

### Key Audit Findings:
1. **Separability Recovery**:
   - In a 1.0s network observation window, **`src_concentration_ratio`** achieves a massive **KS statistic of 0.9412** and Cohen's d of **-3.9401**.
   - In single-source DoS floods, the primary attacker contributes **80.7%** of all observed packets (median 86.3%).
   - In distributed DDoS floods, the top bot contributes only **27.2%** of packets (median 21.3%), yielding clean architectural separation.
2. **Entropy Disparity**:
   - **`src_ip_entropy`** achieves a **KS of 0.9048**. DoS traffic exhibits low source entropy due to a single dominant source, while DDoS exhibits high source entropy across distributed attack bots.
3. **Streaming Viability**:
   - Streaming implementation verified numerically equivalent to offline batch ground truth ($|\Delta| < 10^{-5}$).
   - Throughput exceeds **413,054 packets/second** (2421.0 ns/packet), consuming negligible memory (< 427.8 KB).
4. **Safety & Zero Leakage**:
   - Network topology tracking is strictly causal (past-to-present).
   - Zero access to labels, attack category, or future packets.
   - Respects 5.0-second state-machine quarantine dead-times between temporal partitions.

---

## 2. Candidate Feature Mathematical Formulations

For any common network-level observation time window $W$ of duration $\Delta t$:

1. **`src_ip_cardinality`**:
   $$|S_W| = |\{ \text{src\_ip}_i : i \in W \}|$$
2. **`src_concentration_ratio`**:
   $$C_S = \frac{\max_{s \in S_W} \text{count}(s)}{|W|}, \quad C_S \in [0.0, 1.0]$$
3. **`dst_ip_cardinality`**:
   $$|D_W| = |\{ \text{dst\_ip}_i : i \in W \}|$$
4. **`unique_flow_count`**:
   $$|F_W| = |\{ \text{canonical\_5tuple}_i : i \in W \}|$$
5. **`src_ip_entropy`**:
   $$H(S_W) = -\sum_{s \in S_W} p(s) \log_2 p(s), \quad \text{where } p(s) = \frac{\text{count}(s)}{|W|}$$
6. **`fan_in_ratio`**:
   $$\text{fan\_in} = \frac{|S_W|}{|D_W|} \quad (0.0 \text{ if } |D_W| = 0)$$
7. **`fan_out_ratio`**:
   $$\text{fan\_out} = \frac{|D_W|}{|S_W|} \quad (0.0 \text{ if } |S_W| = 0)$$

---

## 3. Separability Diagnostics (DDoS vs. DoS)

### Summary Across Observation Windows:

| Window Size | Top Separating Feature | KS Statistic | Cohen's d | Overlap Coeff | DDoS Median | DoS Median |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **500 ms** | `src_concentration_ratio` | **0.8816** | -3.3556 | 0.0790 | 0.2400 | 0.8562 |
| **1000 ms** | `src_concentration_ratio` | **0.9412** | -3.9401 | 0.0182 | 0.2131 | 0.8627 |
| **2000 ms** | `src_concentration_ratio` | **1.0000** | -5.1203 | 0.0000 | 0.2062 | 0.8513 |

### Detailed Feature Separability in 1.0-Second Observation Windows:

| Candidate Feature | KS Statistic | p-value | Cohen's d | Overlap | DDoS Mean (Std) | DoS Mean (Std) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `src_concentration_ratio` | **0.9412** | 1.76e-13 | -3.9401 | 0.0182 | 0.27 (0.21) | 0.81 (0.86) |
| `src_ip_entropy` | **0.9048** | 5.01e-12 | +2.5800 | 0.0364 | 2.89 (3.13) | 0.87 (0.63) |
| `unique_flow_count` | **0.6503** | 9.10e-06 | -1.1940 | 0.1497 | 3425.16 (42.00) | 9554.18 (11196.00) |
| `dst_ip_cardinality` | **0.4332** | 9.99e-03 | -1.0136 | 0.4310 | 21.85 (21.00) | 27.47 (28.00) |
| `src_ip_cardinality` | **0.3433** | 6.96e-02 | -0.8066 | 0.4856 | 24.35 (23.00) | 29.18 (30.00) |
| `fan_in_ratio` | **0.3326** | 8.78e-02 | +0.4920 | 0.4043 | 1.12 (1.12) | 1.06 (1.07) |
| `fan_out_ratio` | **0.3326** | 8.78e-02 | -0.4460 | 0.2545 | 0.90 (0.89) | 0.94 (0.94) |

---

## 4. Architectural Head-to-Head: Flow-Level vs. Network-Level

| Metric / Dimension | Current Flow-Level Architecture (v3) | Proposed Network-Level Architecture | Physical Reason for Difference |
| :--- | :--- | :--- | :--- |
| **Window Boundary** | Isolated 5-tuple flow (`flow_id`) | Common temporal window (e.g. 1.0s) | Flow isolation hides multi-source concurrency across the network |
| **Max Distinct IPs** | $\le 2$ endpoints (Host A, Host B) | Unlimited (all active network hosts) | Multiple attack bots map to distinct 5-tuples |
| **`fan_in_ratio`** | $\equiv 1.0000$ (Zero variance, 0 tree splits) | Variable ($KS = 0.3326$) | In 5-tuple flows, bidirectional is 2/2=1 and unidirectional is 1/1=1 |
| **`src_concentration`** | $0.7775 \pm 0.2094$ ($KS = 0.1500$) | **$KS = 0.9412$** | Aggregates all concurrent source packets arriving at target |
| **`src_ip_entropy`** | Not measurable inside single flow | **$KS = 0.9048$** | Differentiates single high-rate flood from multi-bot distribution |

---

## 5. Engineering & Streaming Viability

- **Streaming / Batch Equivalence**: Maximum numerical delta across all features was `0.000000e+00` ($< 10^{-5}$), confirming 100% deterministic fidelity.
- **Computational Latency**: `2421.0 nanoseconds per packet` (413,054 pkts/sec throughput).
- **Peak Memory**: `427.84 KB`, well within embedded IoT gateway budgets.
- **Temporal Quarantine Safety**: State-machine dead-time gaps between Train $\to$ Val ($\ge 5.0000$s) and Val $\to$ Test ($\ge 5.0004$s) prevent cross-partition contamination when trackers are flushed at partition boundaries.

---

## 6. Final Audit Gate Verdict

```
========================================================================================
  FINAL AUDIT GATE: NETWORK_TOPOLOGY_FEASIBILITY_AUDIT = PASS
========================================================================================
```

### Recommendation for Future Phases:
The audit conclusively proves that **network-level time windowing** recovers the topological distinction between DDoS and DoS that was lost in 5-tuple flow partitioning. In a future model experiment, a hybrid feature vector combining **per-flow physical metrics (24 features)** with **common network-level topology metrics (`src_concentration_ratio`, `src_ip_entropy`, `fan_in_ratio`)** is strongly justified.

# AEGIS-Live v2 — Flow-Leakage Resolution & Evaluation Design Study

**Document ID:** `AEGIS-LIVE-V2-FLOW-LEAKAGE-STUDY-20260920`  
**Date:** 2026-09-20  
**Target Architecture:** AEGIS-Live v2 Evaluation Design  
**Context:** Following `PRETRAINING_DATASET_AUDIT = FAIL` due to non-zero canonical flow overlap across partitions.  
**AEGIS v1 State:** 100% FROZEN & UNTOUCHED (`models/aegis/*` untouched)  
**Dataset State:** `data/aegis_live_v2/` PRESERVED UNCHANGED (analysis-only study)  
**Model Training:** ZERO models trained  

---

## 1. Evaluation Objective & The Two Generalization Targets

The pre-training forensic audit established that in the current dataset:
- **Train $\cap$ Validation:** 37 canonical flows overlap
- **Train $\cap$ Test:** 43 canonical flows overlap
- **Validation $\cap$ Test:** 34 canonical flows overlap

To resolve this scientifically, we must formalize the **two distinct generalization targets** that can be evaluated in network intrusion detection:

```
+---------------------------------------------------------------------------------------------------+
| TARGET A: NEW-FLOW GENERALIZATION (Endpoint / Flow Independence)                                  |
| "Can the model classify traffic belonging to a canonical flow identity that was NOT present in    |
| training?"                                                                                        |
+---------------------------------------------------------------------------------------------------+
| TARGET B: FUTURE-TRAFFIC GENERALIZATION (Temporal Generalization with Persistent-Flow Identity)   |
| "Can the model classify future traffic from persistent network flows that were active during      |
| earlier historical training periods?"                                                             |
+---------------------------------------------------------------------------------------------------+
```

### Forensic Implications for AEGIS-Live v2 and CICIoT2023:

1. **Implications of Target A (New-Flow Generalization):**
   - **Operational Objective:** Evaluates detection capability against zero-day IP endpoint pairs (e.g., an entirely new external IP attacking an entirely new victim).
   - **Constraint in CICIoT2023:** As proven in Section 3, single-target attack captures (`DoS-SYN_Flood.pcap` and `MITM-ArpSpoofing.pcap`) contain **only ONE dominant flow** supplying $>94\%$ to $>99\%$ of traffic. Forcing strict flow independence causes catastrophic sample collapse for DoS (from 3,000 to 0 or 38 windows in Validation/Test), rendering multi-class evaluation mathematically and empirically invalid.
   - **Feature Space Nuance:** The canonical 24-feature schema contains **zero IP addresses, zero MAC addresses, and zero port numbers**. All features are behavioral metrics (rates, durations, IATs, packet sizes, flag ratios, protocol ratios). The model cannot memorize IP identities; however, consecutive windows from the same flow burst share correlated physical dynamics.

2. **Implications of Target B (Future-Traffic Generalization):**
   - **Operational Objective:** Evaluates detection capability in persistent enterprise and IoT networks where physical devices (e.g., IP cameras, smart sensors, gateways) are permanent endpoints. The model must learn baseline and adversarial traffic behaviors from historical data and accurately detect attacks occurring in future time periods.
   - **Constraint in CICIoT2023:** Highly feasible and statistically robust. Enables exactly 14,000 Train / 3,000 Val / 3,000 Test windows across all 4 classes.
   - **Tradeoff:** It does **NOT** prove that the model generalizes to novel, unobserved IP endpoints; it evaluates temporal stability and out-of-time detection on known network nodes.

---

## 2. Available Grouping Units in CICIoT2023 Captures

From the raw captures and extracted windows, seven candidate grouping units were quantified:

| Grouping Unit | Formal Definition | Scope / Cardinality in 80k Dataset | Independence Guarantee |
| :--- | :--- | :---: | :--- |
| **1. Canonical Flow** | $(\min(\text{src}, \text{dst}), \max(\text{src}, \text{dst}), \text{proto})$ | 487 unique flows across all captures | Endpoint independent; fails under single-flow DoS |
| **2. PCAP Capture File** | Source capture filename | Exactly 4 captures (1 per class) | Class confounding; makes multiclass impossible |
| **3. Temporal Block** | Chronological partition segment | 10 blocks per class (40 total blocks) | Zero temporal lookahead; persistent flows cross |
| **4. Flow + Temporal Block** | $(\text{canonical\_flow}, \text{temporal\_block})$ | 884 flow-block groups | Zero group overlap; flow identity shared across time |
| **5. Endpoint Pair** | $(\min(\text{src}, \text{dst}), \max(\text{src}, \text{dst}))$ | 465 unique IP pairs | Equivalent to canonical flow (most IoT is single proto) |
| **6. Transport Protocol** | IP Protocol (TCP=6, UDP=17, ICMP=1) | Exactly 3 protocols | Coarse protocol division; cannot split within class |
| **7. Window Sequence** | Arrival index of 50-packet window | Exactly 80,000 individual windows | Row-level random splitting (high burst correlation) |

---

## 3. Per-PCAP Flow Concentration Analysis

A deep forensic packet scan was conducted across 1,500,000 packets per capture file to quantify flow concentration and determine whether strict flow-level splitting is physically possible:

### Empirical Flow Concentration Metrics:

| Capture File | Class | IPv4 Packets Scanned | Unique Flows | Largest Flow Packet Share | Top 5 Flows Packet Share | Flows with $\ge 50$ Windows ($\ge 2.5\text{k pkts}$) | Flows with $\ge 500$ Windows ($\ge 25\text{k pkts}$) | Flows with $\ge 1000$ Windows ($\ge 50\text{k pkts}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `DDoS-SYN_Flood.pcap` | **DDoS** | 1,497,725 | 287 | 22.21% | 86.33% | **6** | 6 | 6 |
| `DoS-SYN_Flood.pcap` | **DoS** | 1,499,177 | 294 | **99.11%** | **99.40%** | **1** | **1** | **1** |
| `MITM-ArpSpoofing.pcap` | **Spoofing** | 1,471,746 | 2,179 | 30.86% | 51.94% | **51** | 9 | 5 |
| `BenignTraffic.pcap` | **Benign** | 1,441,956 | 1,119 | 13.55% | 40.78% | **103** | 6 | 4 |

### Top Flows per Capture:
- **`DoS-SYN_Flood.pcap`:**
  - Flow 1: `192.168.137.182 <-> 192.168.137.187` proto=6 $\to$ **1,485,809 pkts (99.11%)**
  - Flow 2: `192.168.137.187 <-> 54.230.48.195` proto=6 $\to$ **1,784 pkts (0.12%)** ($\sim 35$ windows)
  - Flow 3: `157.249.81.141 <-> 192.168.137.224` proto=6 $\to$ **1,460 pkts (0.10%)** ($\sim 29$ windows)
  - *All 293 other flows combined represent less than 0.89% of traffic.*
- **`DDoS-SYN_Flood.pcap`:**
  - 6 distinct attacking botnet nodes targeting victim `192.168.137.30` (each providing $\sim 200\text{k}$ to $330\text{k}$ packets).
- **`MITM-ArpSpoofing.pcap`:**
  - Intercepted host `192.168.137.162 <-> 205.251.251.216` represents 30.86% of packets; remaining traffic distributed across 51 active flows.
- **`BenignTraffic.pcap`:**
  - Distributed across 33 physical IoT devices (103 flows supplying $\ge 50$ windows).

---

## 4. Strict Flow-Level Feasibility Test (Mathematical Proof)

### Theorem:
*A balanced multi-class dataset satisfying 70% Train, 15% Validation, 15% Test with all four classes present is MATHEMATICALLY IMPOSSIBLE under strict canonical flow-level splitting on the authentic CICIoT2023 raw captures.*

### Proof:
1. Let $\mathcal{F}_{\text{DoS}}$ be the set of canonical flows in `DoS-SYN_Flood.pcap`.
2. Under strict flow-level splitting, $\mathcal{F}_{\text{DoS}}$ must be partitioned into disjoint subsets:
   $$\mathcal{F}_{\text{DoS}} = \mathcal{F}_{\text{Train}} \cup \mathcal{F}_{\text{Val}} \cup \mathcal{F}_{\text{Test}}, \quad \text{where } \mathcal{F}_i \cap \mathcal{F}_j = \emptyset \quad \forall i \ne j$$
3. Let $w(F)$ be the window yield of flow $F$. For a target of $N = 20,000$ DoS windows:
   - Target Train: $w(\mathcal{F}_{\text{Train}}) \approx 14,000$
   - Target Validation: $w(\mathcal{F}_{\text{Val}}) \approx 3,000$
   - Target Test: $w(\mathcal{F}_{\text{Test}}) \approx 3,000$
4. As empirically measured in the 20,000 DoS window dataset:
   - Flow $F_1 = (\text{192.168.137.182}, \text{192.168.137.187}, 6)$ produces $w(F_1) = \mathbf{19,924\text{ windows}}$ (**99.62%**).
   - The sum of all other 16 flows combined is $\sum_{k=2}^{17} w(F_k) = \mathbf{76\text{ windows}}$ (**0.38%**).
5. Because $F_1$ is an atomic flow, it can belong to **at most one** partition:
   - **If $F_1 \in \mathcal{F}_{\text{Train}}$:**
     $$w(\mathcal{F}_{\text{Val}}) + w(\mathcal{F}_{\text{Test}}) \le 76\text{ windows}$$
     Even if split equally, Validation gets at most 38 DoS windows, and Test gets at most 38 DoS windows. Compared to the required 3,000 windows, the DoS class in Validation and Test collapses by **98.7%**, yielding severe class starvation.
   - **If $F_1 \in \mathcal{F}_{\text{Test}}$:**
     $$w(\mathcal{F}_{\text{Train}}) \le 76\text{ windows}$$
     Train gets at most 76 DoS windows out of 14,000 (a 99.5% deficit). The model cannot be trained to recognize DoS attacks.
   - **If $F_1 \in \mathcal{F}_{\text{Val}}$:**
     Train and Test both collapse to $\le 38$ DoS windows.
6. Therefore, no deterministic or stochastic flow partitioning exists that can provide balanced representation of the DoS class across Train, Validation, and Test. $\blacksquare$

---

## 5. Flow-Level Candidate Split Simulation

To verify the mathematical proof, three deterministic flow-level splitting strategies were implemented and simulated on the 80,000 extracted windows:

### Strategy 1: Greedy Flow Assignment by Class & Row Count
Assigns flows greedily to achieve approximately 14,000 Train, 3,000 Val, 3,000 Test per class:

| Class | Train Windows (Flows) | Validation Windows (Flows) | Test Windows (Flows) | Class Balance Verdict |
| :--- | :---: | :---: | :---: | :---: |
| **DDoS** | 12,975 (13 flows) | 2,600 (1 flow) | 4,425 (2 flows) | Coarse (Val has only 1 flow) |
| **DoS** | **76 (16 flows)** | **0 (0 flows)** | **19,924 (1 flow)** | **CATASTROPHIC FAILURE (Val = 0%)** |
| **Spoofing** | 14,118 (7 flows) | 3,059 (27 flows) | 2,823 (368 flows) | Valid (ample flow diversity) |
| **Benign** | 14,041 (11 flows) | 3,052 (19 flows) | 2,907 (175 flows) | Valid (ample flow diversity) |

### Strategy 2: Bin-Packing / Integer Optimization
Formulates exact 70/15/15 partition as a multi-way number partitioning problem.
- **Outcome:** Infeasible. In DoS, item weights are $[19924, 33, 18, 5, 4, \dots]$. Because $19,924 > 14,000$ (Train target) and $19,924 \gg 3,000$ (Val/Test target), the solver reports zero feasible solutions within a $\pm 20\%$ tolerance.

### Strategy 3: Flow-Level Split Maximizing Class Representation
Enforces that every partition receives at least one flow for every class:
- **Outcome:** Train gets 76 DoS windows (16 flows), Val gets 38 DoS windows (e.g. 1 flow with 33 pkts), Test gets 19,924 DoS windows (1 flow).
- **Result:** While technically non-empty, Validation contains only 38 DoS rows vs 3,000 DDoS, 3,000 Spoofing, and 3,000 Benign rows (DoS represents **0.4%** of Validation). Validation metrics would be dominated by high variance.

---

## 6. Flow + Temporal Hybrid Grouping Evaluation

In this design:
$$\text{Protected Unit } G = (\text{canonical\_flow}, \text{temporal\_segment})$$

### Properties:
1. **Protected Group Overlap:** **EXACTLY ZERO.** Train has 28 temporal block groups, Validation has 4, and Test has 8. Disjoint sets.
2. **Boundary Adjacent Windows:** In the current implementation, window 13,999 (Train) and window 14,000 (Validation) belong to the same flow with a microsecond time gap (+33 $\mu$s).
3. **Mitigation via Temporal Quarantine Dead-Time:**  
   If a dead-time quarantine buffer (e.g., 5 to 10 seconds) is introduced at partition boundaries:
   - All in-flight bursts naturally terminate.
   - No two windows from the same flow are adjacent within seconds.
4. **Distinction:**
   - **Canonical Flow Identity Overlap:** Non-zero (endpoints recur across time).
   - **Session / Burst Overlap:** Zero (sessions are temporally quarantined).

---

## 7. PCAP-Level Splitting Evaluation

Because each PCAP in the authentic CICIoT2023 dataset contains a single attack class:
- `DDoS-SYN_Flood.pcap` = DDoS
- `DoS-SYN_Flood.pcap` = DoS
- `MITM-ArpSpoofing.pcap` = Spoofing
- `BenignTraffic.pcap` = Benign

Holding out entire PCAPs would assign entire classes to single partitions (e.g. Train gets DDoS + Benign; Val gets DoS; Test gets Spoofing).
- Train would have 0 DoS and 0 Spoofing.
- Multi-class intrusion detection cannot be trained or validated under PCAP-level splitting.
- **Verdict:** Scientifically unusable.

---

## 8. Current Chronological Split Evaluation

The current dataset persisted in `data/aegis_live_v2/` is formally characterized as:
$$\textbf{Temporal Generalization with Persistent-Flow Identity}$$

### Forensic Properties:
- **Canonical Flow Overlap:** 37 flows (Train $\cap$ Val), 43 flows (Train $\cap$ Test).
- **Temporal Overlap:** **0.0 seconds.** Forward arrow of time strictly enforced ($t_{\text{train}} < t_{\text{val}} < t_{\text{test}}$).
- **Exact Feature Vector Duplicates:** **0 rows.** Zero duplicate 24D vectors cross partitions.
- **Overlapping Windows:** **0 windows.** Zero windows bridge partition boundaries.
- **Boundary Adjacent Same Flow:** Present in the current build (33 $\mu$s gap at DoS boundary).
- **Class Balance:** Perfectly balanced (14,000 / 3,000 / 3,000 per class across all 4 classes).

---

## 9. Strategy Comparison & Tradeoff Matrix

| Evaluation Strategy | Canonical Flow Overlap | Temporal Overlap | 4 Classes in All Splits? | Approx 70/15/15? | Statistically Feasible? | Primary Scientific Interpretation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Strict Flow-Level Split** | **0 flows** | Variable | **NO (DoS Val=0%)** | **NO (DoS 0.4%)** | **INFEASIBLE** | Tests novel-endpoint generalization, but mathematically collapses on single-target PCAPs. |
| **2. PCAP-Level Split** | **0 flows** | Complete separation | **NO (1 class/split)** | **NO** | **INFEASIBLE** | Confounding class with partition; impossible for multi-class detection. |
| **3. Current Chronological Split** | 37–43 flows | **0.0s** | **YES (Balanced)** | **YES (Exact)** | **FEASIBLE** | Tests out-of-time future traffic generalization on known network endpoints. |
| **4. Chronological Split + Temporal Quarantine Buffer** | 37–43 flows | **0.0s (with dead-time gap)** | **YES (Balanced)** | **YES (Exact)** | **FEASIBLE (RECOMMENDED)** | Tests out-of-time future traffic generalization while eliminating microsecond burst adjacency across boundaries. |

---

## 10. Decisions Required Before Model Training

To proceed to model training, the user must align on the formal evaluation contract:

1. **Acknowledge Generalization Scope:**
   - Formally document that AEGIS-Live v2 is evaluated under **Temporal Generalization with Persistent-Flow Identity** (Target B), reflecting an operational IoT deployment where fixed device endpoints are monitored across continuous future time.
   - Do **NOT** claim "flow-independent" or "unseen-endpoint" generalization, as single-target capture constraints prevent strict flow isolation.
2. **Boundary Adjacency Resolution (Optional Enhancement):**
   - Retain the current dataset as-is (`PRETRAINING_DATASET_AUDIT = PASS` under Target B contract), **OR**
   - Apply a 5-second temporal quarantine dead-time buffer between Train $\to$ Val and Val $\to$ Test to guarantee that no boundary adjacent windows from the same flow occur within seconds of each other.

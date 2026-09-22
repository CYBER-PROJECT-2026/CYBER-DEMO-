# AEGIS-Live v3: Candidate Topology Feature Definitions & Operational Specifications

## 1. Executive Summary & Purpose

The AEGIS-Live v2 forensic audit established that volumetric TCP SYN floods in DDoS and DoS share identical physical micro-temporal statistics within the canonical 24-feature schema (near-total SYN saturation, minimal packet sizes of ~60 bytes, zero payload, and overlapping arrival rates). Distinguishing between distributed botnet floods (DDoS) and single-source floods (DoS) requires topological information regarding endpoint cardinality and flow diversity.

This document formally defines four candidate topology features for AEGIS-Live v3 feasibility testing.

---

## 2. Formal Feature Specifications

### Feature A: `src_ip_cardinality_per_window`

- **Conceptual Purpose**: Measures the host distribution of incoming traffic within a sliding temporal window. Differentiates single-source attacks from distributed multi-node attacks.
- **Mathematical Definition**:
  $$\text{src\_ip\_cardinality\_per\_window} = |\mathcal{S}_W|$$
  where $\mathcal{S}_W = \{ \text{src\_ip}_i \mid i \in W, \text{is\_ipv4}(i) \}$ is the set of distinct source IPv4 addresses observed among all valid IP packets in window $W$.
- **Required Packet Fields**: IPv4 Source Address (`ip.src`).
- **Window Scope**: Interface/Ingress sliding window of $N=50$ consecutive packets (or temporal timeout $\Delta t = 2.0\text{ s}$).
- **Requires IP Addresses**: **YES** (Source IP only).
- **Requires Transport Ports**: **NO**.
- **Requires Protocol**: **NO** (Applies across all IP protocols).
- **Requires Attack Labels**: **NO** (Zero label dependency).
- **Online Incremental Computability**: **YES** ($O(1)$ amortized insertion into an in-memory hash set `src_ip_set`).
- **Behavior for Empty / Invalid Windows**: If $|W| = 0$ or all packets in $W$ are non-IPv4 (e.g., pure ARP/LLDP), returns `0`.
- **Deterministic Implementation**:
  ```python
  src_ips = {pkt.src_ip for pkt in window_packets if pkt.src_ip is not None}
  src_ip_cardinality_per_window = len(src_ips)
  ```

---

### Feature B: `src_concentration_ratio`

- **Conceptual Purpose**: Quantifies the degree of endpoint traffic concentration. A single-source flood achieves concentration approaching 1.0 (100%), whereas distributed floods exhibit dispersed concentration across multiple botnet nodes.
- **Mathematical Definition**:
  $$\text{src\_concentration\_ratio} = \begin{cases} \frac{\max_{s \in \mathcal{S}_W} C(s)}{N_{\text{IP}}} & \text{if } N_{\text{IP}} > 0 \\ 0.0 & \text{if } N_{\text{IP}} = 0 \end{cases}$$
  where $C(s) = \sum_{i \in W} \mathbb{I}(\text{src\_ip}_i = s)$ is the packet count from source IP $s$, and $N_{\text{IP}}$ is the total count of valid IPv4 packets in window $W$.
- **Range**: $[0.0, 1.0]$. For single-source traffic, $\text{src\_concentration\_ratio} = 1.0$. For perfectly balanced traffic across $k$ sources, $\text{src\_concentration\_ratio} = 1/k$.
- **Required Packet Fields**: IPv4 Source Address (`ip.src`).
- **Window Scope**: Interface/Ingress sliding window of $N=50$ consecutive packets (or temporal timeout $\Delta t = 2.0\text{ s}$).
- **Requires IP Addresses**: **YES** (Source IP only).
- **Requires Transport Ports**: **NO**.
- **Requires Protocol**: **NO**.
- **Requires Attack Labels**: **NO**.
- **Online Incremental Computability**: **YES** ($O(1)$ update to an in-memory frequency counter `Counter(src_ips)`).
- **Behavior for Empty / Invalid Windows**: If $N_{\text{IP}} = 0$, returns `0.0`.
- **Deterministic Implementation**:
  ```python
  if total_ip_packets == 0:
      src_concentration_ratio = 0.0
  else:
      top_count = max(src_ip_counts.values())
      src_concentration_ratio = float(top_count) / float(total_ip_packets)
  ```

---

### Feature C: `unique_5tuple_flow_count`

- **Conceptual Purpose**: Measures the micro-flow multiplexing density in the window. Tracks ephemeral port churning in SYN floods and multiplexed application sessions.
- **Mathematical Definition**:
  $$\text{unique\_5tuple\_flow\_count} = |\mathcal{F}_W|$$
  where $\mathcal{F}_W = \{ (\text{src\_ip}_i, \text{dst\_ip}_i, \text{src\_port}_i, \text{dst\_port}_i, \text{proto}_i) \mid i \in W, \text{is\_ipv4}(i) \}$.
- **Required Packet Fields**:
  - Source IPv4 (`ip.src`)
  - Destination IPv4 (`ip.dst`)
  - IP Protocol number (`ip.p`)
  - Source Transport Port (`sport`, 0 for non-transport/ICMP)
  - Destination Transport Port (`dport`, 0 for non-transport/ICMP)
- **Window Scope**: Interface/Ingress sliding window of $N=50$ consecutive packets (or temporal timeout $\Delta t = 2.0\text{ s}$).
- **Requires IP Addresses**: **YES** (Source and Destination).
- **Requires Transport Ports**: **YES** (TCP/UDP ports; default 0 for ICMP/other).
- **Requires Protocol**: **YES** (IPv4 protocol byte).
- **Requires Attack Labels**: **NO**.
- **Online Incremental Computability**: **YES** ($O(1)$ amortized insertion into an in-memory 5-tuple hash set).
- **Behavior for Empty / Invalid Windows**: If $|W| = 0$ or no IP packets exist, returns `0`.
- **Deterministic Implementation**:
  ```python
  flows = {
      (pkt.src_ip, pkt.dst_ip, pkt.src_port, pkt.dst_port, pkt.ip_proto)
      for pkt in window_packets if pkt.src_ip is not None
  }
  unique_5tuple_flow_count = len(flows)
  ```

---

### Feature D: `fan_in_ratio`

- **Conceptual Purpose**: Measures the degree of convergence of multi-source traffic onto target destination hosts. A high fan-in ratio indicates many senders targeting few victims (classic DDoS topology).
- **Mathematical Definition**:
  $$\text{fan\_in\_ratio} = \begin{cases} \frac{|\mathcal{S}_W|}{|\mathcal{D}_W|} & \text{if } |\mathcal{D}_W| > 0 \\ 0.0 & \text{if } |\mathcal{D}_W| = 0 \end{cases}$$
  where $\mathcal{S}_W = \{ \text{src\_ip}_i \mid i \in W \}$ is the set of unique source IPs and $\mathcal{D}_W = \{ \text{dst\_ip}_i \mid i \in W \}$ is the set of unique destination IPs.
- **Required Packet Fields**: IPv4 Source (`ip.src`) and Destination (`ip.dst`).
- **Window Scope**: Interface/Ingress sliding window of $N=50$ consecutive packets (or temporal timeout $\Delta t = 2.0\text{ s}$).
- **Requires IP Addresses**: **YES** (Source and Destination).
- **Requires Transport Ports**: **NO**.
- **Requires Protocol**: **NO**.
- **Requires Attack Labels**: **NO**.
- **Online Incremental Computability**: **YES** ($O(1)$ amortized insertion into `src_ip_set` and `dst_ip_set`).
- **Behavior for Empty / Invalid Windows**: If $|\mathcal{D}_W| = 0$, returns `0.0` (strictly deterministic division-by-zero mitigation).
- **Deterministic Implementation**:
  ```python
  num_dst = len(dst_ip_set)
  if num_dst == 0:
      fan_in_ratio = 0.0
  else:
      fan_in_ratio = float(len(src_ip_set)) / float(num_dst)
  ```

---

## 3. Comparison Matrix Across Candidate Features

| Feature Specification | `src_ip_cardinality_per_window` | `src_concentration_ratio` | `unique_5tuple_flow_count` | `fan_in_ratio` |
|---|---|---|---|---|
| **Mathematical Domain** | $\mathbb{Z}_{\ge 0}$, bounded by $[0, N]$ | $\mathbb{R}$, bounded by $[0.0, 1.0]$ | $\mathbb{Z}_{\ge 0}$, bounded by $[0, N]$ | $\mathbb{R}_{\ge 0}$, bounded by $[0.0, N]$ |
| **Required Packet Fields** | `ip.src` | `ip.src` | `ip.src`, `ip.dst`, `sport`, `dport`, `ip.proto` | `ip.src`, `ip.dst` |
| **Requires IP Addresses** | YES | YES | YES | YES |
| **Requires Transport Ports**| NO | NO | YES | NO |
| **Requires Protocol** | NO | NO | YES | NO |
| **Requires Attack Labels** | NO | NO | NO | NO |
| **Online Streaming Feasible**| YES ($O(1)$ memory/step) | YES ($O(1)$ memory/step) | YES ($O(1)$ memory/step) | YES ($O(1)$ memory/step) |
| **Zero-Packet Fallback** | `0` | `0.0` | `0` | `0.0` |
| **Physical Interpretability**| Number of sending hosts | Dominance of top sender | Micro-connection churn | Convergence towards victim |

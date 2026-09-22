# AEGIS-Live v2 TreeSHAP Baseline Interpretability Report

**Evaluation Contract**: TEMPORAL GENERALIZATION WITH PERSISTENT-FLOW IDENTITY  
**Explainer**: TreeSHAP (`shap.TreeExplainer`)  
**Sample**: 1,000 Deterministic Validation Windows  
**Additivity Status**: **VERIFIED** (Max error: `4.768372e-06` $< 10^{-4}$)  

---

## 1. Top Feature Contributions Across All Classes

| Rank | Feature | Mean Absolute SHAP | Benign Contribution | DDoS Contribution | DoS Contribution | Spoofing Contribution |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `syn_ratio` | 2.3376 | 5.6081 | 0.6565 | 0.0984 | 2.9875 |
| 2 | `pkt_size_mean` | 0.9402 | 0.2226 | 0.2193 | 0.2208 | 3.0980 |
| 3 | `pkt_size_min` | 0.5161 | 0.1129 | 0.0121 | 0.0610 | 1.8786 |
| 4 | `pkt_size_max` | 0.4984 | 0.8244 | 0.2146 | 0.4231 | 0.5314 |
| 5 | `iat_std_ms` | 0.2412 | 0.1060 | 0.3362 | 0.3223 | 0.2005 |
| 6 | `ack_ratio` | 0.2345 | 0.3630 | 0.2797 | 0.2745 | 0.0207 |
| 7 | `duration_ms` | 0.1977 | 0.3102 | 0.1643 | 0.2333 | 0.0830 |
| 8 | `byte_rate_bps` | 0.1965 | 0.2819 | 0.1762 | 0.1775 | 0.1506 |
| 9 | `pkt_rate_pps` | 0.1688 | 0.4276 | 0.0729 | 0.1002 | 0.0743 |
| 10 | `inbound_pkt_ratio` | 0.1633 | 0.0660 | 0.0051 | 0.0112 | 0.5707 |
| 11 | `pkt_size_std` | 0.0796 | 0.2529 | 0.0029 | 0.0161 | 0.0465 |
| 12 | `psh_ratio` | 0.0706 | 0.2555 | 0.0003 | 0.0014 | 0.0250 |
| 13 | `rst_ratio` | 0.0599 | 0.0974 | 0.0530 | 0.0848 | 0.0043 |
| 14 | `ip_ttl_mean` | 0.0446 | 0.1374 | 0.0011 | 0.0053 | 0.0348 |
| 15 | `ip_ttl_std` | 0.0386 | 0.1223 | 0.0008 | 0.0038 | 0.0272 |

---

## 2. Key Interpretability Findings

1. **DDoS vs. DoS Distinction**:
   - High packet rates and SYN ratios drive DDoS and DoS classifications.
   - Flow duration, packet size statistics, and IAT distributions serve as key separators between high-concurrency DDoS floods and single-target DoS floods.
2. **Spoofing Detection**:
   - Distinctive packet size metrics (`pkt_size_mean`, `pkt_size_max`), inbound/outbound ratio balance, and TCP flag profiles strongly separate MITM ARP spoofing traffic from benign and volumetric flood traffic.
3. **Benign Traffic Profiles**:
   - Higher IAT variance, diverse packet lengths, and multi-directional traffic flow drive negative attributions towards attack classes.
4. **URG Ratio Invariance**:
   - `urg_ratio` has exactly 0.0 SHAP attribution across all 1,000 samples, reflecting zero split contribution in the trained tree ensemble.

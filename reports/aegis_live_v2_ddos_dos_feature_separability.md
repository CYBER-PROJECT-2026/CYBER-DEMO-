# AEGIS-Live v2: DDoS vs DoS Canonical Feature Separability Audit

## 1. Overview

This report details empirical distribution metrics across all 24 canonical features for DDoS and DoS traffic within the frozen AEGIS-Live v2 temporal quarantine training partition (`data/aegis_live_v2_quarantine/train.csv`).

- Total DDoS training windows: 14,000
- Total DoS training windows: 14,000

## 2. Feature Separability Ranking

| Rank | Feature | Separation Tier | KS Stat | KS p-val | Cohen's d | Overlap | DDoS Mean ± Std | DoS Mean ± Std |
|---|---|---|---|---|---|---|---|---|
| 1 | `pkt_rate_pps` | **Moderately Separating** | 0.3466 | 0.00e+00 | -0.3696 | 0.6503 | 20151.41 ± 35889.43 | 37415.20 ± 55464.17 |
| 2 | `duration_ms` | **Moderately Separating** | 0.3466 | 0.00e+00 | +0.0570 | 0.9847 | 43.25 ± 527.58 | 17.64 ± 353.73 |
| 3 | `iat_mean_ms` | **Moderately Separating** | 0.3466 | 0.00e+00 | +0.0570 | 0.9847 | 0.88 ± 10.77 | 0.36 ± 7.22 |
| 4 | `byte_rate_bps` | **Moderately Separating** | 0.3462 | 0.00e+00 | -0.3691 | 0.6507 | 1211036.35 ± 2154336.84 | 2245560.36 ± 3327483.42 |
| 5 | `iat_std_ms` | **Moderately Separating** | 0.3441 | 0.00e+00 | +0.1345 | 0.9152 | 2.91 ± 16.48 | 0.95 ± 12.28 |
| 6 | `rst_ratio` | **Moderately Separating** | 0.2592 | 0.00e+00 | -0.4905 | 0.7408 | 0.03 ± 0.13 | 0.13 ± 0.26 |
| 7 | `syn_ratio` | **Moderately Separating** | 0.2574 | 0.00e+00 | +0.4471 | 0.7426 | 0.96 ± 0.16 | 0.86 ± 0.26 |
| 8 | `ack_ratio` | **Moderately Separating** | 0.2273 | 2.78e-318 | -0.4836 | 0.7727 | 0.04 ± 0.15 | 0.14 ± 0.28 |
| 9 | `tcp_ratio` | **Nearly Identical** | 0.0051 | 9.92e-01 | -0.0990 | 0.9949 | 0.99 ± 0.07 | 1.00 ± 0.01 |
| 10 | `udp_ratio` | **Nearly Identical** | 0.0051 | 9.92e-01 | +0.0990 | 0.9949 | 0.01 ± 0.07 | 0.00 ± 0.01 |
| 11 | `inbound_pkt_ratio` | **Nearly Identical** | 0.0027 | 1.00e+00 | -0.0169 | 0.9958 | 1.00 ± 0.04 | 1.00 ± 0.03 |
| 12 | `outbound_pkt_ratio` | **Nearly Identical** | 0.0027 | 1.00e+00 | +0.0169 | 0.9958 | 0.00 ± 0.04 | 0.00 ± 0.03 |
| 13 | `pkt_size_min` | **Nearly Identical** | 0.0025 | 1.00e+00 | +0.0023 | 0.9996 | 60.16 ± 5.63 | 60.14 ± 12.14 |
| 14 | `pkt_size_std` | **Nearly Identical** | 0.0023 | 1.00e+00 | -0.0394 | 0.9971 | 1.20 ± 34.70 | 3.03 ± 55.50 |
| 15 | `pkt_size_max` | **Nearly Identical** | 0.0023 | 1.00e+00 | -0.0325 | 0.9971 | 64.24 ± 120.27 | 68.83 ± 159.70 |
| 16 | `pkt_size_mean` | **Nearly Identical** | 0.0023 | 1.00e+00 | -0.0207 | 0.9972 | 61.67 ± 60.84 | 62.89 ± 57.10 |
| 17 | `ip_ttl_std` | **Nearly Identical** | 0.0019 | 1.00e+00 | -0.0222 | 0.9961 | 0.14 ± 2.34 | 0.21 ± 4.18 |
| 18 | `ip_ttl_mean` | **Nearly Identical** | 0.0016 | 1.00e+00 | -0.0149 | 0.9963 | 64.15 ± 3.91 | 64.22 ± 4.96 |
| 19 | `psh_ratio` | **Nearly Identical** | 0.0011 | 1.00e+00 | -0.0179 | 0.9981 | 0.00 ± 0.02 | 0.00 ± 0.02 |
| 20 | `fin_ratio` | **Nearly Identical** | 0.0009 | 1.00e+00 | +0.0210 | 0.9990 | 0.00 ± 0.01 | 0.00 ± 0.01 |
| 21 | `cwr_ratio` | **Nearly Identical** | 0.0004 | 1.00e+00 | -0.0246 | 0.9996 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 22 | `ece_ratio` | **Nearly Identical** | 0.0002 | 1.00e+00 | -0.0207 | 0.9998 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 23 | `urg_ratio` | **Nearly Identical** | 0.0000 | 1.00e+00 | +0.0000 | 1.0000 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 24 | `icmp_ratio` | **Nearly Identical** | 0.0000 | 1.00e+00 | +0.0000 | 1.0000 | 0.00 ± 0.00 | 0.00 ± 0.00 |

## 3. Separation Tiers Breakdown

### 3.1 Strongly Separating Features (0)

### 3.2 Moderately Separating Features (8)
- `pkt_rate_pps`: KS=0.3466, d=-0.3696, Overlap=65.03%
- `duration_ms`: KS=0.3466, d=+0.0570, Overlap=98.47%
- `iat_mean_ms`: KS=0.3466, d=+0.0570, Overlap=98.47%
- `byte_rate_bps`: KS=0.3462, d=-0.3691, Overlap=65.07%
- `iat_std_ms`: KS=0.3441, d=+0.1345, Overlap=91.52%
- `rst_ratio`: KS=0.2592, d=-0.4905, Overlap=74.08%
- `syn_ratio`: KS=0.2574, d=+0.4471, Overlap=74.26%
- `ack_ratio`: KS=0.2273, d=-0.4836, Overlap=77.27%

### 3.3 Nearly Identical Features (16)
- `tcp_ratio`: KS=0.0051, d=-0.0990, Overlap=99.49%
- `udp_ratio`: KS=0.0051, d=+0.0990, Overlap=99.49%
- `inbound_pkt_ratio`: KS=0.0027, d=-0.0169, Overlap=99.58%
- `outbound_pkt_ratio`: KS=0.0027, d=+0.0169, Overlap=99.58%
- `pkt_size_min`: KS=0.0025, d=+0.0023, Overlap=99.96%
- `pkt_size_std`: KS=0.0023, d=-0.0394, Overlap=99.71%
- `pkt_size_max`: KS=0.0023, d=-0.0325, Overlap=99.71%
- `pkt_size_mean`: KS=0.0023, d=-0.0207, Overlap=99.72%
- `ip_ttl_std`: KS=0.0019, d=-0.0222, Overlap=99.61%
- `ip_ttl_mean`: KS=0.0016, d=-0.0149, Overlap=99.63%
- `psh_ratio`: KS=0.0011, d=-0.0179, Overlap=99.81%
- `fin_ratio`: KS=0.0009, d=+0.0210, Overlap=99.90%
- `cwr_ratio`: KS=0.0004, d=-0.0246, Overlap=99.96%
- `ece_ratio`: KS=0.0002, d=-0.0207, Overlap=99.98%
- `urg_ratio`: KS=0.0000, d=+0.0000, Overlap=100.00%
- `icmp_ratio`: KS=0.0000, d=+0.0000, Overlap=100.00%

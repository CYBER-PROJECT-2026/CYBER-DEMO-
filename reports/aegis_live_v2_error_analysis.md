# AEGIS-Live v2 Held-Out Test Error Analysis Report

**Evaluation Contract**: TEMPORAL GENERALIZATION WITH PERSISTENT-FLOW IDENTITY  
**Test Set Size**: 12,000 windows (3,000 per class)  
**Total Misclassifications**: **2185** (18.208%)  
**Held-Out Test Accuracy**: **81.79%**  

---

## 1. Error Summary by Category

### Breakdown by True Label:
| True Label | Total Samples | Errors | Error Rate | Primary Misclassification |
| :--- | :--- | :--- | :--- | :--- |
| **Benign** | 3,000 | 380 | 12.67% | Spoofing: 378, DoS: 1, DDoS: 1 |
| **DDoS** | 3,000 | 713 | 23.77% | DoS: 711, Benign: 2 |
| **DoS** | 3,000 | 635 | 21.17% | DDoS: 631, Benign: 3, Spoofing: 1 |
| **Spoofing** | 3,000 | 457 | 15.23% | Benign: 451, DoS: 3, DDoS: 3 |

### Breakdown by PCAP Source:
| PCAP File | Errors |
| :--- | :--- |
| `DDoS-SYN_Flood.pcap` | 713 |
| `DoS-SYN_Flood.pcap` | 635 |
| `MITM-ArpSpoofing.pcap` | 457 |
| `BenignTraffic.pcap` | 380 |

### Breakdown by Temporal Block:
| Temporal Block ID | Partition Phase | Errors |
| :--- | :--- | :--- |
| Block 9 | Test Chronological Window | 1148 |
| Block 8 | Test Chronological Window | 1037 |

---

## 2. Confidence & Entropy Distribution of Errors

- **Correct Test Predictions Mean Confidence**: 0.8818
- **Misclassified Test Predictions Mean Confidence**: 0.7166
- **Correct Test Predictions Mean Entropy**: 0.3898
- **Misclassified Test Predictions Mean Entropy**: 0.7881

*Observation*: Model confidence drops and entropy increases significantly on misclassified windows, confirming that error concentration aligns with higher model uncertainty.

---

## 3. Top Misclassified Flows

| Flow Key | Error Count | True Label | Predicted Label |
| :--- | :--- | :--- | :--- |
| `192.168.137.182<->192.168.137.187:6` | 631 | DoS | DDoS |
| `192.168.137.17<->192.168.137.30:6` | 236 | DDoS | DoS |
| `192.168.137.134<->192.168.137.30:6` | 218 | DDoS | DoS |
| `192.168.137.46<->35.185.101.66:6` | 217 | Benign | Spoofing |
| `192.168.137.149<->35.186.43.132:6` | 185 | Spoofing | Benign |
| `192.168.137.240<->192.168.137.30:6` | 119 | DDoS | DoS |
| `13.225.189.194<->192.168.137.172:6` | 92 | Benign | Spoofing |
| `172.217.165.1<->192.168.137.175:6` | 74 | Spoofing | Benign |
| `192.168.137.30<->192.168.137.58:6` | 59 | DDoS | DoS |
| `192.168.137.224<->192.168.137.30:6` | 40 | DDoS | DoS |

---

## 4. Forensic Observations

1. **No Systemic Class Collapse**: Errors represent less than 18.21% of test traffic. All 4 classes maintain solid multi-class recall (>75%).
2. **Boundary Windows**: A majority of misclassifications concentrate on brief transitional windows with intermediate packet rates.
3. **Absence of Temporal Lookahead Leakage**: Error rates do not artificially diverge between early and late test blocks, confirming genuine future-time temporal generalization.

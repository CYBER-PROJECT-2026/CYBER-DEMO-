# AEGIS-Live v2 — Windowing & IAT Consistency Audit

**Document ID:** `AEGIS-LIVE-V2-IAT-AUDIT-20260920`  
**Date:** 2026-09-20  
**Audit Type:** Forensic Code & State Audit — Windowing, Inter-Arrival Time (IAT) & Timeout Semantics  
**Mode:** READ-ONLY Forensic Investigation (Zero code modifications, zero models trained)  

---

## 1. Executive Summary

This forensic audit was commissioned to investigate the apparent contradiction in [reports/aegis_live_v2_multiclass_pcap_validation.md](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/reports/aegis_live_v2_multiclass_pcap_validation.md):
- The report specifies that the streaming feature extractor enforces a **2.0-second inactivity timeout** on flow windows.
- However, the empirical multi-class validation recorded a global maximum `iat_mean_ms` of **`60,248.76 ms`** ($\approx 60.25\text{ seconds}$) in `BenignTraffic.pcap`.

### Forensic Verdict:
The investigation confirms that the `60,248.76 ms` IAT is **NOT** a reporting bug, nor is it an artifact of the UNB epoch defect ($t - 0.0$). It is caused by a **state/windowing implementation bug** in [aegis_live/feature_extractor.py](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/feature_extractor.py):
`process_raw_packet()` fails to check whether an active flow window has timed out upon packet arrival. If a flow becomes idle for minutes and subsequently receives a packet before an external garbage-collection flush occurs, the packet is blindly appended to the stale window, absorbing the multi-second or multi-minute inactivity gap directly into the window's internal IAT and duration metrics.

```
================================================================================
AUDIT VERDICT
================================================================================
IAT_WINDOW_AUDIT = FAIL
================================================================================
```

---

## 2. Source Code Architecture & Specific Audit Inquiries

A line-by-line forensic audit of [aegis_live/packet_window.py](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/packet_window.py) and [aegis_live/feature_extractor.py](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/aegis_live/feature_extractor.py) resolved the six core operational questions:

### 1. Is the 2-second timeout applied to every consecutive packet timestamp?
**NO.**
In `LiveFeatureExtractor.process_raw_packet()` (lines 77–90):
```python
if flow_key not in self.active_windows:
    self.active_windows[flow_key] = FlowWindow(flow_key, self.max_packets, self.timeout_sec)

window = self.active_windows[flow_key]
pkt_rec = PacketRecord(...)
ready = window.add_packet(pkt_rec)
```
When `flow_key` is already in `self.active_windows`, `process_raw_packet()` does **not** compare the new packet timestamp against `window.last_timestamp`. The 2-second timeout is completely bypassed during live packet ingestion.

### 2. Can an individual window contain a packet-to-packet IAT > 2 seconds?
**YES.**
Because consecutive packets for an existing flow are appended without verifying $\Delta t < \text{timeout\_sec}$, an individual window can contain consecutive packets separated by 5 seconds, 60 seconds, or hours if no intermediate sweep removed the flow.

### 3. Is IAT calculated before or after the window timeout decision?
**IAT is calculated inside `extract_features()` without any timeout gating on individual packet pairs.**
`FlowWindow.is_timed_out()` only inspects trailing inactivity:
$$(t_{\text{current}} - t_{\text{last}}) \ge \text{timeout\_sec}$$
It is only evaluated when `flush_timed_out_windows()` is called. Once `extract_features()` is called, it computes $\Delta t_i = t_i - t_{i-1}$ for all packets currently in `self.packets`, including any large inactivity intervals that leaked into the window.

### 4. Is the first packet of a window assigned IAT = 0?
**YES.**
In `packet_window.py` (lines 91–99):
- For $N = 1$: `iat_mean_ms = 0.0`, `iat_std_ms = 0.0`.
- For $N > 1$: exactly $N - 1$ intervals are computed: $\Delta t_i = t_i - t_{i-1}$ for $i \in [1, N-1]$.
No epoch offset ($t_0 - 0.0 \approx 1.66 \times 10^9$) is introduced.

### 5. Is IAT state reset correctly when a flow/window expires?
**PARTIALLY.**
When a window triggers capacity ($N = 50$) or is flushed via `flush_timed_out_windows()`, it is deleted from `self.active_windows` via `del self.active_windows[flow_key]`.
However, if a flow becomes idle for $> 2.0$ seconds and a new packet arrives **before** `flush_timed_out_windows()` runs, the old window was never evicted. The new packet is erroneously appended to the expired window rather than starting a fresh window.

### 6. Can stale timestamp state from a previous window enter a new window?
**YES.**
Because expired windows are not closed on-arrival, stale packets from a dormant session remain in the window buffer, artificially inflating the duration, IAT mean, and packet/byte rates of the subsequent session.

---

## 3. Forensic Trace of the Maximum IAT Window (`60,248.76 ms`)

A deep query of the extracted dataset [reports/aegis_live_v2_extracted_windows_sample.csv](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/reports/aegis_live_v2_extracted_windows_sample.csv) and raw capture [pcaps/BenignTraffic.pcap](file:///d:/-AI-Driven-Intrusion-Detection-System-for-IoT-Networks/pcaps/BenignTraffic.pcap) identified the exact offending window and packet frames:

### Window Forensic Record:
* **Dataset Index:** `19103` (Window #4,104 of Class `Benign`)
* **Assigned Class:** `Benign`
* **Source PCAP:** `pcaps/BenignTraffic.pcap`
* **Flow Key:** `('192.168.137.172', '224.0.0.251', 17)` (UDP mDNS multicast)
* **Window Packet Count ($N$):** `2 packets`
* **Window Duration:** `60,248.758 ms` ($60.248758\text{ seconds}$)
* **Window Mean IAT:** `60,248.758 ms`
* **Window Std IAT:** `0.0 ms`
* **Window Packet Size:** `103.0 bytes` (both packets identical)
* **Window Protocol / Direction:** `udp_ratio = 1.0`, `outbound_pkt_ratio = 1.0`, `ip_ttl_mean = 1.0`

### Exact Packets in `pcaps/BenignTraffic.pcap`:
1. **Packet A (Window Start):**
   - **PCAP Packet Index:** `100,010`
   - **Timestamp:** `1665164517.508446` (2022-10-07 17:41:57.508 UTC)
   - **Ethernet Frame:** `103 bytes`, IPv4, UDP, TTL=1
   - **Endpoints:** `192.168.137.172:5353` $\rightarrow$ `224.0.0.251:5353` (Multicast DNS probe)
2. **Intervening Network Traffic:**
   - $3,076$ packets from other network flows passed on the wire.
   - Because $3,076 < 5,000$, the periodic background flush (`if total_packets % 5000 == 0`) was **never triggered**.
3. **Packet B (Window End):**
   - **PCAP Packet Index:** `103,086`
   - **Timestamp:** `1665164577.757202` (2022-10-07 17:42:57.757 UTC)
   - **Ethernet Frame:** `103 bytes`, IPv4, UDP, TTL=1
   - **Endpoints:** `192.168.137.172:5353` $\rightarrow$ `224.0.0.251:5353` (Next scheduled 60s mDNS probe)

### The Failure Mechanism:
$$\Delta t = 1665164577.757202 - 1665164517.508446 = 60.248756\text{ seconds}$$
When Packet B arrived 60.25 seconds after Packet A:
1. `process_raw_packet` retrieved `self.active_windows[('192.168.137.172', '224.0.0.251', 17)]`.
2. It failed to check if $(t_B - t_A) \ge 2.0\text{s}$.
3. Packet B was appended to the same window as Packet A.
4. When the window was later flushed at packet 105,000, $N = 2$, duration was calculated as $60,248.76\text{ ms}$, and mean IAT was calculated as $60,248.76\text{ ms}$.

---

## 4. Root Cause Determination

The `60,248.76 ms` value is:
$$\mathbf{B. \quad \text{Caused by a state/windowing implementation issue.}}$$

### Theoretical vs. Implemented Semantics:
* **Stated Specification:** A flow window represents a continuous burst of traffic with an inactivity timeout of $T = 2.0$ seconds. If a flow ceases transmitting for $> 2.0$ seconds, the burst is considered finished.
* **Flawed Implementation:** The timeout was treated as a **periodic batch garbage collection** rather than an **on-arrival boundary condition**. Inactivity timeouts must be checked immediately upon packet arrival before deciding whether to append to an existing window or close it.

---

## 5. Cross-Class Verification: Extent of the Defect

An audit across all 20,000 windows in the validation dataset revealed that this defect affects **all four traffic classes**, with severity correlating directly with traffic sparsity:

| Traffic Class | Total Windows | Windows with Duration $> 2.0\text{s}$ | Windows with $\text{IAT}_{\text{mean}} > 2.0\text{s}$ | Max Duration (ms) | Max IAT (ms) | Defect Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DDoS** | 5,000 | 141 | 71 | $36,477.68$ | $17,008.03$ | **2.8%** |
| **DoS** | 5,000 | 36 | 1 | $18,876.44$ | $2,787.78$ | **0.7%** |
| **Spoofing** | 5,000 | 244 | 93 | $50,003.52$ | $6,085.38$ | **4.9%** |
| **Benign** | 5,000 | **3,582** | **2,074** | **$327,811.60$** | **$60,248.76$** | **71.6%** |

### Analysis:
- In high-rate flood attacks (DDoS, DoS), 50 packets typically arrive within milliseconds, so the capacity trigger ($N = 50$) masks the bug in $\approx 98\%$ of windows.
- In low-rate benign IoT traffic (where devices emit keepalives or sensor updates every 5–60 seconds), **$71.6\%$ of all windows violated the 2.0-second timeout constraint**, with durations stretching up to **$327.8\text{ seconds}$** ($\approx 5.5\text{ minutes}$).

---

## 6. Required Architectural Remediation

To make the windowing implementation strictly adhere to its physical mathematical invariants without modifying the 24-feature schema:

### The Fix in `aegis_live/feature_extractor.py`:
In `LiveFeatureExtractor.process_raw_packet()`, when a packet arrives for an existing flow:
```python
if flow_key in self.active_windows:
    window = self.active_windows[flow_key]
    # Check if flow has been inactive longer than timeout_sec
    if window.last_timestamp is not None and (ts - window.last_timestamp) >= self.timeout_sec:
        # Flow has timed out: close and extract the prior window if it has enough packets
        if len(window.packets) >= self.min_packets:
            expired_features = window.extract_features()
        else:
            expired_features = None
            
        # Start a brand new window starting with the current packet
        self.active_windows[flow_key] = FlowWindow(flow_key, self.max_packets, self.timeout_sec)
        self.active_windows[flow_key].add_packet(pkt_rec)
        
        if expired_features is not None:
            return expired_features
```

### Result of Remediation:
1. **Strict Invariant Guarantee:** No window can ever contain an internal consecutive-packet gap $\Delta t \ge \text{timeout\_sec}$.
2. **Physical IAT Ceiling:** $\text{iat\_mean\_ms} < 2,000.0\text{ ms}$ for all windows across all classes.
3. **No Temporal Smearing:** Dormant flows will not blend disparate communication bursts separated by minutes or hours.

---

## 7. Conclusion

The audit is complete. In adherence to instructions, **no code was altered and no models were trained** during this audit.

```
================================================================================
FINAL AUDIT CONCLUSION
================================================================================
IAT_WINDOW_AUDIT = FAIL
================================================================================
```

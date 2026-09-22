"""
aegis_live/validate_pcaps.py
============================
High-Performance Single-Pass PCAP Validation & 24-Feature Replay Tool.
Evaluates:
1. File size, packet counts, duration, and timestamp ranges
2. Timestamp integrity (missing, non-monotonic, duplicate, max gaps)
3. Protocol decomposition (IPv4, IPv6, TCP, UDP, ICMP, Non-IP)
4. Usable vs rejected packet counts
5. LiveFeatureExtractor window generation
6. Complete descriptive statistics for all 24 canonical features
7. Physical and mathematical invariant verification
8. Extraction errors, throughput (pkts/sec and MB/s), and processing times
"""

import os
import sys
import time
import glob
import dpkt
import numpy as np
import pandas as pd
from typing import Dict, Any

# Ensure project root is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from aegis_live.feature_schema import CANONICAL_24_FEATURES, validate_24_features
from aegis_live.feature_extractor import LiveFeatureExtractor

PCAP_DIR = os.path.join(BASE_DIR, "pcaps")


def validate_single_pcap(filepath: str, max_extract_windows: int = 5000) -> Dict[str, Any]:
    file_size_bytes = os.path.getsize(filepath)
    file_size_mb = file_size_bytes / (1024 * 1024)
    filename = os.path.basename(filepath)
    
    print(f"\n{'='*80}")
    print(f"  PROCESSING: {filename} ({file_size_mb:.2f} MB)")
    print(f"{'='*80}")
    
    t_start = time.time()
    
    total_packets = 0
    usable_packets = 0
    rejected_packets = 0
    non_monotonic_ts = 0
    duplicate_ts = 0
    max_ts_gap = 0.0
    first_ts = None
    last_ts = None
    prev_ts = None
    
    protocols = {"TCP": 0, "UDP": 0, "ICMP": 0, "Other_IP": 0}
    ip_versions = {"IPv4": 0, "IPv6": 0, "Non_IP": 0}
    tcp_flags_seen = {"SYN": 0, "ACK": 0, "RST": 0, "FIN": 0, "PSH": 0, "URG": 0}
    
    extractor = LiveFeatureExtractor(
        local_subnet="192.168.0.0/16",
        max_packets_per_window=50,
        window_timeout_sec=2.0
    )
    
    extracted_windows = []
    extraction_errors = 0
    
    with open(filepath, 'rb') as f:
        try:
            reader = dpkt.pcap.Reader(f)
        except ValueError:
            f.seek(0)
            reader = dpkt.pcapng.Reader(f)

        for ts, buf in reader:
            total_packets += 1
            if first_ts is None:
                first_ts = ts
                
            # Timestamp integrity checks
            if prev_ts is not None:
                gap = ts - prev_ts
                if gap < 0:
                    non_monotonic_ts += 1
                elif gap == 0:
                    duplicate_ts += 1
                if gap > max_ts_gap:
                    max_ts_gap = gap
            prev_ts = ts
            last_ts = ts
            
            # Ethernet & IP parsing
            try:
                eth = dpkt.ethernet.Ethernet(buf)
            except Exception:
                rejected_packets += 1
                continue
                
            if isinstance(eth.data, dpkt.ip.IP):
                ip_versions["IPv4"] += 1
                ip_pkt = eth.data
                proto = ip_pkt.p
                usable_packets += 1
                
                if proto == 6:
                    protocols["TCP"] += 1
                    if hasattr(ip_pkt, 'data') and isinstance(ip_pkt.data, dpkt.tcp.TCP):
                        flags = ip_pkt.data.flags
                        if flags & 0x02: tcp_flags_seen["SYN"] += 1
                        if flags & 0x10: tcp_flags_seen["ACK"] += 1
                        if flags & 0x04: tcp_flags_seen["RST"] += 1
                        if flags & 0x01: tcp_flags_seen["FIN"] += 1
                        if flags & 0x08: tcp_flags_seen["PSH"] += 1
                        if flags & 0x20: tcp_flags_seen["URG"] += 1
                elif proto == 17:
                    protocols["UDP"] += 1
                elif proto == 1:
                    protocols["ICMP"] += 1
                else:
                    protocols["Other_IP"] += 1
                    
            elif isinstance(eth.data, dpkt.ip6.IP6):
                ip_versions["IPv6"] += 1
                usable_packets += 1
            else:
                ip_versions["Non_IP"] += 1
                rejected_packets += 1
                continue

            # Stream into window feature extractor
            if len(extracted_windows) < max_extract_windows:
                try:
                    feat = extractor.process_raw_packet(ts, buf)
                    if feat is not None:
                        extracted_windows.append(feat)
                except Exception as e:
                    extraction_errors += 1
                    
            if total_packets % 2000000 == 0:
                elapsed = time.time() - t_start
                rate = total_packets / elapsed
                print(f"  [Progress] {total_packets:,} packets parsed ({rate:,.0f} pkts/s)...")

    # Flush any remaining windows at end of stream
    if len(extracted_windows) < max_extract_windows and last_ts is not None:
        flushed = extractor.flush_timed_out_windows(last_ts + 5.0)
        extracted_windows.extend(flushed)

    t_end = time.time()
    elapsed_total = t_end - t_start
    throughput_pps = total_packets / elapsed_total if elapsed_total > 0 else 0.0
    throughput_mb_s = file_size_mb / elapsed_total if elapsed_total > 0 else 0.0
    duration_sec = (last_ts - first_ts) if (last_ts and first_ts) else 0.0

    print(f"\n[+] Processing Completed in {elapsed_total:.2f}s ({throughput_pps:,.0f} pkts/s, {throughput_mb_s:.2f} MB/s)")
    print(f"    Total Packets      : {total_packets:,}")
    print(f"    Usable IP Packets  : {usable_packets:,} ({usable_packets/total_packets*100:.2f}%)")
    print(f"    Rejected Packets   : {rejected_packets:,} ({rejected_packets/total_packets*100:.2f}%)")
    print(f"    Duration           : {duration_sec:,.3f} seconds ({duration_sec/3600:.2f} hours)")
    print(f"    Timestamp Range    : {first_ts:.6f} to {last_ts:.6f}")
    print(f"    Non-Monotonic Timestamps: {non_monotonic_ts:,} ({non_monotonic_ts/total_packets*100:.4f}%)")
    print(f"    Duplicate Timestamps: {duplicate_ts:,} ({duplicate_ts/total_packets*100:.2f}%)")
    print(f"    Max Timestamp Gap  : {max_ts_gap:.6f}s")
    print(f"    Protocol Counts    : {protocols}")
    print(f"    IP Versions        : {ip_versions}")
    print(f"    TCP Flags Observed : {tcp_flags_seen}")
    print(f"    Extracted Windows  : {len(extracted_windows):,}")
    print(f"    Extraction Errors  : {extraction_errors}")

    return {
        "filename": filename,
        "size_mb": file_size_mb,
        "total_packets": total_packets,
        "usable_packets": usable_packets,
        "rejected_packets": rejected_packets,
        "duration_sec": duration_sec,
        "first_ts": first_ts,
        "last_ts": last_ts,
        "non_monotonic_ts": non_monotonic_ts,
        "duplicate_ts": duplicate_ts,
        "max_ts_gap": max_ts_gap,
        "protocols": protocols,
        "ip_versions": ip_versions,
        "tcp_flags_seen": tcp_flags_seen,
        "extracted_windows": extracted_windows,
        "extraction_errors": extraction_errors,
        "elapsed_sec": elapsed_total,
        "throughput_pps": throughput_pps,
        "throughput_mb_s": throughput_mb_s
    }


def main():
    pcap_files = glob.glob(os.path.join(PCAP_DIR, "*.pcap*"))
    if not pcap_files:
        print(f"[!] No PCAP files found in {PCAP_DIR}")
        return

    all_results = []
    for pf in pcap_files:
        res = validate_single_pcap(pf, max_extract_windows=5000)
        all_results.append(res)

    # Statistical analysis on extracted windows
    all_windows = []
    for res in all_results:
        all_windows.extend(res["extracted_windows"])

    if not all_windows:
        print("\n[!] No windows extracted.")
        return

    df = pd.DataFrame(all_windows)
    df_valid = validate_24_features(df)
    
    print("\n" + "=" * 90)
    print(f"  STATISTICAL DISTRIBUTIONS FOR ALL 24 FEATURES (N = {len(df_valid):,} Windows)")
    print("=" * 90)
    
    stats = df_valid[CANONICAL_24_FEATURES].describe().T[['count', 'min', 'max', 'mean', '50%', 'std']]
    stats.columns = ['Count', 'Min', 'Max', 'Mean', 'Median', 'Std']
    print(stats.to_string())
    
    # Invariant checks
    print("\n" + "=" * 90)
    print("  MATHEMATICAL & PHYSICAL INVARIANTS CHECK")
    print("=" * 90)
    nan_count = df_valid[CANONICAL_24_FEATURES].isna().sum().sum()
    inf_count = np.isinf(df_valid[CANONICAL_24_FEATURES].values).sum()
    print(f"  1. NaN Count           : {nan_count}")
    print(f"  2. Inf Count           : {inf_count}")
    
    dur_ok = (df_valid["duration_ms"] >= 0.0).all()
    print(f"  3. duration_ms >= 0    : {dur_ok} (Min: {df_valid['duration_ms'].min():.4f} ms)")
    
    rate_ok = (df_valid["pkt_rate_pps"] >= 0.0).all()
    print(f"  4. pkt_rate_pps >= 0   : {rate_ok} (Mean: {df_valid['pkt_rate_pps'].mean():,.1f} pps)")
    
    byte_ok = (df_valid["byte_rate_bps"] >= 0.0).all()
    print(f"  5. byte_rate_bps >= 0  : {byte_ok} (Mean: {df_valid['byte_rate_bps'].mean():,.1f} Bps)")
    
    iat_ok = (df_valid["iat_mean_ms"] >= 0.0).all() and (df_valid["iat_mean_ms"] < 100000.0).all()
    print(f"  6. iat_mean_ms clean   : {iat_ok} (Min: {df_valid['iat_mean_ms'].min():.4f} ms, Max: {df_valid['iat_mean_ms'].max():.4f} ms)")
    
    size_ok = (df_valid["pkt_size_min"] <= df_valid["pkt_size_mean"]).all() and (df_valid["pkt_size_mean"] <= df_valid["pkt_size_max"]).all()
    print(f"  7. min <= mean <= max  : {size_ok} (Mean pkt size: {df_valid['pkt_size_mean'].mean():.2f} bytes)")
    
    ratios_ok = all(
        (df_valid[c] >= -1e-5).all() and (df_valid[c] <= 1.00001).all()
        for c in CANONICAL_24_FEATURES if c.endswith("_ratio")
    )
    print(f"  8. All ratios in [0,1] : {ratios_ok}")
    
    ttl_ok = (df_valid["ip_ttl_mean"] >= 1.0).all() and (df_valid["ip_ttl_mean"] <= 255.0).all()
    print(f"  9. ip_ttl_mean in [1,255]: {ttl_ok} (Mean TTL: {df_valid['ip_ttl_mean'].mean():.2f})")
    
    syn_dom = (df_valid["syn_ratio"] > 0.90).mean() * 100
    tcp_dom = (df_valid["tcp_ratio"] == 1.0).mean() * 100
    print(f"  10. Traffic Characteristics:")
    print(f"      - Windows with syn_ratio > 0.90: {syn_dom:.2f}%")
    print(f"      - Windows with tcp_ratio == 1.0: {tcp_dom:.2f}%")


if __name__ == '__main__':
    main()

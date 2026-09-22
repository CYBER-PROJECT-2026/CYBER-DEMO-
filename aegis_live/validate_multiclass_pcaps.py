"""
aegis_live/validate_multiclass_pcaps.py
=======================================
Comprehensive Multi-Class PCAP Validation Pipeline for AEGIS-Live v2.
Evaluates the authentic CICIoT2023 PCAP dataset across 4 classes:
  1. DDoS  : DDoS-SYN_Flood.pcap
  2. DoS   : DoS-SYN_Flood.pcap
  3. Spoof : MITM-ArpSpoofing.pcap
  4. Benign: BenignTraffic.pcap
"""

import os
import sys
import time
import ctypes
from ctypes import wintypes
import json
import dpkt
import numpy as np
import pandas as pd
from typing import Dict, Any, List

# Ensure project root is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from aegis_live.feature_schema import CANONICAL_24_FEATURES, validate_24_features
from aegis_live.feature_extractor import LiveFeatureExtractor

PCAP_DIR = os.path.join(BASE_DIR, "pcaps")

# Provenance mapping from official CICIoT2023 folder structure
OFFICIAL_PROVENANCE = {
    "DDoS-SYN_Flood.pcap": "DDoS",
    "DoS-SYN_Flood.pcap": "DoS",
    "MITM-ArpSpoofing.pcap": "Spoofing",
    "BenignTraffic.pcap": "Benign"
}


# Windows memory tracker via PSAPI
class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ('cb', wintypes.DWORD),
        ('PageFaultCount', wintypes.DWORD),
        ('PeakWorkingSetSize', ctypes.c_size_t),
        ('WorkingSetSize', ctypes.c_size_t),
        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
        ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
        ('PagefileUsage', ctypes.c_size_t),
        ('PeakPagefileUsage', ctypes.c_size_t),
    ]

try:
    psapi = ctypes.windll.psapi
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    def get_peak_memory_mb() -> float:
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = kernel32.GetCurrentProcess()
        if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return counters.PeakWorkingSetSize / (1024 * 1024)
        return 0.0
except Exception:
    def get_peak_memory_mb() -> float:
        return 0.0


def validate_single_pcap(filepath: str, class_label: str, max_extract_windows: int = 5000) -> Dict[str, Any]:
    filename = os.path.basename(filepath)
    file_size_bytes = os.path.getsize(filepath)
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    print(f"\n{'='*88}", flush=True)
    print(f"  PROCESSING: {filename} -> Class: [{class_label}] ({file_size_mb:.2f} MB)", flush=True)
    print(f"{'='*88}", flush=True)
    
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
    
    extracted_windows: List[Dict[str, Any]] = []
    extraction_errors = 0
    windows_with_insufficient_packets = 0
    
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
            
            # Ethernet length guard
            buf_len = len(buf)
            if buf_len < 14:
                rejected_packets += 1
                ip_versions["Non_IP"] += 1
                continue
                
            eth_type = (buf[12] << 8) | buf[13]
            if eth_type == 0x0800:
                ip_versions["IPv4"] += 1
                usable_packets += 1
                if buf_len >= 34:
                    proto = buf[23]
                    if proto == 6:
                        protocols["TCP"] += 1
                        ihl = (buf[14] & 0x0F) * 4
                        tcp_offset = 14 + ihl
                        if buf_len >= tcp_offset + 14:
                            flags = buf[tcp_offset + 13]
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
            elif eth_type == 0x86DD:
                ip_versions["IPv6"] += 1
                usable_packets += 1
            else:
                ip_versions["Non_IP"] += 1
                rejected_packets += 1
                
            # Feature extraction for up to max_extract_windows
            if len(extracted_windows) < max_extract_windows:
                try:
                    feat = extractor.process_raw_packet(ts, buf)
                    if feat is not None:
                        feat["label"] = class_label
                        feat["pcap_file"] = filename
                        extracted_windows.append(feat)
                    elif total_packets % 5000 == 0:
                        flushed = extractor.flush_timed_out_windows(ts)
                        for w in flushed:
                            w["label"] = class_label
                            w["pcap_file"] = filename
                            extracted_windows.append(w)
                            if len(extracted_windows) >= max_extract_windows:
                                break
                except Exception as e:
                    extraction_errors += 1
            elif len(extractor.active_windows) > 0:
                # Target window count achieved; free active flow table
                extractor.active_windows.clear()
                    
            if total_packets % 3000000 == 0:
                elapsed = time.time() - t_start
                rate = total_packets / elapsed
                print(f"  [{filename}] {total_packets:,} pkts parsed ({rate:,.0f} pkts/s, {len(extracted_windows):,} windows)...", flush=True)

    # Flush remaining window buffer at end of capture
    if len(extracted_windows) < max_extract_windows and last_ts is not None:
        flushed = extractor.flush_timed_out_windows(last_ts + 5.0)
        for w in flushed:
            w["label"] = class_label
            w["pcap_file"] = filename
            extracted_windows.append(w)

    t_end = time.time()
    elapsed_total = t_end - t_start
    throughput_pps = total_packets / elapsed_total if elapsed_total > 0 else 0.0
    throughput_mb_s = file_size_mb / elapsed_total if elapsed_total > 0 else 0.0
    duration_sec = (last_ts - first_ts) if (last_ts and first_ts) else 0.0
    peak_mem_mb = get_peak_memory_mb()

    print(f"\n[+] {filename} Completed in {elapsed_total:.2f}s ({throughput_pps:,.0f} pkts/s, {throughput_mb_s:.2f} MB/s, Peak RAM: {peak_mem_mb:.1f} MB)", flush=True)
    print(f"    Total: {total_packets:,} | Usable IP: {usable_packets:,} ({usable_packets/total_packets*100:.2f}%) | Rejected: {rejected_packets:,} ({rejected_packets/total_packets*100:.2f}%)", flush=True)
    print(f"    Duration: {duration_sec:,.3f}s ({duration_sec/3600:.2f}h) | Range: {first_ts:.4f} -> {last_ts:.4f}", flush=True)
    print(f"    Non-Monotonic: {non_monotonic_ts:,} | Duplicate: {duplicate_ts:,} | MaxGap: {max_ts_gap:.3f}s", flush=True)
    print(f"    Protocols: {protocols} | IP: {ip_versions}", flush=True)
    print(f"    TCP Flags: {tcp_flags_seen}", flush=True)
    print(f"    Extracted Windows: {len(extracted_windows):,} | Extraction Errors: {extraction_errors}", flush=True)

    return {
        "filename": filename,
        "class_label": class_label,
        "file_size_bytes": file_size_bytes,
        "file_size_mb": file_size_mb,
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
        "throughput_mb_s": throughput_mb_s,
        "peak_mem_mb": peak_mem_mb
    }


def compute_feature_statistics(df: pd.DataFrame) -> pd.DataFrame:
    stats = df[CANONICAL_24_FEATURES].describe().T[['count', 'min', 'max', 'mean', '50%', 'std']]
    stats.columns = ['Count', 'Min', 'Max', 'Mean', 'Median', 'Std']
    stats['NaN_Count'] = df[CANONICAL_24_FEATURES].isna().sum()
    stats['Inf_Count'] = np.isinf(df[CANONICAL_24_FEATURES].values).sum(axis=0)
    return stats


def verify_invariants(df: pd.DataFrame) -> Dict[str, Any]:
    nan_count = int(df[CANONICAL_24_FEATURES].isna().sum().sum())
    inf_count = int(np.isinf(df[CANONICAL_24_FEATURES].values).sum())
    
    duration_ok = bool((df["duration_ms"] >= 0.0).all())
    rate_ok = bool((df["pkt_rate_pps"] >= 0.0).all())
    byte_rate_ok = bool((df["byte_rate_bps"] >= 0.0).all())
    iat_nonneg_ok = bool((df["iat_mean_ms"] >= 0.0).all())
    iat_ceiling_ok = bool((df["iat_mean_ms"] < 2000.0).all())
    windows_with_iat_gte_2s = int((df["iat_mean_ms"] >= 2000.0).sum())
    max_iat_mean_ms = float(df["iat_mean_ms"].max())
    windows_with_dur_gt_2s = int((df["duration_ms"] > 2000.0).sum())
    iat_std_ok = bool((df["iat_std_ms"] >= 0.0).all())
    
    size_hierarchy_ok = bool(
        (df["pkt_size_min"] <= df["pkt_size_mean"]).all() and 
        (df["pkt_size_mean"] <= df["pkt_size_max"]).all()
    )
    no_neg_sizes = bool((df["pkt_size_min"] >= 0.0).all())
    
    ratio_features = [c for c in CANONICAL_24_FEATURES if c.endswith("_ratio")]
    ratios_ok = bool(all(
        (df[c] >= -1e-5).all() and (df[c] <= 1.00001).all()
        for c in ratio_features
    ))
    
    dir_sum = df["inbound_pkt_ratio"] + df["outbound_pkt_ratio"]
    dir_conservation_ok = bool(((dir_sum - 1.0).abs() < 1e-4).all())
    
    ttl_ok = bool((df["ip_ttl_mean"] >= 1.0).all() and (df["ip_ttl_mean"] <= 255.0).all())
    
    proto_ratios = [c for c in ["tcp_ratio", "udp_ratio", "icmp_ratio"] if c in df.columns]
    proto_ratios_ok = bool(all(
        (df[c] >= 0.0).all() and (df[c] <= 1.0).all()
        for c in proto_ratios
    ))
    
    all_passed = (
        nan_count == 0 and
        inf_count == 0 and
        duration_ok and
        rate_ok and
        byte_rate_ok and
        iat_nonneg_ok and
        iat_ceiling_ok and
        iat_std_ok and
        size_hierarchy_ok and
        no_neg_sizes and
        ratios_ok and
        dir_conservation_ok and
        ttl_ok and
        proto_ratios_ok
    )
    
    return {
        "all_passed": all_passed,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "duration_ok": duration_ok,
        "rate_ok": rate_ok,
        "byte_rate_ok": byte_rate_ok,
        "iat_nonneg_ok": iat_nonneg_ok,
        "iat_ceiling_ok": iat_ceiling_ok,
        "windows_with_iat_gte_2s": windows_with_iat_gte_2s,
        "max_iat_mean_ms": max_iat_mean_ms,
        "windows_with_dur_gt_2s": windows_with_dur_gt_2s,
        "iat_std_ok": iat_std_ok,
        "size_hierarchy_ok": size_hierarchy_ok,
        "no_neg_sizes": no_neg_sizes,
        "ratios_ok": ratios_ok,
        "dir_conservation_ok": dir_conservation_ok,
        "ttl_ok": ttl_ok,
        "proto_ratios_ok": proto_ratios_ok
    }


def main():
    print("=" * 88)
    print("  AEGIS-Live v2 — MULTI-CLASS PCAP VALIDATION PIPELINE")
    print("=" * 88, flush=True)
    
    results = {}
    all_dfs = []
    
    for pcap_filename, class_label in OFFICIAL_PROVENANCE.items():
        filepath = os.path.join(PCAP_DIR, pcap_filename)
        if not os.path.exists(filepath):
            print(f"[!] MISSING REQUIRED PCAP: {filepath}")
            print("PCAP_MULTICLASS_VALIDATION = INSUFFICIENT_DATA")
            return
            
        pcap_res = validate_single_pcap(filepath, class_label, max_extract_windows=5000)
        results[class_label] = pcap_res
        
        df_class = pd.DataFrame(pcap_res["extracted_windows"])
        if len(df_class) > 0:
            df_class_valid = validate_24_features(df_class)
            df_class_valid["label"] = class_label
            all_dfs.append(df_class_valid)
            
    if len(all_dfs) < 4:
        print(f"[!] Error: Only {len(all_dfs)}/4 classes produced valid windows.")
        print("PCAP_MULTICLASS_VALIDATION = INSUFFICIENT_DATA")
        return
        
    df_all = pd.concat(all_dfs, ignore_index=True)
    
    # Per-Class Statistics & Invariant Checks
    print("\n" + "=" * 88)
    print("  PER-CLASS 24-FEATURE INVARIANT CHECKS")
    print("=" * 88, flush=True)
    
    class_stats = {}
    class_invariants = {}
    for class_label in ["DDoS", "DoS", "Spoofing", "Benign"]:
        df_c = df_all[df_all["label"] == class_label]
        stats_c = compute_feature_statistics(df_c)
        inv_c = verify_invariants(df_c)
        class_stats[class_label] = stats_c
        class_invariants[class_label] = inv_c
        print(f"  Class [{class_label:8s}] Invariants All Passed: {inv_c['all_passed']} (N = {len(df_c):,} windows)", flush=True)
        print(f"       - Windows with IAT >= 2.0s: {inv_c['windows_with_iat_gte_2s']} | Max Mean IAT: {inv_c['max_iat_mean_ms']:.2f} ms", flush=True)
        print(f"       - Windows with duration > 2.0s: {inv_c['windows_with_dur_gt_2s']} ({inv_c['windows_with_dur_gt_2s']/len(df_c)*100:.1f}%)", flush=True)

    # Data Quality Checks
    print("\n" + "=" * 88)
    print("  DATA QUALITY & ANOMALY AUDIT")
    print("=" * 88, flush=True)
    
    dup_windows_all = int(df_all[CANONICAL_24_FEATURES].duplicated().sum())
    print(f"  1. Duplicate Windows Across Entire Dataset: {dup_windows_all:,} ({dup_windows_all/len(df_all)*100:.2f}%)", flush=True)
    dup_by_class = {}
    for class_label in ["DDoS", "DoS", "Spoofing", "Benign"]:
        df_c = df_all[df_all["label"] == class_label]
        dup_c = int(df_c[CANONICAL_24_FEATURES].duplicated().sum())
        dup_by_class[class_label] = dup_c
        print(f"     - Class [{class_label:8s}] Duplicates: {dup_c:,} ({dup_c/len(df_c)*100:.2f}%)", flush=True)
        
    print("  2. Suspiciously Constant Features (Std == 0):", flush=True)
    const_by_class = {}
    for class_label in ["DDoS", "DoS", "Spoofing", "Benign"]:
        df_c = df_all[df_all["label"] == class_label]
        const_feats = [c for c in CANONICAL_24_FEATURES if df_c[c].std() == 0.0]
        const_by_class[class_label] = const_feats
        print(f"     - Class [{class_label:8s}]: {len(const_feats)} features: {const_feats}", flush=True)
        
    print("  3. Windows Crossing Large Timestamp Gaps:", flush=True)
    for class_label, pcap_res in results.items():
        print(f"     - Class [{class_label:8s}]: Max Gap in PCAP = {pcap_res['max_ts_gap']:.3f}s", flush=True)
        
    print("  4. Class Balance (Extracted Windows):", flush=True)
    for class_label in ["DDoS", "DoS", "Spoofing", "Benign"]:
        count_c = len(df_all[df_all["label"] == class_label])
        print(f"     - Class [{class_label:8s}]: {count_c:,} windows ({count_c/len(df_all)*100:.1f}%)", flush=True)

    # Cross-Class Distribution Comparison
    print("\n" + "=" * 88)
    print("  CROSS-CLASS COMPARATIVE DISTRIBUTIONS (Mean +/- Std)")
    print("=" * 88, flush=True)
    
    comp_rows = []
    for feat in CANONICAL_24_FEATURES:
        row = {"Feature": feat}
        means = {}
        for class_label in ["DDoS", "DoS", "Spoofing", "Benign"]:
            m = float(df_all[df_all["label"] == class_label][feat].mean())
            s = float(df_all[df_all["label"] == class_label][feat].std())
            means[class_label] = m
            row[class_label] = f"{m:,.2f} +/- {s:,.2f}"
            
        class_mean_vals = list(means.values())
        differs = np.std(class_mean_vals) > 1e-4
        row["Differs"] = "YES" if differs else "NO (Constant)"
        comp_rows.append(row)
        
    df_comp = pd.DataFrame(comp_rows)
    print(df_comp.to_string(index=False), flush=True)

    # Save summary JSON
    summary_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "pcap_metrics": {
            cls: {k: v for k, v in res.items() if k != "extracted_windows"}
            for cls, res in results.items()
        },
        "invariants": class_invariants,
        "data_quality": {
            "total_windows": len(df_all),
            "duplicate_windows_total": dup_windows_all,
            "duplicate_by_class": dup_by_class,
            "constant_features_by_class": const_by_class
        },
        "feature_stats": {
            cls: class_stats[cls].to_dict(orient="index")
            for cls in ["DDoS", "DoS", "Spoofing", "Benign"]
        }
    }
    
    json_path = os.path.join(BASE_DIR, "reports", "aegis_live_v2_multiclass_validation.json")
    with open(json_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\n[+] Saved validation summary to {json_path}", flush=True)
    
    # Save CSV of all extracted windows
    csv_path = os.path.join(BASE_DIR, "reports", "aegis_live_v2_extracted_windows_sample.csv")
    df_all.to_csv(csv_path, index=False)
    print(f"[+] Saved extracted windows dataset (N = {len(df_all):,}) to {csv_path}", flush=True)

    # Determine final verdict
    all_pcaps_ok = len(results) == 4
    all_inv_ok = all(inv["all_passed"] for inv in class_invariants.values())
    no_errors = all(res["extraction_errors"] == 0 for res in results.values())
    
    print("\n" + "=" * 88)
    print("  FINAL AUDIT VERDICT")
    print("=" * 88)
    if all_pcaps_ok and all_inv_ok and no_errors:
        print("PCAP_MULTICLASS_VALIDATION = PASS", flush=True)
    else:
        print("PCAP_MULTICLASS_VALIDATION = FAIL", flush=True)


if __name__ == '__main__':
    main()

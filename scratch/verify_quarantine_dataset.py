"""
Audit and verify AEGIS-Live v2 Final Temporal Quarantine Dataset
"""
import os
import sys
import json
import hashlib
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DATA_DIR = r"D:\-AI-Driven-Intrusion-Detection-System-for-IoT-Networks\data\aegis_live_v2_quarantine"
PCAP_DIR = r"D:\-AI-Driven-Intrusion-Detection-System-for-IoT-Networks\pcaps"

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024 * 16):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("--- 1. LOADING DATASETS ---")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "validation.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    
    with open(os.path.join(DATA_DIR, "dataset_manifest.json"), "r") as f:
        manifest = json.load(f)
        
    print(f"Train rows: {len(train_df):,}")
    print(f"Validation rows: {len(val_df):,}")
    print(f"Test rows: {len(test_df):,}")
    print(f"Total: {len(train_df) + len(val_df) + len(test_df):,}")
    
    print("\n--- 2. CLASS BALANCE ---")
    for part_name, df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        print(f"{part_name}:")
        print(df["label"].value_counts())
        
    print("\n--- 3. PER-PCAP TIMESTAMPS & QUARANTINE INTERVALS ---")
    for pcap, q in manifest["quarantine_audit"].items():
        print(f"\nPCAP: {pcap}")
        print(f"  Train End TS:                {q['train_end_ts']:.6f}")
        print(f"  Train->Val Quarantine Start: {q['quarantine_train_val_start']:.6f}")
        print(f"  Train->Val Quarantine End:   {q['quarantine_train_val_end']:.6f}")
        print(f"  Val Start TS:                {q['val_start_ts']:.6f}")
        print(f"  Actual Train->Val Gap:       {q['train_val_actual_gap_sec']:.6f} s")
        print(f"  Quarantined Train->Val Pkts: {q['quarantined_train_val_packets']:,}")
        print(f"  Val End TS:                  {q['val_end_ts']:.6f}")
        print(f"  Val->Test Quarantine Start:  {q['quarantine_val_test_start']:.6f}")
        print(f"  Val->Test Quarantine End:    {q['quarantine_val_test_end']:.6f}")
        print(f"  Test Start TS:               {q['test_start_ts']:.6f}")
        print(f"  Actual Val->Test Gap:        {q['val_test_actual_gap_sec']:.6f} s")
        print(f"  Quarantined Val->Test Pkts:  {q['quarantined_val_test_packets']:,}")

    print("\n--- 4. EXACT 24-FEATURE CROSS-PARTITION DUPLICATES ---")
    from aegis_live.feature_schema import CANONICAL_24_FEATURES
    
    tr_feats = set(tuple(r) for r in train_df[CANONICAL_24_FEATURES].values)
    va_feats = set(tuple(r) for r in val_df[CANONICAL_24_FEATURES].values)
    te_feats = set(tuple(r) for r in test_df[CANONICAL_24_FEATURES].values)
    
    tr_va_dup = len(tr_feats.intersection(va_feats))
    tr_te_dup = len(tr_feats.intersection(te_feats))
    va_te_dup = len(va_feats.intersection(te_feats))
    
    print(f"Train INTERSECT Val duplicates:  {tr_va_dup}")
    print(f"Train INTERSECT Test duplicates: {tr_te_dup}")
    print(f"Val INTERSECT Test duplicates:   {va_te_dup}")
    
    # Within partition duplicates
    print(f"\nWithin-partition duplicates:")
    print(f"Train unique: {len(tr_feats):,} / {len(train_df):,} (Duplicates: {len(train_df) - len(tr_feats):,})")
    print(f"Val unique:   {len(va_feats):,} / {len(val_df):,} (Duplicates: {len(val_df) - len(va_feats):,})")
    print(f"Test unique:  {len(te_feats):,} / {len(test_df):,} (Duplicates: {len(test_df) - len(te_feats):,})")

    print("\n--- 5. FLOW IDENTITY OVERLAP (PERSISTENT-FLOW EVALUATION DESIGN) ---")
    tr_flows = set(train_df["flow_id"].unique())
    va_flows = set(val_df["flow_id"].unique())
    te_flows = set(test_df["flow_id"].unique())
    
    print(f"Train unique flows: {len(tr_flows)}")
    print(f"Val unique flows:   {len(va_flows)}")
    print(f"Test unique flows:  {len(te_flows)}")
    print(f"Train INTERSECT Val flows:  {len(tr_flows.intersection(va_flows))}")
    print(f"Train INTERSECT Test flows: {len(tr_flows.intersection(te_flows))}")
    print(f"Val INTERSECT Test flows:   {len(va_flows.intersection(te_flows))}")

    print("\n--- 6. FEATURE INTEGRITY AUDIT ---")
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    nans = all_df[CANONICAL_24_FEATURES].isna().sum().sum()
    infs = np.isinf(all_df[CANONICAL_24_FEATURES].values).sum()
    neg_dur = (all_df["duration_ms"] < 0).sum()
    neg_pkt_rate = (all_df["pkt_rate_pps"] < 0).sum()
    neg_byte_rate = (all_df["byte_rate_bps"] < 0).sum()
    size_bad = ((all_df["pkt_size_min"] < 14) | (all_df["pkt_size_max"] > 65535)).sum()
    ttl_bad = ((all_df["ip_ttl_mean"] < 1.0) | (all_df["ip_ttl_mean"] > 255.0)).sum()
    dir_bad = (np.abs((all_df["inbound_pkt_ratio"] + all_df["outbound_pkt_ratio"]) - 1.0) > 1e-4).sum()
    
    print(f"NaNs:                  {nans}")
    print(f"Infs:                  {infs}")
    print(f"Negative duration:     {neg_dur}")
    print(f"Negative packet rate:  {neg_pkt_rate}")
    print(f"Negative byte rate:    {neg_byte_rate}")
    print(f"Invalid packet sizes:  {size_bad}")
    print(f"Invalid TTL:           {ttl_bad}")
    print(f"Direction sum != 1.0:  {dir_bad}")

    print("\n--- 7. EXCLUSION AUDIT ---")
    for pcap, exc in manifest["exclusion_counts"].items():
        print(f"PCAP: {pcap} -> {exc}")

    print("\n--- 8. PCAP HASHES ---")
    pcaps = [
        "DDoS-SYN_Flood.pcap",
        "DoS-SYN_Flood.pcap",
        "MITM-ArpSpoofing.pcap",
        "BenignTraffic.pcap"
    ]
    for p in pcaps:
        fpath = os.path.join(PCAP_DIR, p)
        h = compute_sha256(fpath)
        print(f"  {p:25s}: {h}")

if __name__ == "__main__":
    main()

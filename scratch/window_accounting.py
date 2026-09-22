"""
Calculate candidate, quarantined, and remaining windows for each PCAP
"""
import os
import json
import pandas as pd

DATA_DIR = r"D:\-AI-Driven-Intrusion-Detection-System-for-IoT-Networks\data\aegis_live_v2_quarantine"

def main():
    with open(os.path.join(DATA_DIR, "dataset_manifest.json"), "r") as f:
        manifest = json.load(f)
        
    print("=== WINDOW ACCOUNTING PER PCAP ===")
    for pcap, q in manifest["quarantine_audit"].items():
        exc = manifest["exclusion_counts"][pcap]
        # In each PCAP, 20,000 windows were selected.
        # Packets quarantined during Tr->Val and Val->Te:
        q_pkts = q["quarantined_train_val_packets"] + q["quarantined_val_test_packets"]
        # Potential windows if 50 packets per window:
        q_est_windows = q_pkts // 50
        print(f"\nPCAP: {pcap}")
        print(f"  Selected Windows:          20,000")
        print(f"  Quarantined Packets:       {q_pkts:,} (Tr->Val: {q['quarantined_train_val_packets']:,}, Val->Te: {q['quarantined_val_test_packets']:,})")
        print(f"  Quarantined Potential Wins: ~{q_est_windows:,} (based on 50-pkt capacity)")
        print(f"  Candidate Processed Wins:  20,000 accepted + {exc.get('cross_partition_duplicate_train_vs_test', 0)} cross-partition excluded")
        print(f"  Remaining Valid Windows:   20,000")
        print(f"  Exclusions:                {exc}")

if __name__ == "__main__":
    main()

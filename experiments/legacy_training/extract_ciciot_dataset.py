"""
Script to extract exactly 1,500,000 records from CIC-IoT-2023-neto-full train split
Categories: DDoS (~500k), DoS (~500k), Spoofing (~500k / all available in train)
Memory-efficient streaming execution with deterministic sampling.
"""

import os
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

# Configuration
REPO_ID = "lacg030175/CIC-IoT-2023-neto-full"
SPLIT_FILE = "random/train-00000-of-00001.parquet"
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CIC_IoT_2023_DDoS_DoS_Spoofing_1.5M.csv")
RANDOM_SEED = 42

TARGET_TOTAL = 1_500_000
# Train split has 389,321 total Spoofing records.
# Remaining 1,110,679 records are split evenly between DDoS and DoS.
TARGET_SPOOFING = 389_321
TARGET_DDOS = 555_340
TARGET_DOS = 555_339

def main():
    print("=" * 70)
    print("CIC-IoT-2023 Dataset Extraction: DDoS, DoS, Spoofing (1.5M Total)")
    print("=" * 70)
    
    # 1. Download / Locate cached parquet file
    print("\n[Step 1/4] Ensuring dataset parquet is available locally...")
    parquet_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=SPLIT_FILE,
        repo_type="dataset"
    )
    print(f"Parquet source: {parquet_path}")
    print(f"Parquet file size: {os.path.getsize(parquet_path) / (1024*1024):.2f} MB")
    
    # 2. Inspect Schema and Row Groups
    pf = pq.ParquetFile(parquet_path)
    num_rgs = pf.num_row_groups
    total_rows = pf.metadata.num_rows
    print(f"Total row groups: {num_rgs}, Total rows in split: {total_rows:,}")
    print(f"Total columns in schema: {len(pf.schema.names)}")
    
    # 3. Pass 1: Scan class counts across row groups for exact stratified allocation
    print("\n[Step 2/4] Scanning class distributions across row groups...")
    t0 = time.time()
    rg_counts = []  # list of (ddos_count, dos_count, spoofing_count)
    
    for rg_idx in range(num_rgs):
        col = pf.read_row_group(rg_idx, columns=['attack_class']).column('attack_class')
        # Fast counting using numpy / pandas
        s = col.to_pandas().value_counts()
        rg_counts.append({
            'DDoS': s.get('DDoS', 0),
            'DoS': s.get('DoS', 0),
            'Spoofing': s.get('Spoofing', 0)
        })
        if (rg_idx + 1) % 50 == 0 or rg_idx == num_rgs - 1:
            print(f"  Scanned row group {rg_idx + 1}/{num_rgs}...")
            
    t1 = time.time()
    
    total_ddos_avail = sum(c['DDoS'] for c in rg_counts)
    total_dos_avail = sum(c['DoS'] for c in rg_counts)
    total_spoofing_avail = sum(c['Spoofing'] for c in rg_counts)
    
    print(f"Scan complete in {t1 - t0:.2f}s.")
    print(f"Available in Train Split -> DDoS: {total_ddos_avail:,}, DoS: {total_dos_avail:,}, Spoofing: {total_spoofing_avail:,}")
    
    # Compute exact sampling quota per row group using fixed seed
    rng = np.random.RandomState(RANDOM_SEED)
    
    # Stratified proportional quota with remainder distribution
    def compute_quotas(counts_list, target_total):
        total_avail = sum(counts_list)
        if total_avail <= target_total:
            return counts_list[:]
        
        # Proportional float allocation
        exact_alloc = [c * target_total / total_avail for c in counts_list]
        base_alloc = [int(np.floor(x)) for x in exact_alloc]
        remainder = target_total - sum(base_alloc)
        
        # Distribute remaining slots based on fractional parts
        fractional_parts = [exact_alloc[i] - base_alloc[i] for i in range(len(counts_list))]
        # Use argsort of fractional parts (with random perturbation for deterministic tie breaking)
        noise = rng.uniform(0, 1e-6, size=len(fractional_parts))
        sorted_indices = np.argsort([-(f + n) for f, n in zip(fractional_parts, noise)])
        
        for idx in sorted_indices[:remainder]:
            base_alloc[idx] += 1
            
        assert sum(base_alloc) == target_total, f"Quota mismatch: {sum(base_alloc)} != {target_total}"
        return base_alloc

    ddos_quotas = compute_quotas([c['DDoS'] for c in rg_counts], TARGET_DDOS)
    dos_quotas = compute_quotas([c['DoS'] for c in rg_counts], TARGET_DOS)
    spoofing_quotas = [c['Spoofing'] for c in rg_counts]  # take all available
    
    print(f"\nTarget Sampling Quotas:")
    print(f"  DDoS quota:     {sum(ddos_quotas):,}")
    print(f"  DoS quota:      {sum(dos_quotas):,}")
    print(f"  Spoofing quota: {sum(spoofing_quotas):,}")
    print(f"  Total quota:    {sum(ddos_quotas) + sum(dos_quotas) + sum(spoofing_quotas):,}")
    
    # 4. Pass 2: Stream row groups, sample exact rows, write incrementally to CSV
    print(f"\n[Step 3/4] Streaming and writing records to CSV: {OUTPUT_CSV}")
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)
        
    t_start = time.time()
    records_written = 0
    ddos_written = 0
    dos_written = 0
    spoofing_written = 0
    first_chunk = True
    
    for rg_idx in range(num_rgs):
        k_ddos = ddos_quotas[rg_idx]
        k_dos = dos_quotas[rg_idx]
        k_spoof = spoofing_quotas[rg_idx]
        
        if k_ddos == 0 and k_dos == 0 and k_spoof == 0:
            continue
            
        table = pf.read_row_group(rg_idx)
        df = table.to_pandas()
        
        selected_parts = []
        
        if k_spoof > 0:
            spoof_df = df[df['attack_class'] == 'Spoofing']
            selected_parts.append(spoof_df)
            spoofing_written += len(spoof_df)
            
        if k_dos > 0:
            dos_df = df[df['attack_class'] == 'DoS']
            if len(dos_df) > k_dos:
                dos_df = dos_df.sample(n=k_dos, random_state=rng.randint(0, 100000000))
            selected_parts.append(dos_df)
            dos_written += len(dos_df)
            
        if k_ddos > 0:
            ddos_df = df[df['attack_class'] == 'DDoS']
            if len(ddos_df) > k_ddos:
                ddos_df = ddos_df.sample(n=k_ddos, random_state=rng.randint(0, 100000000))
            selected_parts.append(ddos_df)
            ddos_written += len(ddos_df)
            
        if selected_parts:
            batch_df = pd.concat(selected_parts, ignore_index=True)
            batch_df.to_csv(
                OUTPUT_CSV,
                mode='a',
                index=False,
                header=first_chunk
            )
            first_chunk = False
            records_written += len(batch_df)
            
        if (rg_idx + 1) % 25 == 0 or rg_idx == num_rgs - 1:
            elapsed = time.time() - t_start
            print(f"  Processed row group {rg_idx + 1}/{num_rgs} | Records written: {records_written:,} | Elapsed: {elapsed:.1f}s")
            
    t_end = time.time()
    print(f"\nCSV writing complete in {t_end - t_start:.2f}s!")
    print(f"Total records written: {records_written:,}")
    print(f"  - DDoS:     {ddos_written:,}")
    print(f"  - DoS:      {dos_written:,}")
    print(f"  - Spoofing: {spoofing_written:,}")
    
    # 5. Verification Phase
    print("\n[Step 4/4] Verifying generated CSV...")
    verify_csv(OUTPUT_CSV)

def verify_csv(file_path):
    assert os.path.exists(file_path), f"File not found: {file_path}"
    file_size_bytes = os.path.getsize(file_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    file_size_gb = file_size_bytes / (1024 * 1024 * 1024)
    
    print(f"\nVerifying file: {file_path}")
    print(f"File Size: {file_size_mb:.2f} MB ({file_size_gb:.2f} GB)")
    
    total_rows = 0
    class_counts = {}
    label_counts = {}
    col_names = []
    
    chunk_size = 200_000
    for chunk_idx, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
        if chunk_idx == 0:
            col_names = list(chunk.columns)
        total_rows += len(chunk)
        for k, v in chunk['attack_class'].value_counts().items():
            class_counts[k] = class_counts.get(k, 0) + v
        for k, v in chunk['Label'].value_counts().items():
            label_counts[k] = label_counts.get(k, 0) + v
            
    print("\n" + "=" * 70)
    print("FINAL VERIFICATION & SUMMARY")
    print("=" * 70)
    print(f"1. Total Number of Rows:      {total_rows:,}")
    print(f"2. Number of DDoS Records:    {class_counts.get('DDoS', 0):,}")
    print(f"3. Number of DoS Records:     {class_counts.get('DoS', 0):,}")
    print(f"4. Number of Spoofing Records:{class_counts.get('Spoofing', 0):,}")
    print(f"5. Other Attack Categories:   {sum(v for k, v in class_counts.items() if k not in ['DDoS', 'DoS', 'Spoofing'])}")
    print(f"6. Total Number of Columns:   {len(col_names)}")
    print(f"7. CSV File Path:             {file_path}")
    print(f"8. CSV File Size:             {file_size_mb:.2f} MB ({file_size_gb:.3f} GB)")
    print("\nSub-attack breakdown:")
    for k, v in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  - {k}: {v:,}")
    print("=" * 70)
    
    # Assertions for 100% compliance
    assert total_rows == 1_500_000, f"Expected 1,500,000 rows, got {total_rows}"
    assert class_counts.get('DDoS', 0) == TARGET_DDOS, f"DDoS count mismatch"
    assert class_counts.get('DoS', 0) == TARGET_DOS, f"DoS count mismatch"
    assert class_counts.get('Spoofing', 0) == TARGET_SPOOFING, f"Spoofing count mismatch"
    assert set(class_counts.keys()) == {'DDoS', 'DoS', 'Spoofing'}, "Unexpected attack categories found!"
    assert len(col_names) == 50, f"Expected 50 columns, got {len(col_names)}"
    print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY! [OK]")

if __name__ == "__main__":
    main()

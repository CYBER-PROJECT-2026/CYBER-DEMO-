"""
Script to create a genuinely UNSEEN 5,000-record holdout dataset (testing_data.csv)
from the official Hugging Face CIC-IoT-2023 random/test split with 0% overlap
against the 1,500,000 training records.

Target:
- DDoS: 1,667
- DoS: 1,667
- Spoofing: 1,666
- Total: 5,000 records
- Columns: 46 model features in exact order + attack_class (47 total)
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

# Ensure UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cic_iot_xgboost_model.pkl")
TRAIN_CSV_PATH = os.path.join(BASE_DIR, "CIC_IoT_2023_DDoS_DoS_Spoofing_1.5M.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "Testing_data")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "testing_data.csv")

REPO_ID = "lacg030175/CIC-IoT-2023-neto-full"
TEST_SPLIT_FILE = "random/test-00000-of-00001.parquet"

RANDOM_SEED = 42
TARGET_DDOS = 1667
TARGET_DOS = 1667
TARGET_SPOOFING = 1666
TARGET_TOTAL = 5000

def main():
    print("=" * 85)
    print("  CREATING UNSEEN 5,000-RECORD HOLDOUT DATASET (testing_data.csv)")
    print("=" * 85)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Model to extract exact 46 feature names and order
    print(f"\n[Step 1/6] Loading 46-feature model: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    features_in_model = list(model.feature_names_in_)
    print(f"✓ Extracted {len(features_in_model)} feature names from model.")
    print("✓ Exact 46 feature list:")
    for idx, f_name in enumerate(features_in_model, 1):
        print(f"   {idx:>2}. {f_name}")
        
    final_columns = features_in_model + ["attack_class"]
    print(f"\n✓ Total final columns ({len(final_columns)}): 46 features + 'attack_class'")
    
    # 2. Build Hash Set of all 1,500,000 Training Records
    print(f"\n[Step 2/6] Hashing all 1,500,000 records from training CSV: {TRAIN_CSV_PATH}")
    t0_hash = time.perf_counter()
    train_hashes = set()
    total_train_rows = 0
    
    for chunk_idx, chunk in enumerate(pd.read_csv(TRAIN_CSV_PATH, chunksize=250000)):
        arr = chunk[features_in_model].values.astype(np.float32)
        for r in arr:
            train_hashes.add(r.tobytes())
        total_train_rows += len(chunk)
        print(f"  Processed {total_train_rows:,} / 1,500,000 training rows (distinct hashes: {len(train_hashes):,})...")
        
    t1_hash = time.perf_counter()
    print(f"✓ Training set hashing completed in {t1_hash - t0_hash:.2f}s.")
    print(f"✓ Total training records hashed: {total_train_rows:,}")
    print(f"✓ Total unique 46-feature hashes in training set: {len(train_hashes):,}")
    
    # 3. Locate & Stream Hugging Face Test Parquet
    print(f"\n[Step 3/6] Locating Hugging Face test split parquet ({TEST_SPLIT_FILE})...")
    test_parquet_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=TEST_SPLIT_FILE,
        repo_type="dataset"
    )
    print(f"✓ Test parquet source: {test_parquet_path}")
    print(f"✓ File size: {os.path.getsize(test_parquet_path) / (1024*1024):.2f} MB")
    
    pf = pq.ParquetFile(test_parquet_path)
    print(f"✓ Total rows in test parquet: {pf.metadata.num_rows:,} across {pf.num_row_groups} row groups")
    
    # 4. Stream and filter non-overlapping candidate rows for each class
    print(f"\n[Step 4/6] Streaming test parquet and filtering eligible non-overlapping records...")
    t0_filter = time.perf_counter()
    
    # We collect a pool of eligible records (up to a safe buffer, e.g., 50,000 per class)
    POOL_LIMIT_PER_CLASS = 20000
    candidate_pools = {
        "DDoS": [],
        "DoS": [],
        "Spoofing": []
    }
    
    overlap_detected = 0
    scanned_samples = 0
    
    for rg_idx in range(pf.num_row_groups):
        # Stop early if we have collected enough candidates for all 3 classes
        if (len(candidate_pools["DDoS"]) >= POOL_LIMIT_PER_CLASS and
            len(candidate_pools["DoS"]) >= POOL_LIMIT_PER_CLASS and
            len(candidate_pools["Spoofing"]) >= POOL_LIMIT_PER_CLASS):
            print(f"  Collected ample candidate buffer across all classes by row group {rg_idx}. Stopping stream scan.")
            break
            
        table = pf.read_row_group(rg_idx, columns=final_columns)
        df_chunk = table.to_pandas()
        scanned_samples += len(df_chunk)
        
        for cls_name in ["DDoS", "DoS", "Spoofing"]:
            if len(candidate_pools[cls_name]) >= POOL_LIMIT_PER_CLASS:
                continue
                
            sub_df = df_chunk[df_chunk['attack_class'] == cls_name]
            if len(sub_df) == 0:
                continue
                
            # Filter rows that are NOT in training set
            feature_arr = sub_df[features_in_model].values.astype(np.float32)
            valid_indices = []
            for i, r in enumerate(feature_arr):
                if r.tobytes() in train_hashes:
                    overlap_detected += 1
                else:
                    valid_indices.append(i)
                    
            if valid_indices:
                valid_df = sub_df.iloc[valid_indices]
                candidate_pools[cls_name].append(valid_df)
                
        if (rg_idx + 1) % 10 == 0 or rg_idx == pf.num_row_groups - 1:
            curr_ddos = sum(len(c) for c in candidate_pools["DDoS"])
            curr_dos = sum(len(c) for c in candidate_pools["DoS"])
            curr_spoof = sum(len(c) for c in candidate_pools["Spoofing"])
            print(f"  Row group {rg_idx + 1:>2}/{pf.num_row_groups} | Non-overlapping candidates -> DDoS: {curr_ddos:,}, DoS: {curr_dos:,}, Spoofing: {curr_spoof:,}")
            
    t1_filter = time.perf_counter()
    print(f"✓ Filtering completed in {t1_filter - t0_filter:.2f}s.")
    print(f"✓ Overlapping records excluded: {overlap_detected}")
    
    # Concatenate candidate pools
    df_candidates = {}
    for cls in ["DDoS", "DoS", "Spoofing"]:
        df_candidates[cls] = pd.concat(candidate_pools[cls], ignore_index=True)
        print(f"✓ Available unseen candidate pool for {cls:<10}: {len(df_candidates[cls]):,} rows")
        
    # 5. Reproducible Random Sampling
    print(f"\n[Step 5/6] Performing reproducible sampling with random_seed = {RANDOM_SEED}...")
    rng = np.random.RandomState(RANDOM_SEED)
    
    sample_ddos = df_candidates["DDoS"].sample(n=TARGET_DDOS, random_state=rng.randint(0, 10000000))
    sample_dos = df_candidates["DoS"].sample(n=TARGET_DOS, random_state=rng.randint(0, 10000000))
    sample_spoof = df_candidates["Spoofing"].sample(n=TARGET_SPOOFING, random_state=rng.randint(0, 10000000))
    
    print(f"✓ Sampled DDoS     : {len(sample_ddos):,} records")
    print(f"✓ Sampled DoS      : {len(sample_dos):,} records")
    print(f"✓ Sampled Spoofing : {len(sample_spoof):,} records")
    
    # Combine and shuffle
    final_df = pd.concat([sample_ddos, sample_dos, sample_spoof], ignore_index=True)
    final_df = final_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    
    # Ensure exact column ordering (46 features + attack_class)
    final_df = final_df[final_columns]
    
    # Save to CSV
    final_df.to_csv(OUTPUT_CSV, index=False)
    file_size_bytes = os.path.getsize(OUTPUT_CSV)
    file_size_kb = file_size_bytes / 1024
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"✓ Successfully saved holdout dataset to: {OUTPUT_CSV}")
    print(f"✓ File size: {file_size_kb:.2f} KB ({file_size_mb:.4f} MB)")
    
    # 6. Rigorous Verification Phase
    print(f"\n[Step 6/6] Verifying dataset integrity and zero overlap...")
    
    # Read saved file back
    verify_df = pd.read_csv(OUTPUT_CSV)
    
    # Verify dimensions
    assert len(verify_df) == TARGET_TOTAL, f"Row count mismatch: {len(verify_df)} != {TARGET_TOTAL}"
    assert len(verify_df.columns) == 47, f"Column count mismatch: {len(verify_df.columns)} != 47"
    assert list(verify_df.columns) == final_columns, "Column names or ordering mismatch!"
    
    # Verify class counts
    counts = verify_df['attack_class'].value_counts().to_dict()
    assert counts.get('DDoS') == TARGET_DDOS, f"DDoS count mismatch: {counts.get('DDoS')}"
    assert counts.get('DoS') == TARGET_DOS, f"DoS count mismatch: {counts.get('DoS')}"
    assert counts.get('Spoofing') == TARGET_SPOOFING, f"Spoofing count mismatch: {counts.get('Spoofing')}"
    assert set(counts.keys()) == {"DDoS", "DoS", "Spoofing"}, f"Unexpected classes: {counts.keys()}"
    
    # Verify 0 overlap with training set
    test_feature_arr = verify_df[features_in_model].values.astype(np.float32)
    overlap_in_final = 0
    for r in test_feature_arr:
        if r.tobytes() in train_hashes:
            overlap_in_final += 1
            
    assert overlap_in_final == 0, f"Critical error: {overlap_in_final} overlapping records detected in final dataset!"
    
    print("=" * 85)
    print("                      DATASET VERIFICATION PASSED (100% SUCCESS)")
    print("=" * 85)
    print(f"1.  Total Rows                : {len(verify_df):,} (Exactly {TARGET_TOTAL})")
    print(f"2.  Total Columns             : {len(verify_df.columns)} (46 features + attack_class)")
    print(f"3.  Target Column             : attack_class")
    print(f"4.  DDoS Records              : {counts['DDoS']:,}")
    print(f"5.  DoS Records               : {counts['DoS']:,}")
    print(f"6.  Spoofing Records          : {counts['Spoofing']:,}")
    print(f"7.  Other Attack Categories   : 0")
    print(f"8.  Training-Source Overlap   : 0 (Zero overlap with 1.5M training set)")
    print(f"9.  Random Seed               : {RANDOM_SEED}")
    print(f"10. Source Split              : Hugging Face random/test split")
    print(f"11. Output File Path          : {OUTPUT_CSV}")
    print(f"12. File Size                 : {file_size_kb:.2f} KB ({file_size_mb:.4f} MB)")
    print("\nStatement:")
    print("These 5,000 records were selected from the official Hugging Face test split and verified to have 0 overlap with the 1,500,000 records used for model training.")
    print("=" * 85)

if __name__ == "__main__":
    main()

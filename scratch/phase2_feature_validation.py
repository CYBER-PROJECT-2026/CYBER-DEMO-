"""
scratch/phase2_feature_validation.py
====================================
Comprehensive read-only forensic validation for Phase 2:
1. Validates candidate formulas across 10,000 rows from the official Parquet test partition.
2. Performs the IAT Model Compatibility Diagnostic on the frozen AEGIS XGBoost model.
3. Quantifies mathematical bounds, distributions, errors, and failure modes.
"""

import os
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Add root to sys.path to allow importing backend components
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.detection.detector import IoTIntrusionDetector
from backend.detection.abstention_gate import OODAbstentionGate
from backend.detection.ood_config import STATE_UNKNOWN_OR_REVIEW


def run_parquet_10k_validation():
    print("=" * 80)
    print("STEP 1: 10,000-ROW STATISTICAL FORMULA VALIDATION ACROSS OFFICIAL PARQUET")
    print("=" * 80)
    
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub/datasets--lacg030175--CIC-IoT-2023-neto-full")
    parquet_path = os.path.join(hf_cache, "snapshots", "bc3fc53b0537cf558f404cf379434b925d14233a", "random", "test-00000-of-00001.parquet")
    
    pf = pq.ParquetFile(parquet_path)
    cols = [
        'flow_duration', 'Header_Length', 'Protocol Type', 'Duration', 'Rate', 'Srate', 'Drate',
        'ack_count', 'syn_count', 'fin_count', 'urg_count', 'rst_count',
        'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number',
        'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight', 'attack_class'
    ]
    # Read first 10,000 rows
    table = pf.read_row_group(0, columns=cols)
    df = table.to_pandas().iloc[:10000].copy()
    n = len(df)
    print(f"Successfully loaded {n} official rows from {parquet_path}")

    # ----------------------------------------------------
    # Feature 1 & 2: Magnitue and Radius
    # ----------------------------------------------------
    print("\n--- A. Magnitue & Radius Validation (10,000 rows) ---")
    cand_mag = np.sqrt(2 * df['AVG'])
    err_mag = np.abs(df['Magnitue'] - cand_mag)
    exact_mag = (err_mag < 1e-3).sum()
    tol_mag = (err_mag < 0.1).sum()
    mae_mag = float(err_mag.mean())
    rmse_mag = float(np.sqrt(np.mean(err_mag**2)))
    max_err_mag = float(err_mag.max())
    print(f"Magnitue candidate: sqrt(2 * AVG)")
    print(f"  Exact (<1e-3):     {exact_mag}/{n} ({exact_mag/n*100:.2f}%)")
    print(f"  Tolerance (<0.1):  {tol_mag}/{n} ({tol_mag/n*100:.2f}%)")
    print(f"  MAE: {mae_mag:.4f}, RMSE: {rmse_mag:.4f}, Max Error: {max_err_mag:.4f}")

    cand_rad = df['Std'] * np.sqrt(2)
    err_rad = np.abs(df['Radius'] - cand_rad)
    exact_rad = (err_rad < 1e-3).sum()
    tol_rad = (err_rad < 0.1).sum()
    mae_rad = float(err_rad.mean())
    rmse_rad = float(np.sqrt(np.mean(err_rad**2)))
    max_err_rad = float(err_rad.max())
    print(f"Radius candidate: Std * sqrt(2)")
    print(f"  Exact (<1e-3):     {exact_rad}/{n} ({exact_rad/n*100:.2f}%)")
    print(f"  Tolerance (<0.1):  {tol_rad}/{n} ({tol_rad/n*100:.2f}%)")
    print(f"  MAE: {mae_rad:.4f}, RMSE: {rmse_rad:.4f}, Max Error: {max_err_rad:.4f}")

    # ----------------------------------------------------
    # Feature 3: Tot sum / AVG ratio
    # ----------------------------------------------------
    print("\n--- B. Tot sum & Tot size vs AVG Validation ---")
    ratio_totsum = df['Tot sum'] / df['AVG']
    exact_105 = (np.abs(ratio_totsum - 10.50) < 0.05).sum()
    print(f"Tot sum / AVG == 10.50 (+-0.05): {exact_105}/{n} ({exact_105/n*100:.2f}%)")
    print("Tot sum / AVG quantiles:\n", ratio_totsum.quantile([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]))

    diff_totsize = np.abs(df['Tot size'] - df['AVG'])
    exact_totsize = (diff_totsize < 1e-2).sum()
    print(f"Tot size == AVG (<0.01): {exact_totsize}/{n} ({exact_totsize/n*100:.2f}%)")
    print(f"Tot size vs AVG MAE: {diff_totsize.mean():.4f}, Max Diff: {diff_totsize.max():.4f}")

    # ----------------------------------------------------
    # Feature 4: Number & Weight
    # ----------------------------------------------------
    print("\n--- C. Number & Weight Distribution in 10,000 rows ---")
    print("Number value distribution:")
    print(df['Number'].value_counts(normalize=True).head(5) * 100)
    print("\nWeight value distribution:")
    print(df['Weight'].value_counts(normalize=True).head(5) * 100)

    # ----------------------------------------------------
    # Feature 5: Variance & Covariance
    # ----------------------------------------------------
    print("\n--- D. Variance & Covariance Bounds & Discrete Frequencies ---")
    print(f"Variance min={df['Variance'].min():.6f}, max={df['Variance'].max():.6f}")
    print("Variance top 10 values:")
    print(df['Variance'].value_counts().head(10))
    zero_std = (df['Std'] == 0)
    print(f"When Std == 0 ({zero_std.sum()}/{n}):")
    print(f"  Variance == 0: {(df.loc[zero_std, 'Variance'] == 0).sum()}/{zero_std.sum()}")
    print(f"  Covariance == 0: {(df.loc[zero_std, 'Covariance'] == 0).sum()}/{zero_std.sum()}")

    # ----------------------------------------------------
    # Feature 6: Rate, Srate, Drate
    # ----------------------------------------------------
    print("\n--- E. Rate, Srate, Drate Relationships ---")
    rate_eq_srate = (df['Rate'] == df['Srate']).sum()
    drate_zero = (df['Drate'] == 0.0).sum()
    flow_dur_zero = (df['flow_duration'] == 0.0).sum()
    print(f"Rate == Srate: {rate_eq_srate}/{n} ({rate_eq_srate/n*100:.2f}%)")
    print(f"Drate == 0.0:  {drate_zero}/{n} ({drate_zero/n*100:.2f}%)")
    print(f"flow_duration == 0.0: {flow_dur_zero}/{n} ({flow_dur_zero/n*100:.2f}%)")

    # ----------------------------------------------------
    # Feature 7: TCP Flag Counts
    # ----------------------------------------------------
    print("\n--- F. TCP Flag Counts (Window-Normalized Frequencies) ---")
    for f in ['ack_count', 'syn_count', 'fin_count', 'urg_count', 'rst_count']:
        sub_frac = ((df[f] > 0) & (df[f] < 1.0)).sum()
        sub_int = (df[f] >= 1.0).sum()
        sub_zero = (df[f] == 0.0).sum()
        print(f"{f:10s}: Zero={sub_zero} ({sub_zero/n*100:.1f}%), Fractions in (0,1)={sub_frac} ({sub_frac/n*100:.1f}%), Integer>=1={sub_int} ({sub_int/n*100:.1f}%)")


def run_iat_model_compatibility_experiment():
    print("\n" + "=" * 80)
    print("STEP 2: IAT MODEL COMPATIBILITY DIAGNOSTIC EXPERIMENT")
    print("=" * 80)
    
    detector = IoTIntrusionDetector()
    gate = OODAbstentionGate()
    
    test_csv = os.path.join(BASE_DIR, "Testing_data", "testing_data.csv")
    df = pd.read_csv(test_csv)
    
    # Stratified 1,000 samples (DDoS: 334, DoS: 333, Spoofing: 333)
    sample_indices = []
    for cls in ['DDoS', 'DoS', 'Spoofing']:
        idx = df[df['attack_class'] == cls].index[:334].tolist()
        sample_indices.extend(idx[:334 if cls=='DDoS' else 333])
        
    sub_df = df.iloc[sample_indices].copy().reset_index(drop=True)
    m = len(sub_df)
    print(f"Testing 1,000 stratified holdout samples across DDoS, DoS, Spoofing...")

    # Condition A: Original IAT (ground truth from dataset)
    print("\nEvaluating Condition A: Original Dataset IAT...")
    res_a = []
    gate_a = []
    for i in range(m):
        row = sub_df.iloc[i:i+1]
        pred = detector.predict(row)
        g = gate.evaluate(pred, row)
        res_a.append(pred)
        gate_a.append(g)

    # Condition B: True Live Physical IAT
    # Replace IAT with realistic line-rate packet intervals:
    # 10 microseconds for DDoS (high-rate flood: 1e-5 s)
    # 500 microseconds for DoS (5e-4 s)
    # 10 milliseconds for Spoofing (1e-2 s)
    print("Evaluating Condition B: Realistic Physical Live IAT (1e-5s - 1e-2s)...")
    sub_df_b = sub_df.copy()
    for i in range(m):
        cls = sub_df.iloc[i]['attack_class']
        if cls == 'DDoS':
            sub_df_b.loc[i, 'IAT'] = 1e-5
        elif cls == 'DoS':
            sub_df_b.loc[i, 'IAT'] = 5e-4
        else:
            sub_df_b.loc[i, 'IAT'] = 1e-2

    res_b = []
    gate_b = []
    for i in range(m):
        row = sub_df_b.iloc[i:i+1]
        pred = detector.predict(row)
        g = gate.evaluate(pred, row)
        res_b.append(pred)
        gate_b.append(g)

    # Condition C: Reconstructed Zero-Bounded IAT
    # If connection has single packet (flow_duration == 0), set IAT = 0.0
    # Otherwise set IAT = flow_duration / (packets - 1)
    print("Evaluating Condition C: Cleaned Normalized Zero-Bounded IAT...")
    sub_df_c = sub_df.copy()
    for i in range(m):
        fd = sub_df.iloc[i]['flow_duration']
        if fd == 0:
            sub_df_c.loc[i, 'IAT'] = 0.0
        else:
            sub_df_c.loc[i, 'IAT'] = min(fd / 10.0, 1.0)

    res_c = []
    gate_c = []
    for i in range(m):
        row = sub_df_c.iloc[i:i+1]
        pred = detector.predict(row)
        g = gate.evaluate(pred, row)
        res_c.append(pred)
        gate_c.append(g)

    # ----------------------------------------------------
    # Comparative Diagnostic Analysis
    # ----------------------------------------------------
    print("\n" + "-" * 70)
    print("IAT COMPATIBILITY COMPARATIVE SUMMARY (1,000 SAMPLES)")
    print("-" * 70)

    def analyze_results(cond_name, res_list, gate_list):
        preds = [r['predicted_class'] for r in res_list]
        confs = [r['confidence'] for r in res_list]
        abstained = [g['is_abstained'] for g in gate_list]
        decisions = [g['decision_state'] for g in gate_list]
        
        # Accuracy against sub_df ground truth
        gt = sub_df['attack_class'].tolist()
        acc = sum(p == g for p, g in zip(preds, gt)) / len(gt) * 100
        
        # Breakdown by attack class
        class_acc = {}
        for c in ['DDoS', 'DoS', 'Spoofing']:
            c_idx = [i for i, g in enumerate(gt) if g == c]
            c_acc = sum(preds[i] == c for i in c_idx) / len(c_idx) * 100
            class_acc[c] = c_acc
            
        print(f"\nCondition [{cond_name}]:")
        print(f"  Overall Raw Model Accuracy: {acc:.2f}%")
        print(f"  Class Accuracies -> DDoS: {class_acc['DDoS']:.2f}%, DoS: {class_acc['DoS']:.2f}%, Spoofing: {class_acc['Spoofing']:.2f}%")
        print(f"  Mean Confidence:            {np.mean(confs)*100:.2f}%")
        print(f"  Abstention Count:           {sum(abstained)}/1000 ({sum(abstained)/10:.1f}%)")
        print(f"  Decision States:            {pd.Series(decisions).value_counts().to_dict()}")

    analyze_results("Condition A: Original Dataset IAT (with Artifact)", res_a, gate_a)
    analyze_results("Condition B: True Physical Live IAT (1e-5s to 1e-2s)", res_b, gate_b)
    analyze_results("Condition C: Normalized Zero-Bounded IAT (0s to 1s)", res_c, gate_c)

    # Measure Class Flips between A and B
    flips_b = sum(r_a['predicted_class'] != r_b['predicted_class'] for r_a, r_b in zip(res_a, res_b))
    flips_c = sum(r_a['predicted_class'] != r_c['predicted_class'] for r_a, r_c in zip(res_a, res_c))
    print(f"\nModel Class Prediction Flips when substituting IAT:")
    print(f"  Condition A -> Condition B (Live physical IAT):   {flips_b}/1000 predictions flipped ({flips_b/10:.1f}%)")
    print(f"  Condition A -> Condition C (Zero-bounded IAT):    {flips_c}/1000 predictions flipped ({flips_c/10:.1f}%)")

    # Specific inspection of DDoS samples under Condition B
    ddos_indices = [i for i, g in enumerate(sub_df['attack_class']) if g == 'DDoS']
    ddos_flips_b = [i for i in ddos_indices if res_a[i]['predicted_class'] != res_b[i]['predicted_class']]
    print(f"  DDoS samples flipped from DDoS under physical IAT: {len(ddos_flips_b)}/{len(ddos_indices)}")
    if ddos_flips_b:
        sample_flip = ddos_flips_b[0]
        print(f"    Example flipped DDoS sample (idx {sample_flip}):")
        print(f"      Original IAT: {sub_df.iloc[sample_flip]['IAT']} -> Predicted: {res_a[sample_flip]['predicted_class']} ({res_a[sample_flip]['confidence']*100:.2f}%)")
        print(f"      Physical IAT: {sub_df_b.iloc[sample_flip]['IAT']} -> Predicted: {res_b[sample_flip]['predicted_class']} ({res_b[sample_flip]['confidence']*100:.2f}%)")


if __name__ == "__main__":
    run_parquet_10k_validation()
    run_iat_model_compatibility_experiment()

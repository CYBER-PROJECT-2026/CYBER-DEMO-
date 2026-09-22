"""
scratch/phase3_blocker_investigation.py
=======================================
Deep forensic validation for Phase 3:
1. Detailed statistical testing of all 16 blocker features across 10,000 official Parquet rows.
2. End-to-end model replay testing into frozen AEGIS (XGBoost + Temperature Calibrator + Mahalanobis OOD Gate).
3. Evaluates 4 conditions:
   - Condition A: Ground-truth official features
   - Condition B: Physical live IAT (0.0001s) with ground-truth other features
   - Condition C: DPKT epoch-emulated IAT with ground-truth other features
   - Condition D: Formula-reconstructed features (Magnitue, Radius, Tot sum, Tot size) + official IAT
   - Condition E: Full live extraction hypothesis (all reconstructed formulas + physical live IAT)
"""

import os
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Add project root to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.detection.detector import IoTIntrusionDetector
from backend.detection.abstention_gate import OODAbstentionGate
from backend.detection.ood_config import STATE_UNKNOWN_OR_REVIEW


def analyze_16_blockers(df):
    print("=" * 90)
    print("SECTION 1: STATISTICAL FORENSICS ON THE 16 BLOCKER FEATURES (N = 10,000 ROWS)")
    print("=" * 90)
    
    # 1. IAT
    iat = df['IAT']
    large_iat = (iat > 1e6).sum()
    zero_iat = (iat == 0.0).sum()
    print(f"\n1. IAT (Inter-Arrival Time):")
    print(f"   Min: {iat.min():.4f}, Max: {iat.max():.4f}, Median: {iat.median():.4f}, Mean: {iat.mean():.4f}")
    print(f"   Values > 1,000,000 (Epoch artifacts): {large_iat} / {len(df)} ({large_iat/len(df)*100:.2f}%)")
    print(f"   Values == 0.0: {zero_iat} / {len(df)} ({zero_iat/len(df)*100:.2f}%)")
    iat_by_class = df.groupby('attack_class')['IAT'].agg(['count', 'median', lambda x: (x > 1e6).mean() * 100])
    iat_by_class.columns = ['Count', 'Median_IAT', 'Pct_Epoch_Artifact']
    print("   Breakdown by Attack Class:")
    print(iat_by_class.to_string())

    # 2. Number
    num = df['Number']
    print(f"\n2. Number:")
    print(f"   Top 5 most frequent values:")
    print(num.value_counts().head(5).to_dict())
    
    # 3. Weight
    wt = df['Weight']
    print(f"\n3. Weight:")
    print(f"   Top 5 most frequent values:")
    print(wt.value_counts().head(5).to_dict())

    # 4. Variance
    var = df['Variance']
    std = df['Std']
    print(f"\n4. Variance:")
    print(f"   Min: {var.min():.4f}, Max: {var.max():.4f}, Median: {var.median():.4f}, Mean: {var.mean():.4f}")
    zero_var = (var == 0.0).sum()
    zero_std = (std == 0.0).sum()
    print(f"   Variance == 0: {zero_var} / {len(df)} ({zero_var/len(df)*100:.2f}%)")
    print(f"   Std == 0: {zero_std} / {len(df)} ({zero_std/len(df)*100:.2f}%)")
    non_zero_var = var[var > 0]
    print(f"   Top 5 non-zero Variance values:")
    print(non_zero_var.round(4).value_counts().head(5).to_dict())

    # 5. Covariance
    cov = df['Covariance']
    print(f"\n5. Covariance:")
    print(f"   Min: {cov.min():.4f}, Max: {cov.max():.4f}, Median: {cov.median():.4f}, Mean: {cov.mean():.4f}")
    zero_cov = (cov == 0.0).sum()
    print(f"   Covariance == 0: {zero_cov} / {len(df)} ({zero_cov/len(df)*100:.2f}%)")

    # 6 & 7. Rate & Srate
    rate = df['Rate']
    srate = df['Srate']
    drate = df['Drate']
    print(f"\n6 & 7. Rate and Srate:")
    exact_rate_srate = (rate == srate).sum()
    diff_rate_srate = (rate - srate).abs().max()
    print(f"   Exact matches (Rate == Srate): {exact_rate_srate} / {len(df)} ({exact_rate_srate/len(df)*100:.2f}%)")
    print(f"   Max absolute difference: {diff_rate_srate:.6f}")
    zero_drate = (drate == 0.0).sum()
    print(f"   Drate == 0.0: {zero_drate} / {len(df)} ({zero_drate/len(df)*100:.2f}%)")

    # 8. Tot sum
    tot_sum = df['Tot sum']
    avg = df['AVG']
    ratio_sum_avg = np.where(avg > 0, tot_sum / avg, np.nan)
    median_ratio = np.nanmedian(ratio_sum_avg)
    print(f"\n8. Tot sum:")
    print(f"   Median ratio (Tot sum / AVG): {median_ratio:.6f}")
    match_10_5 = (np.abs(ratio_sum_avg - 10.5) < 0.05).sum()
    valid_ratios = (~np.isnan(ratio_sum_avg)).sum()
    print(f"   Rows matching Tot sum == 10.5 * AVG (within 0.05): {match_10_5} / {valid_ratios} ({match_10_5/valid_ratios*100:.2f}%)")

    # 9. Tot size
    tot_size = df['Tot size']
    exact_size_avg = (np.abs(tot_size - avg) < 1e-4).sum()
    mae_size_avg = np.mean(np.abs(tot_size - avg))
    print(f"\n9. Tot size:")
    print(f"   Exact match with AVG (|Tot size - AVG| < 1e-4): {exact_size_avg} / {len(df)} ({exact_size_avg/len(df)*100:.2f}%)")
    print(f"   MAE(Tot size, AVG): {mae_size_avg:.6f}")

    # 10. Magnitue
    mag = df['Magnitue']
    cand_mag = np.sqrt(2.0 * avg)
    diff_mag = np.abs(mag - cand_mag)
    exact_mag = (diff_mag < 1e-4).sum()
    tol_mag = (diff_mag < 0.1).sum()
    print(f"\n10. Magnitue:")
    print(f"    Exact match (|Mag - sqrt(2*AVG)| < 1e-4): {exact_mag} / {len(df)} ({exact_mag/len(df)*100:.2f}%)")
    print(f"    Tolerance match (|Mag - sqrt(2*AVG)| < 0.1): {tol_mag} / {len(df)} ({tol_mag/len(df)*100:.2f}%)")
    print(f"    MAE: {np.mean(diff_mag):.6f}, RMSE: {np.sqrt(np.mean(diff_mag**2)):.6f}, Max Error: {np.max(diff_mag):.6f}")

    # 11. Radius
    rad = df['Radius']
    cand_rad = std * np.sqrt(2.0)
    diff_rad = np.abs(rad - cand_rad)
    exact_rad = (diff_rad < 1e-4).sum()
    tol_rad = (diff_rad < 0.1).sum()
    print(f"\n11. Radius:")
    print(f"    Exact match (|Radius - Std*sqrt(2)| < 1e-4): {exact_rad} / {len(df)} ({exact_rad/len(df)*100:.2f}%)")
    print(f"    Tolerance match (|Radius - Std*sqrt(2)| < 0.1): {tol_rad} / {len(df)} ({tol_rad/len(df)*100:.2f}%)")
    print(f"    MAE: {np.mean(diff_rad):.6f}, RMSE: {np.sqrt(np.mean(diff_rad**2)):.6f}, Max Error: {np.max(diff_rad):.6f}")

    # 12-16. TCP Flag Counts
    tcp_cols = ['ack_count', 'syn_count', 'fin_count', 'urg_count', 'rst_count']
    print(f"\n12-16. TCP Flag Counts:")
    for col in tcp_cols:
        val = df[col]
        zero_cnt = (val == 0.0).sum()
        max_val = val.max()
        print(f"    {col:10s} -> Min: {val.min():.2f}, Max: {max_val:.2f}, Zero %: {zero_cnt/len(df)*100:.2f}%, Top 3: {val.value_counts().head(3).to_dict()}")


def run_end_to_end_replay(df_full):
    print("\n" + "=" * 90)
    print("SECTION 2: END-TO-END FROZEN MODEL REPLAY EXPERIMENT (N = 1,000 SAMPLES)")
    print("=" * 90)
    
    detector = IoTIntrusionDetector()
    gate = OODAbstentionGate()
    
    features_46 = [
        'flow_duration', 'Header_Length', 'Protocol Type', 'Duration', 'Rate', 'Srate', 'Drate',
        'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number', 'ack_flag_number',
        'ece_flag_number', 'cwr_flag_number', 'ack_count', 'syn_count', 'fin_count', 'urg_count',
        'rst_count', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP', 'UDP',
        'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size',
        'IAT', 'Number', 'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight'
    ]
    
    sample_df = df_full.head(1000).copy()
    X_orig = sample_df[features_46].values.astype(np.float32)
    
    # Generate conditions:
    # Cond A: Original
    # Cond B: Physical live IAT (0.0001s)
    X_condB = X_orig.copy()
    iat_idx = features_46.index('IAT')
    X_condB[:, iat_idx] = 0.0001
    
    # Cond C: DPKT epoch-emulated IAT (83016449.0)
    X_condC = X_orig.copy()
    X_condC[:, iat_idx] = 83016449.0
    
    # Cond D: Formula-reconstructed (Magnitue, Radius, Tot sum, Tot size) + Original IAT
    X_condD = X_orig.copy()
    avg_idx = features_46.index('AVG')
    std_idx = features_46.index('Std')
    mag_idx = features_46.index('Magnitue')
    rad_idx = features_46.index('Radius')
    totsum_idx = features_46.index('Tot sum')
    totsize_idx = features_46.index('Tot size')
    
    X_condD[:, mag_idx] = np.sqrt(2.0 * X_condD[:, avg_idx])
    X_condD[:, rad_idx] = X_condD[:, std_idx] * np.sqrt(2.0)
    X_condD[:, totsum_idx] = 10.5 * X_condD[:, avg_idx]
    X_condD[:, totsize_idx] = X_condD[:, avg_idx]
    
    # Cond E: Full live extraction hypothesis (Formulas + Physical Live IAT)
    X_condE = X_condD.copy()
    X_condE[:, iat_idx] = 0.0001
    
    conditions = {
        'A: Ground-Truth Official Features': X_orig,
        'B: Physical Live IAT (0.0001s)': X_condB,
        'C: DPKT Epoch-Emulated IAT (8.3e7)': X_condC,
        'D: Reconstructed Formulas + Official IAT': X_condD,
        'E: Reconstructed Formulas + Physical Live IAT': X_condE,
    }
    
    import xgboost as xgb
    
    results = {}
    for name, X_data in conditions.items():
        # 1. XGBoost inference with logits for proper temperature calibration
        dmat = xgb.DMatrix(X_data, feature_names=features_46)
        logits = detector.booster.predict(dmat, output_margin=True)
        # 2. Temperature calibration on logits
        cal_probs = detector.calibrator.predict_proba(logits)
        preds = np.argmax(cal_probs, axis=1)
        pred_labels = detector.encoder.inverse_transform(preds)
        confs = np.max(cal_probs, axis=1)
        
        # 3. Vectorized Class-Conditional Mahalanobis check
        dists = np.zeros(len(X_data), dtype=np.float64)
        for cls_name in ["DDoS", "DoS", "Spoofing"]:
            cls_mask = (pred_labels == cls_name)
            if np.any(cls_mask):
                mu = gate.detector.params[cls_name]["mu"]
                VI = gate.detector.params[cls_name]["VI"]
                diff = X_data[cls_mask].astype(np.float64) - mu
                mah_sq = np.sum((diff @ VI) * diff, axis=1)
                dists[cls_mask] = np.sqrt(np.maximum(mah_sq, 0.0))
                
        ood_breach = dists > gate.mah_threshold
        conf_breach = confs < gate.conf_threshold
        abstentions = ood_breach | conf_breach
        
        results[name] = {
            'preds': pred_labels,
            'confs': confs,
            'cal_probs': cal_probs,
            'min_dists': dists,
            'abstention_rate': np.mean(abstentions) * 100,
            'ood_breach_rate': np.mean(ood_breach) * 100,
            'conf_breach_rate': np.mean(conf_breach) * 100,
            'mean_conf': np.mean(confs),
            'mean_dist': np.mean(dists),
            'class_dist': pd.Series(pred_labels).value_counts().to_dict()
        }

    # Print comparative results
    base_preds = results['A: Ground-Truth Official Features']['preds']
    print(f"\n{'Condition':<46} | {'Abstain %':<10} | {'OOD %':<8} | {'Mean Dist':<10} | {'Mean Conf':<10} | {'Class Agreement':<16}")
    print("-" * 115)
    for name, res in results.items():
        agree_pct = np.mean(res['preds'] == base_preds) * 100
        print(f"{name:<46} | {res['abstention_rate']:>8.2f}% | {res['ood_breach_rate']:>6.2f}% | {res['mean_dist']:>10.4f} | {res['mean_conf']:>10.4f} | {agree_pct:>14.2f}%")
        
    print("\nDetailed Class Predictions by Condition:")
    for name, res in results.items():
        print(f"  {name}: {res['class_dist']}")


def main():
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub/datasets--lacg030175--CIC-IoT-2023-neto-full")
    parquet_path = os.path.join(hf_cache, "snapshots", "bc3fc53b0537cf558f404cf379434b925d14233a", "random", "test-00000-of-00001.parquet")
    
    print(f"Reading 10,000 rows from {parquet_path}...")
    df_full = pq.read_table(parquet_path).to_pandas().head(10000)
    
    analyze_16_blockers(df_full)
    run_end_to_end_replay(df_full)
    print("\nPhase 3 forensic analysis complete.")


if __name__ == '__main__':
    main()

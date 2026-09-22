"""
scratch/forensic_feature_investigation.py
=========================================
Deep forensic analysis of CICIoT2023 feature relationships across
both Testing_data/testing_data.csv (5k holdout) and the official 9.3M test parquet.
"""

import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

def run_investigation():
    csv_path = "Testing_data/testing_data.csv"
    df = pd.read_csv(csv_path)
    n = len(df)
    print(f"=== Loaded {n} rows from {csv_path} ===")

    # ----------------------------------------------------
    # 1. Magnitue vs AVG investigation
    # ----------------------------------------------------
    print("\n--- 1. MAGNITUE INVESTIGATION ---")
    # Candidate A: sqrt(2 * AVG)
    cand_mag_a = np.sqrt(2 * df['AVG'])
    # Candidate B: sqrt(2 * AVG**2) = AVG * sqrt(2)
    cand_mag_b = df['AVG'] * np.sqrt(2)
    # Candidate C: sqrt(AVG**2 + Rate**2)
    cand_mag_c = np.sqrt(df['AVG']**2 + df['Rate']**2)
    
    diff_a = np.abs(df['Magnitue'] - cand_mag_a)
    diff_b = np.abs(df['Magnitue'] - cand_mag_b)
    diff_c = np.abs(df['Magnitue'] - cand_mag_c)

    print(f"Cand A sqrt(2 * AVG): exact matches (<1e-3) = {(diff_a < 1e-3).sum()}/{n} ({(diff_a < 1e-3).mean()*100:.2f}%), MAE={diff_a.mean():.4f}, max_err={diff_a.max():.4f}")
    print(f"Cand B AVG * sqrt(2): exact matches (<1e-3) = {(diff_b < 1e-3).sum()}/{n} ({(diff_b < 1e-3).mean()*100:.2f}%), MAE={diff_b.mean():.4f}, max_err={diff_b.max():.4f}")
    print(f"Cand C sqrt(AVG^2 + Rate^2): exact matches (<1e-3) = {(diff_c < 1e-3).sum()}/{n} ({(diff_c < 1e-3).mean()*100:.2f}%), MAE={diff_c.mean():.4f}")

    # Inspect cases where diff_a is small vs large
    close_a = diff_a < 0.1
    print(f"Cand A close (<0.1): {close_a.sum()}/{n} ({close_a.mean()*100:.2f}%)")
    if (~close_a).any():
        idx = np.where(~close_a)[0][:3]
        for i in idx:
            print(f"  Row {i}: AVG={df['AVG'].iloc[i]}, Magnitue={df['Magnitue'].iloc[i]}, sqrt(2*AVG)={cand_mag_a.iloc[i]:.4f}, diff={diff_a.iloc[i]:.4f}")

    # ----------------------------------------------------
    # 2. Radius vs Std investigation
    # ----------------------------------------------------
    print("\n--- 2. RADIUS INVESTIGATION ---")
    # Candidate A: sqrt(2 * Std**2) = Std * sqrt(2)
    cand_rad_a = df['Std'] * np.sqrt(2)
    # Candidate B: sqrt(2 * Std)
    cand_rad_b = np.sqrt(2 * df['Std'])
    # Candidate C: Std * sqrt(2) when Std > 0, else 0
    diff_rad_a = np.abs(df['Radius'] - cand_rad_a)
    diff_rad_b = np.abs(df['Radius'] - cand_rad_b)
    print(f"Cand A Std * sqrt(2): exact (<1e-3) = {(diff_rad_a < 1e-3).sum()}/{n} ({(diff_rad_a < 1e-3).mean()*100:.2f}%), MAE={diff_rad_a.mean():.4f}, max_err={diff_rad_a.max():.4f}")
    print(f"Cand B sqrt(2 * Std): exact (<1e-3) = {(diff_rad_b < 1e-3).sum()}/{n} ({(diff_rad_b < 1e-3).mean()*100:.2f}%), MAE={diff_rad_b.mean():.4f}")

    # Check zero-std records
    zero_std = df['Std'] == 0
    print(f"Std == 0 records: {zero_std.sum()}/{n}; Radius == 0 in those: {(df.loc[zero_std, 'Radius'] == 0).sum()}")

    # ----------------------------------------------------
    # 3. Variance & Covariance bounds and distributions
    # ----------------------------------------------------
    print("\n--- 3. VARIANCE & COVARIANCE INVESTIGATION ---")
    print(f"Variance: min={df['Variance'].min():.6f}, max={df['Variance'].max():.6f}, mean={df['Variance'].mean():.6f}")
    print(f"Variance value counts top 10:\n{df['Variance'].value_counts().head(10)}")
    print(f"Covariance: min={df['Covariance'].min():.6f}, max={df['Covariance'].max():.6f}, mean={df['Covariance'].mean():.6f}")
    print(f"Covariance == 0 count: {(df['Covariance'] == 0).sum()}/{n}")

    # ----------------------------------------------------
    # 4. Number & Weight investigation
    # ----------------------------------------------------
    print("\n--- 4. NUMBER & WEIGHT INVESTIGATION ---")
    print("Number value counts:")
    print(df['Number'].value_counts().head(10))
    print("Weight value counts:")
    print(df['Weight'].value_counts().head(10))
    print("Cross-tabulation of Number vs Weight:")
    print(pd.crosstab(df['Number'], df['Weight']).head(10))

    # ----------------------------------------------------
    # 5. Rate, Srate, Drate relationship
    # ----------------------------------------------------
    print("\n--- 5. RATE, SRATE, DRATE INVESTIGATION ---")
    srate_eq_rate = (df['Rate'] == df['Srate']).sum()
    drate_zero = (df['Drate'] == 0.0).sum()
    print(f"Rate == Srate: {srate_eq_rate}/{n} ({srate_eq_rate/n*100:.2f}%)")
    print(f"Drate == 0.0: {drate_zero}/{n} ({drate_zero/n*100:.2f}%)")
    rate_zero = (df['Rate'] == 0.0).sum()
    print(f"Rate == 0.0: {rate_zero}/{n} ({rate_zero/n*100:.2f}%)")
    flow_dur_zero = (df['flow_duration'] == 0.0).sum()
    print(f"flow_duration == 0.0: {flow_dur_zero}/{n} ({flow_dur_zero/n*100:.2f}%)")
    
    # Check rate when flow_duration == 0 vs > 0
    fd_zero_rate_nonzero = ((df['flow_duration'] == 0.0) & (df['Rate'] > 0.0)).sum()
    print(f"flow_duration == 0.0 but Rate > 0.0: {fd_zero_rate_nonzero}/{n}")

    # ----------------------------------------------------
    # 6. Duration (IP TTL) investigation
    # ----------------------------------------------------
    print("\n--- 6. DURATION (IP TTL) INVESTIGATION ---")
    print(f"Duration min={df['Duration'].min()}, max={df['Duration'].max()}, mean={df['Duration'].mean():.2f}")
    print("Duration top 15 values:")
    print(df['Duration'].value_counts().head(15))

    # ----------------------------------------------------
    # 7. Tot sum, Tot size, AVG, Min, Max relationships
    # ----------------------------------------------------
    print("\n--- 7. PACKET SIZE STATS INVESTIGATION ---")
    tot_size_eq_avg = np.isclose(df['Tot size'], df['AVG'], atol=1e-2).sum()
    print(f"Tot size == AVG: {tot_size_eq_avg}/{n} ({tot_size_eq_avg/n*100:.2f}%)")
    min_eq_max = (df['Min'] == df['Max']).sum()
    print(f"Min == Max (uniform packet length): {min_eq_max}/{n} ({min_eq_max/n*100:.2f}%)")
    
    # When Min == Max:
    if min_eq_max > 0:
        sub = df[df['Min'] == df['Max']]
        print(f"  When Min == Max:")
        print(f"    AVG == Min: {np.isclose(sub['AVG'], sub['Min'], atol=1e-4).sum()}/{len(sub)}")
        print(f"    Std == 0: {(sub['Std'] == 0).sum()}/{len(sub)}")
        print(f"    Tot size == Min: {np.isclose(sub['Tot size'], sub['Min'], atol=1e-4).sum()}/{len(sub)}")
        print(f"    Radius == 0: {(sub['Radius'] == 0).sum()}/{len(sub)}")
        print(f"    Covariance == 0: {(sub['Covariance'] == 0).sum()}/{len(sub)}")
        print(f"    Variance == 0: {(sub['Variance'] == 0).sum()}/{len(sub)}")

    # Tot sum vs AVG and window count:
    # Does Tot sum / AVG equal an integer?
    ratio = df['Tot sum'] / df['AVG']
    print(f"Tot sum / AVG: min={ratio.min():.2f}, max={ratio.max():.2f}, mean={ratio.mean():.2f}")
    print("Tot sum / AVG rounded top 10:")
    print(ratio.round(2).value_counts().head(10))

    # ----------------------------------------------------
    # 8. IAT deep investigation
    # ----------------------------------------------------
    print("\n--- 8. IAT DEEP INVESTIGATION ---")
    iat_micro = (df['IAT'] < 1000.0).sum()
    iat_large = (df['IAT'] > 1e6).sum()
    print(f"IAT < 1,000s: {iat_micro}/{n} ({iat_micro/n*100:.2f}%)")
    print(f"IAT > 1,000,000: {iat_large}/{n} ({iat_large/n*100:.2f}%)")
    print("Large IAT quantiles:")
    print(df[df['IAT'] > 1e6]['IAT'].quantile([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]))
    print("Small IAT quantiles:")
    print(df[df['IAT'] < 1000.0]['IAT'].quantile([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]))

    # Classwise breakdown of large IAT
    print("\nClasswise distribution of large IAT:")
    print(pd.crosstab(df['attack_class'], df['IAT'] > 1e6, margins=True))

if __name__ == "__main__":
    run_investigation()

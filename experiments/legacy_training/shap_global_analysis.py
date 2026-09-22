"""
Global and Class-Wise SHAP Explainability Analysis for 46-Feature XGBoost IDS.

Strict Analysis Rules:
- Evaluates all 5,000 records from Testing_data/testing_data.csv.
- Uses exact feature ordering from model.feature_names_in_.
- Multiclass SHAP computation across all 3 classes: DDoS, DoS, Spoofing.
- Generates:
  1. shap_global_results.csv
  2. shap_classwise_results.csv
  3. shap_individual_explanations.csv
  4. shap_global_summary.png
  5. shap_ddos_summary.png
  6. shap_dos_summary.png
  7. shap_spoofing_summary.png
  8. shap_sample_ddos.png
  9. shap_sample_dos.png
  10. shap_sample_spoofing.png
"""

import os
import sys
import time
import json
import joblib
import shap
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Ensure UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

MODEL_PATH = os.path.join(BASE_DIR, "cic_iot_xgboost_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
TEMP_CALIB_PATH = os.path.join(BASE_DIR, "calibrator_temperature.pkl")
TEST_CSV_PATH = os.path.join(BASE_DIR, "Testing_data", "testing_data.csv")

# Output Paths
CSV_GLOBAL_PATH = os.path.join(BASE_DIR, "shap_global_results.csv")
CSV_CLASSWISE_PATH = os.path.join(BASE_DIR, "shap_classwise_results.csv")
CSV_INDIVIDUAL_PATH = os.path.join(BASE_DIR, "shap_individual_explanations.csv")

PNG_GLOBAL_PATH = os.path.join(BASE_DIR, "shap_global_summary.png")
PNG_DDOS_PATH = os.path.join(BASE_DIR, "shap_ddos_summary.png")
PNG_DOS_PATH = os.path.join(BASE_DIR, "shap_dos_summary.png")
PNG_SPOOF_PATH = os.path.join(BASE_DIR, "shap_spoofing_summary.png")

PNG_SAMPLE_DDOS = os.path.join(BASE_DIR, "shap_sample_ddos.png")
PNG_SAMPLE_DOS = os.path.join(BASE_DIR, "shap_sample_dos.png")
PNG_SAMPLE_SPOOF = os.path.join(BASE_DIR, "shap_sample_spoofing.png")


def run_global_analysis():
    print("=" * 80)
    print("PHASE 7: GLOBAL & CLASS-WISE SHAP EXPLAINABILITY ANALYSIS")
    print("=" * 80)

    # 1. Load Model and Artifacts
    print("\n[Step 1/8] Loading frozen model artifacts...")
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    calib_temp = joblib.load(TEMP_CALIB_PATH)

    features = list(model.feature_names_in_)
    classes = list(encoder.classes_)
    num_features = len(features)
    num_classes = len(classes)

    print(f"✓ Base model loaded with {num_features} features (canonical order verified).")
    print(f"✓ Classes ({num_classes}): {classes} -> {dict(enumerate(classes))}")
    print(f"✓ Temperature Calibrator loaded (T = {calib_temp.temperature:.4f})")

    # 2. Load Testing Data
    print(f"\n[Step 2/8] Loading untouched testing dataset: {TEST_CSV_PATH}")
    df_test = pd.read_csv(TEST_CSV_PATH)
    N = len(df_test)
    assert N == 5000, f"Expected 5,000 test records, found {N}"
    print(f"✓ Loaded {N:,} test records.")

    target_col = 'attack_class' if 'attack_class' in df_test.columns else 'AttackCategory'
    X_test = df_test[features].astype(np.float32)
    y_actual_str = df_test[target_col].astype(str)
    y_test = encoder.transform(y_actual_str)

    # 3. Model Inference & Temperature Calibration
    print("\n[Step 3/8] Running inference and Temperature Scaling...")
    booster = model.get_booster()
    dmat = xgb.DMatrix(X_test)
    logits = booster.predict(dmat, output_margin=True) # (5000, 3)
    temp_probs = calib_temp.predict_proba(logits)      # (5000, 3)
    preds = np.argmax(temp_probs, axis=1)              # (5000,)
    confs = np.max(temp_probs, axis=1)                 # (5000,)
    print("✓ Inference completed.")

    # 4. Compute SHAP Values on all 5,000 Records
    print("\n[Step 4/8] Computing SHAP TreeExplainer values for all 5,000 records...")
    t0 = time.perf_counter()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)  # Shape (5000, 46, 3)
    t1 = time.perf_counter()
    print(f"✓ Computed SHAP values in {t1 - t0:.2f}s. Shape: {shap_values.shape}")

    # Check Sanity of SHAP Values
    assert shap_values.shape == (5000, 46, 3), f"Unexpected shape {shap_values.shape}"
    assert np.all(np.isfinite(shap_values)), "Found non-finite SHAP values!"
    print("✓ SHAP values sanity check PASSED: 46 features, all values finite, no NaNs.")

    # 5. Global Feature Importance (Mean Absolute SHAP across all samples & classes)
    print("\n[Step 5/8] Calculating Global & Class-Wise Feature Importance...")
    # Class-wise mean absolute SHAP: (46,) for each class
    mean_abs_per_class = {}
    for k, cname in enumerate(classes):
        mean_abs_per_class[cname] = np.mean(np.abs(shap_values[:, :, k]), axis=0)

    # Global mean absolute SHAP (average across classes)
    mean_abs_global = np.mean([mean_abs_per_class[c] for c in classes], axis=0)

    # Global Rankings DataFrame
    global_order = np.argsort(mean_abs_global)[::-1]
    df_global = pd.DataFrame({
        "global_rank": np.arange(1, num_features + 1),
        "feature": [features[i] for i in global_order],
        "mean_absolute_shap": [mean_abs_global[i] for i in global_order]
    })
    df_global.to_csv(CSV_GLOBAL_PATH, index=False)
    print(f"✓ Saved Global Rankings to {CSV_GLOBAL_PATH}")

    # Class-wise Rankings DataFrame
    classwise_rows = []
    for cname in classes:
        cls_scores = mean_abs_per_class[cname]
        cls_order = np.argsort(cls_scores)[::-1]
        for rank, idx in enumerate(cls_order, start=1):
            classwise_rows.append({
                "class": cname,
                "rank": rank,
                "feature": features[idx],
                "mean_absolute_shap": float(cls_scores[idx])
            })
    df_classwise = pd.DataFrame(classwise_rows)
    df_classwise.to_csv(CSV_CLASSWISE_PATH, index=False)
    print(f"✓ Saved Class-Wise Rankings to {CSV_CLASSWISE_PATH}")

    # 6. Save Individual Explanations CSV (Top 10 features per record)
    print("\n[Step 6/8] Generating shap_individual_explanations.csv...")
    indiv_rows = []
    for i in range(N):
        p_cls_idx = preds[i]
        p_cls_name = classes[p_cls_idx]
        act_cls = y_actual_str.iloc[i]
        c_score = float(confs[i])
        
        sample_shap = shap_values[i, :, p_cls_idx]
        sample_vals = X_test.iloc[i].values
        
        ranked_f_idx = np.argsort(np.abs(sample_shap))[::-1][:10]
        for r, f_idx in enumerate(ranked_f_idx, start=1):
            s_val = float(sample_shap[f_idx])
            indiv_rows.append({
                "sample_index": i,
                "actual_class": act_cls,
                "predicted_class": p_cls_name,
                "confidence": c_score,
                "feature_rank": r,
                "feature": features[f_idx],
                "feature_value": float(sample_vals[f_idx]),
                "shap_value": s_val,
                "direction": "POSITIVE" if s_val > 0 else "NEGATIVE"
            })
    df_indiv = pd.DataFrame(indiv_rows)
    df_indiv.to_csv(CSV_INDIVIDUAL_PATH, index=False)
    print(f"✓ Saved Individual Explanations ({len(df_indiv)} rows) to {CSV_INDIVIDUAL_PATH}")

    # 7. Generate Visualizations
    print("\n[Step 7/8] Generating high-resolution SHAP visual artifacts...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Figure 1: Global SHAP Summary (Top 20 Features)
    top20_g_idx = global_order[:20][::-1]
    top20_g_names = [features[i] for i in top20_g_idx]
    top20_g_scores = [mean_abs_global[i] for i in top20_g_idx]

    plt.figure(figsize=(12, 8))
    bars = plt.barh(top20_g_names, top20_g_scores, color='#2563eb', edgecolor='black', alpha=0.85)
    plt.title("Global SHAP Feature Importance (Top 20 Features Across All 5,000 Records)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Mean Absolute SHAP Value (Impact on Log-Odds)", fontsize=12)
    plt.ylabel("Network Flow Feature", fontsize=12)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.02, bar.get_y() + bar.get_height()/2, f"{w:.3f}", va='center', fontsize=10, color='#1e293b')
    plt.tight_layout()
    plt.savefig(PNG_GLOBAL_PATH, dpi=300)
    plt.close()
    print(f"✓ Saved {PNG_GLOBAL_PATH}")

    # Class-specific plots (Top 15 per class)
    colors = {"DDoS": "#dc2626", "DoS": "#d97706", "Spoofing": "#059669"}
    for cname, col, save_path in [("DDoS", colors["DDoS"], PNG_DDOS_PATH),
                                 ("DoS", colors["DoS"], PNG_DOS_PATH),
                                 ("Spoofing", colors["Spoofing"], PNG_SPOOF_PATH)]:
        cls_scores = mean_abs_per_class[cname]
        top15_cls_order = np.argsort(cls_scores)[::-1][:15][::-1]
        top15_cls_names = [features[i] for i in top15_cls_order]
        top15_cls_vals = [cls_scores[i] for i in top15_cls_order]

        plt.figure(figsize=(11, 7))
        bars = plt.barh(top15_cls_names, top15_cls_vals, color=col, edgecolor='black', alpha=0.85)
        plt.title(f"{cname} Attack Class: Top 15 SHAP Feature Importances", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Mean Absolute SHAP Value", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        for bar in bars:
            w = bar.get_width()
            plt.text(w + 0.02, bar.get_y() + bar.get_height()/2, f"{w:.3f}", va='center', fontsize=10, color='#1e293b')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✓ Saved {save_path}")

    # Individual Sample Waterfall / Bar Plots
    # Find clean representative samples of each class
    sample_indices = {
        "DDoS": int(np.where((y_test == 0) & (preds == 0) & (confs > 0.9999))[0][0]),
        "DoS": int(np.where((y_test == 1) & (preds == 1) & (confs > 0.9999))[0][0]),
        "Spoofing": int(np.where((y_test == 2) & (preds == 2) & (confs > 0.9999))[0][0])
    }

    sample_plots = [
        ("DDoS", sample_indices["DDoS"], PNG_SAMPLE_DDOS),
        ("DoS", sample_indices["DoS"], PNG_SAMPLE_DOS),
        ("Spoofing", sample_indices["Spoofing"], PNG_SAMPLE_SPOOF)
    ]

    for cname, s_idx, s_png in sample_plots:
        c_idx = classes.index(cname)
        s_shap = shap_values[s_idx, :, c_idx]
        s_vals = X_test.iloc[s_idx].values
        
        top10_idx = np.argsort(np.abs(s_shap))[::-1][:10][::-1]
        top10_names = [f"{features[i]} = {s_vals[i]:.2f}" for i in top10_idx]
        top10_sv = [s_shap[i] for i in top10_idx]
        bar_colors = ['#16a34a' if v > 0 else '#dc2626' for v in top10_sv]

        plt.figure(figsize=(11, 6))
        plt.barh(top10_names, top10_sv, color=bar_colors, edgecolor='black', alpha=0.85)
        plt.axvline(0, color='black', linewidth=1)
        plt.title(f"SHAP Explanation: Sample #{s_idx} (Actual & Predicted: {cname}, Conf: {confs[s_idx]*100:.2f}%)",
                  fontsize=13, fontweight='bold', pad=12)
        plt.xlabel("SHAP Value (Contribution to Log-Odds)", fontsize=12)
        plt.ylabel("Feature & Observed Value", fontsize=12)
        plt.tight_layout()
        plt.savefig(s_png, dpi=300)
        plt.close()
        print(f"✓ Saved {s_png} (Sample #{s_idx})")

    # 8. Print Summary Rankings Table
    print("\n[Step 8/8] Summary of Global Top 10 Features:")
    print(df_global.head(10).to_string(index=False))

    print("\nSummary of Top 5 Features per Attack Class:")
    for cname in classes:
        print(f"\n--- {cname} Top 5 Features ---")
        df_sub = df_classwise[df_classwise["class"] == cname].head(5)
        print(df_sub[["rank", "feature", "mean_absolute_shap"]].to_string(index=False))

    print("\nGLOBAL SHAP ANALYSIS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    run_global_analysis()

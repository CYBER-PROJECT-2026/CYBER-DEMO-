"""
AEGIS-IoT Phase 13: OOD + Unknown-Input + Abstention Design & Validation Audit
================================================================================
PRODUCTION FREEZE: Model, schema, calibration, RAG, inference are all READ-ONLY.
NO retraining. NO synthetic OOD data. NO test-set threshold tuning.

Version 2: Skip kNN recomputation (uses prior forensic audit distance statistics);
all Mahalanobis computations remain. kNN summary stats from forensic_near_duplicate_results.json.
"""

import os, sys, json, time, warnings
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from offline_feature_validator import CANONICAL_46_FEATURES

MODEL_PATH   = os.path.join(BASE_DIR, "cic_iot_xgboost_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
CALIB_PATH   = os.path.join(BASE_DIR, "calibrator_temperature.pkl")
TRAIN_CSV    = os.path.join(BASE_DIR, "CIC_IoT_2023_DDoS_DoS_Spoofing_1.5M.csv")
TEST_CSV     = os.path.join(BASE_DIR, "Testing_data", "testing_data.csv")
CALIB_CSV    = os.path.join(BASE_DIR, "calibration_data.csv")
PREV_JSON    = os.path.join(BASE_DIR, "forensic_near_duplicate_results.json")
OUT_JSON     = os.path.join(BASE_DIR, "phase13_ood_abstention_results.json")
OUT_CSV      = os.path.join(BASE_DIR, "phase13_ood_candidate_metrics.csv")
OUT_MD       = os.path.join(BASE_DIR, "phase13_ood_abstention_design_audit.md")

print("=" * 75)
print("AEGIS-IoT PHASE 13: OOD + ABSTENTION AUDIT  (v2 — Mahalanobis + Conf)")
print("=" * 75)
print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# ===== LOAD ARTIFACTS =====
print("\n[1] Loading artifacts...")
model         = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)
calibrator    = joblib.load(CALIB_PATH)
classes       = list(label_encoder.classes_)
booster       = model.get_booster()

# ===== LOAD TEST SET =====
print("[2] Loading test set...")
test_df    = pd.read_csv(TEST_CSV)
tc         = "attack_class" if "attack_class" in test_df.columns else "label"
y_test_str = test_df[tc].astype(str)
X_test_raw = test_df[CANONICAL_46_FEATURES].astype(np.float32)
N_TEST     = len(X_test_raw)
print(f"    Test: {N_TEST} records")

# ===== LOAD CALIBRATION SET =====
print("[3] Loading calibration set...")
calib_df    = pd.read_csv(CALIB_CSV)
cc          = "attack_class" if "attack_class" in calib_df.columns else "label"
X_calib_raw = calib_df[CANONICAL_46_FEATURES].astype(np.float32)
y_calib_str = calib_df[cc].astype(str) if cc in calib_df.columns else None
N_CALIB     = len(X_calib_raw)
print(f"    Calibration: {N_CALIB} records")

# ===== LOAD TRAINING SET =====
print("[4] Loading training set (1.5M)...")
train_df    = pd.read_csv(TRAIN_CSV)
tc2         = "attack_class" if "attack_class" in train_df.columns else "label"
y_train_str = train_df[tc2].astype(str)
X_train_raw = train_df[CANONICAL_46_FEATURES].astype(np.float32)
N_TRAIN     = len(X_train_raw)
print(f"    Train: {N_TRAIN} records")

# ===== BASELINE INFERENCE =====
print("[5] Baseline inference on test set...")
raw_probs        = model.predict_proba(X_test_raw)
dmat             = xgb.DMatrix(X_test_raw)
logits           = booster.predict(dmat, output_margin=True)
calib_probs      = calibrator.predict_proba(logits)
pred_idx         = np.argmax(calib_probs, axis=1)
y_pred_labels    = label_encoder.inverse_transform(pred_idx)
calib_top1       = np.max(calib_probs, axis=1)
sorted_probs     = np.sort(calib_probs, axis=1)[:, ::-1]
top2_margin      = sorted_probs[:, 0] - sorted_probs[:, 1]
correct_mask     = (y_pred_labels == y_test_str.values)
errors_mask      = ~correct_mask
n_errors         = int(errors_mask.sum())
accuracy_base    = float(correct_mask.mean())
print(f"    Accuracy={accuracy_base*100:.4f}%, Errors={n_errors}")

# ===== CALIBRATION SET INFERENCE =====
print("[6] Calibration set inference...")
raw_p_c      = model.predict_proba(X_calib_raw)
dmat_c       = xgb.DMatrix(X_calib_raw)
logits_c     = booster.predict(dmat_c, output_margin=True)
calib_p_c    = calibrator.predict_proba(logits_c)
pred_idx_c   = np.argmax(calib_p_c, axis=1)
y_pred_c     = label_encoder.inverse_transform(pred_idx_c)
calib_top1_c = np.max(calib_p_c, axis=1)
if y_calib_str is not None:
    calib_acc = float((y_pred_c == y_calib_str.values).mean())
    print(f"    Calibration accuracy: {calib_acc*100:.4f}%")
else:
    calib_acc = None

# ===== MAHALANOBIS =====
print("[7] Computing GLOBAL Mahalanobis distances (train→test)...")
X_train_np = X_train_raw.values
X_test_np  = X_test_raw.values
X_calib_np = X_calib_raw.values

mu_global = X_train_np.mean(axis=0)
print("    Computing 46x46 covariance matrix...")
cov = np.cov(X_train_np.astype(np.float64).T)
cov_reg = cov + 1e-6 * np.eye(46, dtype=np.float64)
print("    Inverting covariance matrix...")
try:
    VI = np.linalg.inv(cov_reg)
except np.linalg.LinAlgError:
    VI = np.linalg.pinv(cov_reg)

def mah_dist_from_mu(X, mu, VI):
    diff = X.astype(np.float64) - mu
    mah_sq = np.einsum('ij,jk,ik->i', diff, VI, diff)
    return np.sqrt(np.maximum(mah_sq, 0.0))

print("    Computing test set global Mah distances...")
global_mah_test  = mah_dist_from_mu(X_test_np,  mu_global, VI)
print("    Computing calib set global Mah distances...")
global_mah_calib = mah_dist_from_mu(X_calib_np, mu_global, VI)
print(f"    Test Mah: min={global_mah_test.min():.4f}, max={global_mah_test.max():.4f}, mean={global_mah_test.mean():.4f}")
print(f"    Calib Mah: min={global_mah_calib.min():.4f}, max={global_mah_calib.max():.4f}, mean={global_mah_calib.mean():.4f}")

print("[8] Computing CLASS-CONDITIONAL Mahalanobis distances...")
cc_mah_test = np.zeros(N_TEST, dtype=np.float64)
class_mah_params = {}
for cls in classes:
    train_mask  = (y_train_str.values == cls)
    X_cls       = X_train_np[train_mask].astype(np.float64)
    mu_cls      = X_cls.mean(axis=0)
    cov_cls     = np.cov(X_cls.T) + 1e-6 * np.eye(46, dtype=np.float64)
    try:
        VI_cls = np.linalg.inv(cov_cls)
    except:
        VI_cls = np.linalg.pinv(cov_cls)
    class_mah_params[cls] = {"mu": mu_cls, "VI": VI_cls, "n_train": int(train_mask.sum())}

    # Compute for predicted = this class
    test_pred_mask = (y_pred_labels == cls)
    idx_cls = np.where(test_pred_mask)[0]
    if len(idx_cls) > 0:
        diff = X_test_np[idx_cls].astype(np.float64) - mu_cls
        mah_sq = np.einsum('ij,jk,ik->i', diff, VI_cls, diff)
        cc_mah_test[idx_cls] = np.sqrt(np.maximum(mah_sq, 0.0))
    print(f"    {cls}: n_train={train_mask.sum()}, n_test_predicted={len(idx_cls)}")

print(f"    CC Mah: min={cc_mah_test.min():.4f}, max={cc_mah_test.max():.4f}, mean={cc_mah_test.mean():.4f}")

# ===== kNN DISTANCE: LOAD FROM PRIOR AUDIT =====
print("[9] Loading kNN min-distance statistics from prior forensic audit...")
prior = {}
if os.path.exists(PREV_JSON):
    with open(PREV_JSON) as f:
        prior = json.load(f)

# From forensic_near_duplicate_evaluation.md (verified empirical results):
knn_stats_from_prior = {
    "source": "forensic_near_duplicate_results.json + phase11_12_distribution_ood_results.json",
    "note"  : "Min standardized Euclidean distances from full forensic audit (Phase 10 & 11-12)",
    "n_test": 5000,
    "near_dup_d_lt_1e6"   : {"n": 1075, "pct": 21.50},
    "near_dup_d_lt_0.1"   : {"n": 3438, "pct": 68.76},
    "near_dup_d_lt_0.5"   : {"n": 4336, "pct": 86.72},
    "min"    : 0.0,
    "median" : 0.002760,   # P50 from phase11_12 report
    "mean"   : 0.173498,   # from phase11_12 report
    "p90"    : 0.6150,     # from phase11_12 report
    "p95"    : 0.8501,     # from phase11_12 report
    "p99"    : 1.4694,     # from phase11_12 report
    "max"    : 4.5660,     # from phase11_12 report
    "error_sample_657_min_dist": 0.00000118,  # from forensic report
    "distance_vs_accuracy": {
        "lt_1e6"        : {"n": 1075, "errors": 0, "accuracy_pct": 100.0, "mean_conf_pct": 99.9999},
        "1e6_to_0.01"   : {"n": 1675, "errors": 1, "accuracy_pct": 99.9403, "mean_conf_pct": 99.9818},
        "0.01_to_0.1"   : {"n": 688,  "errors": 0, "accuracy_pct": 100.0,   "mean_conf_pct": 99.9998},
        "0.1_to_0.5"    : {"n": 898,  "errors": 0, "accuracy_pct": 100.0,   "mean_conf_pct": 99.9997},
        "gte_0.5"       : {"n": 664,  "errors": 0, "accuracy_pct": 100.0,   "mean_conf_pct": 99.9902},
    },
}
print("    kNN stats loaded from prior forensic audit.")

# ===== BUILD PER-SAMPLE TABLE =====
print("[10] Building per-sample analysis table...")
results_table = pd.DataFrame({
    "sample_idx"      : np.arange(N_TEST),
    "actual_class"    : y_test_str.values,
    "predicted_class" : y_pred_labels,
    "correct"         : correct_mask,
    "calib_top1_prob" : calib_top1,
    "top2_prob"       : sorted_probs[:, 1],
    "top1_top2_margin": top2_margin,
    "global_mah_dist" : global_mah_test,
    "cc_mah_dist"     : cc_mah_test,
    "ddos_prob"       : calib_probs[:, 0],
    "dos_prob"        : calib_probs[:, 1],
    "spoofing_prob"   : calib_probs[:, 2],
})

# ===== CONFIDENCE THRESHOLD ANALYSIS (PART 6) =====
print("[11] Part 6: Confidence threshold analysis...")
conf_thresholds = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
conf_rows = []
for thr in conf_thresholds:
    acc_mask = calib_top1 >= thr
    abs_mask = ~acc_mask
    n_acc    = int(acc_mask.sum())
    n_abs    = int(abs_mask.sum())
    coverage = n_acc / N_TEST
    err_acc  = int((errors_mask & acc_mask).sum())
    err_abs  = int((errors_mask & abs_mask).sum())
    cor_abs  = int((correct_mask & abs_mask).sum())
    acc_acc  = (n_acc - err_acc) / n_acc if n_acc > 0 else 1.0
    conf_rows.append({
        "threshold"          : thr,
        "total_N"            : N_TEST,
        "n_accepted"         : n_acc,
        "n_abstained"        : n_abs,
        "coverage_pct"       : round(coverage * 100, 4),
        "errors_in_accepted" : err_acc,
        "accuracy_accepted_pct": round(acc_acc * 100, 6),
        "error_rate_pct"     : round((1 - acc_acc) * 100, 6),
        "errors_to_human"    : err_abs,
        "correct_abstained"  : cor_abs,
        "correct_abstained_pct": round(cor_abs / N_TEST * 100, 4),
    })
conf_df = pd.DataFrame(conf_rows)

# ===== CONFIDENCE + MAHALANOBIS COMBINATION (PART 7) =====
print("[12] Part 7: Confidence + Mahalanobis combination analysis...")
HIGH_CONF_THR = 0.90
mah_p90 = float(np.percentile(global_mah_test, 90))

high_conf   = calib_top1 >= HIGH_CONF_THR
low_conf    = ~high_conf
close_mah   = global_mah_test <= mah_p90
far_mah     = ~close_mah

combo_labels = {
    "HIGH_conf_CLOSE_mah (likely in-dist, confident)" : (high_conf & close_mah),
    "HIGH_conf_FAR_mah (confident but outlier)"       : (high_conf & far_mah),
    "LOW_conf_CLOSE_mah (uncertain despite proximity)": (low_conf  & close_mah),
    "LOW_conf_FAR_mah (uncertain AND distant)"        : (low_conf  & far_mah),
}

combo_results = {}
for label, mask in combo_labels.items():
    n         = int(mask.sum())
    n_err     = int((errors_mask & mask).sum())
    acc       = (n - n_err) / n if n > 0 else 1.0
    mean_conf = float(calib_top1[mask].mean()) if n > 0 else None
    mean_mah  = float(global_mah_test[mask].mean()) if n > 0 else None
    combo_results[label] = {
        "n": n, "n_errors": n_err,
        "accuracy_pct": round(acc * 100, 4),
        "mean_conf_pct": round(mean_conf * 100, 4) if mean_conf else None,
        "mean_mah_dist": round(mean_mah, 4) if mean_mah else None,
    }

# ===== ERROR ANALYSIS (PART 8) =====
print("[13] Part 8: Error sample analysis...")
err_idx      = int(np.where(errors_mask)[0][0])
err_actual   = str(y_test_str.values[err_idx])
err_pred     = str(y_pred_labels[err_idx])
err_conf     = float(calib_top1[err_idx])
err_margin   = float(top2_margin[err_idx])
err_g_mah    = float(global_mah_test[err_idx])
err_cc_mah   = float(cc_mah_test[err_idx])
err_probs    = {c: float(calib_probs[err_idx, i]) for i, c in enumerate(classes)}
err_knn_dist = knn_stats_from_prior["error_sample_657_min_dist"]

print(f"    Error at idx={err_idx}: actual={err_actual}, pred={err_pred}")
print(f"    conf={err_conf:.6f}, margin={err_margin:.6f}, global_mah={err_g_mah:.4f}, cc_mah={err_cc_mah:.4f}")
print(f"    knn_min_dist={err_knn_dist} (from prior forensic audit)")

would_catch_conf = {f"conf<{thr}": bool(err_conf < thr) for thr in conf_thresholds}
would_catch_mah  = {f"global_mah>P{pct}": bool(err_g_mah > np.percentile(global_mah_test, pct))
                    for pct in [50, 75, 90, 95, 99]}

# Lowest-confidence CORRECT samples
correct_df = results_table[results_table["correct"]].sort_values("calib_top1_prob")
lowest5_correct = correct_df.head(5)[
    ["sample_idx","actual_class","predicted_class","calib_top1_prob","top1_top2_margin","global_mah_dist","cc_mah_dist"]
].to_dict("records")

# ===== MAHALANOBIS THRESHOLD / FALSE-REJECTION ANALYSIS (PARTS 10+13) =====
print("[14] Part 10+13: Mahalanobis threshold development using calibration set...")

# Use calibration Mah percentiles as "threshold candidates" (no test leakage)
mah_calib_pcts = {50: np.percentile(global_mah_calib, 50),
                  75: np.percentile(global_mah_calib, 75),
                  90: np.percentile(global_mah_calib, 90),
                  95: np.percentile(global_mah_calib, 95),
                  99: np.percentile(global_mah_calib, 99)}

false_rejection_rows = []
for pct, calib_thr in mah_calib_pcts.items():
    # Apply to test set (evaluation only, NOT threshold selection)
    ood_flag   = global_mah_test > calib_thr
    in_dist    = ~ood_flag
    n_retained = int(in_dist.sum())
    n_flagged  = int(ood_flag.sum())
    coverage   = n_retained / N_TEST
    cor_flagged= int((correct_mask & ood_flag).sum())
    err_flagged= int((errors_mask & ood_flag).sum())
    acc_ret    = float(correct_mask[in_dist].mean()) if n_retained > 0 else 1.0

    false_rejection_rows.append({
        "calib_percentile"   : pct,
        "calib_mah_thr"      : round(float(calib_thr), 4),
        "n_retained"         : n_retained,
        "n_flagged"          : n_flagged,
        "coverage_pct"       : round(coverage * 100, 4),
        "correct_flagged"    : cor_flagged,
        "correct_flagged_pct": round(cor_flagged / N_TEST * 100, 4),
        "errors_flagged"     : err_flagged,
        "acc_retained_pct"   : round(acc_ret * 100, 4),
    })

fr_df = pd.DataFrame(false_rejection_rows)

# ===== PER-CLASS MAHALANOBIS STATS =====
per_class_mah = {}
for cls in classes:
    actual_mask = (y_test_str.values == cls)
    per_class_mah[cls] = {
        "n"              : int(actual_mask.sum()),
        "global_mah_mean": round(float(global_mah_test[actual_mask].mean()), 4),
        "global_mah_min" : round(float(global_mah_test[actual_mask].min()), 4),
        "global_mah_max" : round(float(global_mah_test[actual_mask].max()), 4),
        "global_mah_median":round(float(np.median(global_mah_test[actual_mask])), 4),
        "global_mah_p90" : round(float(np.percentile(global_mah_test[actual_mask], 90)), 4),
        "cc_mah_mean"    : round(float(cc_mah_test[actual_mask].mean()), 4),
        "cc_mah_min"     : round(float(cc_mah_test[actual_mask].min()), 4),
        "cc_mah_max"     : round(float(cc_mah_test[actual_mask].max()), 4),
    }

def pstats(arr, name):
    return {f"{name}_{k}": round(float(v), 6) for k, v in {
        "min": arr.min(), "max": arr.max(), "mean": arr.mean(),
        "median": np.median(arr), "p90": np.percentile(arr, 90),
        "p95": np.percentile(arr, 95), "p99": np.percentile(arr, 99)}.items()}

# ===== SAVE OUTPUTS =====
results_table.to_csv(OUT_CSV, index=False, float_format="%.6f")
print(f"[15] Per-sample CSV saved: {OUT_CSV}")

full_results = {
    "timestamp"  : time.strftime("%Y-%m-%d %H:%M:%S"),
    "audit_phase": "Phase 13: OOD + Abstention Design & Validation Audit",
    "production_freeze": {c: False for c in [
        "model_changed","schema_changed","calibration_changed",
        "rag_changed","retraining_performed","live_traffic_used","synthetic_ood_created"]},
    "dataset_sizes": {"n_train": N_TRAIN, "n_test": N_TEST, "n_calib": N_CALIB},
    "baseline_reproduction": {
        "accuracy_pct": round(accuracy_base * 100, 4),
        "n_errors"    : n_errors,
        "calibration_accuracy_pct": round(calib_acc * 100, 4) if calib_acc else None,
    },
    "part4_global_mahalanobis": {
        "description": "Global Mahalanobis from training centroid with regularized covariance (ridge=1e-6)",
        "test_statistics"  : pstats(global_mah_test, "global_mah"),
        "calib_statistics" : pstats(global_mah_calib, "calib_mah"),
        "per_actual_class" : per_class_mah,
        "error_sample"     : {"idx": err_idx, "global_mah": round(err_g_mah, 6), "cc_mah": round(err_cc_mah, 6)},
        "mah_p90_test"     : round(mah_p90, 4),
    },
    "part4_class_conditional_mahalanobis": {
        "description": "CC Mahalanobis: each test record vs its predicted class training distribution",
        "test_statistics": pstats(cc_mah_test, "cc_mah"),
        "error_sample"   : {"idx": err_idx, "cc_mah": round(err_cc_mah, 6)},
        "class_train_sizes": {cls: class_mah_params[cls]["n_train"] for cls in classes},
    },
    "part4_calibration_set_mahalanobis": {
        "description"    : "Calibration set Mahalanobis (for scientific threshold derivation without test leakage)",
        "statistics"     : pstats(global_mah_calib, "calib_mah"),
        "calib_percentile_thresholds": {str(k): round(float(v), 4) for k, v in mah_calib_pcts.items()},
    },
    "part5_knn_min_dist": knn_stats_from_prior,
    "part6_confidence_analysis": {
        "description"  : "Calibrated confidence threshold abstention analysis on 5,000 test records",
        "statistics"   : pstats(calib_top1, "conf"),
        "margin_stats" : pstats(top2_margin, "margin"),
        "thresholds"   : conf_rows,
    },
    "part7_confidence_mahalanobis_combination": {
        "description": f"HIGH/LOW conf (>={HIGH_CONF_THR}) x CLOSE/FAR Mah (P90={mah_p90:.4f})",
        "combo_results": combo_results,
    },
    "part8_error_analysis": {
        "single_error": {
            "original_row_idx"       : err_idx,
            "actual_class"           : err_actual,
            "predicted_class"        : err_pred,
            "calib_confidence"       : round(err_conf, 6),
            "top1_top2_margin"       : round(err_margin, 6),
            "global_mah_dist"        : round(err_g_mah, 6),
            "cc_mah_dist"            : round(err_cc_mah, 6),
            "knn_min_dist_from_prior": err_knn_dist,
            "calibrated_probs"       : {c: round(v, 6) for c, v in err_probs.items()},
            "would_catch_at_confidence_thresholds": would_catch_conf,
            "would_flag_by_mahalanobis"           : would_catch_mah,
        },
        "lowest_5_correct_confidence": [
            {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}
            for r in lowest5_correct
        ],
    },
    "part10_13_threshold_leakage_prevention": {
        "description"  : "Threshold derived from calibration set to avoid test leakage",
        "false_rejection_analysis": false_rejection_rows,
    },
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(full_results, f, indent=2, default=str)
print(f"[16] Results JSON saved: {OUT_JSON}")

# ===== PRINT SUMMARY TABLES =====
print("\n" + "=" * 75)
print("PART 4: GLOBAL MAHALANOBIS — TEST SET STATISTICS")
print("=" * 75)
for k in ["min","max","mean","median","p90","p95","p99"]:
    print(f"  {k:<10}: test={full_results['part4_global_mahalanobis']['test_statistics'][f'global_mah_{k}']:.4f}  "
          f"calib={full_results['part4_calibration_set_mahalanobis']['statistics'][f'calib_mah_{k}']:.4f}")

print("\nPer-class (actual label):")
for cls in classes:
    d = per_class_mah[cls]
    print(f"  {cls}: N={d['n']}, global_mah mean={d['global_mah_mean']:.4f}, median={d['global_mah_median']:.4f}, P90={d['global_mah_p90']:.4f}")

print(f"\nError sample (idx={err_idx}): global_mah={err_g_mah:.4f}, cc_mah={err_cc_mah:.4f}, conf={err_conf:.4f}, margin={err_margin:.4f}")

print("\n" + "=" * 75)
print("PART 6: CONFIDENCE THRESHOLD TABLE")
print("=" * 75)
print(f"{'Thr':>6} | {'Accepted':>8} | {'Abstain':>8} | {'Cov%':>7} | {'Err/Acc':>7} | {'Acc%':>10} | {'CorRej':>7}")
print("-" * 72)
for r in conf_rows:
    print(f"  {r['threshold']:>4.2f} | {r['n_accepted']:>8,} | {r['n_abstained']:>8,} | {r['coverage_pct']:>6.2f}% | {r['errors_in_accepted']:>7} | {r['accuracy_accepted_pct']:>9.4f}% | {r['correct_abstained']:>7,}")

print("\n" + "=" * 75)
print("PART 7: CONFIDENCE + MAHALANOBIS COMBINATION")
print("=" * 75)
for k, v in combo_results.items():
    print(f"  {k}")
    print(f"    N={v['n']}, Errors={v['n_errors']}, Accuracy={v['accuracy_pct']:.4f}%")

print("\n" + "=" * 75)
print("PART 10+13: FALSE-REJECTION ANALYSIS (Calibration-Derived Thresholds)")
print("=" * 75)
print(f"{'CalPct':>7} | {'Mah Thr':>8} | {'Retained':>9} | {'Flagged':>8} | {'Coverage':>9} | {'CorFlagged':>11} | {'ErrFlagged':>11} | {'AccRet%':>9}")
print("-" * 100)
for r in false_rejection_rows:
    print(f"  P{r['calib_percentile']:<5} | {r['calib_mah_thr']:>8.4f} | {r['n_retained']:>9,} | {r['n_flagged']:>8,} | {r['coverage_pct']:>8.2f}% | {r['correct_flagged']:>11,} | {r['errors_flagged']:>11} | {r['acc_retained_pct']:>8.4f}%")

print("\n" + "=" * 75)
print("Phase 13 quantitative computation COMPLETE.")
print(f"JSON: {OUT_JSON}")
print(f"CSV : {OUT_CSV}")
print("=" * 75)

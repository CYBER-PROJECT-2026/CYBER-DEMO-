"""
aegis_live/train_v4_baseline.py
===============================
Independent Model Training and Baseline Comparison Pipeline for AEGIS-Live v4.
Evaluation Contract: TEMPORAL GENERALIZATION WITH PERSISTENT-FLOW IDENTITY +
                     NETWORK-LEVEL TOPOLOGY OBSERVATION (1.0s Window).

Uses ONLY:
  - data/aegis_live_v4/train.csv (56,000 rows, 14,000/class)
  - data/aegis_live_v4/validation.csv (12,000 rows, 3,000/class)
  - data/aegis_live_v4/test.csv (12,000 rows, 3,000/class)

Schema: CANONICAL_29_FEATURES (Canonical 24 Flow + 5 Network Topology Features):
  1-24: CANONICAL_24_FEATURES (duration_ms through outbound_pkt_ratio)
  25: src_concentration_ratio
  26: src_ip_entropy
  27: src_ip_cardinality
  28: dst_ip_cardinality
  29: fan_in_ratio

Baseline Comparisons:
  - Evaluates frozen v2 model (models/aegis_live_v2/aegis_live_xgboost_baseline.pkl)
    on the exact same v4 test set using ONLY the original 24 canonical features.
  - Compares against frozen v3 baseline metrics.
  - Diagnostic ablation: trains 24-feature baseline on v4 train data to isolate network topology gain.
"""

import os
import sys
import time
import json
import pickle
import hashlib
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    classification_report
)

# Project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from aegis_live.feature_schema import CANONICAL_24_FEATURES, CANONICAL_29_FEATURES, V4_TOPOLOGY_FEATURES

DATA_DIR = os.path.join(BASE_DIR, "data", "aegis_live_v4")
V2_MODEL_DIR = os.path.join(BASE_DIR, "models", "aegis_live_v2")
V3_MODEL_DIR = os.path.join(BASE_DIR, "models", "aegis_live_v3")
V4_MODEL_DIR = os.path.join(BASE_DIR, "models", "aegis_live_v4")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

CLASSES = ["Benign", "DDoS", "DoS", "Spoofing"]


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024 * 16):
            h.update(chunk)
    return h.hexdigest()


def compute_entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(probs, eps, 1.0)
    return -np.sum(p * np.log2(p), axis=1)


def evaluate_predictions(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         y_prob: np.ndarray,
                         classes: List[str]) -> Dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    try:
        auc_ovr = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except Exception:
        auc_ovr = float("nan")
    loss = float(log_loss(y_true, y_prob))

    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes))).tolist()
    clf_rep = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)

    per_class = {}
    for c in classes:
        per_class[c] = {
            "precision": float(clf_rep[c]["precision"]),
            "recall": float(clf_rep[c]["recall"]),
            "f1_score": float(clf_rep[c]["f1-score"]),
            "support": int(clf_rep[c]["support"])
        }

    # Extract DDoS <-> DoS cross-talk specifically
    # Class mapping: Benign=0, DDoS=1, DoS=2, Spoofing=3
    ddos_idx = classes.index("DDoS")
    dos_idx = classes.index("DoS")
    ddos_to_dos = int(cm[ddos_idx][dos_idx])
    dos_to_ddos = int(cm[dos_idx][ddos_idx])
    total_cross_talk = ddos_to_dos + dos_to_ddos

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "precision_weighted": prec_weighted,
        "recall_weighted": rec_weighted,
        "f1_weighted": f1_weighted,
        "cohen_kappa": kappa,
        "mcc": mcc,
        "roc_auc_ovr_macro": auc_ovr,
        "log_loss": loss,
        "confusion_matrix": cm,
        "per_class": per_class,
        "cross_talk": {
            "ddos_to_dos": ddos_to_dos,
            "dos_to_ddos": dos_to_ddos,
            "total_ddos_dos_cross_talk": total_cross_talk
        }
    }


def compute_distribution_stats(series: pd.Series) -> Dict[str, float]:
    return {
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "p25": float(series.quantile(0.25)),
        "median": float(series.median()),
        "p75": float(series.quantile(0.75)),
        "max": float(series.max())
    }


def main():
    print("=" * 88)
    print("  AEGIS-Live v4: MULTICLASS MODEL TRAINING & BASELINE COMPARISON")
    print("  Evaluation Contract: TEMPORAL GENERALIZATION WITH PERSISTENT-FLOW IDENTITY")
    print("  Features: Raw Physical 29 Features (Canonical 24 + 5 Network Topology Features)")
    print("=" * 88)

    os.makedirs(V4_MODEL_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 0. VERIFY IMMUTABLE BASELINES BEFORE EXECUTION
    # -------------------------------------------------------------------------
    print("\n--- 0. CHECKING IMMUTABLE BASELINES ---")
    v2_model_path = os.path.join(V2_MODEL_DIR, "aegis_live_xgboost_baseline.pkl")
    v3_model_path = os.path.join(V3_MODEL_DIR, "aegis_live_xgboost_baseline.pkl")
    if not os.path.exists(v2_model_path):
        raise FileNotFoundError(f"Missing frozen v2 model at {v2_model_path}")
    if not os.path.exists(v3_model_path):
        raise FileNotFoundError(f"Missing frozen v3 model at {v3_model_path}")

    v2_model_hash_before = compute_sha256(v2_model_path)
    v3_model_hash_before = compute_sha256(v3_model_path)
    print(f"  Frozen v2 model SHA256: {v2_model_hash_before}")
    print(f"  Frozen v3 model SHA256: {v3_model_hash_before}")

    # -------------------------------------------------------------------------
    # 1. LOAD V4 DATASETS & INTEGRITY CHECK
    # -------------------------------------------------------------------------
    print("\n--- 1. LOADING V4 DATASETS ---")
    train_path = os.path.join(DATA_DIR, "train.csv")
    val_path = os.path.join(DATA_DIR, "validation.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")

    for p in [train_path, val_path, test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required v4 dataset: {p}")

    train_hash = compute_sha256(train_path)
    val_hash = compute_sha256(val_path)
    test_hash = compute_sha256(test_path)

    print(f"  train.csv      ({os.path.getsize(train_path):,} bytes) SHA256: {train_hash}")
    print(f"  validation.csv ({os.path.getsize(val_path):,} bytes) SHA256: {val_hash}")
    print(f"  test.csv       ({os.path.getsize(test_path):,} bytes) SHA256: {test_hash}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print(f"  Train samples:      {len(train_df):,}")
    print(f"  Validation samples: {len(val_df):,}")
    print(f"  Test samples:       {len(test_df):,}")

    assert len(train_df) == 56000, f"Train row count {len(train_df)} != 56,000"
    assert len(val_df) == 12000, f"Validation row count {len(val_df)} != 12,000"
    assert len(test_df) == 12000, f"Test row count {len(test_df)} != 12,000"

    # Verify class balance
    for name, df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        counts = df["label"].value_counts().to_dict()
        expected = 14000 if name == "Train" else 3000
        for c in CLASSES:
            assert counts[c] == expected, f"{name} {c} count {counts.get(c, 0)} != {expected}"

    # Verify exact 29 features in strict canonical order
    for col in CANONICAL_29_FEATURES:
        assert col in train_df.columns, f"Missing {col} in train_df"
        assert col in val_df.columns, f"Missing {col} in val_df"
        assert col in test_df.columns, f"Missing {col} in test_df"

    assert "unique_flow_count" not in train_df.columns, "unique_flow_count must NOT be in train_df"
    assert "unique_5tuple_flow_count" not in train_df.columns, "unique_5tuple_flow_count must NOT be in train_df"
    assert "fan_out_ratio" not in train_df.columns, "fan_out_ratio must NOT be in train_df"

    X_train_v4 = train_df[CANONICAL_29_FEATURES].copy()
    X_val_v4 = val_df[CANONICAL_29_FEATURES].copy()
    X_test_v4 = test_df[CANONICAL_29_FEATURES].copy()

    # Preprocessing audit: No scaler, no NaN, no Inf
    assert X_train_v4.isna().sum().sum() == 0, "NaN in X_train_v4"
    assert X_val_v4.isna().sum().sum() == 0, "NaN in X_val_v4"
    assert X_test_v4.isna().sum().sum() == 0, "NaN in X_test_v4"
    assert np.isinf(X_train_v4.values).sum() == 0, "Inf in X_train_v4"
    assert np.isinf(X_val_v4.values).sum() == 0, "Inf in X_val_v4"
    assert np.isinf(X_test_v4.values).sum() == 0, "Inf in X_test_v4"

    print("  [+] Preprocessing Audit: Raw physical values, zero NaN, zero Inf, NO scaler.")

    # -------------------------------------------------------------------------
    # 2. DETERMINISTIC LABEL ENCODING
    # -------------------------------------------------------------------------
    print("\n--- 2. LABEL ENCODING ---")
    le = LabelEncoder()
    le.fit(sorted(CLASSES))
    label_mapping = {cls: int(idx) for idx, cls in enumerate(le.classes_)}
    print(f"  Class mapping: {label_mapping}")
    assert label_mapping == {"Benign": 0, "DDoS": 1, "DoS": 2, "Spoofing": 3}

    y_train = le.transform(train_df["label"])
    y_val = le.transform(val_df["label"])
    y_test = le.transform(test_df["label"])

    # -------------------------------------------------------------------------
    # 3. XGBOOST MODEL TRAINING (IDENTICAL HYPERPARAMETERS TO V2 / V3)
    # -------------------------------------------------------------------------
    print("\n--- 3. V4 MODEL TRAINING (SEED 42) ---")
    xgb_params = {
        "objective": "multi:softprob",
        "num_class": 4,
        "n_estimators": 300,
        "max_depth": 8,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1
    }
    print(f"  Hyperparameters: {xgb_params}")

    t0 = time.time()
    v4_model = xgb.XGBClassifier(**xgb_params)
    v4_model.fit(X_train_v4, y_train)
    t_train_sec = time.time() - t0
    print(f"  [+] v4 model trained on 56,000 samples (29 features) in {t_train_sec:.2f}s")

    # -------------------------------------------------------------------------
    # 4. REPRODUCIBILITY VERIFICATION
    # -------------------------------------------------------------------------
    print("\n--- 4. REPRODUCIBILITY VERIFICATION ---")
    print("  Training second instance with seed 42 to verify deterministic behavior...")
    v4_model_rep = xgb.XGBClassifier(**xgb_params)
    v4_model_rep.fit(X_train_v4, y_train)

    val_probs_1 = v4_model.predict_proba(X_val_v4)
    val_probs_2 = v4_model_rep.predict_proba(X_val_v4)
    val_preds_1 = np.argmax(val_probs_1, axis=1)
    val_preds_2 = np.argmax(val_probs_2, axis=1)

    preds_match = bool(np.array_equal(val_preds_1, val_preds_2))
    max_prob_diff = float(np.abs(val_probs_1 - val_probs_2).max())
    print(f"  Validation predictions identical: {preds_match}")
    print(f"  Maximum class probability difference: {max_prob_diff:.6e}")
    assert preds_match and max_prob_diff < 1e-5, f"Reproducibility failed: diff={max_prob_diff}"
    print("  [+] Reproducibility verification: PASS")

    # -------------------------------------------------------------------------
    # 5. VALIDATION EVALUATION (MODEL DIAGNOSTICS ONLY)
    # -------------------------------------------------------------------------
    print("\n--- 5. VALIDATION EVALUATION ---")
    val_metrics_v4 = evaluate_predictions(y_val, val_preds_1, val_probs_1, CLASSES)
    print(f"  Validation Accuracy:          {val_metrics_v4['accuracy'] * 100:.2f}%")
    print(f"  Validation Balanced Accuracy: {val_metrics_v4['balanced_accuracy'] * 100:.2f}%")
    print(f"  Validation Macro F1:          {val_metrics_v4['f1_macro']:.4f}")
    print(f"  Validation Log Loss:          {val_metrics_v4['log_loss']:.4f}")
    print(f"  Validation Cohen Kappa:       {val_metrics_v4['cohen_kappa']:.4f}")
    print(f"  Validation MCC:               {val_metrics_v4['mcc']:.4f}")
    print(f"  Validation DDoS->DoS:         {val_metrics_v4['cross_talk']['ddos_to_dos']}")
    print(f"  Validation DoS->DDoS:         {val_metrics_v4['cross_talk']['dos_to_ddos']}")
    print(f"  Validation Total Cross-Talk:  {val_metrics_v4['cross_talk']['total_ddos_dos_cross_talk']}")

    # -------------------------------------------------------------------------
    # 6. HELD-OUT TEST EVALUATION (EXACTLY ONCE ON V4 TEST SET)
    # -------------------------------------------------------------------------
    print("\n--- 6. HELD-OUT TEST EVALUATION (EXACTLY ONCE) ---")
    test_probs_v4 = v4_model.predict_proba(X_test_v4)
    test_preds_v4 = np.argmax(test_probs_v4, axis=1)
    test_max_conf_v4 = np.max(test_probs_v4, axis=1)
    test_entropy_v4 = compute_entropy(test_probs_v4)

    test_metrics_v4 = evaluate_predictions(y_test, test_preds_v4, test_probs_v4, CLASSES)

    print(f"  [V4 TEST] Accuracy:          {test_metrics_v4['accuracy'] * 100:.2f}%")
    print(f"  [V4 TEST] Balanced Accuracy: {test_metrics_v4['balanced_accuracy'] * 100:.2f}%")
    print(f"  [V4 TEST] Macro Precision:   {test_metrics_v4['precision_macro']:.4f}")
    print(f"  [V4 TEST] Macro Recall:      {test_metrics_v4['recall_macro']:.4f}")
    print(f"  [V4 TEST] Macro F1:          {test_metrics_v4['f1_macro']:.4f}")
    print(f"  [V4 TEST] Weighted F1:       {test_metrics_v4['f1_weighted']:.4f}")
    print(f"  [V4 TEST] Cohen Kappa:       {test_metrics_v4['cohen_kappa']:.4f}")
    print(f"  [V4 TEST] MCC:               {test_metrics_v4['mcc']:.4f}")
    print(f"  [V4 TEST] Log Loss:          {test_metrics_v4['log_loss']:.4f}")
    print(f"  [V4 TEST] ROC-AUC (OVR):     {test_metrics_v4['roc_auc_ovr_macro']:.4f}")
    print("  [V4 TEST] Confusion Matrix:")
    for row in test_metrics_v4["confusion_matrix"]:
        print(f"    {row}")
    print("  [V4 TEST] Per-Class Performance:")
    for c, m in test_metrics_v4["per_class"].items():
        print(f"    {c:10s} | P: {m['precision']:.4f} | R: {m['recall']:.4f} | F1: {m['f1_score']:.4f} | N: {m['support']}")
    print(f"  [V4 TEST] DDoS -> DoS errors: {test_metrics_v4['cross_talk']['ddos_to_dos']}")
    print(f"  [V4 TEST] DoS -> DDoS errors: {test_metrics_v4['cross_talk']['dos_to_ddos']}")
    print(f"  [V4 TEST] Total Cross-Talk:   {test_metrics_v4['cross_talk']['total_ddos_dos_cross_talk']}")

    # -------------------------------------------------------------------------
    # 7. FROZEN V2 MODEL EVALUATION ON SAME V4 TEST DATA (24 FEATURES)
    # -------------------------------------------------------------------------
    print("\n--- 7. EVALUATING FROZEN V2 MODEL ON SAME V4 TEST DATA ---")
    with open(v2_model_path, "rb") as f:
        v2_model = pickle.load(f)

    # Supply exact canonical 24 features in original order
    X_test_v2_features = test_df[CANONICAL_24_FEATURES].copy()
    test_probs_v2 = v2_model.predict_proba(X_test_v2_features)
    test_preds_v2 = np.argmax(test_probs_v2, axis=1)
    test_metrics_v2 = evaluate_predictions(y_test, test_preds_v2, test_probs_v2, CLASSES)

    print(f"  [V2 on v4 test set] Accuracy:          {test_metrics_v2['accuracy'] * 100:.2f}%")
    print(f"  [V2 on v4 test set] Balanced Accuracy: {test_metrics_v2['balanced_accuracy'] * 100:.2f}%")
    print(f"  [V2 on v4 test set] Macro F1:          {test_metrics_v2['f1_macro']:.4f}")
    print(f"  [V2 on v4 test set] Log Loss:          {test_metrics_v2['log_loss']:.4f}")
    print(f"  [V2 on v4 test set] Cohen Kappa:       {test_metrics_v2['cohen_kappa']:.4f}")
    print(f"  [V2 on v4 test set] MCC:               {test_metrics_v2['mcc']:.4f}")
    print(f"  [V2 on v4 test set] DDoS -> DoS:       {test_metrics_v2['cross_talk']['ddos_to_dos']}")
    print(f"  [V2 on v4 test set] DoS -> DDoS:       {test_metrics_v2['cross_talk']['dos_to_ddos']}")
    print(f"  [V2 on v4 test set] Total Cross-Talk:  {test_metrics_v2['cross_talk']['total_ddos_dos_cross_talk']}")

    # -------------------------------------------------------------------------
    # 8. LOAD FROZEN V3 EVALUATION METRICS FOR COMPARISON
    # -------------------------------------------------------------------------
    print("\n--- 8. LOADING FROZEN V3 EVALUATION METRICS ---")
    v3_eval_path = os.path.join(V3_MODEL_DIR, "evaluation_results.json")
    with open(v3_eval_path, "r") as f:
        v3_eval_data = json.load(f)
    v3_test_metrics = v3_eval_data["v3_metrics"]["test"]
    print(f"  [V3 Frozen Test] Accuracy:         {v3_test_metrics['accuracy'] * 100:.2f}%")
    print(f"  [V3 Frozen Test] Macro F1:         {v3_test_metrics['f1_macro']:.4f}")
    print(f"  [V3 Frozen Test] Log Loss:         {v3_test_metrics['log_loss']:.4f}")
    print(f"  [V3 Frozen Test] Total Cross-Talk: {v3_test_metrics['cross_talk']['total_ddos_dos_cross_talk']}")

    # -------------------------------------------------------------------------
    # 9. V4 VS V2 HEAD-TO-HEAD COMPARISON METRICS
    # -------------------------------------------------------------------------
    print("\n--- 9. APPLES-TO-APPLES COMPARISON (SAME V4 TEST PARTITION) ---")
    acc_diff_v2 = test_metrics_v4["accuracy"] - test_metrics_v2["accuracy"]
    f1_diff_v2 = test_metrics_v4["f1_macro"] - test_metrics_v2["f1_macro"]
    logloss_diff_v2 = test_metrics_v4["log_loss"] - test_metrics_v2["log_loss"]
    kappa_diff_v2 = test_metrics_v4["cohen_kappa"] - test_metrics_v2["cohen_kappa"]
    mcc_diff_v2 = test_metrics_v4["mcc"] - test_metrics_v2["mcc"]

    v2_ct = test_metrics_v2["cross_talk"]["total_ddos_dos_cross_talk"]
    v4_ct = test_metrics_v4["cross_talk"]["total_ddos_dos_cross_talk"]
    ct_reduction_v2 = v2_ct - v4_ct
    ct_reduction_pct_v2 = (ct_reduction_v2 / v2_ct * 100.0) if v2_ct > 0 else 0.0

    v2_ddos_to_dos = test_metrics_v2["cross_talk"]["ddos_to_dos"]
    v4_ddos_to_dos = test_metrics_v4["cross_talk"]["ddos_to_dos"]
    ddos_dos_red_v2 = v2_ddos_to_dos - v4_ddos_to_dos
    ddos_dos_red_pct_v2 = (ddos_dos_red_v2 / v2_ddos_to_dos * 100.0) if v2_ddos_to_dos > 0 else 0.0

    v2_dos_to_ddos = test_metrics_v2["cross_talk"]["dos_to_ddos"]
    v4_dos_to_ddos = test_metrics_v4["cross_talk"]["dos_to_ddos"]
    dos_ddos_red_v2 = v2_dos_to_ddos - v4_dos_to_ddos
    dos_ddos_red_pct_v2 = (dos_ddos_red_v2 / v2_dos_to_ddos * 100.0) if v2_dos_to_ddos > 0 else 0.0

    print(f"  Accuracy:      v2 = {test_metrics_v2['accuracy']*100:.2f}% | v4 = {test_metrics_v4['accuracy']*100:.2f}% (Diff: {acc_diff_v2*100:+.2f}%)")
    print(f"  Macro F1:      v2 = {test_metrics_v2['f1_macro']:.4f}  | v4 = {test_metrics_v4['f1_macro']:.4f}  (Diff: {f1_diff_v2:+.4f})")
    print(f"  Log Loss:      v2 = {test_metrics_v2['log_loss']:.4f}  | v4 = {test_metrics_v4['log_loss']:.4f}  (Diff: {logloss_diff_v2:+.4f})")
    print(f"  DDoS -> DoS:   v2 = {v2_ddos_to_dos:4d}      | v4 = {v4_ddos_to_dos:4d}      (Reduced by {ddos_dos_red_v2} / {ddos_dos_red_pct_v2:.1f}%)")
    print(f"  DoS -> DDoS:   v2 = {v2_dos_to_ddos:4d}      | v4 = {v4_dos_to_ddos:4d}      (Reduced by {dos_ddos_red_v2} / {dos_ddos_red_pct_v2:.1f}%)")
    print(f"  Total Conf:    v2 = {v2_ct:4d}      | v4 = {v4_ct:4d}      (Reduced by {ct_reduction_v2} / {ct_reduction_pct_v2:.1f}%)")

    # -------------------------------------------------------------------------
    # 10. DIAGNOSTIC ABLATION: 24-FEATURE BASELINE TRAINED ON V4 DATA
    # -------------------------------------------------------------------------
    print("\n--- 10. DIAGNOSTIC ABLATION (24 FEATURES ON V4 DATASET) ---")
    X_train_v4_24 = train_df[CANONICAL_24_FEATURES].copy()
    X_test_v4_24 = test_df[CANONICAL_24_FEATURES].copy()
    v4_ablation_24 = xgb.XGBClassifier(**xgb_params)
    v4_ablation_24.fit(X_train_v4_24, y_train)
    test_probs_ablation = v4_ablation_24.predict_proba(X_test_v4_24)
    test_preds_ablation = np.argmax(test_probs_ablation, axis=1)
    test_metrics_ablation = evaluate_predictions(y_test, test_preds_ablation, test_probs_ablation, CLASSES)

    print(f"    v4 (24 feats, v4-trained) Test Acc: {test_metrics_ablation['accuracy']*100:.2f}% | Macro F1: {test_metrics_ablation['f1_macro']:.4f} | Cross-Talk: {test_metrics_ablation['cross_talk']['total_ddos_dos_cross_talk']}")
    print(f"    v4 (29 feats, v4-trained) Test Acc: {test_metrics_v4['accuracy']*100:.2f}% | Macro F1: {test_metrics_v4['f1_macro']:.4f} | Cross-Talk: {test_metrics_v4['cross_talk']['total_ddos_dos_cross_talk']}")
    net_topo_acc_gain = (test_metrics_v4['accuracy'] - test_metrics_ablation['accuracy']) * 100
    net_topo_f1_gain = test_metrics_v4['f1_macro'] - test_metrics_ablation['f1_macro']
    net_topo_ct_reduction = test_metrics_ablation['cross_talk']['total_ddos_dos_cross_talk'] - test_metrics_v4['cross_talk']['total_ddos_dos_cross_talk']
    print(f"    -> Net Network Topology Contribution: Acc {net_topo_acc_gain:+.2f}%, F1 {net_topo_f1_gain:+.4f}, Cross-Talk Reduction: {net_topo_ct_reduction} errors")

    # -------------------------------------------------------------------------
    # 11. FEATURE IMPORTANCE & NETWORK TOPOLOGY FEATURE FORENSICS
    # -------------------------------------------------------------------------
    print("\n--- 11. FEATURE IMPORTANCE & NETWORK TOPOLOGY FORENSICS ---")
    booster_v4 = v4_model.get_booster()
    score_gain = booster_v4.get_score(importance_type="gain")
    score_weight = booster_v4.get_score(importance_type="weight")
    score_cover = booster_v4.get_score(importance_type="cover")

    fi_records = []
    for feat in CANONICAL_29_FEATURES:
        g = float(score_gain.get(feat, 0.0))
        w = int(score_weight.get(feat, 0))
        c = float(score_cover.get(feat, 0.0))
        is_topo = feat in V4_TOPOLOGY_FEATURES
        fi_records.append({
            "feature": feat,
            "is_topology": is_topo,
            "gain": g,
            "weight_split_count": w,
            "cover": c
        })

    fi_df = pd.DataFrame(fi_records).sort_values(by="gain", ascending=False).reset_index(drop=True)
    fi_path = os.path.join(V4_MODEL_DIR, "feature_importance.csv")
    fi_df.to_csv(fi_path, index=False)
    print(f"  Saved {fi_path}")

    print("  Top 10 Features by XGBoost Gain:")
    for rank, (_, r) in enumerate(fi_df.head(10).iterrows(), start=1):
        topo_flag = " [NETWORK TOPOLOGY]" if r["is_topology"] else ""
        print(f"    {rank:2d}. {r['feature']:30s}{topo_flag} | Gain: {r['gain']:10.4f} | Splits: {int(r['weight_split_count']):4d}")

    print("\n  Network Topology Features Specific Ranks & Gains:")
    topo_summary_dict = {}
    for tf in V4_TOPOLOGY_FEATURES:
        row_tf = fi_df[fi_df["feature"] == tf].iloc[0]
        rank_tf = int(fi_df[fi_df["feature"] == tf].index[0]) + 1
        topo_summary_dict[tf] = {
            "rank": rank_tf,
            "gain": float(row_tf["gain"]),
            "splits": int(row_tf["weight_split_count"]),
            "cover": float(row_tf["cover"])
        }
        print(f"    {tf:30s} | Rank {rank_tf:2d}/29 | Gain: {row_tf['gain']:10.4f} | Splits: {int(row_tf['weight_split_count']):4d}")

    # Diagnostic comparison of DDoS vs DoS distributions for the 5 topology features
    print("\n  Diagnostic Distribution Comparison (DDoS vs DoS on Test Set):")
    topo_dist_comparison = {}
    ddos_test_df = test_df[test_df["label"] == "DDoS"]
    dos_test_df = test_df[test_df["label"] == "DoS"]
    for tf in V4_TOPOLOGY_FEATURES:
        ddos_stats = compute_distribution_stats(ddos_test_df[tf])
        dos_stats = compute_distribution_stats(dos_test_df[tf])
        topo_dist_comparison[tf] = {
            "DDoS": ddos_stats,
            "DoS": dos_stats
        }
        print(f"    {tf:25s} | DDoS median: {ddos_stats['median']:8.4f} (mean={ddos_stats['mean']:8.4f}) | DoS median: {dos_stats['median']:8.4f} (mean={dos_stats['mean']:8.4f})")

    # -------------------------------------------------------------------------
    # 12. TREESHAP DIAGNOSTIC EVALUATION (VALIDATION SET ONLY)
    # -------------------------------------------------------------------------
    print("\n--- 12. TREESHAP DIAGNOSTIC (VALIDATION SET SAMPLE ONLY) ---")
    shap_records = []
    try:
        import shap
        # Fixed diagnostic sample from VALIDATION ONLY (never test!)
        np.random.seed(42)
        val_sample_idx = np.random.choice(len(X_val_v4), size=400, replace=False)
        X_val_sample = X_val_v4.iloc[val_sample_idx]
        explainer = shap.TreeExplainer(v4_model)
        shap_values = explainer.shap_values(X_val_sample)
        # For multiclass, shap_values is a list of arrays (one per class) or (N, features, classes)
        if isinstance(shap_values, list):
            mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
        else:
            mean_abs_shap = np.abs(shap_values).mean(axis=0)

        shap_df = pd.DataFrame({
            "feature": CANONICAL_29_FEATURES,
            "mean_abs_shap": mean_abs_shap
        }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

        print("  Top 10 Features by Mean Absolute SHAP (Validation Sample):")
        for rank, (_, r) in enumerate(shap_df.head(10).iterrows(), start=1):
            is_t = " [TOPOLOGY]" if r["feature"] in V4_TOPOLOGY_FEATURES else ""
            print(f"    {rank:2d}. {r['feature']:30s}{is_t} | Mean |SHAP|: {r['mean_abs_shap']:.6f}")
        for tf in V4_TOPOLOGY_FEATURES:
            r_tf = shap_df[shap_df["feature"] == tf].iloc[0]
            rank_tf = int(shap_df[shap_df["feature"] == tf].index[0]) + 1
            shap_records.append({
                "feature": tf,
                "shap_rank": rank_tf,
                "mean_abs_shap": float(r_tf["mean_abs_shap"])
            })
            print(f"    [Topology SHAP] {tf:25s} | Rank {rank_tf:2d}/29 | Mean |SHAP|: {r_tf['mean_abs_shap']:.6f}")
    except Exception as e:
        print(f"  SHAP diagnostic skipped / error: {e}")

    # -------------------------------------------------------------------------
    # 13. ERROR ANALYSIS & CONFUSION FORENSICS
    # -------------------------------------------------------------------------
    print("\n--- 13. ERROR ANALYSIS & CONFUSION FORENSICS ---")
    test_df_eval = test_df.copy()
    test_df_eval["v2_pred"] = le.inverse_transform(test_preds_v2)
    test_df_eval["v4_pred"] = le.inverse_transform(test_preds_v4)
    for idx, c in enumerate(CLASSES):
        test_df_eval[f"v4_prob_{c}"] = test_probs_v4[:, idx]
        test_df_eval[f"v2_prob_{c}"] = test_probs_v2[:, idx]
    test_df_eval["v4_confidence"] = test_max_conf_v4
    test_df_eval["v4_entropy"] = test_entropy_v4

    # Error flags
    test_df_eval["v2_correct"] = (test_df_eval["label"] == test_df_eval["v2_pred"])
    test_df_eval["v4_correct"] = (test_df_eval["label"] == test_df_eval["v4_pred"])

    # Extract DDoS <-> DoS confusions in v4
    v4_ddos_as_dos = test_df_eval[(test_df_eval["label"] == "DDoS") & (test_df_eval["v4_pred"] == "DoS")]
    v4_dos_as_ddos = test_df_eval[(test_df_eval["label"] == "DoS") & (test_df_eval["v4_pred"] == "DDoS")]

    # Extract DDoS <-> DoS confusions in v2
    v2_ddos_as_dos = test_df_eval[(test_df_eval["label"] == "DDoS") & (test_df_eval["v2_pred"] == "DoS")]
    v2_dos_as_ddos = test_df_eval[(test_df_eval["label"] == "DoS") & (test_df_eval["v2_pred"] == "DDoS")]

    print(f"  v2 DDoS -> DoS count: {len(v2_ddos_as_dos)} | v4 count: {len(v4_ddos_as_dos)}")
    print(f"  v2 DoS -> DDoS count: {len(v2_dos_as_ddos)} | v4 count: {len(v4_dos_as_ddos)}")

    forensic_features = V4_TOPOLOGY_FEATURES + [
        "pkt_rate_pps",
        "syn_ratio",
        "ack_ratio",
        "iat_mean_ms",
        "iat_std_ms"
    ]

    forensic_summary = {}
    print("\n  Distribution Statistics for Misclassifications vs Correct:")
    for group_name, group_df in [
        ("v4_DDoS_misclassified_as_DoS", v4_ddos_as_dos),
        ("v4_DoS_misclassified_as_DDoS", v4_dos_as_ddos),
        ("v4_DDoS_correct", test_df_eval[(test_df_eval["label"] == "DDoS") & (test_df_eval["v4_pred"] == "DDoS")]),
        ("v4_DoS_correct", test_df_eval[(test_df_eval["label"] == "DoS") & (test_df_eval["v4_pred"] == "DoS")]),
        ("v2_DDoS_misclassified_as_DoS", v2_ddos_as_dos),
        ("v2_DoS_misclassified_as_DDoS", v2_dos_as_ddos)
    ]:
        forensic_summary[group_name] = {"count": len(group_df), "features": {}}
        print(f"\n  --- Group: {group_name} (N={len(group_df)}) ---")
        if len(group_df) > 0:
            for feat in forensic_features:
                stats = compute_distribution_stats(group_df[feat])
                forensic_summary[group_name]["features"][feat] = stats
                print(f"    {feat:28s} | Mean: {stats['mean']:10.4f} | Median: {stats['median']:10.4f} | Std: {stats['std']:10.4f} | Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

    # Save detailed confusion forensics CSV
    forensic_subset = test_df_eval[
        (test_df_eval["label"].isin(["DDoS", "DoS"])) |
        (test_df_eval["v4_pred"].isin(["DDoS", "DoS"])) |
        (test_df_eval["v2_pred"].isin(["DDoS", "DoS"]))
    ].copy()

    forensic_csv_path = os.path.join(REPORTS_DIR, "aegis_live_v4_confusion_forensics.csv")
    cols_to_save = [
        "temporal_block_id", "window_start_ts", "pcap_file", "label",
        "v2_pred", "v4_pred", "v2_correct", "v4_correct",
        "v4_confidence", "v4_entropy"
    ] + forensic_features
    forensic_subset[cols_to_save].to_csv(forensic_csv_path, index=False)
    print(f"\n  Saved confusion forensics: {forensic_csv_path} ({len(forensic_subset):,} rows)")

    # -------------------------------------------------------------------------
    # 14. SAVE MODEL & METADATA ARTIFACTS
    # -------------------------------------------------------------------------
    print("\n--- 14. SAVING V4 MODEL AND METADATA ---")
    model_save_path = os.path.join(V4_MODEL_DIR, "aegis_live_xgboost_baseline.pkl")
    with open(model_save_path, "wb") as f:
        pickle.dump(v4_model, f)
    v4_model_hash = compute_sha256(model_save_path)
    print(f"  Saved v4 model: {model_save_path} (SHA256: {v4_model_hash})")

    # Training metadata
    training_metadata = {
        "model_name": "AEGIS-Live v4 Raw Baseline Classifier (Network-Level Topology Augmented)",
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "evaluation_contract": "TEMPORAL GENERALIZATION WITH PERSISTENT-FLOW IDENTITY + NETWORK TOPOLOGY",
        "observation_window_seconds": 1.0,
        "feature_schema": {
            "feature_count": len(CANONICAL_29_FEATURES),
            "features": CANONICAL_29_FEATURES,
            "canonical_24_features": CANONICAL_24_FEATURES,
            "topology_features": V4_TOPOLOGY_FEATURES,
            "excluded_features": ["unique_flow_count", "unique_5tuple_flow_count", "fan_out_ratio"]
        },
        "model_parameters": xgb_params,
        "preprocessing": {
            "scaler": None,
            "smote": False,
            "resampling": None,
            "note": "Raw 29-feature physical values with no scaler or synthetic generation."
        },
        "training_row_count": len(train_df),
        "validation_row_count": len(val_df),
        "test_row_count": len(test_df),
        "class_mapping": label_mapping,
        "classes": CLASSES,
        "random_seed": 42,
        "training_duration_seconds": float(t_train_sec),
        "software_versions": {
            "python": sys.version,
            "xgboost": xgb.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__
        },
        "reproducibility": {
            "verified": preds_match,
            "max_prob_diff": max_prob_diff
        },
        "artifact_hashes": {
            "model_pkl_sha256": v4_model_hash,
            "train_csv_sha256": train_hash,
            "validation_csv_sha256": val_hash,
            "test_csv_sha256": test_hash
        },
        "topology_features_summary": topo_summary_dict
    }

    metadata_path = os.path.join(V4_MODEL_DIR, "training_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(training_metadata, f, indent=2)
    print(f"  Saved metadata: {metadata_path}")

    # Evaluation results JSON
    eval_results = {
        "evaluation_contract": "TEMPORAL GENERALIZATION WITH PERSISTENT-FLOW IDENTITY + NETWORK TOPOLOGY",
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "v4_metrics": {
            "validation": val_metrics_v4,
            "test": test_metrics_v4
        },
        "v2_metrics_on_same_v4_test": test_metrics_v2,
        "v3_frozen_test_metrics": v3_test_metrics,
        "v4_diagnostic_ablation_24features": test_metrics_ablation,
        "comparison_v4_vs_v2": {
            "accuracy": {
                "v2": test_metrics_v2["accuracy"],
                "v4": test_metrics_v4["accuracy"],
                "difference": acc_diff_v2
            },
            "macro_f1": {
                "v2": test_metrics_v2["f1_macro"],
                "v4": test_metrics_v4["f1_macro"],
                "difference": f1_diff_v2
            },
            "log_loss": {
                "v2": test_metrics_v2["log_loss"],
                "v4": test_metrics_v4["log_loss"],
                "difference": logloss_diff_v2
            },
            "cohen_kappa": {
                "v2": test_metrics_v2["cohen_kappa"],
                "v4": test_metrics_v4["cohen_kappa"],
                "difference": kappa_diff_v2
            },
            "mcc": {
                "v2": test_metrics_v2["mcc"],
                "v4": test_metrics_v4["mcc"],
                "difference": mcc_diff_v2
            },
            "cross_talk": {
                "ddos_to_dos": {
                    "v2": v2_ddos_to_dos,
                    "v4": v4_ddos_to_dos,
                    "reduction": ddos_dos_red_v2,
                    "reduction_pct": ddos_dos_red_pct_v2
                },
                "dos_to_ddos": {
                    "v2": v2_dos_to_ddos,
                    "v4": v4_dos_to_ddos,
                    "reduction": dos_ddos_red_v2,
                    "reduction_pct": dos_ddos_red_pct_v2
                },
                "total": {
                    "v2": v2_ct,
                    "v4": v4_ct,
                    "reduction": ct_reduction_v2,
                    "reduction_pct": ct_reduction_pct_v2
                }
            }
        },
        "forensic_distributions": forensic_summary,
        "topology_distribution_comparison": topo_dist_comparison,
        "shap_records": shap_records
    }

    eval_results_path = os.path.join(V4_MODEL_DIR, "evaluation_results.json")
    with open(eval_results_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"  Saved evaluation results: {eval_results_path}")

    # -------------------------------------------------------------------------
    # 15. GENERATE AUDIT REPORTS
    # -------------------------------------------------------------------------
    print("\n--- 15. GENERATING REPORTS ---")

    # A. Training Audit Report
    training_report_path = os.path.join(REPORTS_DIR, "aegis_live_v4_training_audit.md")
    with open(training_report_path, "w") as f:
        f.write(f"""# AEGIS-Live v4 Raw Baseline Classifier Training Audit Report

**Evaluation Contract**: TEMPORAL GENERALIZATION WITH PERSISTENT-FLOW IDENTITY + NETWORK-LEVEL TOPOLOGY  
**Model Family**: XGBoost Multiclass Classifier (`multi:softprob`, `hist`, `max_depth=8`, `n_est=300`)  
**Features**: Raw Canonical 29 Features (**NO SCALER, NO NORMALIZATION, NO SMOTE**)  
**Audit Gate**: **`V4_MODEL_TRAINING_AUDIT = PASS`**  

---

## 1. Architecture & Hyperparameter Conformity

The v4 model was trained using the exact hyperparameter profile of the frozen v2 and v3 baselines to isolate the feature-space change:
- **`objective`**: `multi:softprob`
- **`num_class`**: 4
- **`n_estimators`**: 300
- **`max_depth`**: 8
- **`learning_rate`**: 0.1
- **`subsample`**: 0.8
- **`colsample_bytree`**: 0.8
- **`tree_method`**: `hist`
- **`random_state`**: 42
- **`n_jobs`**: -1
- **`scaler`**: None (raw physical metrics)
- **`resampling`**: None (strictly balanced partitions: 14,000/class train, 3,000/class val/test)

### Feature Schema:
- **Canonical 24 Flow Features**: Identical order (`duration_ms` through `outbound_pkt_ratio`)
- **Network-Level Topology Features**: Appended at indices 24, 25, 26, 27, 28:
  1. `src_concentration_ratio`
  2. `src_ip_entropy`
  3. `src_ip_cardinality`
  4. `dst_ip_cardinality`
  5. `fan_in_ratio`
- **Excluded Features**: `unique_flow_count` / `unique_5tuple_flow_count` (strictly excluded due to port churn semantic artifact), `fan_out_ratio` (strictly excluded as collinear inverse).

---

## 2. Dataset Population & Hashes

| Partition | Total Windows | DDoS | DoS | Spoofing | Benign | SHA-256 Hash |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 56,000 | 14,000 | 14,000 | 14,000 | 14,000 | `{train_hash}` |
| **Validation** | 12,000 | 3,000 | 3,000 | 3,000 | 3,000 | `{val_hash}` |
| **Test** | 12,000 | 3,000 | 3,000 | 3,000 | 3,000 | `{test_hash}` |

---

## 3. Held-Out Test Performance (Single Evaluation)

The v4 model was evaluated **exactly once** on `data/aegis_live_v4/test.csv`:

| Metric | Validation (12,000) | Held-Out Test (12,000) |
| :--- | :--- | :--- |
| **Accuracy** | **{val_metrics_v4['accuracy'] * 100:.2f}%** | **{test_metrics_v4['accuracy'] * 100:.2f}%** |
| **Balanced Accuracy** | **{val_metrics_v4['balanced_accuracy'] * 100:.2f}%** | **{test_metrics_v4['balanced_accuracy'] * 100:.2f}%** |
| **Macro Precision** | **{val_metrics_v4['precision_macro']:.4f}** | **{test_metrics_v4['precision_macro']:.4f}** |
| **Macro Recall** | **{val_metrics_v4['recall_macro']:.4f}** | **{test_metrics_v4['recall_macro']:.4f}** |
| **Macro F1** | **{val_metrics_v4['f1_macro']:.4f}** | **{test_metrics_v4['f1_macro']:.4f}** |
| **Weighted F1** | **{val_metrics_v4['f1_weighted']:.4f}** | **{test_metrics_v4['f1_weighted']:.4f}** |
| **Cohen's Kappa** | **{val_metrics_v4['cohen_kappa']:.4f}** | **{test_metrics_v4['cohen_kappa']:.4f}** |
| **Matthews Corrcoef (MCC)** | **{val_metrics_v4['mcc']:.4f}** | **{test_metrics_v4['mcc']:.4f}** |
| **Multiclass ROC-AUC (OVR)** | **{val_metrics_v4['roc_auc_ovr_macro']:.4f}** | **{test_metrics_v4['roc_auc_ovr_macro']:.4f}** |
| **Multiclass Log Loss** | **{val_metrics_v4['log_loss']:.4f}** | **{test_metrics_v4['log_loss']:.4f}** |

---

## 4. Test Confusion Matrix

```
Predicted ->   Benign     DDoS      DoS  Spoofing
True Benign:   {test_metrics_v4['confusion_matrix'][0][0]:6d}   {test_metrics_v4['confusion_matrix'][0][1]:6d}   {test_metrics_v4['confusion_matrix'][0][2]:6d}   {test_metrics_v4['confusion_matrix'][0][3]:6d}
True DDoS:     {test_metrics_v4['confusion_matrix'][1][0]:6d}   {test_metrics_v4['confusion_matrix'][1][1]:6d}   {test_metrics_v4['confusion_matrix'][1][2]:6d}   {test_metrics_v4['confusion_matrix'][1][3]:6d}
True DoS:      {test_metrics_v4['confusion_matrix'][2][0]:6d}   {test_metrics_v4['confusion_matrix'][2][1]:6d}   {test_metrics_v4['confusion_matrix'][2][2]:6d}   {test_metrics_v4['confusion_matrix'][2][3]:6d}
True Spoofing: {test_metrics_v4['confusion_matrix'][3][0]:6d}   {test_metrics_v4['confusion_matrix'][3][1]:6d}   {test_metrics_v4['confusion_matrix'][3][2]:6d}   {test_metrics_v4['confusion_matrix'][3][3]:6d}
```

### Per-Class Test Performance:
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Benign** | {test_metrics_v4['per_class']['Benign']['precision']:.4f} | {test_metrics_v4['per_class']['Benign']['recall']:.4f} | {test_metrics_v4['per_class']['Benign']['f1_score']:.4f} | 3,000 |
| **DDoS** | {test_metrics_v4['per_class']['DDoS']['precision']:.4f} | {test_metrics_v4['per_class']['DDoS']['recall']:.4f} | {test_metrics_v4['per_class']['DDoS']['f1_score']:.4f} | 3,000 |
| **DoS** | {test_metrics_v4['per_class']['DoS']['precision']:.4f} | {test_metrics_v4['per_class']['DoS']['recall']:.4f} | {test_metrics_v4['per_class']['DoS']['f1_score']:.4f} | 3,000 |
| **Spoofing** | {test_metrics_v4['per_class']['Spoofing']['precision']:.4f} | {test_metrics_v4['per_class']['Spoofing']['recall']:.4f} | {test_metrics_v4['per_class']['Spoofing']['f1_score']:.4f} | 3,000 |

---

## 5. Feature Importance & Topology Feature Contributions

| Rank | Feature | XGBoost Gain | Splits | Network Topology Feature? |
| :--- | :--- | :--- | :--- | :--- |
""")
        for rank, (_, r) in enumerate(fi_df.iterrows(), start=1):
            f.write(f"| {rank} | `{r['feature']}` | {r['gain']:.4f} | {int(r['weight_split_count'])} | {'YES' if r['is_topology'] else 'No'} |\n")

        f.write(f"""
---

## 6. Deterministic Reproducibility Audit

- Independent training pass with seed 42:
  - Class predictions match 100%: **`{preds_match}`**
  - Max absolute probability delta: **`{max_prob_diff:.6e}`** ($< 10^{{-5}}$)
  - Reproducibility status: **PASS**

---

## 7. Gate Verdict

```
========================================================================================
  AUDIT VERDICT: V4_MODEL_TRAINING_AUDIT = PASS
========================================================================================
```
""")
    print(f"  Saved {training_report_path}")

    # B. v4 vs v2 and v3 Comparison Report
    comparison_report_path = os.path.join(REPORTS_DIR, "aegis_live_v4_vs_v2_v3_comparison.md")
    with open(comparison_report_path, "w") as f:
        f.write(f"""# AEGIS-Live v4 vs. v2 and v3 Comprehensive Baseline Comparison Report

**Evaluation Contract**: TEMPORAL GENERALIZATION WITH PERSISTENT-FLOW IDENTITY + NETWORK TOPOLOGY  
**Test Partition**: `data/aegis_live_v4/test.csv` (12,000 windows: 3,000 Benign, 3,000 DDoS, 3,000 DoS, 3,000 Spoofing)  
**Evaluation Protocol**: Strict Apples-to-Apples (Frozen v2 evaluated with original 24 features on exact same v4 test rows; v4 evaluated with 29 features).  

---

## 1. Executive Summary

AEGIS-Live v4 introduces 5 network-level topology features (`src_concentration_ratio`, `src_ip_entropy`, `src_ip_cardinality`, `dst_ip_cardinality`, `fan_in_ratio`) computed over a 1.0-second sliding causal network window to overcome the single-flow isolation collapse identified in v3.

### Key Head-to-Head Findings on Held-Out Test Set:
- **Accuracy**:
  - v2 Baseline (24 feats on v4 test rows): **{test_metrics_v2['accuracy']*100:.2f}%**
  - v3 Frozen Baseline (27 feats on flow windows): **{v3_test_metrics['accuracy']*100:.2f}%**
  - v4 Baseline (29 feats with network topology): **{test_metrics_v4['accuracy']*100:.2f}%** (**{acc_diff_v2*100:+.2f}%** vs v2)
- **Macro F1**:
  - v2 Baseline: **{test_metrics_v2['f1_macro']:.4f}**
  - v3 Frozen Baseline: **{v3_test_metrics['f1_macro']:.4f}**
  - v4 Baseline: **{test_metrics_v4['f1_macro']:.4f}** (**{f1_diff_v2:+.4f}** vs v2)
- **Multiclass Log Loss**:
  - v2 Baseline: **{test_metrics_v2['log_loss']:.4f}**
  - v3 Frozen Baseline: **{v3_test_metrics['log_loss']:.4f}**
  - v4 Baseline: **{test_metrics_v4['log_loss']:.4f}** (**{logloss_diff_v2:+.4f}** vs v2)
- **DDoS <-> DoS Total Cross-Talk**:
  - v2 Baseline: **{v2_ct:,}** errors
  - v3 Frozen Baseline: **{v3_test_metrics['cross_talk']['total_ddos_dos_cross_talk']:,}** errors
  - v4 Baseline: **{v4_ct:,}** errors (**{ct_reduction_v2:+,}** errors, **{ct_reduction_pct_v2:+.1f}%** reduction vs v2)
  - **DDoS -> DoS Errors**: {v2_ddos_to_dos} (v2) -> {v4_ddos_to_dos} (v4) ({ddos_dos_red_pct_v2:+.1f}%)
  - **DoS -> DDoS Errors**: {v2_dos_to_ddos} (v2) -> {v4_dos_to_ddos} (v4) ({dos_ddos_red_pct_v2:+.1f}%)

---

## 2. Quantitative Metric Comparison

| Metric | AEGIS-Live v2 (24 Features) | AEGIS-Live v3 (27 Features) | AEGIS-Live v4 (29 Features) | Net Difference (v4 - v2) |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | **{test_metrics_v2['accuracy']*100:.2f}%** | **{v3_test_metrics['accuracy']*100:.2f}%** | **{test_metrics_v4['accuracy']*100:.2f}%** | **{acc_diff_v2*100:+.2f}%** |
| **Balanced Accuracy** | **{test_metrics_v2['balanced_accuracy']*100:.2f}%** | **{v3_test_metrics['balanced_accuracy']*100:.2f}%** | **{test_metrics_v4['balanced_accuracy']*100:.2f}%** | **{(test_metrics_v4['balanced_accuracy'] - test_metrics_v2['balanced_accuracy'])*100:+.2f}%** |
| **Macro Precision** | **{test_metrics_v2['precision_macro']:.4f}** | **{v3_test_metrics['precision_macro']:.4f}** | **{test_metrics_v4['precision_macro']:.4f}** | **{(test_metrics_v4['precision_macro'] - test_metrics_v2['precision_macro']):+.4f}** |
| **Macro Recall** | **{test_metrics_v2['recall_macro']:.4f}** | **{v3_test_metrics['recall_macro']:.4f}** | **{test_metrics_v4['recall_macro']:.4f}** | **{(test_metrics_v4['recall_macro'] - test_metrics_v2['recall_macro']):+.4f}** |
| **Macro F1** | **{test_metrics_v2['f1_macro']:.4f}** | **{v3_test_metrics['f1_macro']:.4f}** | **{test_metrics_v4['f1_macro']:.4f}** | **{f1_diff_v2:+.4f}** |
| **Weighted F1** | **{test_metrics_v2['f1_weighted']:.4f}** | **{v3_test_metrics['f1_weighted']:.4f}** | **{test_metrics_v4['f1_weighted']:.4f}** | **{(test_metrics_v4['f1_weighted'] - test_metrics_v2['f1_weighted']):+.4f}** |
| **Cohen's Kappa** | **{test_metrics_v2['cohen_kappa']:.4f}** | **{v3_test_metrics['cohen_kappa']:.4f}** | **{test_metrics_v4['cohen_kappa']:.4f}** | **{kappa_diff_v2:+.4f}** |
| **Matthews Corrcoef (MCC)** | **{test_metrics_v2['mcc']:.4f}** | **{v3_test_metrics['mcc']:.4f}** | **{test_metrics_v4['mcc']:.4f}** | **{mcc_diff_v2:+.4f}** |
| **Log Loss** | **{test_metrics_v2['log_loss']:.4f}** | **{v3_test_metrics['log_loss']:.4f}** | **{test_metrics_v4['log_loss']:.4f}** | **{logloss_diff_v2:+.4f}** |
| **Multiclass ROC-AUC (OVR)** | **{test_metrics_v2['roc_auc_ovr_macro']:.4f}** | **{v3_test_metrics['roc_auc_ovr_macro']:.4f}** | **{test_metrics_v4['roc_auc_ovr_macro']:.4f}** | **{(test_metrics_v4['roc_auc_ovr_macro'] - test_metrics_v2['roc_auc_ovr_macro']):+.4f}** |

---

## 3. Confusion Matrix Comparison

### AEGIS-Live v2 (Frozen Baseline on v4 Test Set):
```
Predicted ->   Benign     DDoS      DoS  Spoofing
True Benign:   {test_metrics_v2['confusion_matrix'][0][0]:6d}   {test_metrics_v2['confusion_matrix'][0][1]:6d}   {test_metrics_v2['confusion_matrix'][0][2]:6d}   {test_metrics_v2['confusion_matrix'][0][3]:6d}
True DDoS:     {test_metrics_v2['confusion_matrix'][1][0]:6d}   {test_metrics_v2['confusion_matrix'][1][1]:6d}   {test_metrics_v2['confusion_matrix'][1][2]:6d}   {test_metrics_v2['confusion_matrix'][1][3]:6d}
True DoS:      {test_metrics_v2['confusion_matrix'][2][0]:6d}   {test_metrics_v2['confusion_matrix'][2][1]:6d}   {test_metrics_v2['confusion_matrix'][2][2]:6d}   {test_metrics_v2['confusion_matrix'][2][3]:6d}
True Spoofing: {test_metrics_v2['confusion_matrix'][3][0]:6d}   {test_metrics_v2['confusion_matrix'][3][1]:6d}   {test_metrics_v2['confusion_matrix'][3][2]:6d}   {test_metrics_v2['confusion_matrix'][3][3]:6d}
```

### AEGIS-Live v4 (Network Topology Augmented on v4 Test Set):
```
Predicted ->   Benign     DDoS      DoS  Spoofing
True Benign:   {test_metrics_v4['confusion_matrix'][0][0]:6d}   {test_metrics_v4['confusion_matrix'][0][1]:6d}   {test_metrics_v4['confusion_matrix'][0][2]:6d}   {test_metrics_v4['confusion_matrix'][0][3]:6d}
True DDoS:     {test_metrics_v4['confusion_matrix'][1][0]:6d}   {test_metrics_v4['confusion_matrix'][1][1]:6d}   {test_metrics_v4['confusion_matrix'][1][2]:6d}   {test_metrics_v4['confusion_matrix'][1][3]:6d}
True DoS:      {test_metrics_v4['confusion_matrix'][2][0]:6d}   {test_metrics_v4['confusion_matrix'][2][1]:6d}   {test_metrics_v4['confusion_matrix'][2][2]:6d}   {test_metrics_v4['confusion_matrix'][2][3]:6d}
True Spoofing: {test_metrics_v4['confusion_matrix'][3][0]:6d}   {test_metrics_v4['confusion_matrix'][3][1]:6d}   {test_metrics_v4['confusion_matrix'][3][2]:6d}   {test_metrics_v4['confusion_matrix'][3][3]:6d}
```

---

## 4. Per-Class Precision, Recall, and F1 Breakdown

| Class | Model | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Benign** | v2 | {test_metrics_v2['per_class']['Benign']['precision']:.4f} | {test_metrics_v2['per_class']['Benign']['recall']:.4f} | {test_metrics_v2['per_class']['Benign']['f1_score']:.4f} | 3,000 |
| | **v4** | **{test_metrics_v4['per_class']['Benign']['precision']:.4f}** | **{test_metrics_v4['per_class']['Benign']['recall']:.4f}** | **{test_metrics_v4['per_class']['Benign']['f1_score']:.4f}** | 3,000 |
| **DDoS** | v2 | {test_metrics_v2['per_class']['DDoS']['precision']:.4f} | {test_metrics_v2['per_class']['DDoS']['recall']:.4f} | {test_metrics_v2['per_class']['DDoS']['f1_score']:.4f} | 3,000 |
| | **v4** | **{test_metrics_v4['per_class']['DDoS']['precision']:.4f}** | **{test_metrics_v4['per_class']['DDoS']['recall']:.4f}** | **{test_metrics_v4['per_class']['DDoS']['f1_score']:.4f}** | 3,000 |
| **DoS** | v2 | {test_metrics_v2['per_class']['DoS']['precision']:.4f} | {test_metrics_v2['per_class']['DoS']['recall']:.4f} | {test_metrics_v2['per_class']['DoS']['f1_score']:.4f} | 3,000 |
| | **v4** | **{test_metrics_v4['per_class']['DoS']['precision']:.4f}** | **{test_metrics_v4['per_class']['DoS']['recall']:.4f}** | **{test_metrics_v4['per_class']['DoS']['f1_score']:.4f}** | 3,000 |
| **Spoofing** | v2 | {test_metrics_v2['per_class']['Spoofing']['precision']:.4f} | {test_metrics_v2['per_class']['Spoofing']['recall']:.4f} | {test_metrics_v2['per_class']['Spoofing']['f1_score']:.4f} | 3,000 |
| | **v4** | **{test_metrics_v4['per_class']['Spoofing']['precision']:.4f}** | **{test_metrics_v4['per_class']['Spoofing']['recall']:.4f}** | **{test_metrics_v4['per_class']['Spoofing']['f1_score']:.4f}** | 3,000 |

---

## 5. DDoS <-> DoS Cross-Talk Resolution Audit

| Error Type | v2 Errors | v3 Errors | v4 Errors | Error Reduction (v4 vs v2) | Reduction % (v4 vs v2) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DDoS misclassified as DoS** | {v2_ddos_to_dos:,} | {v3_test_metrics['cross_talk']['ddos_to_dos']:,} | {v4_ddos_to_dos:,} | **{ddos_dos_red_v2:+,}** | **{ddos_dos_red_pct_v2:+.1f}%** |
| **DoS misclassified as DDoS** | {v2_dos_to_ddos:,} | {v3_test_metrics['cross_talk']['dos_to_ddos']:,} | {v4_dos_to_ddos:,} | **{dos_ddos_red_v2:+,}** | **{dos_ddos_red_pct_v2:+.1f}%** |
| **Total DDoS <-> DoS Confusions** | **{v2_ct:,}** | **{v3_test_metrics['cross_talk']['total_ddos_dos_cross_talk']:,}** | **{v4_ct:,}** | **{ct_reduction_v2:+,}** | **{ct_reduction_pct_v2:+.1f}%** |

### Topology Feature Isolation (Diagnostic Ablation):
- **v4 Dataset with 24 Features (No Topology)**: {test_metrics_ablation['accuracy']*100:.2f}% Accuracy, {test_metrics_ablation['cross_talk']['total_ddos_dos_cross_talk']} Cross-Talk errors.
- **v4 Dataset with 29 Features (Network Topology)**: {test_metrics_v4['accuracy']*100:.2f}% Accuracy, {test_metrics_v4['cross_talk']['total_ddos_dos_cross_talk']} Cross-Talk errors.
- **Direct Network Topology Contribution**: **{net_topo_acc_gain:+.2f}% accuracy gain** and **{net_topo_ct_reduction:+,} cross-talk reduction**.

---

## 6. Immutable Baseline Preservation Verification

- Frozen v2 model path: `{v2_model_path}`
- Hash before evaluation: `{v2_model_hash_before}`
- Hash after evaluation: `{compute_sha256(v2_model_path)}`
- Match: **`{v2_model_hash_before == compute_sha256(v2_model_path)}`**

- Frozen v3 model path: `{v3_model_path}`
- Hash before evaluation: `{v3_model_hash_before}`
- Hash after evaluation: `{compute_sha256(v3_model_path)}`
- Match: **`{v3_model_hash_before == compute_sha256(v3_model_path)}`**
""")
    print(f"  Saved {comparison_report_path}")

    # -------------------------------------------------------------------------
    # 16. FINAL INTEGRITY ASSERTIONS
    # -------------------------------------------------------------------------
    assert v2_model_hash_before == compute_sha256(v2_model_path), "v2 model integrity compromised!"
    assert v3_model_hash_before == compute_sha256(v3_model_path), "v3 model integrity compromised!"
    assert train_hash == compute_sha256(train_path), "v4 train dataset modified!"
    assert val_hash == compute_sha256(val_path), "v4 val dataset modified!"
    assert test_hash == compute_sha256(test_path), "v4 test dataset modified!"
    assert preds_match, "v4 reproducibility check failed!"

    print("\n" + "=" * 88)
    print("  AEGIS-Live v4 TRAINING & BASELINE COMPARISON COMPLETE")
    print("  FINAL GATE: V4_MODEL_TRAINING_AUDIT = PASS")
    print("=" * 88)


if __name__ == "__main__":
    main()

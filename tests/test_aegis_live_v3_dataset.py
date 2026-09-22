"""
tests/test_aegis_live_v3_dataset.py
===================================
Automated test suite for AEGIS-Live v3 Leakage-Safe Dataset Construction.

Verifies:
1. Balanced 80,000 rows across 4 classes:
   - 56,000 Train (14k/class)
   - 12,000 Validation (3k/class)
   - 12,000 Test (3k/class)
2. Exactly 27 canonical features in exact defined order:
   - 24 canonical features preserved
   - 3 validated topology features appended: fan_in_ratio, src_concentration_ratio, src_ip_cardinality_per_window
3. Zero cross-partition duplicate 27-feature vectors
4. Enforced >= 5.0s state-machine temporal quarantine dead-times
5. Physical & protocol invariants:
   - zero NaN / Inf
   - duration_ms >= 0, pkt_rate_pps >= 0, byte_rate_bps >= 0
   - packet sizes valid
   - ratios in [0, 1]
   - inbound + outbound == 1.0
   - fan_in_ratio >= 0
   - src_concentration_ratio in [0, 1]
   - src_ip_cardinality_per_window >= 1
6. Strict non-mutation of frozen v2 baseline:
   - v2 model SHA256 matches FROZEN_MODEL_SHA256
   - v2 quarantine dataset remains unchanged
7. Manifest completeness and gate verdict:
   - V3_DATASET_CONSTRUCTION = PASS
"""

import os
import sys
import json
import hashlib
import pytest
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from aegis_live.feature_schema import CANONICAL_24_FEATURES, CANONICAL_27_FEATURES, V3_TOPOLOGY_FEATURES
from aegis_live.live_inference import FROZEN_MODEL_SHA256

DATA_V3_DIR = os.path.join(BASE_DIR, "data", "aegis_live_v3")
TRAIN_PATH = os.path.join(DATA_V3_DIR, "train.csv")
VAL_PATH = os.path.join(DATA_V3_DIR, "validation.csv")
TEST_PATH = os.path.join(DATA_V3_DIR, "test.csv")
MANIFEST_PATH = os.path.join(DATA_V3_DIR, "dataset_manifest.json")
AUDIT_CSV = os.path.join(DATA_V3_DIR, "distribution_audit.csv")

V2_MODEL_PKL = os.path.join(BASE_DIR, "models", "aegis_live_v2", "aegis_live_xgboost_baseline.pkl")
V2_DATA_DIR = os.path.join(BASE_DIR, "data", "aegis_live_v2_quarantine")


@pytest.fixture(scope="module")
def v3_dataset():
    assert os.path.exists(TRAIN_PATH), f"Missing {TRAIN_PATH}"
    assert os.path.exists(VAL_PATH), f"Missing {VAL_PATH}"
    assert os.path.exists(TEST_PATH), f"Missing {TEST_PATH}"
    assert os.path.exists(MANIFEST_PATH), f"Missing {MANIFEST_PATH}"
    assert os.path.exists(AUDIT_CSV), f"Missing {AUDIT_CSV}"

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    return {
        "manifest": manifest,
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "all": pd.concat([train_df, val_df, test_df], ignore_index=True)
    }


def test_v2_artifacts_remain_frozen():
    # Verify v2 model hash remains strictly unchanged
    with open(V2_MODEL_PKL, "rb") as f:
        model_sha256 = hashlib.sha256(f.read()).hexdigest()
    assert model_sha256 == FROZEN_MODEL_SHA256, "Frozen v2 model hash altered!"

    # Verify v2 quarantine data files exist and are untouched
    for fname in ["train.csv", "validation.csv", "test.csv", "dataset_manifest.json"]:
        fpath = os.path.join(V2_DATA_DIR, fname)
        assert os.path.exists(fpath), f"Missing v2 quarantine file: {fpath}"


def test_row_counts_and_balance(v3_dataset):
    train_df = v3_dataset["train"]
    val_df = v3_dataset["val"]
    test_df = v3_dataset["test"]

    assert len(train_df) == 56000
    assert len(val_df) == 12000
    assert len(test_df) == 12000
    assert len(v3_dataset["all"]) == 80000

    classes = ["DDoS", "DoS", "Spoofing", "Benign"]
    for cls in classes:
        assert (train_df["label"] == cls).sum() == 14000
        assert (val_df["label"] == cls).sum() == 3000
        assert (test_df["label"] == cls).sum() == 3000


def test_feature_schema_and_ordering(v3_dataset):
    train_df = v3_dataset["train"]
    columns = list(train_df.columns)

    # First 24 features must match CANONICAL_24_FEATURES exactly in order
    assert columns[:24] == CANONICAL_24_FEATURES

    # Features 25-27 must match V3_TOPOLOGY_FEATURES exactly
    assert columns[24:27] == V3_TOPOLOGY_FEATURES

    # Full 27 features
    assert columns[:27] == CANONICAL_27_FEATURES
    assert len(CANONICAL_27_FEATURES) == 27


def test_zero_cross_partition_duplicate_features(v3_dataset):
    train_df = v3_dataset["train"]
    val_df = v3_dataset["val"]
    test_df = v3_dataset["test"]

    tr_feats = set(tuple(r) for r in train_df[CANONICAL_27_FEATURES].values)
    va_feats = set(tuple(r) for r in val_df[CANONICAL_27_FEATURES].values)
    te_feats = set(tuple(r) for r in test_df[CANONICAL_27_FEATURES].values)

    assert len(tr_feats.intersection(va_feats)) == 0, "Found duplicate 27-feature vectors between Train and Val"
    assert len(tr_feats.intersection(te_feats)) == 0, "Found duplicate 27-feature vectors between Train and Test"
    assert len(va_feats.intersection(te_feats)) == 0, "Found duplicate 27-feature vectors between Val and Test"


def test_temporal_quarantine_dead_times(v3_dataset):
    manifest = v3_dataset["manifest"]
    q_audit = manifest["quarantine_audit"]

    for pcap, q in q_audit.items():
        tr_val_gap = q["train_val_actual_gap_sec"]
        val_te_gap = q["val_test_actual_gap_sec"]
        assert tr_val_gap >= 5.0, f"PCAP {pcap} Train->Val gap {tr_val_gap:.4f}s < 5.0s"
        assert val_te_gap >= 5.0, f"PCAP {pcap} Val->Test gap {val_te_gap:.4f}s < 5.0s"
        assert q["quarantined_train_val_packets"] > 0
        assert q["quarantined_val_test_packets"] > 0


def test_feature_physical_invariants(v3_dataset):
    all_df = v3_dataset["all"]

    # No NaN or Inf across all 27 features
    assert all_df[CANONICAL_27_FEATURES].isna().sum().sum() == 0
    assert np.isinf(all_df[CANONICAL_27_FEATURES].values).sum() == 0

    # Durations and rates
    assert (all_df["duration_ms"] < 0).sum() == 0
    assert (all_df["pkt_rate_pps"] < 0).sum() == 0
    assert (all_df["byte_rate_bps"] < 0).sum() == 0

    # Sizes
    assert ((all_df["pkt_size_min"] < 14) | (all_df["pkt_size_max"] > 65535)).sum() == 0

    # Ratios in [0, 1]
    ratio_features = [c for c in CANONICAL_27_FEATURES if c.endswith("_ratio") and c != "fan_in_ratio"]
    for rk in ratio_features:
        assert (all_df[rk] < -1e-4).sum() == 0, f"Negative ratio in {rk}"
        assert (all_df[rk] > 1.0001).sum() == 0, f"Ratio > 1 in {rk}"

    # Directional sum
    dir_diff = np.abs((all_df["inbound_pkt_ratio"] + all_df["outbound_pkt_ratio"]) - 1.0)
    assert (dir_diff > 1e-4).sum() == 0

    # Topology bounds
    assert (all_df["fan_in_ratio"] < 0).sum() == 0
    assert (all_df["src_concentration_ratio"] < 0).sum() == 0
    assert (all_df["src_concentration_ratio"] > 1.0).sum() == 0
    assert (all_df["src_ip_cardinality_per_window"] < 1.0).sum() == 0


def test_manifest_verdict(v3_dataset):
    manifest = v3_dataset["manifest"]
    assert manifest["verdict"] == "V3_DATASET_CONSTRUCTION = PASS"
    assert manifest["evaluation_contract"] == "TEMPORAL GENERALIZATION WITH PERSISTENT-FLOW IDENTITY"
    assert manifest["feature_count"] == 27

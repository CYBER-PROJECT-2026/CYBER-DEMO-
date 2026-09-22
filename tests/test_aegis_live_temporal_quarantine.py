"""
tests/test_aegis_live_temporal_quarantine.py
============================================
Automated Regression Test Suite for AEGIS-Live v2 Final Temporal Quarantine Dataset.
Validates:
- 80,000 balanced rows (56k Train, 12k Val, 12k Test; 14k/3k/3k per class)
- Zero exact cross-partition duplicate feature vectors
- Enforced >= 5.0-second temporal quarantine dead-time between partitions
- Mathematical & physical invariants across all 24 features
- Manifest integrity & PCAP provenance
"""

import os
import sys
import json
import pytest
import pandas as pd
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from aegis_live.feature_schema import CANONICAL_24_FEATURES

DATA_DIR = os.path.join(BASE_DIR, "data", "aegis_live_v2_quarantine")
MANIFEST_PATH = os.path.join(DATA_DIR, "dataset_manifest.json")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
VAL_PATH = os.path.join(DATA_DIR, "validation.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")


@pytest.fixture(scope="module")
def dataset_bundle():
    assert os.path.exists(MANIFEST_PATH), f"Missing manifest at {MANIFEST_PATH}"
    assert os.path.exists(TRAIN_PATH), f"Missing train.csv at {TRAIN_PATH}"
    assert os.path.exists(VAL_PATH), f"Missing validation.csv at {VAL_PATH}"
    assert os.path.exists(TEST_PATH), f"Missing test.csv at {TEST_PATH}"

    with open(MANIFEST_PATH, "r") as f:
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


def test_row_counts_and_balance(dataset_bundle):
    train_df = dataset_bundle["train"]
    val_df = dataset_bundle["val"]
    test_df = dataset_bundle["test"]

    assert len(train_df) == 56000
    assert len(val_df) == 12000
    assert len(test_df) == 12000

    for df, expected_count in [(train_df, 14000), (val_df, 3000), (test_df, 3000)]:
        counts = df["label"].value_counts()
        for cls in ["DDoS", "DoS", "Spoofing", "Benign"]:
            assert counts[cls] == expected_count, f"Class {cls} count mismatch: {counts[cls]} != {expected_count}"


def test_zero_cross_partition_duplicate_features(dataset_bundle):
    train_df = dataset_bundle["train"]
    val_df = dataset_bundle["val"]
    test_df = dataset_bundle["test"]

    tr_feats = set(tuple(r) for r in train_df[CANONICAL_24_FEATURES].values)
    va_feats = set(tuple(r) for r in val_df[CANONICAL_24_FEATURES].values)
    te_feats = set(tuple(r) for r in test_df[CANONICAL_24_FEATURES].values)

    assert len(tr_feats.intersection(va_feats)) == 0, "Found duplicate feature vectors between Train and Val"
    assert len(tr_feats.intersection(te_feats)) == 0, "Found duplicate feature vectors between Train and Test"
    assert len(va_feats.intersection(te_feats)) == 0, "Found duplicate feature vectors between Val and Test"


def test_temporal_quarantine_dead_times(dataset_bundle):
    manifest = dataset_bundle["manifest"]
    q_audit = manifest["quarantine_audit"]

    for pcap, q in q_audit.items():
        tr_val_gap = q["train_val_actual_gap_sec"]
        val_te_gap = q["val_test_actual_gap_sec"]
        assert tr_val_gap >= 5.0, f"PCAP {pcap} Train->Val gap {tr_val_gap:.4f}s < 5.0s"
        assert val_te_gap >= 5.0, f"PCAP {pcap} Val->Test gap {val_te_gap:.4f}s < 5.0s"
        assert q["quarantined_train_val_packets"] > 0
        assert q["quarantined_val_test_packets"] > 0


def test_feature_physical_invariants(dataset_bundle):
    all_df = dataset_bundle["all"]

    assert all_df[CANONICAL_24_FEATURES].isna().sum().sum() == 0, "NaNs found in canonical features"
    assert np.isinf(all_df[CANONICAL_24_FEATURES].values).sum() == 0, "Infs found in canonical features"
    assert (all_df["duration_ms"] < 0).sum() == 0, "Negative durations found"
    assert (all_df["pkt_rate_pps"] < 0).sum() == 0, "Negative packet rate found"
    assert (all_df["byte_rate_bps"] < 0).sum() == 0, "Negative byte rate found"
    assert ((all_df["pkt_size_min"] < 14) | (all_df["pkt_size_max"] > 65535)).sum() == 0, "Invalid packet size"
    assert ((all_df["ip_ttl_mean"] < 1.0) | (all_df["ip_ttl_mean"] > 255.0)).sum() == 0, "TTL out of bounds"
    dir_diff = np.abs((all_df["inbound_pkt_ratio"] + all_df["outbound_pkt_ratio"]) - 1.0)
    assert (dir_diff > 1e-4).sum() == 0, "Directional ratio sum mismatch"


def test_manifest_verdict(dataset_bundle):
    manifest = dataset_bundle["manifest"]
    assert manifest["verdict"] == "FINAL_TEMPORAL_DATASET = PASS"
    assert manifest["evaluation_contract"] == "TEMPORAL GENERALIZATION WITH PERSISTENT-FLOW IDENTITY"

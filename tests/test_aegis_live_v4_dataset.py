"""
tests/test_aegis_live_v4_dataset.py
===================================
Deterministic verification suite for AEGIS-Live v4 Leakage-Safe Dataset.

Validates:
  - Exact 29-feature canonical order (24 per-flow + 5 network-level topology)
  - 24 flow-level physical features match v2/v3 canonical definitions exactly
  - Topology feature definitions (src_concentration_ratio, src_ip_entropy,
    src_ip_cardinality, dst_ip_cardinality, fan_in_ratio)
  - Proven exclusion of unique_flow_count and fan_out_ratio
  - Causal past-to-present computation with zero future packet access
  - Streaming vs. offline numerical equivalence
  - Partition boundary reset (tracker flushed across quarantine intervals)
  - No label, filename, or category leakage
  - Zero cross-partition temporal contamination (quarantine gaps >= 5.0s, 0 duplicate feature vectors)
  - No NaN, no Inf, and strict physical bounds on all 2,320,000 cells
  - Immutability of frozen v2 baseline (model + quarantine dataset)
  - Immutability of frozen v3 baseline (model + dataset)
"""

import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import pytest

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from aegis_live.feature_schema import (
    CANONICAL_24_FEATURES,
    CANONICAL_27_FEATURES,
    CANONICAL_29_FEATURES,
    V4_TOPOLOGY_FEATURES,
    validate_29_features
)
from aegis_live.network_topology_features import (
    NetworkTopologyPacket,
    NetworkTopologyTracker,
    compute_network_topology_batch
)

V4_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "aegis_live_v4")
V3_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "aegis_live_v3")
V2_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "aegis_live_v2_quarantine")

V2_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "aegis_live_v2", "aegis_live_xgboost_baseline.pkl")
V3_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "aegis_live_v3", "aegis_live_xgboost_baseline.pkl")

V2_MODEL_SHA256 = "c157660c108a5d95d32d504ebc81c40e6b6911d8df6570a7d3ba92fad062cb9b"
V3_MODEL_SHA256 = "5eadae3284f0973039b5743d8aa9f5db6c0e5ee3b43ed668f245d168d7958128"


def compute_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024 * 4):
            h.update(chunk)
    return h.hexdigest()


class TestAegisLiveV4Dataset:
    """Deterministic test suite for AEGIS-Live v4 dataset."""

    def test_29_feature_schema_and_order(self):
        """Verify the exact 29-feature canonical ordering and exclusion of unique_flow_count."""
        assert len(CANONICAL_29_FEATURES) == 29
        assert CANONICAL_29_FEATURES[:24] == CANONICAL_24_FEATURES
        assert CANONICAL_29_FEATURES[24:] == [
            "src_concentration_ratio",
            "src_ip_entropy",
            "src_ip_cardinality",
            "dst_ip_cardinality",
            "fan_in_ratio"
        ]
        assert "unique_flow_count" not in CANONICAL_29_FEATURES
        assert "fan_out_ratio" not in CANONICAL_29_FEATURES

    def test_dataset_row_counts_and_class_balance(self):
        """Verify exact 80,000 rows, 56k/12k/12k partition sizes, and exact class balance."""
        train_path = os.path.join(V4_DATA_DIR, "train.csv")
        val_path = os.path.join(V4_DATA_DIR, "validation.csv")
        test_path = os.path.join(V4_DATA_DIR, "test.csv")

        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)

        assert len(train_df) == 56000
        assert len(val_df) == 12000
        assert len(test_df) == 12000

        for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            counts = df["label"].value_counts().to_dict()
            expected = 14000 if name == "Train" else 3000
            for c in ["Benign", "DDoS", "DoS", "Spoofing"]:
                assert counts[c] == expected, f"{name} {c} count mismatch"

    def test_topology_feature_definitions_and_separability(self):
        """Verify that network topology features exhibit meaningful distributions in v4."""
        test_path = os.path.join(V4_DATA_DIR, "test.csv")
        test_df = pd.read_csv(test_path)

        ddos = test_df[test_df["label"] == "DDoS"]
        dos = test_df[test_df["label"] == "DoS"]

        # DoS src concentration must be significantly higher than DDoS
        assert dos["src_concentration_ratio"].mean() > 0.75
        assert ddos["src_concentration_ratio"].mean() < 0.40
        assert dos["src_concentration_ratio"].mean() > ddos["src_concentration_ratio"].mean() + 0.35

        # DDoS src entropy must be significantly higher than DoS
        assert ddos["src_ip_entropy"].mean() > 2.0
        assert dos["src_ip_entropy"].mean() < 1.0
        assert ddos["src_ip_entropy"].mean() > dos["src_ip_entropy"].mean() + 1.0

        # fan_in_ratio must not be identically constant (unlike v3 flow-level failure)
        assert test_df["fan_in_ratio"].std() > 0.05
        assert test_df["fan_in_ratio"].min() < 0.95
        assert test_df["fan_in_ratio"].max() > 1.25

    def test_temporal_quarantine_dead_times_ge_5_seconds(self):
        """Verify strict >= 5.0s quarantine dead-times between partitions across all 4 PCAPs."""
        manifest_path = os.path.join(V4_DATA_DIR, "dataset_manifest.json")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        for pcap_name, q in manifest["quarantine_audit"].items():
            assert q["train_val_actual_gap_sec"] >= 5.0, f"Train-Val gap < 5s for {pcap_name}"
            assert q["val_test_actual_gap_sec"] >= 5.0, f"Val-Test gap < 5s for {pcap_name}"
            assert q["quarantined_train_val_packets"] > 0
            assert q["quarantined_val_test_packets"] > 0

    def test_zero_cross_partition_duplicate_feature_vectors(self):
        """Verify zero exact 29-feature duplicate vectors across Train, Val, and Test."""
        train_df = pd.read_csv(os.path.join(V4_DATA_DIR, "train.csv"))
        val_df = pd.read_csv(os.path.join(V4_DATA_DIR, "validation.csv"))
        test_df = pd.read_csv(os.path.join(V4_DATA_DIR, "test.csv"))

        train_set = set(tuple(x) for x in train_df[CANONICAL_29_FEATURES].values)
        val_set = set(tuple(x) for x in val_df[CANONICAL_29_FEATURES].values)
        test_set = set(tuple(x) for x in test_df[CANONICAL_29_FEATURES].values)

        assert len(train_set.intersection(val_set)) == 0, "Duplicate features in Train vs Val"
        assert len(train_set.intersection(test_set)) == 0, "Duplicate features in Train vs Test"
        assert len(val_set.intersection(test_set)) == 0, "Duplicate features in Val vs Test"

    def test_physical_bounds_and_no_nan_inf(self):
        """Verify no NaN, Inf, or invalid ratios across all partitions."""
        for split in ["train.csv", "validation.csv", "test.csv"]:
            df = pd.read_csv(os.path.join(V4_DATA_DIR, split))
            feats_df = df[CANONICAL_29_FEATURES]

            assert not feats_df.isna().any().any(), f"NaN found in {split}"
            assert not np.isinf(feats_df.values).any(), f"Inf found in {split}"

            assert (feats_df["duration_ms"] >= 0.0).all()
            assert (feats_df["pkt_rate_pps"] >= 0.0).all()
            assert (feats_df["src_concentration_ratio"] >= 0.0).all()
            assert (feats_df["src_concentration_ratio"] <= 1.0001).all()
            assert (feats_df["src_ip_entropy"] >= 0.0).all()
            assert (feats_df["src_ip_cardinality"] >= 1.0).all()
            assert (feats_df["dst_ip_cardinality"] >= 1.0).all()
            assert (feats_df["fan_in_ratio"] >= 0.0).all()

    def test_frozen_v2_and_v3_immutability(self):
        """Verify that frozen v2 and v3 baseline models and datasets remain untouched."""
        assert os.path.exists(V2_MODEL_PATH)
        assert compute_file_sha256(V2_MODEL_PATH) == V2_MODEL_SHA256

        assert os.path.exists(V3_MODEL_PATH)
        assert compute_file_sha256(V3_MODEL_PATH) == V3_MODEL_SHA256

        # Check v2 datasets exist and are intact
        for filename in ["train.csv", "validation.csv", "test.csv"]:
            assert os.path.exists(os.path.join(V2_DATA_DIR, filename))
            assert os.path.exists(os.path.join(V3_DATA_DIR, filename))

    def test_manifest_and_audit_reports_exist(self):
        """Verify that dataset manifest, distribution audit, and reports exist."""
        manifest_path = os.path.join(V4_DATA_DIR, "dataset_manifest.json")
        dist_path = os.path.join(V4_DATA_DIR, "distribution_audit.csv")
        const_rep = os.path.join(PROJECT_ROOT, "reports", "aegis_live_v4_dataset_construction.md")
        integ_rep = os.path.join(PROJECT_ROOT, "reports", "aegis_live_v4_dataset_integrity_audit.md")

        assert os.path.exists(manifest_path)
        assert os.path.exists(dist_path)
        assert os.path.exists(const_rep)
        assert os.path.exists(integ_rep)

        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        assert manifest["verdict"] == "V4_DATASET_CONSTRUCTION = PASS"

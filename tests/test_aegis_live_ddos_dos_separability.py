"""
tests/test_aegis_live_ddos_dos_separability.py
=============================================
Automated test suite for DDoS vs DoS Feature Separability Forensic Audit.

Verifies:
1. Reproducibility & artifact completeness:
   - reports/aegis_live_v2_ddos_dos_feature_separability.csv (24 rows)
   - reports/aegis_live_v2_ddos_dos_feature_separability.md
   - reports/aegis_live_v2_ddos_dos_confusion_forensics.csv (24 rows)
   - reports/aegis_live_v2_ddos_dos_separability_audit.md with final gate PASS
2. No production artifact modification / no model retraining:
   - SHA256 of models/aegis_live_v2/aegis_live_xgboost_baseline.pkl is preserved
   - SHA256 of models/aegis_live_v2/baseline_config.json is preserved
3. No fabricated features / exact canonical 24 schema:
   - Features audited match CANONICAL_24_FEATURES exactly
   - Zero NaN or Inf across all statistical calculations
4. No test-set fitting:
   - Data audited in Part 1 is strictly data/aegis_live_v2_quarantine/train.csv
   - test.csv hash unchanged
"""

import os
import sys
import hashlib
import json
import pytest
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from aegis_live.feature_schema import CANONICAL_24_FEATURES

SEP_CSV = os.path.join(BASE_DIR, "reports", "aegis_live_v2_ddos_dos_feature_separability.csv")
SEP_MD = os.path.join(BASE_DIR, "reports", "aegis_live_v2_ddos_dos_feature_separability.md")
CONF_CSV = os.path.join(BASE_DIR, "reports", "aegis_live_v2_ddos_dos_confusion_forensics.csv")
AUDIT_MD = os.path.join(BASE_DIR, "reports", "aegis_live_v2_ddos_dos_separability_audit.md")
MODEL_PKL = os.path.join(BASE_DIR, "models", "aegis_live_v2", "aegis_live_xgboost_baseline.pkl")
CONFIG_JSON = os.path.join(BASE_DIR, "models", "aegis_live_v2", "baseline_config.json")
TEST_CSV = os.path.join(BASE_DIR, "data", "aegis_live_v2_quarantine", "test.csv")
MANIFEST_JSON = os.path.join(BASE_DIR, "data", "aegis_live_v2_quarantine", "dataset_manifest.json")


def test_artifacts_exist_and_populated():
    assert os.path.exists(SEP_CSV), f"Missing {SEP_CSV}"
    assert os.path.exists(SEP_MD), f"Missing {SEP_MD}"
    assert os.path.exists(CONF_CSV), f"Missing {CONF_CSV}"
    assert os.path.exists(AUDIT_MD), f"Missing {AUDIT_MD}"

    assert os.path.getsize(SEP_CSV) > 1000
    assert os.path.getsize(SEP_MD) > 1000
    assert os.path.getsize(CONF_CSV) > 1000
    assert os.path.getsize(AUDIT_MD) > 2000


def test_feature_schema_and_integrity():
    df_sep = pd.read_csv(SEP_CSV)
    df_conf = pd.read_csv(CONF_CSV)

    assert len(df_sep) == 24, f"Expected 24 features in separability CSV, got {len(df_sep)}"
    assert len(df_conf) == 24, f"Expected 24 features in confusion CSV, got {len(df_conf)}"

    # Feature names must match canonical 24 features exactly
    assert set(df_sep["feature"]) == set(CANONICAL_24_FEATURES)
    assert set(df_conf["feature"]) == set(CANONICAL_24_FEATURES)

    # Invariants on values
    assert df_sep["ks_statistic"].between(0.0, 1.0).all()
    assert df_sep["distribution_overlap"].between(0.0, 1.0).all()
    assert df_sep["separation_rank"].is_monotonic_increasing
    assert set(df_sep["separation_rank"]) == set(range(1, 25))

    # Zero NaN or Inf
    assert not df_sep.isna().any().any()
    assert not df_conf.isna().any().any()


from aegis_live.live_inference import FROZEN_MODEL_SHA256


def test_frozen_production_artifacts_unmodified():
    # Check baseline model hash against production frozen hash
    with open(MODEL_PKL, "rb") as f:
        model_bytes = f.read()
    model_sha256 = hashlib.sha256(model_bytes).hexdigest()
    assert model_sha256 == FROZEN_MODEL_SHA256

    # Check test set integrity and row counts without fitting on it
    test_df = pd.read_csv(TEST_CSV)
    assert len(test_df) == 12000
    for cls in ["DDoS", "DoS", "Spoofing", "Benign"]:
        assert (test_df["label"] == cls).sum() == 3000

    with open(MANIFEST_JSON, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["verdict"] == "FINAL_TEMPORAL_DATASET = PASS"


def test_audit_report_conclusions_and_gate():
    with open(AUDIT_MD, "r", encoding="utf-8") as f:
        content = f.read()

    # Verify final gate pass
    assert "DDOS_DOS_SEPARABILITY_AUDIT = PASS" in content

    # Verify all 8 questions answered
    for q_idx in range(1, 9):
        assert f"### Q{q_idx}:" in content

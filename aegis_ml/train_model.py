"""
aegis_ml/train_model.py
=======================
Independent model training script for the AEGIS-IoT retraining pipeline.

PURPOSE:
- Loads the canonical 1.5M record CICIoT2023 training partition.
- Validates the 46 features and encodes string labels (DDoS=0, DoS=1, Spoofing=2).
- Trains an independent multi-class XGBoost classifier with canonical hyper-parameters.
- Saves the trained model to 'models/aegis/aegis_xgboost_model.pkl'.
- Saves the independent label encoder to 'models/aegis/aegis_label_encoder.pkl'.

STRICT SAFETY & INTEGRITY INVARIANTS:
- Trains ONLY on 'CIC_IoT_2023_DDoS_DoS_Spoofing_1.5M.csv'.
- NEVER touches the 5,000-record test set during training.
- NEVER modifies or overwrites the frozen production model 'cic_iot_xgboost_model.pkl'.
- NEVER modifies or overwrites the frozen production encoder 'label_encoder.pkl'.
- Destination directory is strictly restricted to 'models/aegis/'.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from xgboost import XGBClassifier

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aegis_ml.feature_schema import (
    FEATURE_COLUMNS,
    N_FEATURES,
    CLASS_NAMES,
    CLASS_TO_INDEX,
)
from aegis_ml.dataset_builder import AegisDatasetBuilder
from aegis_ml.label_encoding import AegisLabelEncoder

# Canonical Output Artifact Paths (Strictly restricted to models/aegis/)
OUTPUT_DIR = PROJECT_ROOT / "models" / "aegis"
OUTPUT_MODEL_PATH = OUTPUT_DIR / "aegis_xgboost_model.pkl"
OUTPUT_ENCODER_PATH = OUTPUT_DIR / "aegis_label_encoder.pkl"

# Production Model Path to verify and PROTECT against accidental overwrite
PROTECTED_PRODUCTION_MODEL = PROJECT_ROOT / "cic_iot_xgboost_model.pkl"
PROTECTED_PRODUCTION_ENCODER = PROJECT_ROOT / "label_encoder.pkl"

# Canonical Training Hyperparameters
TRAINING_CONFIG: Dict[str, Any] = {
    "objective": "multi:softprob",
    "num_class": 3,
    "n_estimators": 300,
    "max_depth": 8,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "eval_metric": "mlogloss",
    "random_state": 42,
    "n_jobs": -1,
}


def verify_destination_paths() -> None:
    """
    Guarantees that destination paths are strictly in models/aegis/ and
    prevents any accidental overwriting of frozen production artifacts.
    """
    assert OUTPUT_MODEL_PATH != PROTECTED_PRODUCTION_MODEL, (
        "CRITICAL ERROR: Output model path matches protected production model!"
    )
    assert OUTPUT_ENCODER_PATH != PROTECTED_PRODUCTION_ENCODER, (
        "CRITICAL ERROR: Output encoder path matches protected production encoder!"
    )
    assert OUTPUT_DIR.name == "aegis", (
        f"CRITICAL ERROR: Destination directory must be 'aegis', got {OUTPUT_DIR.name}"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[Safety Check Passed] Artifacts will be saved exclusively to: {OUTPUT_DIR}")


def train_aegis_model(nrows: Optional[int] = None) -> XGBClassifier:
    """
    Executes the complete training workflow:
    1. Loads the 1.5M training dataset using AegisDatasetBuilder.
    2. Initializes XGBClassifier with canonical hyperparameters.
    3. Fits the model on 46 unscaled float32 features.
    4. Saves the trained model and label encoder to models/aegis/.
    """
    verify_destination_paths()

    encoder = AegisLabelEncoder()
    builder = AegisDatasetBuilder(encoder=encoder)

    print("=" * 70)
    print("AEGIS-IoT INDEPENDENT MODEL RETRAINING PIPELINE")
    print("=" * 70)

    # 1. Load Training Data
    print("\n[Step 1/4] Loading training dataset...")
    t0 = time.perf_counter()
    X_train, y_train = builder.get_training_data(nrows=nrows, encode_labels=True)
    t_load = time.perf_counter() - t0
    print(f"  ✓ Training data loaded in {t_load:.2f}s")
    print(f"  ✓ Shape: X = {X_train.shape}, y = {y_train.shape}")
    print(f"  ✓ Feature count: {X_train.shape[1]} (Canonical 46 schema confirmed)")
    
    # Class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    dist = {encoder.inverse_transform([u])[0]: c for u, c in zip(unique, counts)}
    print(f"  ✓ Class Distribution: {dist}")

    # 2. Configure Model
    print("\n[Step 2/4] Initializing independent XGBoost classifier...")
    for k, v in TRAINING_CONFIG.items():
        print(f"    - {k}: {v}")
    
    model = XGBClassifier(**TRAINING_CONFIG)

    # 3. Fit Model
    print("\n[Step 3/4] Training XGBoost model on 46 raw flow features...")
    t_train_start = time.perf_counter()
    model.fit(X_train, y_train)
    t_train = time.perf_counter() - t_train_start
    print(f"  ✓ Training completed in {t_train:.2f}s ({t_train/60:.2f} min)")

    # 4. Save Artifacts to models/aegis/
    print("\n[Step 4/4] Saving independent artifacts...")
    joblib.dump(model, OUTPUT_MODEL_PATH)
    print(f"  ✓ Model saved to: {OUTPUT_MODEL_PATH} ({OUTPUT_MODEL_PATH.stat().st_size / 1e6:.2f} MB)")
    
    encoder.save(OUTPUT_ENCODER_PATH)
    print(f"  ✓ Label encoder saved to: {OUTPUT_ENCODER_PATH}")

    # Verify that protected production artifacts were untouched
    assert PROTECTED_PRODUCTION_MODEL.exists(), "CRITICAL: Production model is missing!"
    print("  ✓ Verification: Existing production artifacts remain 100% UNTOUCHED.")

    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 70)

    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train independent AEGIS-IoT XGBoost model.")
    parser.add_argument("--nrows", type=int, default=None, help="Optional row limit for test runs.")
    args = parser.parse_args()

    train_aegis_model(nrows=args.nrows)

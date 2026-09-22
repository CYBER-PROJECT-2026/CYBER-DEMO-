"""
aegis_ml/dataset_builder.py
===========================
Dataset construction module for the AEGIS-IoT retraining pipeline.

PURPOSE:
- Coordinates data loading, feature validation, and label encoding across the
  three canonical project dataset partitions:
  1. Training Partition:   'CIC_IoT_2023_DDoS_DoS_Spoofing_1.5M.csv' (1.5M records)
  2. Calibration Partition:'calibration_data.csv' (21K records, reserved for calibration)
  3. Final Holdout Test:   'Testing_data/testing_data.csv' (5K records, untouched test)

CRITICAL INTEGRITY INVARIANTS:
- The 5,000-record holdout test dataset must NEVER be used for training or
  hyperparameter tuning.
- The calibration dataset must NOT be mixed into model training.
- No synthetic data generation or random reshuffling of splits.
- Preserves exact 46-feature schema order and float32 dtype.
"""

from pathlib import Path
from typing import Tuple, Optional, Union
import numpy as np
import pandas as pd

from aegis_ml.feature_schema import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    CLASS_NAMES,
    N_FEATURES,
)
from aegis_ml.data_preprocessing import (
    load_dataset,
    prepare_dataset,
)
from aegis_ml.feature_engineering import (
    prepare_feature_matrix,
)
from aegis_ml.label_encoding import (
    AegisLabelEncoder,
)

# Canonical Dataset File Paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAINING_DATASET_PATH = PROJECT_ROOT / "CIC_IoT_2023_DDoS_DoS_Spoofing_1.5M.csv"
CALIBRATION_DATASET_PATH = PROJECT_ROOT / "calibration_data.csv"
TESTING_DATASET_PATH = PROJECT_ROOT / "Testing_data" / "testing_data.csv"


class AegisDatasetBuilder:
    """
    Dataset builder coordinating dataset loading and preparation for AEGIS-IoT.
    """

    def __init__(self, encoder: Optional[AegisLabelEncoder] = None):
        self.encoder = encoder or AegisLabelEncoder()

    def build_dataset_from_path(
        self,
        csv_path: Union[str, Path],
        nrows: Optional[int] = None,
        encode_labels: bool = True,
    ) -> Tuple[pd.DataFrame, Union[np.ndarray, pd.Series]]:
        """
        Loads and prepares any compatible dataset CSV.

        Parameters:
            csv_path: Path to dataset CSV file.
            nrows: Optional row limit.
            encode_labels: If True, returns integer labels y (np.ndarray).
                           If False, returns raw string labels y (pd.Series).

        Returns:
            Tuple of (X, y):
                X: pd.DataFrame with 46 canonical features in exact order.
                y: np.ndarray (integer encoded) or pd.Series (string labels).
        """
        # 1. Load and validate using data_preprocessing module
        X_raw, y_raw = prepare_dataset(csv_path, nrows=nrows)

        # 2. Finalize feature matrix via feature_engineering module
        X = prepare_feature_matrix(X_raw, dtype=np.float32)

        # 3. Handle label encoding if requested
        if encode_labels:
            y = self.encoder.transform(y_raw)
        else:
            y = y_raw

        return X, y

    def get_training_data(
        self,
        nrows: Optional[int] = None,
        encode_labels: bool = True,
    ) -> Tuple[pd.DataFrame, Union[np.ndarray, pd.Series]]:
        """
        Loads the 1.5M training dataset partition.
        Source: CIC_IoT_2023_DDoS_DoS_Spoofing_1.5M.csv
        """
        if not TRAINING_DATASET_PATH.exists():
            raise FileNotFoundError(f"Training dataset not found at: {TRAINING_DATASET_PATH}")
        return self.build_dataset_from_path(
            TRAINING_DATASET_PATH,
            nrows=nrows,
            encode_labels=encode_labels,
        )

    def get_calibration_data(
        self,
        nrows: Optional[int] = None,
        encode_labels: bool = True,
    ) -> Tuple[pd.DataFrame, Union[np.ndarray, pd.Series]]:
        """
        Loads the 21,000-record calibration dataset partition.
        Source: calibration_data.csv
        """
        if not CALIBRATION_DATASET_PATH.exists():
            raise FileNotFoundError(f"Calibration dataset not found at: {CALIBRATION_DATASET_PATH}")
        return self.build_dataset_from_path(
            CALIBRATION_DATASET_PATH,
            nrows=nrows,
            encode_labels=encode_labels,
        )

    def get_test_data(
        self,
        nrows: Optional[int] = None,
        encode_labels: bool = True,
    ) -> Tuple[pd.DataFrame, Union[np.ndarray, pd.Series]]:
        """
        Loads the 5,000-record untouched holdout test dataset partition.
        Source: Testing_data/testing_data.csv

        WARNING: This dataset is reserved exclusively for final evaluation.
                 NEVER use this partition for model training or tuning.
        """
        if not TESTING_DATASET_PATH.exists():
            raise FileNotFoundError(f"Test dataset not found at: {TESTING_DATASET_PATH}")
        return self.build_dataset_from_path(
            TESTING_DATASET_PATH,
            nrows=nrows,
            encode_labels=encode_labels,
        )


if __name__ == "__main__":
    print("Testing aegis_ml.dataset_builder...")
    builder = AegisDatasetBuilder()
    
    # Test loading test dataset (5,000 records)
    X_t, y_t = builder.get_test_data(nrows=100)
    print(f"Test sample loaded successfully: X={X_t.shape}, y={y_t.shape}, y_dtype={y_t.dtype}")
    assert X_t.shape == (100, 46)
    assert list(X_t.columns) == FEATURE_COLUMNS
    assert set(np.unique(y_t)).issubset({0, 1, 2})
    print("Self-check passed!")

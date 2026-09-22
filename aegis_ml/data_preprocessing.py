"""
aegis_ml/data_preprocessing.py
==============================
Independent dataset loading, validation, and preparation module for the
AEGIS-IoT ML retraining and inference pipeline.

PURPOSE:
- Safely loads CICIoT2023-derived CSV datasets using pathlib.Path.
- Extracts and strictly validates the 46 canonical features against the
  independent feature_schema module.
- Validates data integrity: checks for missing features, non-numeric values,
  NaN values, and infinite (Inf) values.
- Validates target labels: confirms labels match only the 3 canonical classes
  ('DDoS', 'DoS', 'Spoofing').
- Returns clean (X, y) data structures without modifying underlying raw values.

STRICT SAFETY RULES:
- NO silent zero-filling or imputation (NO fillna).
- NO silent row removal or silent feature reordering.
- NO feature scaling (NO StandardScaler, NO RobustScaler).
- NO train/test splitting or feature selection in this module.
- NO integer label encoding in this module (labels remain strings).
"""

from pathlib import Path
from typing import Union, Tuple, Optional, List
import numpy as np
import pandas as pd

from aegis_ml.feature_schema import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    CLASS_NAMES,
    N_FEATURES,
    validate_feature_schema,
)


def load_dataset(path: Union[str, Path], nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Loads a CSV dataset from a file path using pathlib.Path.

    Parameters:
        path: Path to the CSV file (str or Path object).
        nrows: Optional maximum number of rows to load (useful for validation tests).

    Returns:
        pd.DataFrame: Raw loaded dataset.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file is empty or cannot be parsed as a CSV.
    """
    file_path = Path(path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Path is not a valid file: {file_path}")

    try:
        df = pd.read_csv(file_path, nrows=nrows)
    except Exception as e:
        raise ValueError(f"Failed to load CSV from {file_path}: {e}") from e

    if df.empty:
        raise ValueError(f"Loaded dataset is empty: {file_path}")

    return df


def validate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts the canonical 46 features from the DataFrame and strictly
    validates column presence, ordering, datatypes, and numerical integrity.

    Parameters:
        df: Input DataFrame containing at least the 46 canonical feature columns.

    Returns:
        pd.DataFrame: Clean DataFrame containing exactly the 46 canonical features
                      in canonical order, without any modifications to underlying values.

    Raises:
        ValueError: If any canonical features are missing, if feature values are
                    non-numeric, or if NaN or infinite (Inf) values are present.
    """
    # 1. Verify that all 46 canonical feature columns exist
    missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Feature validation failed: Missing {len(missing_cols)} canonical column(s): {missing_cols}"
        )

    # 2. Select ONLY the 46 canonical feature columns in exact canonical order
    X = df[FEATURE_COLUMNS].copy()

    # 3. Verify ordering via the feature_schema validator
    validate_feature_schema(X.columns)

    # 4. Verify all feature columns are numeric
    non_numeric_cols: List[str] = []
    for col in FEATURE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(X[col]):
            non_numeric_cols.append(col)

    if non_numeric_cols:
        raise ValueError(
            f"Feature validation failed: Non-numeric data found in column(s): {non_numeric_cols}"
        )

    # 5. Check for NaN values (Strict rule: NO silent imputation, NO fillna)
    nan_counts = X.isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0].to_dict()
    if cols_with_nan:
        raise ValueError(
            f"Feature validation failed: Found NaN values in {len(cols_with_nan)} column(s). "
            f"Automatic imputation is strictly forbidden. Details: {cols_with_nan}"
        )

    # 6. Check for infinite (Inf / -Inf) values
    arr = X.to_numpy(dtype=np.float64)
    if np.isinf(arr).any():
        inf_mask = np.isinf(arr)
        inf_per_col = {
            FEATURE_COLUMNS[i]: int(np.sum(inf_mask[:, i]))
            for i in range(N_FEATURES)
            if np.sum(inf_mask[:, i]) > 0
        }
        raise ValueError(
            f"Feature validation failed: Found infinite (Inf) values in {len(inf_per_col)} column(s). "
            f"Details: {inf_per_col}"
        )

    return X


def validate_target(target_series: pd.Series) -> pd.Series:
    """
    Validates that the target series contains valid string class labels and
    has no NaN or unexpected labels.

    Parameters:
        target_series: Pandas Series representing the target class labels.

    Returns:
        pd.Series: Validated target series of string labels.

    Raises:
        ValueError: If target contains NaN values or unexpected class labels.
    """
    # 1. Check for missing / NaN values in target
    nan_count = int(target_series.isna().sum())
    if nan_count > 0:
        raise ValueError(
            f"Target validation failed: Found {nan_count} missing (NaN) target values."
        )

    # 2. Check that labels are strings and match only the allowed canonical classes
    series_str = target_series.astype(str)
    unique_labels = set(series_str.unique())
    allowed_labels = set(CLASS_NAMES)

    unexpected_labels = unique_labels - allowed_labels
    if unexpected_labels:
        raise ValueError(
            f"Target validation failed: Found unexpected class label(s): {unexpected_labels}. "
            f"Allowed classes are strictly: {CLASS_NAMES}"
        )

    return series_str


def prepare_dataset(
    path: Union[str, Path],
    nrows: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads and completely validates a dataset from a CSV file.

    Execution Flow:
    1. Loads raw CSV data using load_dataset.
    2. Verifies presence of TARGET_COLUMN ('attack_class').
    3. Validates and extracts canonical 46 features using validate_features.
    4. Validates target labels using validate_target.
    5. Returns (X, y) with unmodified feature values and string labels.

    Parameters:
        path: Path to the dataset CSV file.
        nrows: Optional row limit for fast testing or batch validation.

    Returns:
        Tuple[pd.DataFrame, pd.Series]:
            - X: DataFrame of exactly 46 canonical features in canonical order.
            - y: Series of string target labels ('DDoS', 'DoS', 'Spoofing').

    Raises:
        ValueError: If schema, numerical, or label validation fails.
    """
    raw_df = load_dataset(path, nrows=nrows)

    # Verify target column exists
    if TARGET_COLUMN not in raw_df.columns:
        raise ValueError(
            f"Dataset preparation failed: Required target column '{TARGET_COLUMN}' not found in {path}. "
            f"Available columns: {list(raw_df.columns[:10])}..."
        )

    # Validate features (returns exact 46 columns in order)
    X = validate_features(raw_df)

    # Validate target labels (returns clean string series)
    y = validate_target(raw_df[TARGET_COLUMN])

    return X, y


if __name__ == "__main__":
    print("Running aegis_ml.data_preprocessing self-check...")
    # Quick self-test with a synthetic valid DataFrame
    dummy_data = {col: np.random.rand(10).astype(np.float32) for col in FEATURE_COLUMNS}
    dummy_data[TARGET_COLUMN] = ["DDoS", "DoS", "Spoofing"] * 3 + ["DDoS"]
    df_test = pd.DataFrame(dummy_data)

    X_test = validate_features(df_test)
    y_test = validate_target(df_test[TARGET_COLUMN])
    print(f"Self-check passed: X shape = {X_test.shape}, y unique = {list(y_test.unique())}")

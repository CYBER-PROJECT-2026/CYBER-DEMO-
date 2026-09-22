"""
aegis_ml/feature_engineering.py
===============================
Independent feature preparation module for the AEGIS-IoT retraining pipeline.

IMPORTANT CLARIFICATION:
"Feature preparation" in this pipeline means deterministically selecting,
ordering, and casting the existing canonical 46 flow-level features present
in the CICIoT2023 dataset. It does NOT mean recreating CICFlowMeter feature
extraction from raw packet captures.

STRICT PRINCIPLES & PROHIBITIONS:
- The source dataset already contains all 46 engineered flow fields.
- DO NOT invent mathematical formulas to fabricate features.
- DO NOT fill missing values with zero (NO fillna).
- DO NOT perform automatic imputation.
- DO NOT apply scalers (NO StandardScaler, NO RobustScaler, NO MinMaxScaler).
- DO NOT perform feature selection or dimensionality reduction (NO ExtraTrees, NO PCA).
- The raw unscaled 46 float features are passed directly to XGBoost.
"""

from typing import Union, Sequence, Optional
import numpy as np
import pandas as pd

from aegis_ml.feature_schema import (
    FEATURE_COLUMNS,
    N_FEATURES,
    validate_feature_schema,
)


def prepare_feature_matrix(
    data: Union[pd.DataFrame, pd.Series, dict],
    dtype: np.dtype = np.float32,
) -> pd.DataFrame:
    """
    Extracts, strictly orders, and types the canonical 46 features for XGBoost.

    Execution Steps:
    1. Converts input dict or Series into a single-row DataFrame if needed.
    2. Verifies presence of all 46 canonical columns.
    3. Selects columns in exact canonical order defined by FEATURE_COLUMNS.
    4. Validates schema order using validate_feature_schema.
    5. Converts feature values to the specified numeric float dtype.
    6. Validates that no NaN or infinite values exist in the resulting matrix.
    7. Returns clean DataFrame with verified column names and exact order.

    Parameters:
        data: Input DataFrame, Series, or dictionary containing feature records.
        dtype: Numerical float precision (default np.float32 for model inference).

    Returns:
        pd.DataFrame: Clean feature matrix ready for XGBoost inference/training.

    Raises:
        ValueError: If features are missing, non-numeric, NaN, or infinite.
    """
    # 1. Standardize input to DataFrame
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, pd.Series):
        df = pd.DataFrame([data])
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError(f"Unsupported data type for feature matrix preparation: {type(data)}")

    # 2. Check for missing canonical columns
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Feature preparation failed: Missing {len(missing)} required feature(s): {missing}"
        )

    # 3. Select ONLY the 46 canonical features in canonical order
    X = df[FEATURE_COLUMNS].copy()

    # 4. Strictly validate schema ordering
    validate_feature_schema(X.columns)

    # 5. Convert to target numeric float dtype
    try:
        X = X.astype(dtype)
    except Exception as e:
        raise ValueError(f"Feature preparation failed during numeric conversion to {dtype}: {e}") from e

    # 6. Strict check: Zero NaNs permitted
    nan_count = int(X.isna().sum().sum())
    if nan_count > 0:
        nan_cols = X.isna().sum()[X.isna().sum() > 0].to_dict()
        raise ValueError(
            f"Feature preparation failed: Found {nan_count} NaN values in feature matrix. "
            f"Zero-imputation is strictly forbidden. Problematic columns: {nan_cols}"
        )

    # 7. Strict check: Zero Infs permitted
    arr = X.to_numpy()
    if np.isinf(arr).any():
        inf_mask = np.isinf(arr)
        inf_cols = {
            FEATURE_COLUMNS[i]: int(np.sum(inf_mask[:, i]))
            for i in range(N_FEATURES)
            if np.sum(inf_mask[:, i]) > 0
        }
        raise ValueError(
            f"Feature preparation failed: Found infinite (Inf) values. Problematic columns: {inf_cols}"
        )

    return X


if __name__ == "__main__":
    print("Testing aegis_ml.feature_engineering...")
    dummy = {col: 1.0 for col in FEATURE_COLUMNS}
    prepared = prepare_feature_matrix(dummy)
    assert prepared.shape == (1, 46)
    assert list(prepared.columns) == FEATURE_COLUMNS
    assert prepared.dtypes.iloc[0] == np.float32
    print(f"Self-check passed: Shape = {prepared.shape}, columns verified.")

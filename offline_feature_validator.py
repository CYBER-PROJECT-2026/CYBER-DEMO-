"""
offline_feature_validator.py
=============================
Strict 46-Feature Validator for AEGIS-IoT Production Offline ML Pipeline.

Enforces 10 Strict Validation Invariants:
1. Exactly 46 features.
2. Exact canonical feature names matching models/aegis/aegis_xgboost_model.pkl.
3. Exact canonical feature order.
4. Numeric values where expected.
5. Zero missing values (None / null rejected).
6. Zero NaN values rejected.
7. Zero Infinite values (+Inf / -Inf rejected).
8. Zero unexpected / extra columns rejected.
9. Zero missing columns rejected.
10. Data type validation and strict conversion to float32.

If any invariant is violated, the input is REJECTED immediately with a descriptive ValueError.
"""

from typing import Union, Dict, Any, List
import numpy as np
import pandas as pd

# Canonical 46 Features in exact order from models/aegis/aegis_xgboost_model.pkl (model.feature_names_in_)
CANONICAL_46_FEATURES: List[str] = [
    "flow_duration",
    "Header_Length",
    "Protocol Type",
    "Duration",
    "Rate",
    "Srate",
    "Drate",
    "fin_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "psh_flag_number",
    "ack_flag_number",
    "ece_flag_number",
    "cwr_flag_number",
    "ack_count",
    "syn_count",
    "fin_count",
    "urg_count",
    "rst_count",
    "HTTP",
    "HTTPS",
    "DNS",
    "Telnet",
    "SMTP",
    "SSH",
    "IRC",
    "TCP",
    "UDP",
    "DHCP",
    "ARP",
    "ICMP",
    "IPv",
    "LLC",
    "Tot sum",
    "Min",
    "Max",
    "AVG",
    "Std",
    "Tot size",
    "IAT",
    "Number",
    "Magnitue",
    "Radius",
    "Covariance",
    "Variance",
    "Weight"
]

CANONICAL_FEATURE_SET = set(CANONICAL_46_FEATURES)
CANONICAL_FEATURE_COUNT = len(CANONICAL_46_FEATURES)  # 46


class FeatureValidationError(ValueError):
    """Raised when an input fails the strict 46-feature validation."""
    pass


def validate_46_features(
    input_data: Union[Dict[str, Any], pd.Series, pd.DataFrame, List[Any], np.ndarray],
    strict_columns: bool = True
) -> pd.DataFrame:
    """
    Validates and standardizes input to a canonical 46-feature float32 DataFrame.
    
    Parameters:
        input_data: Dict, Series, DataFrame, list, or ndarray representing one or more flow records.
        strict_columns: If True, rejects any unexpected extra columns.
        
    Returns:
        pd.DataFrame: Validated DataFrame with exactly 46 columns in canonical order, dtype float32.
        
    Raises:
        FeatureValidationError: If any of the 10 invariants are violated.
    """
    # 1. Type normalization to DataFrame
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.Series):
        df = pd.DataFrame([input_data.to_dict()])
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    elif isinstance(input_data, (list, np.ndarray)):
        arr = np.array(input_data)
        if arr.ndim == 1:
            if len(arr) != CANONICAL_FEATURE_COUNT:
                raise FeatureValidationError(
                    f"REJECT INPUT: Expected exactly {CANONICAL_FEATURE_COUNT} features, received {len(arr)}."
                )
            df = pd.DataFrame([arr], columns=CANONICAL_46_FEATURES)
        elif arr.ndim == 2:
            if arr.shape[1] != CANONICAL_FEATURE_COUNT:
                raise FeatureValidationError(
                    f"REJECT INPUT: Expected exactly {CANONICAL_FEATURE_COUNT} columns, received {arr.shape[1]}."
                )
            df = pd.DataFrame(arr, columns=CANONICAL_46_FEATURES)
        else:
            raise FeatureValidationError(
                f"REJECT INPUT: Array has unsupported dimension {arr.ndim} (must be 1D or 2D)."
            )
    else:
        raise FeatureValidationError(
            f"REJECT INPUT: Unsupported input data type '{type(input_data).__name__}'. "
            f"Must be dict, pd.Series, pd.DataFrame, list, or np.ndarray."
        )

    # If df contains target/label columns, remove or flag
    # In offline test sets, 'attack_class', 'label', 'Label', 'Label_orig' might be present
    # We only strip target/label if explicitly present and strict_columns is set
    metadata_cols = {"attack_class", "label", "Label", "Label_orig", "AttackCategory", "target"}
    present_metadata = metadata_cols.intersection(df.columns)
    if present_metadata:
        df = df.drop(columns=list(present_metadata))

    input_cols = list(df.columns)
    input_col_set = set(input_cols)

    # 2. Check for missing columns
    missing_cols = [f for f in CANONICAL_46_FEATURES if f not in input_col_set]
    if missing_cols:
        raise FeatureValidationError(
            f"REJECT INPUT: Missing {len(missing_cols)} required feature(s): {missing_cols}. "
            f"Total present: {len(input_cols)}, expected: {CANONICAL_FEATURE_COUNT}."
        )

    # 3. Check for unexpected / extra columns
    if strict_columns:
        unexpected_cols = [c for c in input_cols if c not in CANONICAL_FEATURE_SET]
        if unexpected_cols:
            raise FeatureValidationError(
                f"REJECT INPUT: Found {len(unexpected_cols)} unexpected column(s): {unexpected_cols}. "
                f"The production model strictly requires only the canonical 46 features."
            )

    # 4. Check feature count
    if len(df.columns) != CANONICAL_FEATURE_COUNT:
        raise FeatureValidationError(
            f"REJECT INPUT: Expected exactly {CANONICAL_FEATURE_COUNT} features, received {len(df.columns)}."
        )

    # 5. Enforce exact canonical feature ordering
    df = df[CANONICAL_46_FEATURES]

    # 6. Check for None / Null / Missing values
    null_counts = df.isnull().sum()
    if null_counts.any():
        cols_with_null = null_counts[null_counts > 0].to_dict()
        raise FeatureValidationError(
            f"REJECT INPUT: Found missing/null values in features: {cols_with_null}."
        )

    # 7. Check numeric conversion and data types
    for col in CANONICAL_46_FEATURES:
        try:
            df[col] = pd.to_numeric(df[col], errors="raise")
        except (ValueError, TypeError) as err:
            raise FeatureValidationError(
                f"REJECT INPUT: Feature '{col}' contains non-numeric values that cannot be converted to float: {err}"
            )

    # Convert to float32
    df = df.astype(np.float32)

    # 8. Check for NaN values
    if np.isnan(df.values).any():
        nan_cols = [c for c in CANONICAL_46_FEATURES if np.isnan(df[c].values).any()]
        raise FeatureValidationError(
            f"REJECT INPUT: NaN values detected in feature(s): {nan_cols}."
        )

    # 9. Check for Infinite values (+Inf / -Inf)
    if np.isinf(df.values).any():
        inf_cols = [c for c in CANONICAL_46_FEATURES if np.isinf(df[c].values).any()]
        raise FeatureValidationError(
            f"REJECT INPUT: Infinite (+Inf / -Inf) values detected in feature(s): {inf_cols}."
        )

    return df

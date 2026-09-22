"""
aegis_ml/feature_schema.py
==========================
Independent canonical feature schema and class mapping definitions for the
AEGIS-IoT retraining and inference pipeline.

These 46 features represent the canonical flow-level fields from the CICIoT2023
dataset. This module defines the schema and provides validation functions to
ensure data integrity.

NOTE:
- This module only defines the schema and validates feature column names/order.
- It does NOT perform feature engineering, imputation, scaling, or transformation.
- Missing columns or incorrect ordering will raise an explicit ValueError.
- The spelling 'Magnitue' (feature #42) is intentional and matches the raw dataset.
"""

from typing import Sequence, List, Dict

# ==============================================================================
# CANONICAL 46 FEATURES (EXACT NAMES AND EXACT ORDER)
# ==============================================================================
FEATURE_COLUMNS: List[str] = [
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
    "Magnitue",  # Intentional spelling matching CICIoT2023 raw schema
    "Radius",
    "Covariance",
    "Variance",
    "Weight",
]

# Total number of features
N_FEATURES: int = len(FEATURE_COLUMNS)

# Target column name in processed datasets
TARGET_COLUMN: str = "attack_class"

# Target class labels
CLASS_NAMES: List[str] = [
    "DDoS",
    "DoS",
    "Spoofing",
]

# Class-to-Index mapping
CLASS_TO_INDEX: Dict[str, int] = {
    "DDoS": 0,
    "DoS": 1,
    "Spoofing": 2,
}

# Index-to-Class mapping
INDEX_TO_CLASS: Dict[int, str] = {
    0: "DDoS",
    1: "DoS",
    2: "Spoofing",
}


# ==============================================================================
# SCHEMA VALIDATION FUNCTION
# ==============================================================================
def validate_feature_schema(columns: Sequence[str]) -> bool:
    """
    Validates that the provided sequence of column names matches the exact 46
    canonical features in the required order.

    Parameters:
        columns: Sequence of column names (e.g. list, pandas Index).

    Returns:
        bool: True if the schema is strictly valid.

    Raises:
        ValueError: If column count differs, required features are missing,
                    unexpected features are present, or ordering is incorrect.
    """
    cols_list = list(columns)
    
    # 1. Check feature count
    if len(cols_list) != N_FEATURES:
        missing = [f for f in FEATURE_COLUMNS if f not in cols_list]
        extra = [c for c in cols_list if c not in FEATURE_COLUMNS]
        msg_parts = [
            f"Schema validation failed: Expected exactly {N_FEATURES} features, got {len(cols_list)}."
        ]
        if missing:
            msg_parts.append(f"Missing {len(missing)} feature(s): {missing}")
        if extra:
            msg_parts.append(f"Unexpected extra {len(extra)} column(s): {extra}")
        raise ValueError(" ".join(msg_parts))

    # 2. Check for missing or unexpected columns
    missing = [f for f in FEATURE_COLUMNS if f not in cols_list]
    extra = [c for c in cols_list if c not in FEATURE_COLUMNS]
    if missing or extra:
        raise ValueError(
            f"Schema mismatch. Missing: {missing}, Unexpected extra: {extra}"
        )

    # 3. Check exact order
    mismatches = []
    for idx, (expected, actual) in enumerate(zip(FEATURE_COLUMNS, cols_list)):
        if expected != actual:
            mismatches.append(f"Position {idx + 1}: expected '{expected}', found '{actual}'")

    if mismatches:
        detail = "; ".join(mismatches[:5])
        if len(mismatches) > 5:
            detail += f" ... ({len(mismatches) - 5} more)"
        raise ValueError(f"Feature column order mismatch. Details: {detail}")

    return True


if __name__ == "__main__":
    print(f"AEGIS-IoT Feature Schema defined with {N_FEATURES} features.")
    print("Classes:", CLASS_NAMES)
    print("Class-to-Index:", CLASS_TO_INDEX)
    
    # Self-test validation
    try:
        validate_feature_schema(FEATURE_COLUMNS)
        print("Self-test schema validation: PASSED")
    except ValueError as e:
        print("Self-test schema validation: FAILED ->", e)

"""
aegis_live/feature_schema.py
============================
Canonical 24-Feature Schema for AEGIS-Live v2.
Every feature is strictly physically measurable from live network packets.
Completely isolated from AEGIS v1 (which uses the 46-feature offline schema).
"""

from typing import List, Dict, Any, Union
import numpy as np
import pandas as pd

# EXACT 24 Features in Canonical Order
CANONICAL_24_FEATURES: List[str] = [
    "duration_ms",
    "pkt_rate_pps",
    "byte_rate_bps",
    "iat_mean_ms",
    "iat_std_ms",
    "pkt_size_mean",
    "pkt_size_std",
    "pkt_size_min",
    "pkt_size_max",
    "syn_ratio",
    "ack_ratio",
    "rst_ratio",
    "fin_ratio",
    "urg_ratio",
    "psh_ratio",
    "ece_ratio",
    "cwr_ratio",
    "ip_ttl_mean",
    "ip_ttl_std",
    "tcp_ratio",
    "udp_ratio",
    "icmp_ratio",
    "inbound_pkt_ratio",
    "outbound_pkt_ratio"
]

FEATURE_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "duration_ms": {"unit": "ms", "min": 0.0, "max": 600000.0, "desc": "Elapsed window duration in milliseconds"},
    "pkt_rate_pps": {"unit": "pps", "min": 0.0, "max": 1e7, "desc": "Packet arrival rate (packets per second)"},
    "byte_rate_bps": {"unit": "Bps", "min": 0.0, "max": 1e10, "desc": "Byte transfer rate (bytes per second)"},
    "iat_mean_ms": {"unit": "ms", "min": 0.0, "max": 60000.0, "desc": "Mean inter-packet arrival time in ms"},
    "iat_std_ms": {"unit": "ms", "min": 0.0, "max": 60000.0, "desc": "Standard deviation of inter-packet arrival time"},
    "pkt_size_mean": {"unit": "bytes", "min": 20.0, "max": 65535.0, "desc": "Mean packet length in bytes"},
    "pkt_size_std": {"unit": "bytes", "min": 0.0, "max": 65535.0, "desc": "Packet length standard deviation"},
    "pkt_size_min": {"unit": "bytes", "min": 20.0, "max": 65535.0, "desc": "Minimum packet length in bytes"},
    "pkt_size_max": {"unit": "bytes", "min": 20.0, "max": 65535.0, "desc": "Maximum packet length in bytes"},
    "syn_ratio": {"unit": "ratio", "min": 0.0, "max": 1.0, "desc": "Proportion of packets with TCP SYN flag"},
    "ack_ratio": {"unit": "ratio", "min": 0.0, "max": 1.0, "desc": "Proportion of packets with TCP ACK flag"},
    "rst_ratio": {"unit": "ratio", "min": 0.0, "max": 1.0, "desc": "Proportion of packets with TCP RST flag"},
    "fin_ratio": {"unit": "ratio", "min": 0.0, "max": 1.0, "desc": "Proportion of packets with TCP FIN flag"},
    "urg_ratio": {"unit": "ratio", "min": 0.0, "max": 1.0, "desc": "Proportion of packets with TCP URG flag"},
    "psh_ratio": {"unit": "ratio", "min": 0.0, "max": 1.0, "desc": "Proportion of packets with TCP PSH flag"},
    "ece_ratio": {"unit": "ratio", "min": 0.0, "max": 1.0, "desc": "Proportion of packets with TCP ECE flag"},
    "cwr_ratio": {"unit": "ratio", "min": 0.0, "max": 1.0, "desc": "Proportion of packets with TCP CWR flag"},
    "ip_ttl_mean": {"unit": "hops", "min": 1.0, "max": 255.0, "desc": "Mean IP Time-To-Live"},
    "ip_ttl_std": {"unit": "hops", "min": 0.0, "max": 255.0, "desc": "Standard deviation of IP Time-To-Live"},
    "tcp_ratio": {"unit": "ratio", "min": 0.0, "max": 1.0, "desc": "Proportion of TCP packets"},
    "udp_ratio": {"unit": "ratio", "min": 0.0, "max": 1.0, "desc": "Proportion of UDP packets"},
    "icmp_ratio": {"unit": "ratio", "min": 0.0, "max": 1.0, "desc": "Proportion of ICMP packets"},
    "inbound_pkt_ratio": {"unit": "ratio", "min": 0.0, "max": 1.0, "desc": "Proportion of inbound packets to local host/network"},
    "outbound_pkt_ratio": {"unit": "ratio", "min": 0.0, "max": 1.0, "desc": "Proportion of outbound packets from local host/network"}
}

TARGET_CLASSES: List[str] = ["DDoS", "DoS", "Spoofing"]


def validate_24_features(raw_input: Union[Dict[str, Any], pd.Series, pd.DataFrame]) -> pd.DataFrame:
    """
    Validates that the input strictly adheres to the 24 canonical features.
    Enforces:
    - Exact 24 columns in canonical order
    - float32 data types
    - No NaN, null, or infinite values
    - Physical bounds checks
    """
    if isinstance(raw_input, dict):
        df = pd.DataFrame([raw_input])
    elif isinstance(raw_input, pd.Series):
        df = pd.DataFrame([raw_input.to_dict()])
    elif isinstance(raw_input, pd.DataFrame):
        df = raw_input.copy()
    else:
        raise TypeError(f"Expected dict, Series, or DataFrame, got {type(raw_input)}")

    missing = set(CANONICAL_24_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required AEGIS-Live v2 features: {sorted(list(missing))}")

    # Reorder exactly to canonical order
    df = df[CANONICAL_24_FEATURES].copy()

    # Check for NaN / Inf
    if df.isna().any().any():
        raise ValueError("Input contains NaN values")
    if np.isinf(df.values).any():
        raise ValueError("Input contains Infinite values")

    # Cast to float32
    df = df.astype(np.float32)

    # Validate ratios in [0.0, 1.0]
    ratio_cols = [c for c in CANONICAL_24_FEATURES if c.endswith("_ratio")]
    for c in ratio_cols:
        if (df[c] < -1e-4).any() or (df[c] > 1.0001).any():
            raise ValueError(f"Ratio column {c} has values outside [0, 1]: min={df[c].min()}, max={df[c].max()}")

    return df


# =============================================================================
# AEGIS-Live v3: 27-Feature Schema (Canonical 24 + 3 Validated Topology Features)
# =============================================================================
V3_TOPOLOGY_FEATURES: List[str] = [
    "fan_in_ratio",
    "src_concentration_ratio",
    "src_ip_cardinality_per_window"
]

CANONICAL_27_FEATURES: List[str] = CANONICAL_24_FEATURES + V3_TOPOLOGY_FEATURES


def validate_27_features(raw_input: Union[Dict[str, Any], pd.Series, pd.DataFrame]) -> pd.DataFrame:
    """
    Validates that the input strictly adheres to the 27 canonical v3 features.
    """
    if isinstance(raw_input, dict):
        df = pd.DataFrame([raw_input])
    elif isinstance(raw_input, pd.Series):
        df = pd.DataFrame([raw_input.to_dict()])
    elif isinstance(raw_input, pd.DataFrame):
        df = raw_input.copy()
    else:
        raise TypeError(f"Expected dict, Series, or DataFrame, got {type(raw_input)}")

    missing = set(CANONICAL_27_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required AEGIS-Live v3 features: {sorted(list(missing))}")

    df = df[CANONICAL_27_FEATURES].copy()

    if df.isna().any().any():
        raise ValueError("Input contains NaN values")
    if np.isinf(df.values).any():
        raise ValueError("Input contains Infinite values")

    df = df.astype(np.float32)

    # Validate ratios in [0.0, 1.0] (excluding fan_in_ratio which can be > 1)
    ratio_cols = [c for c in CANONICAL_27_FEATURES if c.endswith("_ratio") and c != "fan_in_ratio"]
    for c in ratio_cols:
        if (df[c] < -1e-4).any() or (df[c] > 1.0001).any():
            raise ValueError(f"Ratio column {c} has values outside [0, 1]: min={df[c].min()}, max={df[c].max()}")

    if (df["fan_in_ratio"] < -1e-4).any():
        raise ValueError("fan_in_ratio cannot be negative")
    if (df["src_ip_cardinality_per_window"] < 0.0).any():
        raise ValueError("src_ip_cardinality_per_window cannot be negative")

    return df


# =============================================================================
# AEGIS-Live v4: 29-Feature Schema (Canonical 24 + 5 Network Topology Features)
# =============================================================================
V4_TOPOLOGY_FEATURES: List[str] = [
    "src_concentration_ratio",
    "src_ip_entropy",
    "src_ip_cardinality",
    "dst_ip_cardinality",
    "fan_in_ratio"
]

CANONICAL_29_FEATURES: List[str] = CANONICAL_24_FEATURES + V4_TOPOLOGY_FEATURES


def validate_29_features(raw_input: Union[Dict[str, Any], pd.Series, pd.DataFrame]) -> pd.DataFrame:
    """
    Validates that the input strictly adheres to the 29 canonical v4 features:
    - 24 per-flow physical features
    - 5 network-level topology features (src_concentration_ratio, src_ip_entropy,
      src_ip_cardinality, dst_ip_cardinality, fan_in_ratio)
    """
    if isinstance(raw_input, dict):
        df = pd.DataFrame([raw_input])
    elif isinstance(raw_input, pd.Series):
        df = pd.DataFrame([raw_input.to_dict()])
    elif isinstance(raw_input, pd.DataFrame):
        df = raw_input.copy()
    else:
        raise TypeError(f"Expected dict, Series, or DataFrame, got {type(raw_input)}")

    missing = set(CANONICAL_29_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required AEGIS-Live v4 features: {sorted(list(missing))}")

    df = df[CANONICAL_29_FEATURES].copy()

    if df.isna().any().any():
        raise ValueError("Input contains NaN values")
    if np.isinf(df.values).any():
        raise ValueError("Input contains Infinite values")

    df = df.astype(np.float32)

    # Validate ratios in [0.0, 1.0] (excluding fan_in_ratio which can be > 1.0)
    ratio_cols = [c for c in CANONICAL_29_FEATURES if c.endswith("_ratio") and c != "fan_in_ratio"]
    for c in ratio_cols:
        if (df[c] < -1e-4).any() or (df[c] > 1.0001).any():
            raise ValueError(f"Ratio column {c} has values outside [0, 1]: min={df[c].min()}, max={df[c].max()}")

    if (df["fan_in_ratio"] < -1e-4).any():
        raise ValueError("fan_in_ratio cannot be negative")
    if (df["src_ip_cardinality"] < 0.0).any():
        raise ValueError("src_ip_cardinality cannot be negative")
    if (df["dst_ip_cardinality"] < 0.0).any():
        raise ValueError("dst_ip_cardinality cannot be negative")
    if (df["src_ip_entropy"] < -1e-4).any():
        raise ValueError("src_ip_entropy cannot be negative")

    return df


# =============================================================================
# AEGIS-Live v4.3: 39-Feature Schema (Canonical 29 + 10 Causal Context Features)
# =============================================================================
CONTEXT_10_FEATURES: List[str] = [
    "src_ip_cardinality_5s",
    "dst_ip_cardinality_5s",
    "src_ip_entropy_5s",
    "fan_in_ratio_5s",
    "fan_out_ratio_5s",
    "pkt_rate_pps_5s",
    "byte_rate_bps_5s",
    "tcp_syn_ratio_5s",
    "tcp_rst_ratio_5s",
    "protocol_entropy_5s"
]

CANONICAL_39_FEATURES: List[str] = CANONICAL_29_FEATURES + CONTEXT_10_FEATURES


def validate_39_features(raw_input: Union[Dict[str, Any], pd.Series, pd.DataFrame]) -> pd.DataFrame:
    """
    Validates that the input strictly adheres to the 39 canonical v4.3 features:
    - 24 per-flow physical features
    - 5 network topology features
    - 10 causal 5.0-second rolling temporal context features
    """
    if isinstance(raw_input, dict):
        df = pd.DataFrame([raw_input])
    elif isinstance(raw_input, pd.Series):
        df = pd.DataFrame([raw_input.to_dict()])
    elif isinstance(raw_input, pd.DataFrame):
        df = raw_input.copy()
    else:
        raise TypeError(f"Expected dict, Series, or DataFrame, got {type(raw_input)}")

    missing = set(CANONICAL_39_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required AEGIS-Live v4.3 features: {sorted(list(missing))}")

    df = df[CANONICAL_39_FEATURES].copy()

    if df.isna().any().any():
        raise ValueError("Input contains NaN values")
    if np.isinf(df.values).any():
        raise ValueError("Input contains Infinite values")

    df = df.astype(np.float32)
    return df

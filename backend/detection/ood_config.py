"""
backend/detection/ood_config.py
================================
Locked configuration and operational thresholds for Phase 14:
Out-Of-Distribution (OOD), Novelty Scoring, and Decision Abstention.

IMPORTANT:
These thresholds were developed STRICTLY using the 21,000-record calibration dataset
(calibration_data.csv). The 5,000-record test set was NEVER used to select these values.

Parameters derived from:
- 1.5M training records: Class centroids and inverted covariance matrices for 46 raw features.
- 21,000 calibration records: Empirical distribution and percentile cutoffs.
"""

# ==============================================================================
# 1. LOCKED THRESHOLDS
# ==============================================================================

# Confidence Threshold: Calibrated top-1 posterior probability
# Empirically validated on calibration dataset:
# - Catches classification ambiguity at class boundaries
# - False abstention on calibration dataset: 0.0191%
CONFIDENCE_THRESHOLD = 0.70

# Class-Conditional Mahalanobis Threshold: Feature-space novelty
# Empirically set to P99.0 of calibration dataset:
# - Rejects only top 1% extreme feature outliers on in-distribution data
# - Retains 99.0% coverage with only 0.98% false rejection on calibration data
# - Catches samples that are geometrically atypical relative to the predicted class
CC_MAHALANOBIS_THRESHOLD = 23.4341

# ==============================================================================
# 2. DECISION STATES
# ==============================================================================

STATE_KNOWN_DDOS = "KNOWN_DDoS"
STATE_KNOWN_DOS = "KNOWN_DoS"
STATE_KNOWN_SPOOFING = "KNOWN_Spoofing"
STATE_UNKNOWN_OR_REVIEW = "UNKNOWN_OR_REVIEW"

ALL_DECISION_STATES = [
    STATE_KNOWN_DDOS,
    STATE_KNOWN_DOS,
    STATE_KNOWN_SPOOFING,
    STATE_UNKNOWN_OR_REVIEW
]

# ==============================================================================
# 3. ABSTENTION REASONS
# ==============================================================================

REASON_LOW_CONFIDENCE = "LOW_CONFIDENCE"
REASON_OOD_DISTANCE = "OOD_DISTANCE"
REASON_BOTH = "LOW_CONFIDENCE_AND_OOD_DISTANCE"

# ==============================================================================
# 4. OPERATIONAL POLICY
# ==============================================================================

# For UNKNOWN_OR_REVIEW:
# Automated firewall drops and network mutations are STRICTLY DISALLOWED.
# Must be routed to Human Analyst Triage.
POLICY_AUTOMATED_BLOCK_FOR_UNKNOWN = False

# Advisory label prefix for reports
ADVISORY_PREFIX = "[ADVISORY / UNVERIFIED]"

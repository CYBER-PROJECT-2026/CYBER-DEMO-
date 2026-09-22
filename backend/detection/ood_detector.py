"""
backend/detection/ood_detector.py
=================================
Class-Conditional Mahalanobis Novelty Detector for AEGIS-IoT.

Measures the geometric atypicality of an incoming 46-feature telemetry vector
relative to the training distribution cluster of the model-predicted class.

Mathematical Formulation:
    d_{CC}(x, c) = \\sqrt{ (x - \\mu_c)^T (\\Sigma_c + \\lambda I)^{-1} (x - \\mu_c) }

Where:
    \\mu_c: Empirical centroid of class c in 46 unscaled float32 feature space (computed from 1.5M training records)
    \\Sigma_c: Covariance matrix of class c in 46 unscaled float32 feature space
    \\lambda: Regularization factor (1e-6) ensuring numerical invertibility
    c: Predicted class (DDoS, DoS, or Spoofing)

Compatibility Guarantee:
    - Uses the exact 46 canonical raw features (NO SCALER).
    - Completely external to XGBoost; leaves the tree booster 100% frozen.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from offline_feature_validator import validate_46_features, CANONICAL_46_FEATURES
from backend.detection.ood_config import CC_MAHALANOBIS_THRESHOLD

DEFAULT_PARAMS_PATH = os.path.join(BASE_DIR, "cc_mahalanobis_params.pkl")


class OODNoveltyDetector:
    """
    Class-conditional Mahalanobis distance detector for 46 raw IoT telemetry features.
    """
    def __init__(self, params_path: str = DEFAULT_PARAMS_PATH, threshold: float = CC_MAHALANOBIS_THRESHOLD):
        self.params_path = params_path
        self.threshold = threshold
        
        if not os.path.exists(params_path):
            raise FileNotFoundError(
                f"Class-conditional Mahalanobis parameters not found at: {params_path}. "
                f"Run run_phase14_calibration_study.py to compute them."
            )
            
        self.params = joblib.load(params_path)
        self.classes = list(self.params.keys())
        
        # Validate that all required classes exist
        for expected_cls in ["DDoS", "DoS", "Spoofing"]:
            if expected_cls not in self.params:
                raise ValueError(f"Missing parameters for class '{expected_cls}' in {params_path}")

    def compute_distance(self, df_input: pd.DataFrame, target_class: str) -> float:
        """
        Computes the class-conditional Mahalanobis distance for a single sample
        relative to the cluster of `target_class`.
        
        Args:
            df_input: 1-row DataFrame strictly validated to 46 canonical float32 features.
            target_class: Target class name ("DDoS", "DoS", "Spoofing").
            
        Returns:
            float: Class-conditional Mahalanobis distance.
        """
        if target_class not in self.params:
            raise ValueError(f"Unknown target class '{target_class}'. Must be one of {self.classes}")
            
        mu = self.params[target_class]["mu"]
        VI = self.params[target_class]["VI"]
        
        x = df_input[CANONICAL_46_FEATURES].iloc[0].values.astype(np.float64)
        diff = x - mu
        mah_sq = np.dot(diff, np.dot(VI, diff))
        return float(np.sqrt(max(mah_sq, 0.0)))

    def evaluate_novelty(self, input_features: Union[pd.DataFrame, pd.Series, dict], predicted_class: str) -> Dict[str, Any]:
        """
        Evaluates whether a sample is a geometric outlier / atypical in feature space.
        
        Args:
            input_features: Raw input features (will be validated against 10 strict schema invariants).
            predicted_class: Predicted class name from XGBoost model.
            
        Returns:
            Dict[str, Any]: Detailed novelty assessment.
        """
        # Strictly validate input against canonical schema
        df_validated = validate_46_features(input_features)
        
        # Compute distance
        dist = self.compute_distance(df_validated, predicted_class)
        is_atypical = bool(dist > self.threshold)
        
        return {
            "cc_mahalanobis_distance": round(dist, 4),
            "threshold": self.threshold,
            "is_atypical": is_atypical,
            "percentile_reference": "P99.0_calibration",
            "reference_class": predicted_class
        }

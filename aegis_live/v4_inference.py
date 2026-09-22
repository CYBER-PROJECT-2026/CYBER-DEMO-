"""
aegis_live/v4_inference.py
==========================
Runtime Inference and Evidence-Based Abstention Engine for AEGIS-Live v4.
STRICTLY DETECTION-ONLY MODE: No automatic mitigation, no firewall actions, no blocking.

Loaded Artifacts:
  - Model: models/aegis_live_v4/aegis_live_xgboost_baseline.pkl
  - Calibration: models/aegis_live_v4/temperature_scaling.json (T = 1.199787)
  - OOD Parameters: models/aegis_live_v4/ood_parameters.pkl
  - Abstention Config: models/aegis_live_v4/abstention_config.json

Returns:
  {
      "predicted_class": str,
      "calibrated_probabilities": Dict[str, float],
      "confidence": float,
      "mahalanobis_distance": float,
      "trusted": bool,
      "status": "TRUSTED" or "UNKNOWN_OR_REVIEW",
      "reason_codes": List[str]
  }
"""

import os
import sys
import json
import pickle
import hashlib
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union

import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from aegis_live.feature_schema import CANONICAL_29_FEATURES, validate_29_features
from aegis_live.audit_v4_ood import ClassConditionalMahalanobisV4, stable_softmax

V4_MODEL_DIR = os.path.join(BASE_DIR, "models", "aegis_live_v4")


class AEGISLiveV4Inference:
    """
    AEGIS-Live v4 Runtime Inference and Abstention Engine.
    Executes the complete detection-only pipeline:
      29 Features -> XGBoost Logits -> Temperature Scaling -> Mahalanobis OOD -> Abstention Gate
    """
    def __init__(self, model_dir: Optional[str] = None):
        if model_dir is None:
            model_dir = V4_MODEL_DIR

        self.model_path = os.path.join(model_dir, "aegis_live_xgboost_baseline.pkl")
        self.cal_path = os.path.join(model_dir, "temperature_scaling.json")
        self.ood_path = os.path.join(model_dir, "ood_parameters.pkl")
        self.abstention_path = os.path.join(model_dir, "abstention_config.json")

        for p in [self.model_path, self.cal_path, self.ood_path, self.abstention_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing required v4 inference artifact: {p}")

        # 1. Load model
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)

        # 2. Load temperature
        with open(self.cal_path, "r") as f:
            cal_data = json.load(f)
        self.temperature = float(cal_data["learned_temperature"])
        self.classes = list(cal_data.get("classes", ["Benign", "DDoS", "DoS", "Spoofing"]))

        # 3. Load OOD parameters
        with open(self.ood_path, "rb") as f:
            ood_data = pickle.load(f)
        self.class_means: Dict[int, np.ndarray] = ood_data["class_means"]
        self.class_inv_covs: Dict[int, np.ndarray] = ood_data["class_inv_covs"]
        self.lambda_reg = float(ood_data.get("lambda_reg", 1e-4))
        self.features = list(ood_data.get("features", CANONICAL_29_FEATURES))

        # 4. Load abstention thresholds
        with open(self.abstention_path, "r") as f:
            abst_data = json.load(f)
        gate = abst_data["gate_definition"]
        self.confidence_threshold = float(gate["confidence_threshold"])
        self.mahalanobis_threshold = float(gate["mahalanobis_threshold"])

    def compute_min_mahalanobis_distance(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the minimum class-conditional Mahalanobis distance across all classes:
        MD(x) = min_c MD_c(x)
        """
        min_dists = np.full(X.shape[0], np.inf, dtype=np.float64)
        for c in range(len(self.classes)):
            diff = X - self.class_means[c]
            mult = np.dot(diff, self.class_inv_covs[c])
            d_sq = np.sum(mult * diff, axis=1)
            d_sq = np.maximum(d_sq, 0.0)
            dist = np.sqrt(d_sq)
            min_dists = np.minimum(min_dists, dist)
        return min_dists

    def predict_sample(self, raw_input: Union[Dict[str, Any], pd.Series, pd.DataFrame]) -> Dict[str, Any]:
        """
        Processes a single 29-feature vector and applies the abstention gate.
        Returns a standardized dictionary.
        """
        # Validate schema and cast to float32 DataFrame with exact order
        df_29 = validate_29_features(raw_input)
        if len(df_29) != 1:
            raise ValueError(f"predict_sample expects exactly 1 sample, got {len(df_29)}")

        # 1. XGBoost Logits
        logits = self.model.predict(df_29, output_margin=True) # shape (1, 4)

        # 2. Temperature Calibration
        probs = stable_softmax(logits, temperature=self.temperature)[0] # shape (4,)

        pred_idx = int(np.argmax(probs))
        predicted_class = self.classes[pred_idx]
        confidence = float(probs[pred_idx])

        # 3. Minimum Mahalanobis Distance
        X_vec = df_29.values.astype(np.float64)
        min_md = float(self.compute_min_mahalanobis_distance(X_vec)[0])

        # 4. Abstention Gate
        reasons = []
        if confidence < self.confidence_threshold:
            reasons.append(f"LOW_CALIBRATED_CONFIDENCE: {confidence:.4f} < {self.confidence_threshold:.4f}")
        if min_md > self.mahalanobis_threshold:
            reasons.append(f"HIGH_MAHALANOBIS_DISTANCE: {min_md:.4f} > {self.mahalanobis_threshold:.4f}")

        trusted = (len(reasons) == 0)
        status = "TRUSTED" if trusted else "UNKNOWN_OR_REVIEW"
        trusted_class = predicted_class if trusted else None
        advisory_class = predicted_class

        sorted_probs = np.sort(probs)
        top2_margin = float(sorted_probs[-1] - sorted_probs[-2])
        p_clipped = np.clip(probs, 1e-12, 1.0)
        entropy = float(-np.sum(p_clipped * np.log2(p_clipped)))

        cal_probs_dict = {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}
        feat_dict = {col: float(df_29[col].iloc[0]) for col in CANONICAL_29_FEATURES}

        return {
            "predicted_class": predicted_class,
            "calibrated_confidence": confidence,
            "confidence": confidence,
            "mahalanobis_distance": min_md,
            "trusted": trusted,
            "decision": status,
            "status": status,
            "trusted_class": trusted_class,
            "advisory_class": advisory_class,
            "automated_blocking_allowed": False,
            "reasons": reasons,
            "reason_codes": reasons,
            "top2_margin": top2_margin,
            "entropy": entropy,
            "feature_vector": feat_dict,
            "calibrated_probabilities": cal_probs_dict
        }

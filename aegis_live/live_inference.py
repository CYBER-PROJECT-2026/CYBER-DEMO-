"""
aegis_live/live_inference.py
============================
Live Model Inference and Evidence-Based Abstention Engine for AEGIS-Live v2.
STRICTLY DETECTION-ONLY MODE: No automatic mitigation, no firewall actions, no blocking.

FROZEN ARTIFACTS LOADED:
- Model: models/aegis_live_v2/aegis_live_xgboost_baseline.pkl
- Calibration: models/aegis_live_v2/temperature_scaling.json (T = 1.005459)
- OOD: models/aegis_live_v2/ood_parameters.pkl
- Abstention: models/aegis_live_v2/abstention_config.json

DECISION CONTRACT:
- If trusted:
    status = "TRUSTED"
    trusted_class = predicted_class
    advisory_class = predicted_class
    automated_blocking_allowed = False
- If abstained:
    status = "UNKNOWN_OR_REVIEW"
    trusted_class = None
    advisory_class = predicted_class
    automated_blocking_allowed = False
    reasons = [triggered_reasons]
"""

import os
import sys
import json
import pickle
import hashlib
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from aegis_live.feature_schema import CANONICAL_24_FEATURES, validate_24_features

FROZEN_MODEL_SHA256 = "c157660c108a5d95d32d504ebc81c40e6b6911d8df6570a7d3ba92fad062cb9b"
FROZEN_TEMPERATURE = 1.0054586295782204
FROZEN_CONFIDENCE_THRESHOLD = 0.55
FROZEN_MAHALANOBIS_THRESHOLD = 35.0


@dataclass
class DetectionResult:
    status: str                         # "TRUSTED", "UNKNOWN_OR_REVIEW", or "REJECTED_QUARANTINE"
    trusted_class: Optional[str]        # String class name if TRUSTED, None if UNKNOWN_OR_REVIEW / REJECTED
    advisory_class: Optional[str]       # Model argmax prediction preserved as advisory
    raw_predicted_class: Optional[str]  # Raw argmax before abstention
    calibrated_probs: Dict[str, float]  # Softmax(logits / T) for all classes
    confidence: float                   # Max calibrated probability
    top2_margin: float                  # Difference between top-1 and top-2 probabilities
    entropy: float                      # Predictive Shannon entropy
    class_distances: Dict[str, float]   # Mahalanobis distance to each class centroid
    min_mahalanobis_distance: float     # Distance to nearest centroid
    predicted_class_distance: float     # Distance to predicted class centroid
    reasons: List[str]                  # Triggered abstention reasons (empty if TRUSTED)
    automated_blocking_allowed: bool = False  # IMMUTABLY FALSE (Detection-Only Mode)


class LiveInferenceEngine:
    """
    AEGIS-Live v2 Live Inference Engine.
    Executes the complete detection-only pipeline:
    24 Physical Features -> XGBoost -> Temperature Scaling -> Mahalanobis OOD -> Abstention Gate
    """
    def __init__(self,
                 model_dir: Optional[str] = None,
                 enforce_hash_check: bool = True):
        if model_dir is None:
            model_dir = os.path.join(BASE_DIR, "models", "aegis_live_v2")

        self.model_path = os.path.join(model_dir, "aegis_live_xgboost_baseline.pkl")
        self.encoder_path = os.path.join(model_dir, "aegis_live_label_encoder.pkl")
        self.temperature_path = os.path.join(model_dir, "temperature_scaling.json")
        self.ood_path = os.path.join(model_dir, "ood_parameters.pkl")
        self.abstention_path = os.path.join(model_dir, "abstention_config.json")

        # 1. Verify model hash
        actual_hash = self._compute_sha256(self.model_path)
        if enforce_hash_check and actual_hash != FROZEN_MODEL_SHA256:
            raise ValueError(
                f"Model hash mismatch! Expected {FROZEN_MODEL_SHA256}, got {actual_hash}"
            )
        self.model_hash = actual_hash

        # 2. Load model & label encoder
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(self.encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)
        self.classes = list(self.label_encoder.classes_)

        # 3. Load & verify temperature
        with open(self.temperature_path, "r") as f:
            temp_data = json.load(f)
        self.temperature = float(temp_data.get("learned_temperature", FROZEN_TEMPERATURE))
        if abs(self.temperature - FROZEN_TEMPERATURE) > 1e-4:
            raise ValueError(f"Temperature mismatch: expected {FROZEN_TEMPERATURE}, got {self.temperature}")

        # 4. Load OOD parameters
        with open(self.ood_path, "rb") as f:
            ood_data = pickle.load(f)
        self.class_means: Dict[int, np.ndarray] = ood_data["class_means"]
        self.class_inv_covs: Dict[int, np.ndarray] = ood_data["class_inv_covs"]
        self.lambda_reg = float(ood_data.get("lambda_reg", 1e-4))

        # 5. Load & verify abstention thresholds
        with open(self.abstention_path, "r") as f:
            abst_data = json.load(f)
        gate_def = abst_data.get("gate_definition", {})
        self.confidence_threshold = float(gate_def.get("confidence_threshold", FROZEN_CONFIDENCE_THRESHOLD))
        self.mahalanobis_threshold = float(gate_def.get("mahalanobis_threshold", FROZEN_MAHALANOBIS_THRESHOLD))

        if abs(self.confidence_threshold - FROZEN_CONFIDENCE_THRESHOLD) > 1e-4:
            raise ValueError(f"Confidence threshold mismatch: {self.confidence_threshold}")
        if abs(self.mahalanobis_threshold - FROZEN_MAHALANOBIS_THRESHOLD) > 1e-4:
            raise ValueError(f"Mahalanobis threshold mismatch: {self.mahalanobis_threshold}")

    @staticmethod
    def _compute_sha256(filepath: str) -> str:
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        return h.hexdigest()

    def _compute_mahalanobis(self, x: np.ndarray) -> Dict[str, float]:
        """
        Computes Mahalanobis distance from feature vector x (1, 24) to each of the 4 class centroids.
        """
        distances = {}
        for c_idx, c_name in enumerate(self.classes):
            mu = self.class_means[c_idx]
            inv_cov = self.class_inv_covs[c_idx]
            diff = x - mu  # shape (24,)
            d_sq = float(np.dot(np.dot(diff, inv_cov), diff))
            d_sq = max(0.0, d_sq)  # Prevent negative numerical noise
            distances[c_name] = float(np.sqrt(d_sq))
        return distances

    def predict(self, feature_dict: Dict[str, float]) -> DetectionResult:
        """
        Executes end-to-end detection on a 24-feature physical window vector.
        Never mutates network, never performs automated blocking.
        """
        # Step 1: Physical Feature Validation
        try:
            validate_24_features(feature_dict)
        except Exception as e:
            return DetectionResult(
                status="REJECTED_QUARANTINE",
                trusted_class=None,
                advisory_class=None,
                raw_predicted_class=None,
                calibrated_probs={},
                confidence=0.0,
                top2_margin=0.0,
                entropy=0.0,
                class_distances={},
                min_mahalanobis_distance=float("inf"),
                predicted_class_distance=float("inf"),
                reasons=[f"feature_validation_failed: {str(e)}"],
                automated_blocking_allowed=False
            )

        # Step 2: Ordered Feature Array
        x_vec = np.array([float(feature_dict[col]) for col in CANONICAL_24_FEATURES], dtype=np.float64)
        df_in = pd.DataFrame([x_vec], columns=CANONICAL_24_FEATURES)

        # Step 3: Raw XGBoost Logits (output_margin=True)
        raw_logits = self.model.predict(df_in, output_margin=True)[0]  # shape (4,)

        # Step 4: Calibrated Probabilities via Frozen Temperature Scaling
        scaled_logits = raw_logits / self.temperature
        shift_z = scaled_logits - np.max(scaled_logits)
        exp_z = np.exp(shift_z)
        probs = exp_z / np.sum(exp_z)

        # Step 5: Metrics & Uncertainty Signals
        pred_idx = int(np.argmax(probs))
        raw_predicted_class = self.classes[pred_idx]
        confidence = float(np.max(probs))

        sorted_probs = np.sort(probs)
        top2_margin = float(sorted_probs[-1] - sorted_probs[-2])

        p_clipped = np.clip(probs, 1e-12, 1.0)
        entropy = float(-np.sum(p_clipped * np.log2(p_clipped)))

        calibrated_probs_dict = {
            self.classes[i]: float(probs[i]) for i in range(len(self.classes))
        }

        # Step 6: Mahalanobis OOD Calculation
        class_distances = self._compute_mahalanobis(x_vec)
        min_mahalanobis = float(min(class_distances.values()))
        pred_mahalanobis = float(class_distances[raw_predicted_class])

        # Step 7: Evidence-Based Abstention Gate
        reasons = []
        if confidence < self.confidence_threshold:
            reasons.append(f"confidence_below_{self.confidence_threshold}")
        if pred_mahalanobis > self.mahalanobis_threshold:
            reasons.append(f"mahalanobis_above_{self.mahalanobis_threshold}")

        # Step 8: Decision Contract
        if len(reasons) == 0:
            status = "TRUSTED"
            trusted_class = raw_predicted_class
            advisory_class = raw_predicted_class
        else:
            status = "UNKNOWN_OR_REVIEW"
            trusted_class = None
            advisory_class = raw_predicted_class

        return DetectionResult(
            status=status,
            trusted_class=trusted_class,
            advisory_class=advisory_class,
            raw_predicted_class=raw_predicted_class,
            calibrated_probs=calibrated_probs_dict,
            confidence=confidence,
            top2_margin=top2_margin,
            entropy=entropy,
            class_distances=class_distances,
            min_mahalanobis_distance=min_mahalanobis,
            predicted_class_distance=pred_mahalanobis,
            reasons=reasons,
            automated_blocking_allowed=False  # ALWAYS FALSE
        )

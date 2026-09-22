"""
backend/detection/abstention_gate.py
====================================
External OOD-Aware Abstention and Decision Gate for AEGIS-IoT.

Closes the architectural loophole identified in Phase 13:
- Prevents unfamiliar or out-of-distribution inputs from being silently forced
  into one of the 3 closed-world classes (DDoS, DoS, Spoofing).
- Evaluates classification certainty (calibrated confidence) AND feature-space
  novelty (Class-Conditional Mahalanobis distance).
- Assigns formal decision states:
    - KNOWN_DDoS
    - KNOWN_DoS
    - KNOWN_Spoofing
    - UNKNOWN_OR_REVIEW
- Enforces Human-In-The-Loop safety by strictly prohibiting automated blocking
  or network mutations on abstained traffic.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from backend.detection.ood_config import (
    CONFIDENCE_THRESHOLD,
    CC_MAHALANOBIS_THRESHOLD,
    STATE_KNOWN_DDOS,
    STATE_KNOWN_DOS,
    STATE_KNOWN_SPOOFING,
    STATE_UNKNOWN_OR_REVIEW,
    REASON_LOW_CONFIDENCE,
    REASON_OOD_DISTANCE,
    REASON_BOTH,
    POLICY_AUTOMATED_BLOCK_FOR_UNKNOWN
)
from backend.detection.ood_detector import OODNoveltyDetector


class OODAbstentionGate:
    """
    Unified OOD and Confidence Abstention Gate.
    """
    def __init__(self,
                 conf_threshold: float = CONFIDENCE_THRESHOLD,
                 mah_threshold: float = CC_MAHALANOBIS_THRESHOLD,
                 novelty_detector: Optional[OODNoveltyDetector] = None):
        self.conf_threshold = conf_threshold
        self.mah_threshold = mah_threshold
        self.detector = novelty_detector or OODNoveltyDetector(threshold=mah_threshold)

    def evaluate(self, detection_result: Dict[str, Any], raw_features: Any) -> Dict[str, Any]:
        """
        Evaluates an inference result against confidence and OOD criteria.
        
        Args:
            detection_result: Dictionary containing:
                - "attack_type" / "predicted_class": Model's argmax class name.
                - "confidence": Calibrated posterior probability (0.0 to 1.0).
                - "probabilities": Dict of class -> calibrated probability.
            raw_features: 46 raw features (DataFrame, Series, dict, or array).
            
        Returns:
            Dict[str, Any]: Comprehensive routing, decision state, and diagnostic record.
        """
        predicted_class = detection_result.get("attack_type") or detection_result.get("predicted_class", "Unknown")
        confidence = float(detection_result.get("confidence", 0.0))
        probabilities = detection_result.get("probabilities", {})

        # Compute Top-2 Margin if probabilities available
        if isinstance(probabilities, dict) and len(probabilities) >= 2:
            sorted_probs = sorted(probabilities.values(), reverse=True)
            top2_margin = round(float(sorted_probs[0] - sorted_probs[1]), 4)
        elif isinstance(probabilities, (list, np.ndarray)) and len(probabilities) >= 2:
            sorted_probs = sorted(probabilities, reverse=True)
            top2_margin = round(float(sorted_probs[0] - sorted_probs[1]), 4)
        else:
            top2_margin = 0.0

        # 1. Feature-space novelty evaluation (Class-Conditional Mahalanobis)
        novelty_eval = self.detector.evaluate_novelty(raw_features, predicted_class)
        mah_distance = novelty_eval["cc_mahalanobis_distance"]
        is_atypical = novelty_eval["is_atypical"]

        # 2. Decision boundary uncertainty evaluation
        is_low_conf = bool(confidence < self.conf_threshold)

        # 3. Decision Rule & Abstention Logic
        is_abstained = bool(is_low_conf or is_atypical)
        reasons: List[str] = []

        if is_low_conf and is_atypical:
            reasons = [REASON_LOW_CONFIDENCE, REASON_OOD_DISTANCE]
            abstention_summary = REASON_BOTH
        elif is_low_conf:
            reasons = [REASON_LOW_CONFIDENCE]
            abstention_summary = REASON_LOW_CONFIDENCE
        elif is_atypical:
            reasons = [REASON_OOD_DISTANCE]
            abstention_summary = REASON_OOD_DISTANCE
        else:
            abstention_summary = None

        if is_abstained:
            decision_state = STATE_UNKNOWN_OR_REVIEW
            trusted_class = None
            routing = "HUMAN_INVESTIGATION"
            priority = "HIGH"
            action_required = "ANALYST_TRIAGE"
            severity = "HIGH"
            automated_blocking_allowed = POLICY_AUTOMATED_BLOCK_FOR_UNKNOWN
            escalate_to_rag = True  # Escalate for advisory intelligence, but flag as unverified
            advisory_note = (
                f"Model predicted '{predicted_class}' with {confidence*100:.2f}% confidence, but sample "
                f"triggered abstention ({abstention_summary}). Prediction is advisory only and must not "
                f"trigger automated network disruptions."
            )
        else:
            decision_state = f"KNOWN_{predicted_class}"
            trusted_class = predicted_class
            severity = "CRITICAL" if predicted_class == "DDoS" else "HIGH"
            routing = "ADAPTIVE_HYBRID_RAG"
            priority = "HIGH"
            action_required = "EXECUTE_AGENT_GRAPH"
            automated_blocking_allowed = True
            escalate_to_rag = True
            advisory_note = f"Verified in-distribution sample classified as {predicted_class}."

        return {
            "decision_state": decision_state,
            "is_abstained": is_abstained,
            "abstention_reasons": reasons,
            "abstention_summary": abstention_summary,
            "advisory_class": predicted_class,
            "trusted_class": trusted_class,
            "confidence": round(confidence, 4),
            "conf_threshold": self.conf_threshold,
            "top2_margin": top2_margin,
            "cc_mahalanobis_distance": mah_distance,
            "mah_threshold": self.mah_threshold,
            "is_atypical_distance": is_atypical,
            "is_low_confidence": is_low_conf,
            "severity": severity,
            "routing": routing,
            "priority": priority,
            "action_required": action_required,
            "automated_blocking_allowed": automated_blocking_allowed,
            "escalate_to_rag": escalate_to_rag,
            "advisory_note": advisory_note
        }

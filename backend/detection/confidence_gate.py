"""
backend/detection/confidence_gate.py
====================================
Upgraded Confidence and OOD-Aware Gate for AEGIS-IoT.

Preserves full backward-compatibility for existing callers while incorporating:
- Class-Conditional Mahalanobis novelty scoring
- Formal UNKNOWN_OR_REVIEW decision state
- Automated network mutation prohibition for abstained inputs
"""

from typing import Dict, Any, Optional
from backend.detection.ood_config import (
    CONFIDENCE_THRESHOLD,
    CC_MAHALANOBIS_THRESHOLD,
    STATE_UNKNOWN_OR_REVIEW
)
from backend.detection.abstention_gate import OODAbstentionGate


class ConfidenceGate:
    def __init__(self,
                 high_confidence_threshold: float = CONFIDENCE_THRESHOLD,
                 critical_risk_threshold: float = 0.85,
                 mah_threshold: float = CC_MAHALANOBIS_THRESHOLD):
        self.high_threshold = high_confidence_threshold
        self.critical_threshold = critical_risk_threshold
        self.mah_threshold = mah_threshold
        self.abstention_gate = OODAbstentionGate(
            conf_threshold=high_confidence_threshold,
            mah_threshold=mah_threshold
        )

    def evaluate(self, detection_result: Dict[str, Any], raw_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates confidence and feature-space novelty to determine routing:
        - IN-DISTRIBUTION & CONFIDENT -> Escalate to Adaptive Hybrid RAG & Multi-Agent Orchestrator
        - LOW_CONFIDENCE or OOD -> Assign UNKNOWN_OR_REVIEW, queue for Human Analyst Triage,
          strictly block automated network enforcement.
        """
        # Run OOD-aware abstention evaluation
        gate_res = self.abstention_gate.evaluate(detection_result, raw_features)

        confidence = detection_result.get("confidence", 0.0)
        attack_type = detection_result.get("attack_type", "Benign")
        rate = raw_features.get("Rate", 0.0) if isinstance(raw_features, dict) else 0.0

        # Maintain backwards-compatible severity logic
        if gate_res["is_abstained"]:
            severity = "HIGH"
            routing = "HUMAN_INVESTIGATION"
            priority = "HIGH"
            action_required = "ANALYST_TRIAGE"
            escalate_to_rag = True  # Escalate for advisory context, but marked unverified
        elif attack_type == "DDoS" or (confidence >= self.critical_threshold and rate > 500):
            severity = "CRITICAL"
            routing = "ADAPTIVE_HYBRID_RAG"
            priority = "HIGH"
            action_required = "EXECUTE_AGENT_GRAPH"
            escalate_to_rag = True
        elif attack_type in ["DoS", "Spoofing"] and confidence >= self.high_threshold:
            severity = "HIGH"
            routing = "ADAPTIVE_HYBRID_RAG"
            priority = "HIGH"
            action_required = "EXECUTE_AGENT_GRAPH"
            escalate_to_rag = True
        elif attack_type in ["DoS", "Spoofing"]:
            severity = "MEDIUM"
            routing = "HUMAN_INVESTIGATION"
            priority = "MEDIUM"
            action_required = "ANALYST_TRIAGE"
            escalate_to_rag = False
        else:
            severity = "LOW"
            routing = "TELEMETRY_LOG"
            priority = "LOW"
            action_required = "NONE"
            escalate_to_rag = False

        # Merge with all new OOD and decision state fields
        output = {
            "severity": severity,
            "routing": routing,
            "priority": priority,
            "action_required": action_required,
            "escalate_to_rag": escalate_to_rag,
            # Phase 14 extensions:
            "decision_state": gate_res["decision_state"],
            "is_abstained": gate_res["is_abstained"],
            "abstention_reasons": gate_res["abstention_reasons"],
            "abstention_summary": gate_res["abstention_summary"],
            "advisory_class": gate_res["advisory_class"],
            "trusted_class": gate_res["trusted_class"],
            "confidence": gate_res["confidence"],
            "top2_margin": gate_res["top2_margin"],
            "cc_mahalanobis_distance": gate_res["cc_mahalanobis_distance"],
            "is_atypical_distance": gate_res["is_atypical_distance"],
            "is_low_confidence": gate_res["is_low_confidence"],
            "automated_blocking_allowed": gate_res["automated_blocking_allowed"],
            "advisory_note": gate_res["advisory_note"]
        }
        return output

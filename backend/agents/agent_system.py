"""
backend/agents/agent_system.py
==============================
Specialized 5-Agent Architecture + Grounded LLM Reasoning Synthesizer
for IoT Intrusion Detection System.

Agents:
1. DetectionAnalysisAgent: Interprets calibrated ML detection without overriding XGBoost.
2. ExplainabilityAgent: Translates TreeSHAP values into directional feature drivers without hallucination.
3. ThreatIntelligenceAgent: Leverages Adaptive Hybrid RAG across MITRE, NIST, and CISA (consumes full doc content).
4. ThreatAssessmentAgent: Computes risk severity and separates FACT, INFERENCE, RECOMMENDATION.
5. ResponseRecommendationAgent: Formulates prioritized defensive playbooks grounded in retrieved guidance (advisory only).
6. LLMSynthesizer: Assembles unified, grounded, 10-part human-readable security incident dossiers.
"""

import os
import sys
import re
import json
import time
from typing import Dict, Any, List, Optional
from backend.rag.adaptive_hybrid_rag import AdaptiveHybridRAG, NOT_ESTABLISHED


# =====================================================================
# AGENT 1: DETECTION ANALYSIS AGENT
# =====================================================================
class DetectionAnalysisAgent:
    """
    Interprets the existing ML detection and calibrated probabilities.
    DOES NOT perform classification and CANNOT override XGBoost.
    """
    def __init__(self):
        self.name = "Detection Analysis Agent"

    def analyze(self,
                attack_type: str,
                confidence: float,
                probabilities: Dict[str, float],
                features: Dict[str, float]) -> Dict[str, Any]:
        
        # Determine certainty / margin
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        top_class, top_p = sorted_probs[0]
        second_class, second_p = sorted_probs[1] if len(sorted_probs) > 1 else ("None", 0.0)
        margin = top_p - second_p

        if top_p >= 0.99:
            confidence_level = "Near-Definite (>99% Calibrated Posterior)"
            uncertainty = "Negligible calibration margin."
        elif top_p >= 0.90:
            confidence_level = "High Confidence (90-99%)"
            uncertainty = f"Moderate distance to alternative class {second_class} ({second_p*100:.2f}%)."
        elif top_p >= 0.70:
            confidence_level = "Moderate Confidence (70-90%)"
            uncertainty = f"Elevated residual ambiguity with {second_class} ({second_p*100:.2f}%)."
        else:
            confidence_level = "Low / Ambiguous (<70%)"
            uncertainty = f"Substantial ambiguity between {top_class} and {second_class}."

        prob_summary = ", ".join([f"{cls}: {prob*100:.4f}%" for cls, prob in probabilities.items()])

        detection_summary = (
            f"The 46-feature XGBoost IDS detected {attack_type} traffic with {confidence*100:.4f}% "
            f"calibrated confidence ({confidence_level}). Posterior class distribution: [{prob_summary}]."
        )

        evidence_summary = (
            f"Observed network flow exhibits key parameters: "
            f"Rate={features.get('Rate', 0.0):.2f} pkts/s, Srate={features.get('Srate', 0.0):.2f}, "
            f"Duration={features.get('Duration', 0.0):.4f}s, Header_Length={features.get('Header_Length', 0.0):.1f}B."
        )

        return {
            "agent": self.name,
            "detected_attack": attack_type,
            "confidence": float(confidence),
            "confidence_level": confidence_level,
            "probabilities": {k: float(v) for k, v in probabilities.items()},
            "probability_summary": prob_summary,
            "detection_summary": detection_summary,
            "evidence_summary": evidence_summary,
            "uncertainty": uncertainty
        }


# =====================================================================
# AGENT 2: EXPLAINABILITY AGENT
# =====================================================================
class ExplainabilityAgent:
    """
    Interprets the TreeSHAP explanations.
    Uses only mathematical SHAP contributions; never invents or hallucinates features.
    """
    def __init__(self):
        self.name = "Explainability Agent"

    def explain(self,
                predicted_attack: str,
                contributions: List[Dict[str, Any]]) -> Dict[str, Any]:
        
        supporting_evidence = []
        opposing_evidence = []
        top_features = []

        for item in contributions:
            feat_name = item.get("feature", "unknown")
            s_val = float(item.get("shap_value", 0.0))
            f_val = float(item.get("feature_value", 0.0))
            rank = item.get("rank", 0)

            record = {
                "rank": rank,
                "feature": feat_name,
                "feature_value": f_val,
                "shap_value": round(s_val, 6),
                "direction": "SUPPORTING" if s_val > 0 else "OPPOSING",
                "impact": f"Pushes prediction TOWARD {predicted_attack} (+)" if s_val > 0 else f"Pushes prediction AWAY from {predicted_attack} (-)"
            }
            top_features.append(record)

            if s_val > 0:
                supporting_evidence.append(record)
            else:
                opposing_evidence.append(record)

        # Build narrative explanation
        top_pos = [f"{e['feature']} (val={e['feature_value']:.2f}, SHAP=+{e['shap_value']:.4f})" for e in supporting_evidence[:3]]
        top_neg = [f"{e['feature']} (val={e['feature_value']:.2f}, SHAP={e['shap_value']:.4f})" for e in opposing_evidence[:2]]

        narrative_parts = []
        if top_pos:
            narrative_parts.append(f"Primary drivers pushing the model TOWARD {predicted_attack}: {', '.join(top_pos)}.")
        if top_neg:
            narrative_parts.append(f"Features moderating or opposing this decision: {', '.join(top_neg)}.")
        if not narrative_parts:
            narrative_parts.append("SHAP values indicate distributed contribution across network flow metrics.")

        human_readable_explanation = " ".join(narrative_parts)

        return {
            "agent": self.name,
            "predicted_attack": predicted_attack,
            "top_features": top_features,
            "supporting_evidence": supporting_evidence,
            "opposing_evidence": opposing_evidence,
            "human_readable_explanation": human_readable_explanation
        }


# =====================================================================
# AGENT 3: THREAT INTELLIGENCE / RAG AGENT
# =====================================================================
class ThreatIntelligenceAgent:
    """
    Retrieves and consumes authoritative cybersecurity knowledge from MITRE, NIST, and CISA.
    Consumes the structured EvidenceBundle returned by AdaptiveHybridRAG — agents receive
    only attack-relevant evidence (mitre_primary + attack_specific tiers).
    General-context CISA/NIST content is kept in a separate field and NOT used for
    attack-specific reasoning fields (attack_descriptions, mechanisms, iot_impacts).
    """
    def __init__(self, retriever: Optional[AdaptiveHybridRAG] = None):
        self.name = "Threat Intelligence / RAG Agent"
        self.retriever = retriever or AdaptiveHybridRAG()

    def retrieve_intel(self,
                       detected_attack: str,
                       shap_evidence: List[Dict[str, Any]],
                       iot_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieves filtered, attack-specific evidence via AdaptiveHybridRAG and
        organises it into structured intelligence fields.

        Evidence consumption policy:
          mitre_primary      → attack_descriptions, mechanisms, mitre_techniques
          attack_specific    → iot_impacts_specific, mitigation_guidance, indicators
          general_context    → general_security_context (separate field — NOT used
                               as primary attack evidence)
          excluded           → rag_grounding_trace only (provenance, not propagated)
        """
        grounded_result = self.retriever.get_grounded_context(
            attack_type=detected_attack,
            shap_evidence=shap_evidence,
            top_k=6
        )

        evidence_bundle = grounded_result.get("evidence_bundle", {})
        all_retrieved   = grounded_result.get("retrieved_sources", [])

        # Evidence buckets from the bundle
        mitre_docs      = evidence_bundle.get("mitre_context", [])
        specific_docs   = evidence_bundle.get("attack_specific_evidence", [])
        general_docs    = evidence_bundle.get("general_security_evidence", [])
        excluded_docs   = evidence_bundle.get("excluded_evidence", [])
        mitig_docs      = evidence_bundle.get("mitigation_evidence", [])
        grounding_trace = evidence_bundle.get("rag_grounding_trace", "")

        def _grounded_text(doc: Dict[str, Any]) -> str:
            """Use attack-relevant excerpt when present; never unfiltered mixed CISA text."""
            excerpt = (doc.get("grounded_excerpt") or "").strip()
            raw = excerpt if excerpt else (doc.get("content", "") or "")
            return re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', raw)

        # ----------------------------------------------------------------
        # Extract structured fields — consume ONLY mitre_primary and
        # attack_specific tiers for attack-reasoning fields.
        # ----------------------------------------------------------------
        attack_descriptions = []
        mechanisms = []
        iot_impacts_specific = []
        general_security_context = []
        indicators = []
        mitre_techniques = []
        mitigation_guidance = []

        # --- Process MITRE primary documents ---
        for doc in mitre_docs:
            title   = doc.get("title", "")
            chunk_id = doc.get("chunk_id", "")
            url     = doc.get("url", "")
            clean_content = _grounded_text(doc)

            m_tactic = re.search(r'Tactics:\s*(.*?)(?:\.|Platforms:)', clean_content)
            tactic_text = m_tactic.group(1).strip() if m_tactic else "Impact"

            m_desc = re.search(r'Description:\s*(.*?)(?:Detection Guidance:|$)', clean_content, re.DOTALL)
            desc_text = m_desc.group(1).strip() if m_desc else clean_content[:800]

            mitre_techniques.append({
                "technique_id": chunk_id.replace("MITRE-", ""),
                "technique_name": title,
                "tactic": tactic_text,
                "url": url,
                "description": desc_text,
                "evidence_tier": "mitre_primary",
            })
            attack_descriptions.append(f"{title}: {desc_text[:600]}")
            mechanisms.append(f"MITRE ATT&CK ({title}): {desc_text[:600]}")
            tech_id = chunk_id.replace("MITRE-", "")
            iot_impacts_specific.append(
                "[INFERENCE — retrieved MITRE " + tech_id + " (" + title + ")] "
                "The retrieved technique description may affect IoT/OT operations as follows: "
                + desc_text[:400]
                + " This is not a confirmed impact for this specific network flow."
            )

        # --- Process attack_specific documents ---
        for doc in specific_docs:
            src     = doc.get("source", "")
            title   = doc.get("title", "")
            chunk_id = doc.get("chunk_id", "")
            url     = doc.get("url", "")
            acs     = doc.get("attack_context_score", 0.0)
            clean_content = _grounded_text(doc)
            if not clean_content:
                continue

            if "NIST" in src:
                if any(kw in title for kw in ["Section 6.7", "Detection Signatures", "SI-4", "Monitoring"]):
                    indicators.append(
                        f"NIST ({title}) [attack_specific, acs={acs:.2f}]: {clean_content}"
                    )
                elif any(kw in title for kw in ["SC-5", "Denial-of-Service", "SC-7", "Boundary"]):
                    mitigation_guidance.append(
                        f"NIST ({title}) [attack_specific, acs={acs:.2f}]: {clean_content}"
                    )
                    iot_impacts_specific.append(
                        f"[INFERENCE — retrieved NIST guidance {title}] "
                        f"The retrieved control describes effects/controls that may apply: "
                        f"{clean_content[:400]}"
                    )
                elif any(kw in title for kw in ["IR 8259A", "Device Identification"]):
                    mitigation_guidance.append(
                        f"NIST IR 8259A ({title}) [attack_specific]: {clean_content}"
                    )
                else:
                    iot_impacts_specific.append(
                        f"[INFERENCE — retrieved NIST guidance {title}] {clean_content[:400]}"
                    )

            elif "CISA" in src:
                # Use excerpt only (already stripped of other-class sentences)
                m_tp = re.search(r'Threat Profile:\s*(.*?)(?:Mandates:|Controls:|$)', clean_content, re.DOTALL)
                m_mand = re.search(r'(?:Mandates:|Controls:|Description:)\s*(.*)', clean_content, re.DOTALL)
                excerpt_body = m_tp.group(1).strip()[:500] if m_tp else clean_content[:400]
                attack_descriptions.append(
                    f"CISA Advisory ({title}) [attack_specific excerpt, acs={acs:.2f}]: {excerpt_body}"
                )
                iot_impacts_specific.append(
                    f"[INFERENCE — retrieved CISA excerpt {title}] "
                    f"The retrieved guidance describes: {excerpt_body} "
                    f"This assessment is based on retrieved {detected_attack}-relevant passages, "
                    f"not a confirmed operational impact for this flow."
                )
                if m_mand:
                    mitigation_guidance.append(
                        f"CISA ({title}): {m_mand.group(1).strip()[:400]}"
                    )

            elif "MITRE" in src:
                m_desc = re.search(r'Description:\s*(.*?)(?:Detection Guidance:|$)', clean_content, re.DOTALL)
                desc_text = m_desc.group(1).strip() if m_desc else clean_content[:600]
                m_tactic = re.search(r'Tactics:\s*(.*?)(?:\.|Platforms:)', clean_content)
                tactic_text = m_tactic.group(1).strip() if m_tactic else "Impact"

                mitre_techniques.append({
                    "technique_id": chunk_id.replace("MITRE-", ""),
                    "technique_name": title,
                    "tactic": tactic_text,
                    "url": url,
                    "description": desc_text,
                    "evidence_tier": "attack_specific",
                })
                mechanisms.append(
                    f"MITRE ATT&CK ({title}) [attack_specific]: {desc_text[:400]}"
                )

        # --- Process general_context documents ---
        # Kept separate — NOT used as IoT/OT impact or attack mechanism.
        for doc in general_docs:
            src   = doc.get("source", "")
            title = doc.get("title", "")
            acs   = doc.get("attack_context_score", 0.0)
            excerpt = _grounded_text(doc)
            preview = excerpt[:300] if excerpt else "(no attack-specific excerpt; title/metadata only)"
            general_security_context.append(
                f"[{src} — {title}, general_context, acs={acs:.2f}] {preview}"
            )
            if excerpt:
                m_mand = re.search(r'(?:Mandates:|Controls:|Description:|Guidance:)\s*(.*)', excerpt, re.DOTALL)
                if m_mand and any(kw in title for kw in ["SC-", "Control", "CPG", "IR 8259A", "Playbook"]):
                    mitigation_guidance.append(
                        f"[General context — {src} {title}]: {m_mand.group(1).strip()[:300]}"
                    )

        for doc in mitig_docs:
            excerpt = _grounded_text(doc)
            if excerpt and doc.get("evidence_tier") in ("mitre_primary", "attack_specific"):
                label = f"{doc.get('source')} ({doc.get('title')})"
                if not any(label in g for g in mitigation_guidance):
                    mitigation_guidance.append(f"{label}: {excerpt[:400]}")

        if not attack_descriptions:
            attack_descriptions.append(NOT_ESTABLISHED)
        if not mechanisms:
            mechanisms.append(NOT_ESTABLISHED)
        if not iot_impacts_specific:
            iot_impacts_specific.append(NOT_ESTABLISHED)
        if not indicators:
            indicators.append(NOT_ESTABLISHED)
        if not mitigation_guidance:
            mitigation_guidance.append(NOT_ESTABLISHED)

        # Threat profile is an inference from retrieved MITRE context only.
        if mitre_techniques:
            t0 = mitre_techniques[0]
            threat_actor_profile = (
                f"Inference consistent with retrieved MITRE ATT&CK "
                f"{t0['technique_id']} ({t0['technique_name']}). "
                f"Not a confirmed actor attribution for this flow."
            )
        else:
            threat_actor_profile = NOT_ESTABLISHED

        return {
            "agent": self.name,
            "query": grounded_result.get("query", ""),
            "retrieved_sources": all_retrieved,
            "evidence_bundle": evidence_bundle,
            "rag_grounding_trace": grounding_trace,
            "threat_actor_profile": threat_actor_profile,
            "attack_descriptions": attack_descriptions,
            "mechanisms": mechanisms,
            # iot_impacts contains ONLY attack-specific evidence (mitre_primary + attack_specific tier)
            "iot_impacts": iot_impacts_specific,
            # general_security_context holds general-tier docs — kept separate for provenance
            "general_security_context": general_security_context,
            "indicators": indicators,
            "mitre_techniques": mitre_techniques,
            "mitigation_guidance": mitigation_guidance,
            "excluded_evidence": excluded_docs,
            "provenance": evidence_bundle.get("provenance", []),
            "status": grounded_result.get("status", "SUCCESS"),
        }


# =====================================================================
# AGENT 4: THREAT ASSESSMENT AGENT
# =====================================================================
class ThreatAssessmentAgent:
    """
    Assesses security significance, severity, blast radius, and potential impact.
    Strictly distinguishes FACT, INFERENCE, and RECOMMENDATION.
    """
    def __init__(self):
        self.name = "Threat Assessment Agent"

    def assess(self,
               attack_type: str,
               confidence: float,
               detection_data: Dict[str, Any],
               explainability_data: Dict[str, Any],
               threat_intel_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Computes risk severity and separates FACT, INFERENCE, RECOMMENDATION.

        Risk score formula (HEURISTIC — not empirically validated):
            base_severity  = 9.0 (DDoS) | 7.5 (DoS) | 7.0 (Spoofing)  [hand-assigned]
            risk_score     = clamp(base_severity x calibrated_confidence, 1.0, 10.0)

        IoT/OT impact is derived ONLY from attack-specific evidence
        (threat_intel_data['iot_impacts'] which contains mitre_primary and
        attack_specific tier content only — general_context docs are excluded
        from this field by ThreatIntelligenceAgent).

        Epistemic distinction:
          FACT:           XGBoost prediction + calibrated probability + TreeSHAP values
          INFERENCE:      Interpretation consistent with retrieved MITRE/NIST evidence
          RECOMMENDATION: Advisory mitigations (human-in-the-loop required)
        """
        # Heuristic risk score — hand-assigned base severity × calibrated confidence
        base_severity = 9.0 if attack_type == "DDoS" else (7.5 if attack_type == "DoS" else 7.0)
        risk_score = round(min(max(base_severity * confidence, 1.0), 10.0), 1)

        if risk_score >= 8.5:
            severity = "CRITICAL"
        elif risk_score >= 6.5:
            severity = "HIGH"
        elif risk_score >= 4.5:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        # Construct impact description from attack-specific evidence ONLY.
        # threat_intel_data['iot_impacts'] contains only mitre_primary + attack_specific
        # tier content — general-context CISA/NIST docs are NOT present here.
        retrieved_impacts = [
            imp for imp in threat_intel_data.get("iot_impacts", [])
            if imp and imp != NOT_ESTABLISHED
        ]
        if retrieved_impacts:
            raw_impact = " ".join(retrieved_impacts[:2])
            impact = (
                f"[INFERENCE — based on retrieved {attack_type}-specific evidence] "
                f"{raw_impact}"
            )
        else:
            impact = (
                f"[INFERENCE] {NOT_ESTABLISHED} "
                f"No attack-specific IoT/OT impact passage was selected for {attack_type}."
            )

        # Grounded facts (model & explainability observations — these ARE facts)
        top_pos = explainability_data.get("top_features", [])[:3]
        top_pos_str = ", ".join([
            f"{f['feature']} (val={f['feature_value']:.2f}, SHAP={f['shap_value']:+.4f})"
            for f in top_pos
        ])

        facts = [
            f"46-Feature XGBoost classified telemetry as '{attack_type}' with "
            f"{confidence*100:.4f}% calibrated (temperature-scaled) confidence.",
            f"TreeSHAP primary mathematical drivers: {top_pos_str}.",
        ]

        # Inferences — consistent with retrieved evidence but not confirmed for this specific flow
        mitre_techs = threat_intel_data.get("mitre_techniques", [])
        mitre_ref = ""
        if mitre_techs:
            primary = [t for t in mitre_techs if t.get("evidence_tier") == "mitre_primary"]
            if primary:
                mitre_ref = (
                    f" Retrieved MITRE ATT&CK material is consistent with "
                    f"{primary[0]['technique_id']} ({primary[0]['technique_name']}) patterns."
                )

        inferences = [
            f"Flow pattern is consistent with "
            f"{threat_intel_data.get('threat_actor_profile', 'IoT Network Anomaly')}."
            f"{mitre_ref}",
            f"Possible operational impact (retrieved evidence basis): {impact}",
        ]

        # Recommendations preview
        recommendations_preview = [
            f"Implement {attack_type}-specific mitigations per retrieved guidance. "
            f"Human-in-the-loop approval required before enforcement."
        ]

        summary = (
            f"Severity evaluated as {severity} "
            f"(Heuristic Risk Score: {risk_score}/10.0 — "
            f"base_severity={base_severity} x confidence={confidence:.4f}, clamped to [1,10]). "
            f"Model classified this flow as {attack_type} with high confidence."
        )

        limitations = (
            "Assessment is based on offline flow record telemetry and TreeSHAP feature attributions. "
            "Physical device actuation state cannot be independently confirmed without endpoint agent telemetry. "
            "Risk score is a heuristic — not empirically validated."
        )

        return {
            "agent": self.name,
            "severity": severity,
            "composite_risk_score": risk_score,
            "risk_score_formula": (
                f"Heuristic: base_severity({base_severity}) x confidence({confidence:.4f}) "
                f"= {base_severity * confidence:.4f}, clamped to [1.0, 10.0] → {risk_score}"
            ),
            "summary": summary,
            "potential_impact": impact,
            "uncertainty": detection_data.get("uncertainty", "Minimal"),
            "limitations": limitations,
            "epistemic_breakdown": {
                "FACT": facts,
                "INFERENCE": inferences,
                "RECOMMENDATION": recommendations_preview,
            },
        }


# =====================================================================
# AGENT 5: RESPONSE RECOMMENDATION AGENT
# =====================================================================
class ResponseRecommendationAgent:
    """
    Generates actionable, prioritized defensive recommendations.
    Grounded in retrieved NIST, CISA, and MITRE guidance.
    Advisory ONLY; does NOT execute automated network/system mutations.
    """
    def __init__(self):
        self.name = "Response Recommendation Agent"

    def formulate_recommendations(self,
                                  attack_type: str,
                                  confidence: float,
                                  threat_assessment: Dict[str, Any],
                                  threat_intel: Dict[str, Any]) -> List[Dict[str, Any]]:
        
        recs = []
        bundle = threat_intel.get("evidence_bundle", {})
        usable_docs = (
            bundle.get("mitre_context", [])
            + bundle.get("attack_specific_evidence", [])
            + bundle.get("mitigation_evidence", [])
        )
        # Deduplicate by chunk_id while preserving order
        seen = set()
        filtered_docs = []
        for d in usable_docs:
            cid = d.get("chunk_id")
            if cid in seen:
                continue
            seen.add(cid)
            filtered_docs.append(d)

        def _src_label(doc):
            return f"{doc.get('source', '')} ({doc.get('title', '')}) [{doc.get('evidence_tier', '')}]"

        filtered_names = [_src_label(d) for d in filtered_docs]
        cisa_source = next((s for s in filtered_names if "CISA" in s), None)
        nist_source = next((s for s in filtered_names if "NIST" in s), None)
        mitre_source = next((s for s in filtered_names if "MITRE" in s), None)

        mitig_lines = [
            g for g in threat_intel.get("mitigation_guidance", [])
            if g and g != NOT_ESTABLISHED
        ]
        grounded_note = (
            mitig_lines[0][:280] if mitig_lines
            else NOT_ESTABLISHED
        )

        def _sources(*preferred):
            out = [s for s in preferred if s]
            if not out:
                out = filtered_names[:2] if filtered_names else ["No attack-specific retrieved source"]
            return out

        if attack_type == "DDoS":
            recs.append({
                "action_id": "REC-DDOS-01",
                "priority": "P1 - CRITICAL",
                "action_type": "RATE_LIMITING_AND_SCRUBBING",
                "title": "Enforce Dynamic Edge Ingress Rate-Limiting & SYN Cookie Protection",
                "rationale": (
                    "Advisory only. Retrieved DDoS-relevant guidance supports availability "
                    f"protection: {grounded_note}"
                ),
                "precautions": "Avoid hard dropping entire subnets if mixed benign/malicious traffic shares the ingress port. Human-in-the-loop approval required.",
                "guidance": "If supported by retrieved controls: deploy per-source rate-limiting and TCP SYN cookies at the border. Do not treat this as confirmed for this flow until an analyst reviews the evidence.",
                "supporting_sources": _sources(nist_source, cisa_source, mitre_source)
            })
            recs.append({
                "action_id": "REC-DDOS-02",
                "priority": "P2 - HIGH",
                "action_type": "VLAN_ISOLATION",
                "title": "Quarantine Affected IoT Gateway Segment via Micro-Segmentation",
                "rationale": "Advisory isolation guidance from retrieved DDoS/availability context. Not an automated action.",
                "precautions": "Ensure quarantine VLAN maintains telemetry logging for post-incident forensic replay.",
                "guidance": "If retrieved playbooks support it: migrate the affected gateway port to a telemetry quarantine VLAN with human oversight.",
                "supporting_sources": _sources(cisa_source, mitre_source, nist_source)
            })

        elif attack_type == "DoS":
            recs.append({
                "action_id": "REC-DOS-01",
                "priority": "P1 - HIGH",
                "action_type": "CONNECTION_THROTTLING",
                "title": "Enforce Connection Timeouts and Socket Quotas",
                "rationale": (
                    "Advisory only. Retrieved DoS-relevant guidance (resource/service exhaustion) "
                    f"supports connection protection: {grounded_note}"
                ),
                "precautions": "Do not set keep-alive timeouts lower than standard sensor polling intervals without analyst review.",
                "guidance": "If supported by retrieved controls: tighten idle-connection timeouts and restrict concurrent half-open connections per client.",
                "supporting_sources": _sources(nist_source, mitre_source, cisa_source)
            })
            recs.append({
                "action_id": "REC-DOS-02",
                "priority": "P2 - MEDIUM",
                "action_type": "STATEFUL_FIREWALL_FILTER",
                "title": "Filter Malformed and Unexpected Flag Bursts",
                "rationale": "Advisory filtering consistent with retrieved endpoint/application exhaustion context. TreeSHAP values are model facts, not proof of a specific packet-crafting technique.",
                "precautions": "Verify application-layer IoT protocols (MQTT/CoAP) before enforcing header drops.",
                "guidance": "If retrieved controls support it: configure stateful inspection to drop unexpected flag bursts lacking handshake context.",
                "supporting_sources": _sources(nist_source, mitre_source)
            })

        elif attack_type == "Spoofing":
            recs.append({
                "action_id": "REC-SPOOF-01",
                "priority": "P1 - CRITICAL",
                "action_type": "DYNAMIC_ARP_INSPECTION",
                "title": "Enable Dynamic ARP Inspection (DAI) & DHCP Snooping",
                "rationale": (
                    "Advisory only. Retrieved spoofing/AiTM guidance supports ARP/DHCP controls: "
                    f"{grounded_note}"
                ),
                "precautions": "Ensure static IP IoT devices have valid MAC-IP bindings entered in the DHCP snooping database.",
                "guidance": "If retrieved NIST/CISA excerpts support it: enable DAI and drop unauthorized gratuitous ARPs. Do not infer that this flow already performed ARP poisoning.",
                "supporting_sources": _sources(nist_source, mitre_source, cisa_source)
            })
            recs.append({
                "action_id": "REC-SPOOF-02",
                "priority": "P2 - HIGH",
                "action_type": "PORT_SECURITY_8021X",
                "title": "Enforce 802.1X Port Authentication & MAC Pinning",
                "rationale": "Advisory network-access control consistent with retrieved spoofing/boundary-protection evidence.",
                "precautions": "Confirm critical sensor endpoints support 802.1X or MAC Authentication Bypass (MAB).",
                "guidance": "If retrieved controls support it: bind switch-port access to authenticated MACs; isolate unverified MACs.",
                "supporting_sources": _sources(nist_source, mitre_source)
            })

        else:
            recs.append({
                "action_id": "REC-GEN-01",
                "priority": "P2 - MEDIUM",
                "action_type": "MONITORING_AND_LOGGING",
                "title": "Increase Network Telemetry Sampling Rate",
                "rationale": NOT_ESTABLISHED,
                "precautions": "Monitor storage capacity on central syslog server.",
                "guidance": NOT_ESTABLISHED,
                "supporting_sources": _sources(nist_source)
            })

        return recs


# =====================================================================
# LLM SYNTHESIZER
# =====================================================================
class LLMSynthesizer:
    """
    Synthesizes the comprehensive, human-readable security incident dossier.
    Strictly grounded in observed model evidence + retrieved knowledge.
    Constructs the 10-part standardized Security Incident Dossier.

    Grounding policy:
      - FACT:        XGBoost prediction, calibrated confidence, TreeSHAP values
      - INFERENCE:   Interpretations consistent with retrieved evidence
      - RECOMMENDATION: Advisory mitigations (human-in-the-loop required)
    """
    def __init__(self):
        pass

    def synthesize(self,
                   attack_type: str,
                   confidence: float,
                   detection_data: Dict[str, Any],
                   explainability_data: Dict[str, Any],
                   threat_intel_data: Dict[str, Any],
                   threat_assessment_data: Dict[str, Any],
                   recommendations: List[Dict[str, Any]]) -> str:

        # Section 1: Detection Summary [FACT]
        prob_summary = detection_data.get("probability_summary", f"{attack_type}: {confidence*100:.2f}%")
        risk_formula = threat_assessment_data.get(
            "risk_score_formula",
            f"Heuristic: base_severity x confidence={confidence:.4f}"
        )
        if attack_type == "UNKNOWN_OR_REVIEW":
            sec1 = (
                f"1. DETECTION SUMMARY [FACT / ABSTAINED]\n"
                f"• Operational State    : UNKNOWN_OR_REVIEW (Abstained from Automated Action)\n"
                f"• Advisory Model Class : {detection_data.get('detected_attack', 'Unknown')} (UNVERIFIED — NOT TRUSTED)\n"
                f"• Calibrated Confidence: {confidence*100:.4f}% ({detection_data.get('confidence_level', 'High')})\n"
                f"• Threat Severity      : {threat_assessment_data.get('severity', 'HIGH')} "
                f"(Heuristic Risk Score: {threat_assessment_data.get('composite_risk_score', 8.0)}/10.0)\n"
                f"• Automated Mitigation : PROHIBITED (Enforces Mandatory Human Analyst Triage)\n"
                f"• Class Distribution   : [{prob_summary}]"
            )
        else:
            sec1 = (
                f"1. DETECTION SUMMARY [FACT]\n"
                f"• Detected Attack Type : {attack_type}\n"
                f"• Calibrated Confidence: {confidence*100:.4f}% ({detection_data.get('confidence_level', 'High')})\n"
                f"• Threat Severity      : {threat_assessment_data.get('severity', 'HIGH')} "
                f"(Heuristic Risk Score: {threat_assessment_data.get('composite_risk_score', 8.0)}/10.0)\n"
                f"• Risk Score Formula   : {risk_formula}\n"
                f"• Class Distribution   : [{prob_summary}]"
            )

        # Section 2: Attack Description [RETRIEVED EVIDENCE]
        # Grounded in mitre_primary + attack_specific tier docs only
        attack_descs = threat_intel_data.get("attack_descriptions", [])
        if attack_descs:
            desc_text = "\n".join([f"• {d}" for d in attack_descs])
        else:
            desc_text = (
                f"• {attack_type} attack class detected. "
                f"Attack description not established from retrieved evidence."
            )
        sec2 = (
            f"2. ATTACK DESCRIPTION [RETRIEVED EVIDENCE — MITRE ATT&CK / NIST / CISA]\n"
            f"{desc_text}"
        )

        # Section 3: How the Attack Works [RETRIEVED EVIDENCE]
        mechanisms = threat_intel_data.get("mechanisms", [])
        if mechanisms:
            mech_text = "\n".join([f"• {m}" for m in mechanisms])
        else:
            mech_text = "• Attack mechanism not established from retrieved evidence."
        sec3 = (
            f"3. HOW THE ATTACK WORKS [RETRIEVED EVIDENCE — MITRE ATT&CK]\n"
            f"{mech_text}"
        )

        # Section 4: IoT/OT Impact [INFERENCE — attack-specific evidence only]
        # iot_impacts contains ONLY mitre_primary + attack_specific tier content.
        # General-context CISA/NIST docs are NOT propagated here.
        iot_impacts = threat_intel_data.get("iot_impacts", [])
        if iot_impacts:
            impact_text = "\n".join([f"• {imp}" for imp in iot_impacts])
        else:
            # Use ThreatAssessmentAgent's cautious inference fallback
            impact_text = f"• {threat_assessment_data.get('potential_impact', 'Impact not established from retrieved evidence.')}"

        # Also note any general security context (kept separate, not as primary evidence)
        general_ctx = threat_intel_data.get("general_security_context", [])
        gen_note = ""
        if general_ctx:
            gen_note = (
                f"\n\n  [General security context — lower relevance to {attack_type}, "
                f"kept separate from attack-specific evidence:]\n"
                + "\n".join([f"  • {g}" for g in general_ctx[:2]])
            )

        sec4 = (
            f"4. IoT/OT IMPACT [INFERENCE — based on attack-specific retrieved evidence]\n"
            f"[NOTE: The following describes possible impacts based on retrieved {attack_type}-specific "
            f"threat intelligence. These are inferences consistent with the detected class, "
            f"not confirmed facts about this specific network flow.]\n"
            f"{impact_text}"
            f"{gen_note}"
        )

        # Section 5: Sample-Specific Model Evidence [FACT — TreeSHAP]
        top_features = explainability_data.get("top_features", [])[:5]
        shap_lines = [
            f"• Feature '{f['feature']}': Observed Value = {f['feature_value']:.2f} "
            f"| TreeSHAP Contribution = {f['shap_value']:+.4f} ({f['direction']})"
            for f in top_features
        ]
        sec5 = (
            f"5. SAMPLE-SPECIFIC MODEL EVIDENCE [FACT — TreeSHAP Mathematical Attributions]\n"
            f"[NOTE: These values explain why the 46-feature XGBoost model classified THIS "
            f"specific sample as {attack_type}. They are observed telemetry parameters and "
            f"mathematical SHAP contributions — not external threat intelligence.]\n"
            + "\n".join(shap_lines)
        )

        # Section 6: Threat Intelligence Summary [INFERENCE]
        threat_profile = threat_intel_data.get("threat_actor_profile", "IoT Network Anomaly")
        sec6 = (
            f"6. THREAT INTELLIGENCE SUMMARY [INFERENCE — grounded in retrieved evidence]\n"
            f"• Identified Threat Profile : {threat_profile}\n"
            f"• Authoritative Alignment   : Grounded against MITRE ATT&CK, NIST SP 800-82, "
            f"NIST SP 800-53, NIST IR 8259A, and CISA advisories.\n"
            f"• Epistemic Note            : Threat profile is an inference consistent with "
            f"retrieved evidence and the detected attack class — not a confirmed attribution."
        )

        # Section 7: MITRE ATT&CK Context [RETRIEVED EVIDENCE]
        mitre_techs = threat_intel_data.get("mitre_techniques", [])
        if mitre_techs:
            mitre_lines = []
            for t in mitre_techs:
                tier_label = t.get("evidence_tier", "retrieved")
                mitre_lines.append(
                    f"• Technique ID : {t['technique_id']} ({t['technique_name']}) "
                    f"[{tier_label}]\n"
                    f"  - Tactic     : {t['tactic']}\n"
                    f"  - URL        : {t['url']}\n"
                    f"  - Description: {t['description'][:280]}..."
                )
            sec7 = "7. MITRE ATT&CK CONTEXT [RETRIEVED EVIDENCE]\n" + "\n".join(mitre_lines)
        else:
            sec7 = "7. MITRE ATT&CK CONTEXT\n• Technique: Not established from retrieved evidence."

        # Section 8: Indicators / Relevant Characteristics [RETRIEVED EVIDENCE + FACT]
        indicators = threat_intel_data.get("indicators", [])
        if indicators:
            ind_text = "\n".join([f"• {ind}" for ind in indicators])
        else:
            ind_text = (
                "• Telemetry rate deviations, abnormal packet header parameters, and "
                "irregular inter-arrival times consistent with anomalous network traffic "
                "(from SHAP feature analysis — FACT)."
            )
        sec8 = f"8. INDICATORS / RELEVANT CHARACTERISTICS [RETRIEVED EVIDENCE + FACT]\n{ind_text}"

        # Section 9: Mitigation / Response Considerations [RECOMMENDATION]
        recs_lines = []
        for r in recommendations:
            srcs = "; ".join(r.get("supporting_sources", ["Authoritative Guidelines"]))
            recs_lines.append(
                f"• [{r['priority']}] {r['title']}\n"
                f"  - Action Type: {r['action_type']}\n"
                f"  - Guidance   : {r['guidance']}\n"
                f"  - Precautions: {r['precautions']}\n"
                f"  - Grounded in: {srcs}"
            )
        sec9 = (
            f"9. MITIGATION / RESPONSE CONSIDERATIONS [RECOMMENDATION — Advisory Only]\n"
            f"[NOTE: All defensive measures are advisory and require human-in-the-loop "
            f"approval before enforcement. No automated network mutations are triggered.]\n"
            + "\n".join(recs_lines)
        )

        # Section 10: Source Provenance [complete — includes all tiers and excluded docs]
        all_scored = threat_intel_data.get("evidence_bundle", {}).get("all_scored_docs", [])
        sources = all_scored if all_scored else threat_intel_data.get("retrieved_sources", [])
        src_lines = []
        for s in sources:
            tier_label = s.get("evidence_tier", "retrieved")
            acs = s.get("attack_context_score", None)
            acs_str = f" | Attack-Ctx Score: {acs:.4f}" if acs is not None else ""
            src_lines.append(
                f"• [{s.get('source', 'Unknown')}] {s.get('title', 'Document')} "
                f"[{tier_label}]\n"
                f"  - Chunk ID        : {s.get('chunk_id', 'N/A')}"
                f" | Retrieval Score: {s.get('relevance_score', 0.0):.4f}"
                f"{acs_str}\n"
                f"  - Official URL    : {s.get('url', 'N/A')}"
            )
        sec10 = "10. SOURCE PROVENANCE [All retrieved docs — tiered]\n" + "\n".join(src_lines)

        dossier = "\n\n" + "=" * 80 + "\n"
        dossier += "                   AEGIS-IoT SECURITY INCIDENT DOSSIER\n"
        dossier += "=" * 80 + "\n\n"
        dossier += "\n\n".join([sec1, sec2, sec3, sec4, sec5, sec6, sec7, sec8, sec9, sec10])
        dossier += "\n\n" + "=" * 80

        return dossier


# =====================================================================
# MULTI-AGENT ORCHESTRATOR
# =====================================================================
class MultiAgentOrchestrator:
    """
    Central Orchestrator managing structured data flow through the 5 agents
    and grounded LLM synthesizer.
    """
    def __init__(self, retriever: Optional[AdaptiveHybridRAG] = None):
        self.retriever = retriever or AdaptiveHybridRAG()
        self.detection_agent = DetectionAnalysisAgent()
        self.explainability_agent = ExplainabilityAgent()
        self.threat_intel_agent = ThreatIntelligenceAgent(self.retriever)
        self.threat_assessment_agent = ThreatAssessmentAgent()
        self.response_agent = ResponseRecommendationAgent()
        self.llm_synthesizer = LLMSynthesizer()

    def process_ids_alert(self, ids_payload: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
        """
        Executes the complete Multi-Agent workflow:
        1. Detection Analysis Agent
        2. Explainability Agent
        3. Threat Intelligence / Adaptive RAG Agent (Consumes Full Content)
        4. Threat Assessment Agent
        5. Response Recommendation Agent
        6. LLM Synthesizer (Produces 10-part Grounded Dossier)
        """
        t_start = time.perf_counter()

        attack_type = ids_payload.get("attack_type", "Unknown")
        confidence = float(ids_payload.get("calibrated_confidence", ids_payload.get("confidence", 0.99)))
        probabilities = ids_payload.get("probabilities", ids_payload.get("class_probabilities", {}))
        features = ids_payload.get("features", ids_payload.get("46_features", {}))
        shap_contributions = ids_payload.get("shap_evidence", ids_payload.get("contributions", []))

        # 1. Detection Agent
        t0 = time.perf_counter()
        detection_out = self.detection_agent.analyze(
            attack_type=attack_type,
            confidence=confidence,
            probabilities=probabilities,
            features=features
        )
        t_detection = round((time.perf_counter() - t0) * 1000, 3)

        # 2. Explainability Agent
        t0 = time.perf_counter()
        explainability_out = self.explainability_agent.explain(
            predicted_attack=attack_type,
            contributions=shap_contributions
        )
        t_explainability = round((time.perf_counter() - t0) * 1000, 3)

        # 3. Threat Intelligence Agent (RAG - consumes EvidenceBundle)
        t0 = time.perf_counter()
        threat_intel_out = self.threat_intel_agent.retrieve_intel(
            detected_attack=attack_type,
            shap_evidence=explainability_out.get("top_features", [])
        )
        t_rag = round((time.perf_counter() - t0) * 1000, 3)

        # Print RAG grounding trace immediately after retrieval if verbose
        grounding_trace = threat_intel_out.get("rag_grounding_trace", "")
        if verbose and grounding_trace:
            print(grounding_trace)

        # 4. Threat Assessment Agent
        t0 = time.perf_counter()
        threat_assessment_out = self.threat_assessment_agent.assess(
            attack_type=attack_type,
            confidence=confidence,
            detection_data=detection_out,
            explainability_data=explainability_out,
            threat_intel_data=threat_intel_out
        )
        t_assessment = round((time.perf_counter() - t0) * 1000, 3)

        # 5. Response Recommendation Agent
        t0 = time.perf_counter()
        recommendations_out = self.response_agent.formulate_recommendations(
            attack_type=attack_type,
            confidence=confidence,
            threat_assessment=threat_assessment_out,
            threat_intel=threat_intel_out
        )
        t_response = round((time.perf_counter() - t0) * 1000, 3)

        # 6. LLM Synthesizer (produces 10-part grounded dossier)
        t0 = time.perf_counter()
        final_explanation = self.llm_synthesizer.synthesize(
            attack_type=attack_type,
            confidence=confidence,
            detection_data=detection_out,
            explainability_data=explainability_out,
            threat_intel_data=threat_intel_out,
            threat_assessment_data=threat_assessment_out,
            recommendations=recommendations_out
        )
        t_synthesis = round((time.perf_counter() - t0) * 1000, 3)
        t_total = round((time.perf_counter() - t_start) * 1000, 3)

        # Build final structured output according to Part 17 Schema + Phase 14/15 Safety Invariants
        is_abstained = (
            ids_payload.get("is_abstained", False) or
            ids_payload.get("decision_state") == "UNKNOWN_OR_REVIEW" or
            attack_type == "UNKNOWN_OR_REVIEW"
        )
        decision_state = ids_payload.get("decision_state", "UNKNOWN_OR_REVIEW" if is_abstained else f"KNOWN_{attack_type}")
        advisory_class = ids_payload.get("advisory_class", ids_payload.get("predicted_class", attack_type))
        trusted_class = None if is_abstained else ids_payload.get("trusted_class", attack_type)
        automated_blocking_allowed = False if is_abstained else ids_payload.get("automated_blocking_allowed", True)
        abstention_reasons = ids_payload.get("abstention_reasons", [])

        return {
            "decision_state": decision_state,
            "is_abstained": is_abstained,
            "trusted_class": trusted_class,
            "advisory_class": advisory_class,
            "abstention_reasons": abstention_reasons,
            "automated_blocking_allowed": automated_blocking_allowed,
            "attack_type": attack_type,
            "confidence": confidence,
            "detection": {
                "summary": detection_out["detection_summary"],
                "probabilities": detection_out["probabilities"]
            },
            "explainability": {
                "top_features": explainability_out["top_features"],
                "supporting_evidence": explainability_out["supporting_evidence"],
                "opposing_evidence": explainability_out["opposing_evidence"]
            },
            "threat_intelligence": {
                "retrieved_sources": threat_intel_out["retrieved_sources"],
                "evidence_bundle": threat_intel_out.get("evidence_bundle", {}),
                "rag_grounding_trace": threat_intel_out.get("rag_grounding_trace", ""),
                "attack_descriptions": threat_intel_out.get("attack_descriptions", []),
                "mechanisms": threat_intel_out.get("mechanisms", []),
                "iot_impacts": threat_intel_out.get("iot_impacts", []),
                "general_security_context": threat_intel_out.get("general_security_context", []),
                "indicators": threat_intel_out.get("indicators", []),
                "mitre_techniques": threat_intel_out.get("mitre_techniques", []),
                "mitigation_guidance": threat_intel_out.get("mitigation_guidance", []),
                "excluded_evidence": threat_intel_out.get("excluded_evidence", []),
                "threat_actor_profile": threat_intel_out.get("threat_actor_profile", "")
            },
            "threat_assessment": {
                "severity": threat_assessment_out["severity"],
                "summary": threat_assessment_out["summary"],
                "potential_impact": threat_assessment_out["potential_impact"],
                "uncertainty": threat_assessment_out["uncertainty"],
                "epistemic_breakdown": threat_assessment_out["epistemic_breakdown"]
            },
            "recommendations": recommendations_out,
            "final_explanation": final_explanation,
            "_agent_telemetry": {
                "detection_ms": t_detection,
                "explainability_ms": t_explainability,
                "rag_ms": t_rag,
                "threat_assessment_ms": t_assessment,
                "recommendation_ms": t_response,
                "synthesis_ms": t_synthesis,
                "total_pipeline_ms": t_total
            }
        }

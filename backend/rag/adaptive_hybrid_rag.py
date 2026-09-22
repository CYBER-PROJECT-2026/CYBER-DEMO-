"""
backend/rag/adaptive_hybrid_rag.py
==================================
Adaptive Hybrid Retrieval-Augmented Generation (RAG) Engine
for IoT Intrusion Detection System.

Features:
1. Dynamic, context-aware query formulation based on:
   - Detected attack type (DDoS, DoS, Spoofing)
   - Top positive/negative SHAP feature contributions
   - IoT network domain context
2. Hybrid Multi-Stage Retrieval:
   - BM25 sparse lexical ranking
   - TF-IDF sublinear dense vector cosine similarity
   - Reciprocal Rank Fusion (RRF)
   - Source diversity balancing (MITRE, NIST, CISA)
3. Attack-Aware Relevance Filtering (post-retrieval):
   - Deterministic attack-context scoring per document
   - Uses chunk metadata (tactics, technique_id, keywords, content)
   - Classifies docs: mitre_primary / attack_specific / general_context / excluded
   - Returns structured EvidenceBundle consumed by downstream agents
4. Full provenance & traceability for all retrieved intelligence.

SCORING DISCLAIMER:
   attack_context_score weights are heuristic and engineering-chosen.
   They have NOT been empirically validated against a held-out relevance dataset.
"""

import os
import sys
import re
import math
import joblib
import json
import numpy as np
from typing import List, Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VECTOR_DB_DIR = os.path.join(BASE_DIR, "knowledge_base", "vector_db")


# =========================================================================
# ATTACK-CONTEXT KEYWORD MAPS
# Grounded in the actual MITRE technique IDs and content in the 906-chunk KB.
# =========================================================================

# Primary MITRE technique IDs for each attack class.
# Sourced from chunks_lookup.json metadata — DO NOT CHANGE without updating KB.
MITRE_PRIMARY_TECHNIQUES = {
    "ddos":     {"T1498", "T1498.001", "T1498.002"},
    "dos":      {"T1499", "T1499.001", "T1499.002", "T1499.003", "T1499.004"},
    "spoofing": {"T1557", "T1557.001", "T1557.002", "T1557.003", "T1557.004"},
}

# Secondary MITRE techniques (partially relevant, not primary)
MITRE_SECONDARY_TECHNIQUES = {
    "ddos":     {"T1499"},                          # DoS/DDoS share some overlap per MITRE
    "dos":      {"T1498"},                          # Network DoS text references endpoint DoS
    "spoofing": {"T1040", "T1020", "T1565.002"},    # Network sniffing, traffic dup, data manip
}

# NIST chunk IDs that are attack-specific (sourced from actual chunk metadata inspection)
NIST_ATTACK_SPECIFIC = {
    "ddos": {
        "NIST-SP800-53-SC-5",   # DoS Protection: rate-limiting, SYN cookies
        "NIST-SP800-82-SEC4",   # Detection Signatures: Rate, Srate, SYN anomalies
    },
    "dos": {
        "NIST-SP800-53-SC-5",   # DoS Protection
        "NIST-SP800-82-SEC4",   # Detection Signatures: Rate, TCP flags
        "NIST-SP800-53-SI-4",   # Anomaly monitoring
    },
    "spoofing": {
        "NIST-SP800-53-SC-7",   # Boundary Protection, DAI, DHCP snooping
        "NIST-IR8259A-CAP1",    # Device Identity / MAC spoofing prevention
    },
}

# CISA chunk relevance tiers per attack type.
# Based on the actual 5 CISA chunks in the 906-chunk KB (not a rebuilt index).
# Tiers: 'primary' | 'partial' | 'general' | 'mismatch'
#
# AA22-110A is a mixed advisory whose MAIN subject is IoT botnets / SYN-UDP floods.
# ARP/IP spoofing and ACK exhaustion appear as secondary clauses in the same
# threat-profile sentence — those clauses must not dominate DoS or Spoofing dossiers.
# AA20-133A is a mixed advisory: DDoS-swarm enrollment AND a separate ARP-spoofing
# sentence. Passage extraction may keep the matching sentence only.
CISA_ATTACK_RELEVANCE = {
    "CISA-AA22-110A": {
        "ddos":     "primary",
        "dos":      "mismatch",
        "spoofing": "mismatch",
    },
    "CISA-AA20-133A": {
        "ddos":     "partial",
        "dos":      "mismatch",
        "spoofing": "partial",  # has a distinct ARP spoofing / DNS poisoning sentence
    },
    "CISA-CPG-1": {
        "ddos": "general", "dos": "general", "spoofing": "general",
    },
    "CISA-CPG-2": {
        "ddos": "general", "dos": "general", "spoofing": "general",
    },
    "CISA-CPG-3": {
        # Mentions all three classes equally as playbook categories — not attack-specific.
        "ddos": "general", "dos": "general", "spoofing": "general",
    },
}

# Attack-specific keyword sets for content-based scoring.
# Grounded in actual MITRE ATT&CK descriptions present in the 906-chunk KB.
ATTACK_KEYWORDS = {
    "ddos": [
        "network denial of service", "network dos", "volumetric", "bandwidth exhaustion",
        "distributed denial", "direct network flood", "reflection amplification",
        "syn flood", "udp flood", "icmp flood", "packet flood", "botnet",
        "t1498", "t1498.001", "t1498.002",
        "rate-limiting", "scrubbing", "syn cookie", "ingress filtering",
    ],
    "dos": [
        "endpoint denial of service", "endpoint dos", "application exhaustion",
        "service exhaustion", "os exhaustion", "resource starvation",
        "connection exhaustion", "socket descriptor", "cpu saturation",
        "keep-alive", "half-open connection", "slow-rate",
        "t1499", "t1499.001", "t1499.002", "t1499.003", "t1499.004",
        "application layer", "web server", "application exhaustion flood",
    ],
    "spoofing": [
        "arp cache poisoning", "arp spoofing", "adversary-in-the-middle",
        "aitm", "man-in-the-middle", "mitm", "address resolution",
        "gratuitous arp", "dhcp spoofing", "name resolution poisoning",
        "mac address", "mac cloning", "ip forgery", "ip header",
        "t1557", "t1557.001", "t1557.002", "t1557.003", "t1557.004",
        "dynamic arp inspection", "dai", "dhcp snooping", "802.1x",
        "traffic interception", "rogue gateway", "link layer",
    ],
}

# Keywords that strongly indicate a document primarily discusses a DIFFERENT attack type.
# Used to detect cross-attack contamination in mixed-content docs like CISA-AA22-110A.
ATTACK_NEGATIVE_KEYWORDS = {
    "ddos":     ["arp cache", "arp poisoning", "gratuitous arp", "adversary-in-the-middle"],
    "dos":      ["direct network flood", "reflection amplification", "volumetric flood",
                 "syn flood", "udp flood", "botnet swarm"],
    "spoofing": ["volumetric flood", "syn flood", "udp flood", "botnet swarm",
                 "direct network flood", "distributed denial-of-service swarm"],
}

NOT_ESTABLISHED = "Not established from retrieved evidence."


def _split_sentences(text: str) -> List[str]:
    """Split chunk content into sentences for passage-level grounding."""
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _keyword_hit_count(text_lower: str, keywords: List[str]) -> int:
    return sum(1 for kw in keywords if kw in text_lower)


def extract_attack_relevant_passages(content: str, attack_type: str) -> Dict[str, Any]:
    """
    Passage-level filter for mixed CISA/NIST chunks.

    A sentence is kept when it matches the detected class at least as strongly
    as any other class. Mixed one-sentence profiles (e.g. AA22-110A listing
    floods AND ARP spoofing) are kept only when the detected class is the
    dominant topic of that sentence.

    Returns excerpts only — never invents content not present in the chunk.
    """
    atk = attack_type.lower()
    pos_kws = ATTACK_KEYWORDS.get(atk, [])
    sentences = _split_sentences(content)
    kept = []
    dropped = []
    pos_total = 0
    other_totals = {k: 0 for k in ATTACK_KEYWORDS if k != atk}

    for sent in sentences:
        sl = sent.lower()
        pos_hits = _keyword_hit_count(sl, pos_kws)
        other_hits = {
            k: _keyword_hit_count(sl, ATTACK_KEYWORDS[k])
            for k in ATTACK_KEYWORDS if k != atk
        }
        max_other = max(other_hits.values()) if other_hits else 0
        pos_total += pos_hits
        for k, v in other_hits.items():
            other_totals[k] += v

        if pos_hits > 0 and pos_hits >= max_other:
            kept.append(sent)
        elif pos_hits > 0 and pos_hits < max_other:
            dropped.append(sent)
        elif max_other > 0 and pos_hits == 0:
            dropped.append(sent)

    dominant_other = max(other_totals, key=other_totals.get) if other_totals else None
    dominant_other_hits = other_totals.get(dominant_other, 0) if dominant_other else 0
    mismatch = dominant_other_hits > pos_total

    excerpt = " ".join(kept).strip()
    return {
        "excerpt": excerpt,
        "kept_sentences": kept,
        "dropped_cross_attack_sentences": dropped,
        "class_keyword_hits": pos_total,
        "other_class_hits": other_totals,
        "dominant_mismatch": mismatch,
    }


class AdaptiveHybridRAG:
    def __init__(self, vector_db_dir: str = VECTOR_DB_DIR):
        self.vector_db_dir = vector_db_dir
        self.vectorizer_path = os.path.join(vector_db_dir, "tfidf_vectorizer.pkl")
        self.matrix_path = os.path.join(vector_db_dir, "tfidf_matrix.pkl")
        self.bm25_path = os.path.join(vector_db_dir, "bm25_store.pkl")
        self.lookup_path = os.path.join(vector_db_dir, "chunks_lookup.json")

        self.loaded = False
        self._load_indices()

    def _load_indices(self):
        if not (os.path.exists(self.vectorizer_path) and
                os.path.exists(self.matrix_path) and
                os.path.exists(self.bm25_path) and
                os.path.exists(self.lookup_path)):
            # Warning: indices not yet generated
            self.chunks = []
            self.loaded = False
            return

        self.vectorizer = joblib.load(self.vectorizer_path)
        self.tfidf_matrix = joblib.load(self.matrix_path)
        self.bm25 = joblib.load(self.bm25_path)

        with open(self.lookup_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        self.loaded = True

    def build_adaptive_query(self,
                             attack_type: str,
                             shap_evidence: Optional[List[Dict[str, Any]]] = None,
                             iot_context: Optional[str] = None) -> str:
        """
        Dynamically construct retrieval query based on detected attack,
        top positive SHAP drivers, and IoT context.
        """
        query_terms = [attack_type]

        # Extract top positive SHAP feature names
        if shap_evidence:
            pos_features = [
                item.get("feature", "")
                for item in shap_evidence
                if item.get("shap_value", 0) > 0 or item.get("direction") == "POSITIVE"
            ][:4]
            if pos_features:
                query_terms.extend(pos_features)

        # Domain terms based on attack type
        if attack_type.lower() == "ddos":
            query_terms.extend(["volumetric", "flood", "denial of service", "rate limiting", "SYN cookies", "T1498"])
        elif attack_type.lower() == "dos":
            query_terms.extend(["endpoint", "exhaustion", "resource starvation", "connection keep-alive", "T1499"])
        elif attack_type.lower() == "spoofing":
            query_terms.extend(["adversary in the middle", "ARP", "IP forgery", "Dynamic ARP Inspection", "T1557"])
        else:
            query_terms.extend(["intrusion", "anomaly", "mitigation", "NIST", "CISA"])

        if iot_context:
            query_terms.append(iot_context)
        else:
            query_terms.extend(["IoT", "network security", "Purdue segmentation"])

        return " ".join(query_terms)

    def retrieve(self,
                 query: str,
                 top_k: int = 6,
                 required_sources: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Performs hybrid retrieval using BM25 and TF-IDF Cosine Similarity with RRF.
        Returns top_k=6 documents (expanded pool) with chunk metadata exposed
        for downstream attack-context scoring.
        """
        if not self.loaded or not self.chunks:
            return []

        # Tokenize query for BM25
        query_tokens = re.findall(r'\b[a-zA-Z0-9_\-]+\b', query.lower())
        if not query_tokens:
            return []

        # 1. BM25 Sparse Ranking
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_ranked = np.argsort(bm25_scores)[::-1]

        # 2. Dense TF-IDF Cosine Similarity
        query_vec = self.vectorizer.transform([query])
        dense_scores = (self.tfidf_matrix * query_vec.T).toarray().flatten()
        dense_ranked = np.argsort(dense_scores)[::-1]

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_k = 60
        rrf_scores = {}
        for rank, idx in enumerate(bm25_ranked[:100]):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank + 1))

        for rank, idx in enumerate(dense_ranked[:100]):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank + 1))

        # Sort by RRF score
        sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)

        # 4. Source Diversity Balance: Ensure MITRE, NIST, CISA are represented if relevant
        selected_indices = []
        seen_sources = set()

        for idx in sorted_indices:
            chunk = self.chunks[idx]
            src = chunk.get("source", "UNKNOWN")

            # Prioritize first match from each distinct authoritative source
            if src not in seen_sources and len(selected_indices) < top_k:
                selected_indices.append(idx)
                seen_sources.add(src)

        # Fill remaining slots up to top_k with top ranked
        for idx in sorted_indices:
            if idx not in selected_indices:
                selected_indices.append(idx)
            if len(selected_indices) >= top_k:
                break

        # Format output — expose chunk metadata fields for attack-context scoring
        results = []
        for idx in selected_indices[:top_k]:
            chunk = self.chunks[idx]
            # Normalize relevance score between 0.70 and 0.99 for clean reporting
            raw_dense = float(dense_scores[idx]) if idx < len(dense_scores) else 0.5
            norm_score = round(min(max(0.70 + raw_dense * 0.28, 0.70), 0.99), 4)

            results.append({
                "source": chunk.get("source", "Authoritative Repository"),
                "title": chunk.get("title", chunk.get("chunk_id", "Doc")),
                "url": chunk.get("url", ""),
                "chunk_id": chunk.get("chunk_id", f"CHK-{idx}"),
                "relevance_score": norm_score,  # legacy display score (normalized TF-IDF cosine)
                "retrieval_score": norm_score,
                "rrf_score": round(float(rrf_scores.get(idx, 0.0)), 6),
                "bm25_score": round(float(bm25_scores[idx]) if idx < len(bm25_scores) else 0.0, 6),
                "tfidf_cosine": round(raw_dense, 6),
                "category": chunk.get("category", "CYBERSECURITY_KNOWLEDGE"),
                "content": chunk.get("content", ""),
                "supporting_text": chunk.get("content", "")[:320] + "...",
                # Expose chunk metadata for attack-context scoring downstream
                "tactics": chunk.get("tactics", []),
                "technique_id": chunk.get("technique_id", ""),
                "keywords": chunk.get("keywords", ""),
            })

        return results

    # =========================================================================
    # ATTACK-AWARE RELEVANCE FILTERING (POST-RETRIEVAL)
    # =========================================================================

    def _compute_attack_context_score(self,
                                      doc: Dict[str, Any],
                                      attack_type: str) -> Dict[str, Any]:
        """
        Deterministic attack-context relevance scoring for a single retrieved document.

        SCORING FORMULA (heuristic — engineering-chosen, NOT empirically validated):

            attack_context_score = clamp(keyword_match + mitre_tech_match + source_adjustment, 0, 1)

        Components:
          keyword_match [0.0, 1.0]:
            Fraction of attack-specific keywords (from ATTACK_KEYWORDS[atk]) found in
            the concatenated content + title + keywords fields of the document.

          mitre_tech_match [0.0, 0.5]:
            +0.5  if chunk technique_id / chunk_id is a PRIMARY MITRE match (MITRE_PRIMARY_TECHNIQUES)
            +0.25 if SECONDARY match (MITRE_SECONDARY_TECHNIQUES)
            +0.0  otherwise

          source_adjustment [-0.25, +0.1]:
            For NIST chunks:  +0.1 if chunk_id is in NIST_ATTACK_SPECIFIC for this class
            For CISA chunks:  +0.1 if 'primary', 0.0 if 'partial', -0.05 if 'general'
                              Additional -0.2 if negative keywords of another attack type
                              dominate content (cross-attack contamination detection)

        Evidence tier thresholds:
          'mitre_primary':    MITRE source AND mitre_tech_match >= 0.5
          'attack_specific':  attack_context_score >= 0.35
          'general_context':  attack_context_score >= 0.10
          'excluded':         attack_context_score <  0.10
        """
        atk = attack_type.lower()
        chunk_id = doc.get("chunk_id", "")
        source = doc.get("source", "")
        technique_id = doc.get("technique_id", "")
        full_content = doc.get("content", "")
        retrieval_score = float(doc.get("retrieval_score", doc.get("relevance_score", 0.0)))

        passage = extract_attack_relevant_passages(full_content, attack_type)
        excerpt = passage["excerpt"]
        scoring_text = excerpt if excerpt else full_content
        content_lower = (
            scoring_text + " " +
            doc.get("title", "") + " " +
            doc.get("keywords", "")
        ).lower()
        full_lower = (
            full_content + " " + doc.get("title", "") + " " + doc.get("keywords", "")
        ).lower()

        # ---- Component 1: keyword_match (excerpt-first) ----
        kw_list = ATTACK_KEYWORDS.get(atk, [])
        hits = _keyword_hit_count(content_lower, kw_list)
        keyword_match = round(min(hits / max(len(kw_list), 1), 1.0), 4)

        # ---- Component 2: mitre_tech_match ----
        mitre_tech_match = 0.0
        primary_techs = MITRE_PRIMARY_TECHNIQUES.get(atk, set())
        secondary_techs = MITRE_SECONDARY_TECHNIQUES.get(atk, set())

        tid_to_check = technique_id if technique_id else chunk_id.replace("MITRE-", "")
        # Ignore non-ATT&CK IDs stored in CISA/NIST technique_id fields
        if not str(tid_to_check).upper().startswith("T"):
            tid_to_check = chunk_id.replace("MITRE-", "")
        if tid_to_check in primary_techs:
            mitre_tech_match = 0.5
        elif tid_to_check in secondary_techs:
            mitre_tech_match = 0.25

        # ---- Component 3: source_adjustment ----
        source_adjustment = 0.0
        cisa_tier = None

        if "NIST" in source:
            nist_specific = NIST_ATTACK_SPECIFIC.get(atk, set())
            if chunk_id in nist_specific:
                source_adjustment = 0.1

        elif "CISA" in source:
            cisa_rel = CISA_ATTACK_RELEVANCE.get(chunk_id, {})
            cisa_tier = cisa_rel.get(atk, "general")
            if cisa_tier == "primary":
                source_adjustment = 0.1
            elif cisa_tier == "partial":
                source_adjustment = 0.0
            elif cisa_tier == "general":
                source_adjustment = -0.05
            elif cisa_tier == "mismatch":
                source_adjustment = -0.25

            neg_kws = ATTACK_NEGATIVE_KEYWORDS.get(atk, [])
            neg_hits = _keyword_hit_count(full_lower, neg_kws)
            if neg_hits > 0 and keyword_match < 0.15:
                source_adjustment -= 0.2

        if passage["dominant_mismatch"] and "MITRE" not in source:
            source_adjustment -= 0.15

        raw_score = keyword_match + mitre_tech_match + source_adjustment
        attack_context_score = round(min(max(raw_score, 0.0), 1.0), 4)

        # Heuristic combination — NOT empirically validated.
        final_grounded_relevance = round(min(max(
            0.30 * retrieval_score
            + 0.45 * attack_context_score
            + 0.25 * mitre_tech_match,
            0.0), 1.0), 4)

        # ---- Evidence tier classification ----
        if "MITRE" in source and mitre_tech_match >= 0.5:
            tier = "mitre_primary"
        elif cisa_tier == "mismatch" and not excerpt:
            tier = "excluded"
        elif passage["dominant_mismatch"] and "MITRE" not in source and not excerpt:
            tier = "excluded"
        elif attack_context_score >= 0.35 and (excerpt or "NIST" in source or "MITRE" in source):
            tier = "attack_specific"
        elif attack_context_score >= 0.10:
            tier = "general_context"
        else:
            tier = "excluded"

        if "CISA" in source and tier == "attack_specific":
            if cisa_tier not in ("primary", "partial"):
                tier = "general_context"
            elif cisa_tier == "partial" and passage["dominant_mismatch"]:
                tier = "general_context"

        grounded_excerpt = excerpt if excerpt else (
            full_content if tier in ("mitre_primary", "attack_specific") else ""
        )

        return {
            "attack_context_score": attack_context_score,
            "keyword_match": keyword_match,
            "mitre_tech_match": mitre_tech_match,
            "source_adjustment": source_adjustment,
            "final_grounded_relevance": final_grounded_relevance,
            "evidence_tier": tier,
            "grounded_excerpt": grounded_excerpt,
            "dominant_mismatch": passage["dominant_mismatch"],
            "class_keyword_hits": passage["class_keyword_hits"],
            "other_class_hits": passage["other_class_hits"],
            "scoring_note": (
                "Heuristic: attack_context_score = keyword_match + mitre_tech_match "
                "+ source_adjustment. final_grounded_relevance = 0.30*retrieval_score "
                "+ 0.45*attack_context_score + 0.25*mitre_tech_match. "
                "Weights are engineering-chosen and have NOT been empirically validated."
            ),
        }

    def build_evidence_bundle(self,
                              attack_type: str,
                              retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Post-retrieval attack-aware relevance filtering.

        Classifies each retrieved document into one of four evidence tiers
        based on deterministic attack-context scoring. Returns a structured
        EvidenceBundle. Downstream agents consume this bundle instead of the
        raw flat document list.

        EvidenceBundle schema:
        {
            "detected_class":           str,
            "mitre_context":            [docs with tier='mitre_primary'],
            "attack_specific_evidence": [docs with tier='attack_specific'],
            "general_security_evidence":[docs with tier='general_context'],
            "mitigation_evidence":      [docs from attack_specific+general containing
                                         mitigation guidance keywords],
            "excluded_evidence":        [docs with tier='excluded' — kept for provenance,
                                         NOT passed to agent reasoning fields],
            "all_scored_docs":          [all docs with scoring metadata attached],
            "rag_grounding_trace":      str (compact console trace),
        }
        """
        mitre_context = []
        attack_specific = []
        general_context = []
        excluded = []
        mitigation_evidence = []
        all_scored = []

        for doc in retrieved_docs:
            score_info = self._compute_attack_context_score(doc, attack_type)
            scored_doc = dict(doc)
            scored_doc["attack_context_score"] = score_info["attack_context_score"]
            scored_doc["keyword_match"] = score_info["keyword_match"]
            scored_doc["mitre_tech_match"] = score_info["mitre_tech_match"]
            scored_doc["source_adjustment"] = score_info["source_adjustment"]
            scored_doc["final_grounded_relevance"] = score_info["final_grounded_relevance"]
            scored_doc["evidence_tier"] = score_info["evidence_tier"]
            scored_doc["grounded_excerpt"] = score_info["grounded_excerpt"]
            scored_doc["dominant_mismatch"] = score_info["dominant_mismatch"]
            scored_doc["class_keyword_hits"] = score_info["class_keyword_hits"]
            scored_doc["other_class_hits"] = score_info["other_class_hits"]
            scored_doc["scoring_note"] = score_info["scoring_note"]

            all_scored.append(scored_doc)
            tier = score_info["evidence_tier"]

            if tier == "mitre_primary":
                mitre_context.append(scored_doc)
            elif tier == "attack_specific":
                attack_specific.append(scored_doc)
            elif tier == "general_context":
                general_context.append(scored_doc)
            else:
                excluded.append(scored_doc)

            # Mitigation evidence from non-excluded docs; prefer excerpts
            text_for_mitig = (scored_doc.get("grounded_excerpt") or doc.get("content", "")).lower()
            mitigation_kws = [
                "mitigation", "control", "guidance", "protect", "prevent",
                "rate-limit", "firewall", "inspection", "segmentation",
                "playbook", "containment", "isolat", "rate-limiting",
                "syn cookie", "dhcp snooping", "dynamic arp",
            ]
            if tier in ("mitre_primary", "attack_specific", "general_context"):
                if any(kw in text_for_mitig for kw in mitigation_kws):
                    mitigation_evidence.append(scored_doc)

        # Sort each bucket by grounded relevance descending
        mitre_context.sort(key=lambda x: x["final_grounded_relevance"], reverse=True)
        attack_specific.sort(key=lambda x: x["final_grounded_relevance"], reverse=True)
        general_context.sort(key=lambda x: x["final_grounded_relevance"], reverse=True)
        mitigation_evidence.sort(key=lambda x: x["final_grounded_relevance"], reverse=True)

        provenance = []
        for d in all_scored:
            provenance.append({
                "chunk_id": d.get("chunk_id"),
                "source": d.get("source"),
                "title": d.get("title"),
                "url": d.get("url"),
                "evidence_tier": d.get("evidence_tier"),
                "retrieval_score": d.get("retrieval_score", d.get("relevance_score")),
                "attack_context_score": d.get("attack_context_score"),
                "mitre_tech_match": d.get("mitre_tech_match"),
                "final_grounded_relevance": d.get("final_grounded_relevance"),
            })

        trace = self._build_grounding_trace(
            attack_type, mitre_context, attack_specific, general_context, excluded
        )

        return {
            "detected_class": attack_type,
            "mitre_context": mitre_context,
            "attack_specific_evidence": attack_specific,
            "general_security_evidence": general_context,
            "mitigation_evidence": mitigation_evidence,
            "excluded_evidence": excluded,
            "excluded_irrelevant_evidence": excluded,
            "all_scored_docs": all_scored,
            "provenance": provenance,
            "rag_grounding_trace": trace,
        }

    def _build_grounding_trace(self,
                               attack_type: str,
                               mitre_context: List[Dict],
                               attack_specific: List[Dict],
                               general_context: List[Dict],
                               excluded: List[Dict]) -> str:
        """
        Compact, human-readable RAG grounding trace for console output.
        Prints document titles and scores only — no full content.
        """
        lines = [
            "",
            "RAG Grounding Trace",
            "-" * 44,
            f"  Detected class : {attack_type}",
            "",
            "  Retrieved & scored:",
        ]

        def _doc_line(doc, prefix="    "):
            cid   = doc.get("chunk_id", "?")
            title = doc.get("title", "?")[:58]
            acs   = doc.get("attack_context_score", 0.0)
            fgr   = doc.get("final_grounded_relevance", 0.0)
            tier  = doc.get("evidence_tier", "?")
            return (
                f"{prefix}[{tier.upper()}] {cid} | {title} "
                f"(acs={acs:.3f}, grounded={fgr:.3f})"
            )

        for doc in mitre_context:
            lines.append(_doc_line(doc))
        for doc in attack_specific:
            lines.append(_doc_line(doc))
        for doc in general_context:
            lines.append(_doc_line(doc))

        if excluded:
            lines.append("")
            lines.append("  Excluded (attack mismatch / low relevance):")
            for doc in excluded:
                cid   = doc.get("chunk_id", "?")
                title = doc.get("title", "?")[:52]
                acs   = doc.get("attack_context_score", 0.0)
                why = "attack mismatch" if doc.get("dominant_mismatch") else "low relevance"
                lines.append(
                    f"    [EXCLUDED] {cid} | {title} (acs={acs:.3f}, {why})"
                )

        lines.append("")
        lines.append("  Evidence bundle summary:")
        lines.append(f"    mitre_primary      : {len(mitre_context)}")
        lines.append(f"    attack_specific    : {len(attack_specific)}")
        lines.append(f"    general_context    : {len(general_context)}")
        lines.append(f"    excluded           : {len(excluded)}")
        lines.append("-" * 44)

        return "\n".join(lines)

    def get_grounded_context(self,
                              attack_type: str,
                              shap_evidence: Optional[List[Dict[str, Any]]] = None,
                              top_k: int = 6) -> Dict[str, Any]:
        """
        Full end-to-end adaptive retrieval pipeline.

        1. Build attack-aware query (unchanged).
        2. Retrieve expanded pool of top_k=6 docs via BM25 + TF-IDF + RRF (unchanged).
        3. Apply attack-aware relevance filtering to produce structured EvidenceBundle.

        Returns:
        {
            "query":             str,
            "retrieved_sources": [scored doc list — for backward compatibility],
            "evidence_bundle":   EvidenceBundle dict,
            "status":            str,
        }
        """
        adaptive_query = self.build_adaptive_query(attack_type, shap_evidence)
        # Retrieve expanded pool before filtering
        retrieved_docs = self.retrieve(adaptive_query, top_k=top_k)

        if not retrieved_docs:
            empty_bundle = {
                "detected_class": attack_type,
                "mitre_context": [],
                "attack_specific_evidence": [],
                "general_security_evidence": [],
                "mitigation_evidence": [],
                "excluded_evidence": [],
                "excluded_irrelevant_evidence": [],
                "all_scored_docs": [],
                "provenance": [],
                "rag_grounding_trace": f"No documents retrieved for {attack_type}.",
            }
            return {
                "query": adaptive_query,
                "retrieved_sources": [],
                "evidence_bundle": empty_bundle,
                "status": "Relevant cybersecurity knowledge could not be retrieved.",
            }

        # Build attack-aware evidence bundle (post-retrieval filtering)
        evidence_bundle = self.build_evidence_bundle(attack_type, retrieved_docs)

        return {
            "query": adaptive_query,
            "retrieved_sources": retrieved_docs,  # Full scored list (for provenance)
            "evidence_bundle": evidence_bundle,
            "status": "SUCCESS",
        }

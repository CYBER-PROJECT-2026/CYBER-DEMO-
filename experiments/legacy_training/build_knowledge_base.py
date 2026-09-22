"""
build_knowledge_base.py
=======================
Preprocessing and Vector/Hybrid Indexing Pipeline for the Multi-Agent RAG System.

Parses raw data from:
- knowledge_base/raw/mitre/
- knowledge_base/raw/nist/
- knowledge_base/raw/cisa/

Generates:
- knowledge_base/processed/*
- knowledge_base/chunks/chunks.json
- knowledge_base/metadata/kb_index_metadata.json
- knowledge_base/vector_db/{tfidf_vectorizer.pkl, tfidf_matrix.pkl, bm25_store.pkl, chunks_lookup.json}
"""

import os
import sys
import json
import re
import joblib
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

# UTF-8 Output Configuration
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(BASE_DIR, "knowledge_base")
RAW_DIR = os.path.join(KB_DIR, "raw")
PROCESSED_DIR = os.path.join(KB_DIR, "processed")
CHUNKS_DIR = os.path.join(KB_DIR, "chunks")
METADATA_DIR = os.path.join(KB_DIR, "metadata")
VECTOR_DB_DIR = os.path.join(KB_DIR, "vector_db")


def clean_text(text: str) -> str:
    if not text:
        return ""
    # Strip markdown citations like (Citation: ...)
    text = re.sub(r'\(Citation:[^\)]+\)', '', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Normalize whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def parse_mitre_stix(filepath: str, domain_name: str):
    """
    Parses official STIX 2.1 JSON bundle from MITRE ATT&CK.
    Extracts attack-pattern (Techniques) and course-of-action (Mitigations).
    """
    if not os.path.exists(filepath):
        print(f"  [Warning] MITRE file not found: {filepath}")
        return []

    print(f"  Parsing MITRE STIX from: {os.path.basename(filepath)}...")
    with open(filepath, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    objects = bundle.get("objects", [])
    chunks = []

    # Map mitigations
    mitigations_map = {}
    for obj in objects:
        if obj.get("type") == "course-of-action" and not obj.get("revoked", False) and not obj.get("x_mitre_deprecated", False):
            name = obj.get("name", "")
            desc = clean_text(obj.get("description", ""))
            ext_refs = obj.get("external_references", [])
            mitre_id = ""
            url = ""
            for ref in ext_refs:
                if ref.get("source_name") in ["mitre-attack", "mitre-ics-attack"]:
                    mitre_id = ref.get("external_id", "")
                    url = ref.get("url", "")
                    break
            if mitre_id:
                mitigations_map[mitre_id] = {
                    "name": name,
                    "description": desc,
                    "url": url
                }

    # Extract techniques
    for obj in objects:
        if obj.get("type") == "attack-pattern" and not obj.get("revoked", False) and not obj.get("x_mitre_deprecated", False):
            name = obj.get("name", "")
            desc = clean_text(obj.get("description", ""))
            detection = clean_text(obj.get("x_mitre_detection", ""))
            platforms = obj.get("x_mitre_platforms", [])
            tactics = [kc.get("phase_name", "") for kc in obj.get("kill_chain_phases", [])]

            ext_refs = obj.get("external_references", [])
            mitre_id = ""
            url = ""
            for ref in ext_refs:
                if ref.get("source_name") in ["mitre-attack", "mitre-ics-attack"]:
                    mitre_id = ref.get("external_id", "")
                    url = ref.get("url", "")
                    break

            if not mitre_id or not desc:
                continue

            # Prioritize networking, denial of service, spoofing, reconnaissance, IoT/ICS relevant techniques
            content = f"MITRE ATT&CK Technique {mitre_id}: {name}. Tactics: {', '.join(tactics)}. Platforms: {', '.join(platforms)}. Description: {desc}"
            if detection:
                content += f" Detection Guidance: {detection}"

            chunks.append({
                "chunk_id": f"MITRE-{mitre_id}",
                "source": "MITRE ATT&CK",
                "domain": domain_name,
                "technique_id": mitre_id,
                "title": f"{mitre_id}: {name}",
                "url": url or f"https://attack.mitre.org/techniques/{mitre_id}/",
                "category": "ATTACK_TECHNIQUE",
                "tactics": tactics,
                "platforms": platforms,
                "content": content,
                "keywords": f"{name} {mitre_id} {' '.join(tactics)} {' '.join(platforms)}"
            })

    # Also add course-of-action mitigations
    for m_id, m_data in mitigations_map.items():
        if m_data["description"]:
            content = f"MITRE ATT&CK Mitigation {m_id}: {m_data['name']}. Guidance: {m_data['description']}"
            chunks.append({
                "chunk_id": f"MITRE-MITIGATION-{m_id}",
                "source": "MITRE ATT&CK",
                "domain": domain_name,
                "technique_id": m_id,
                "title": f"Mitigation {m_id}: {m_data['name']}",
                "url": m_data["url"] or f"https://attack.mitre.org/mitigations/{m_id}/",
                "category": "MITIGATION",
                "tactics": ["Defensive Control"],
                "platforms": ["Network", "IoT"],
                "content": content,
                "keywords": f"{m_data['name']} {m_id} mitigation defense containment"
            })

    print(f"  Extracted {len(chunks)} knowledge chunks from {domain_name}.")
    return chunks


def parse_nist_files():
    """
    Parses NIST JSON documents into structured chunks.
    """
    chunks = []
    
    # 1. NIST SP 800-82 Rev 3
    sp82_path = os.path.join(RAW_DIR, "nist", "nist_sp800_82_rev3.json")
    if os.path.exists(sp82_path):
        with open(sp82_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        for i, sec in enumerate(doc.get("sections", []), 1):
            chunk_id = f"NIST-SP800-82-SEC{i}"
            content = f"{doc['title']} - {sec['section']}: {sec['content']}"
            chunks.append({
                "chunk_id": chunk_id,
                "source": "NIST",
                "domain": "OT & IoT Security",
                "technique_id": f"SP800-82-S{i}",
                "title": f"NIST SP 800-82: {sec['section']}",
                "url": doc["official_url"],
                "category": "STANDARDS_GUIDANCE",
                "tactics": ["OT/IoT Security Baseline", "Containment"],
                "platforms": ["IoT", "OT", "ICS"],
                "content": content,
                "keywords": f"NIST SP 800-82 OT IoT network segmentation Purdue model Rate anomaly containment {sec['section']}"
            })

    # 2. NIST SP 800-53 Rev 5
    sp53_path = os.path.join(RAW_DIR, "nist", "nist_sp800_53_rev5.json")
    if os.path.exists(sp53_path):
        with open(sp53_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        for ctrl in doc.get("controls", []):
            chunk_id = f"NIST-SP800-53-{ctrl['control_id']}"
            content = f"NIST SP 800-53 Rev 5 Control {ctrl['control_id']}: {ctrl['name']}. Guidance: {ctrl['guidance']}"
            chunks.append({
                "chunk_id": chunk_id,
                "source": "NIST",
                "domain": "Security Controls",
                "technique_id": ctrl['control_id'],
                "title": f"NIST SP 800-53: Control {ctrl['control_id']} ({ctrl['name']})",
                "url": doc["official_url"],
                "category": "SECURITY_CONTROL",
                "tactics": ["Defensive Control", "Incident Handling"],
                "platforms": ["Network", "IoT"],
                "content": content,
                "keywords": f"NIST SP 800-53 {ctrl['control_id']} {ctrl['name']} SYN cookies ARP inspection boundary protection"
            })

    # 3. NIST IR 8259A
    ir_path = os.path.join(RAW_DIR, "nist", "nist_ir_8259a.json")
    if os.path.exists(ir_path):
        with open(ir_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        for i, cap in enumerate(doc.get("capabilities", []), 1):
            chunk_id = f"NIST-IR8259A-CAP{i}"
            content = f"NIST IR 8259A Core Baseline: {cap['capability']}. Details: {cap['description']}"
            chunks.append({
                "chunk_id": chunk_id,
                "source": "NIST",
                "domain": "IoT Device Baseline",
                "technique_id": f"IR8259A-C{i}",
                "title": f"NIST IR 8259A: {cap['capability']}",
                "url": doc["official_url"],
                "category": "DEVICE_BASELINE",
                "tactics": ["Device Hardening"],
                "platforms": ["IoT"],
                "content": content,
                "keywords": f"NIST IR 8259A IoT device identity MAC cloning botnet prevention {cap['capability']}"
            })

    print(f"  Extracted {len(chunks)} knowledge chunks from NIST publications.")
    return chunks


def parse_cisa_files():
    """
    Parses CISA JSON documents into structured chunks.
    """
    chunks = []

    # 1. CISA Alert AA22-110A
    aa22_path = os.path.join(RAW_DIR, "cisa", "cisa_aa22_110a.json")
    if os.path.exists(aa22_path):
        with open(aa22_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        mandates_text = " ".join(doc.get("mitigation_mandates", []))
        content = f"{doc['title']}. Threat Profile: {doc['threat_profile']} Mandates: {mandates_text}"
        chunks.append({
            "chunk_id": "CISA-AA22-110A",
            "source": "CISA",
            "domain": "Cybersecurity Advisory",
            "technique_id": "AA22-110A",
            "title": doc["title"],
            "url": doc["official_url"],
            "category": "CISA_ADVISORY",
            "tactics": ["Threat Defense", "Containment"],
            "platforms": ["IoT", "OT", "Network"],
            "content": content,
            "keywords": "CISA AA22-110A IoT botnet Mirai Mozi SYN flood UDP flood ARP spoofing isolation TreeSHAP"
        })

    # 2. CISA Alert AA20-133A
    aa20_path = os.path.join(RAW_DIR, "cisa", "cisa_aa20_133a.json")
    if os.path.exists(aa20_path):
        with open(aa20_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        findings_text = " ".join(doc.get("findings", []))
        controls_text = " ".join(doc.get("recommended_controls", []))
        content = f"{doc['title']}. Key Findings: {findings_text} Controls: {controls_text}"
        chunks.append({
            "chunk_id": "CISA-AA20-133A",
            "source": "CISA",
            "domain": "Cybersecurity Advisory",
            "technique_id": "AA20-133A",
            "title": doc["title"],
            "url": doc["official_url"],
            "category": "CISA_ADVISORY",
            "tactics": ["Vulnerability Mitigation"],
            "platforms": ["IoT", "Network"],
            "content": content,
            "keywords": "CISA AA20-133A IoT camera router vulnerability zero trust rate anomaly flow metrics"
        })

    # 3. CISA Cross-Sector CPGs
    cpg_path = os.path.join(RAW_DIR, "cisa", "cisa_cpg_iot_ot.json")
    if os.path.exists(cpg_path):
        with open(cpg_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        for i, g in enumerate(doc.get("core_goals", []), 1):
            chunk_id = f"CISA-CPG-{i}"
            content = f"CISA Performance Goal: {g['goal']}. Description: {g['description']}"
            chunks.append({
                "chunk_id": chunk_id,
                "source": "CISA",
                "domain": "Performance Goals",
                "technique_id": f"CPG-{i}",
                "title": f"CISA CPG: {g['goal']}",
                "url": doc["official_url"],
                "category": "GOVERNANCE",
                "tactics": ["Defense in Depth"],
                "platforms": ["IoT", "OT"],
                "content": content,
                "keywords": f"CISA CPG micro-segmentation playbooks asset inventory {g['goal']}"
            })

    print(f"  Extracted {len(chunks)} knowledge chunks from CISA advisories.")
    return chunks


def build_vector_and_bm25_indices(all_chunks):
    """
    Builds:
    1. TfidfVectorizer & TF-IDF Dense Matrix for Cosine Similarity Retrieval
    2. BM25Okapi Sparse Keyword Search Index
    3. Serialized Chunk Lookup Store
    """
    print("\n[Indexing] Building TF-IDF Dense Vector Matrix and BM25 Sparse Index...")

    corpus_texts = []
    tokenized_corpus = []

    for c in all_chunks:
        # Full text representation combining title, content, keywords, and tactics
        full_text = f"{c['title']} {c['content']} {c.get('keywords', '')}"
        corpus_texts.append(full_text)
        # Tokenize for BM25
        tokens = re.findall(r'\b[a-zA-Z0-9_\-]+\b', full_text.lower())
        tokenized_corpus.append(tokens)

    # 1. Fit TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words="english"
    )
    tfidf_matrix = vectorizer.fit_transform(corpus_texts)

    # 2. Build BM25 Index
    bm25 = BM25Okapi(tokenized_corpus)

    # 3. Save artifacts
    vectorizer_path = os.path.join(VECTOR_DB_DIR, "tfidf_vectorizer.pkl")
    matrix_path = os.path.join(VECTOR_DB_DIR, "tfidf_matrix.pkl")
    bm25_path = os.path.join(VECTOR_DB_DIR, "bm25_store.pkl")
    lookup_path = os.path.join(VECTOR_DB_DIR, "chunks_lookup.json")

    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(tfidf_matrix, matrix_path)
    joblib.dump(bm25, bm25_path)

    # Save chunks lookup
    with open(lookup_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    # Save chunks.json in chunks/
    chunks_json_path = os.path.join(CHUNKS_DIR, "chunks.json")
    with open(chunks_json_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    # Save metadata summary
    metadata_summary = {
        "total_chunks": len(all_chunks),
        "embedding_model": "TF-IDF Sublinear N-Gram Vectorizer",
        "embedding_dimension": tfidf_matrix.shape[1],
        "sources_represented": list(set(c["source"] for c in all_chunks)),
        "categories": list(set(c["category"] for c in all_chunks)),
        "artifacts": {
            "vectorizer": "tfidf_vectorizer.pkl",
            "tfidf_matrix": "tfidf_matrix.pkl",
            "bm25_store": "bm25_store.pkl",
            "chunks_lookup": "chunks_lookup.json"
        }
    }
    meta_path = os.path.join(METADATA_DIR, "kb_index_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_summary, f, indent=2)

    print(f"  [Vector DB] Matrix Shape: {tfidf_matrix.shape}")
    print(f"  [Vector DB] Artifacts saved to: {VECTOR_DB_DIR}")
    print(f"  [Metadata] Saved summary to: {meta_path}")


def main():
    print("============================================================")
    print("BUILDING AUTHORITATIVE CYBERSECURITY RAG KNOWLEDGE BASE")
    print("============================================================")

    all_chunks = []

    # 1. MITRE Enterprise ATT&CK
    mitre_ent_path = os.path.join(RAW_DIR, "mitre", "mitre_enterprise_attack.json")
    mitre_ent_chunks = parse_mitre_stix(mitre_ent_path, "Enterprise ATT&CK")
    all_chunks.extend(mitre_ent_chunks)

    # 2. MITRE ICS ATT&CK
    mitre_ics_path = os.path.join(RAW_DIR, "mitre", "mitre_ics_attack.json")
    mitre_ics_chunks = parse_mitre_stix(mitre_ics_path, "ICS / Embedded ATT&CK")
    all_chunks.extend(mitre_ics_chunks)

    # 3. NIST Publications
    nist_chunks = parse_nist_files()
    all_chunks.extend(nist_chunks)

    # 4. CISA Advisories
    cisa_chunks = parse_cisa_files()
    all_chunks.extend(cisa_chunks)

    print(f"\n[Summary] Aggregated {len(all_chunks)} total authoritative knowledge chunks.")

    # Save domain splits in processed/
    mitre_processed_path = os.path.join(PROCESSED_DIR, "mitre", "processed_mitre_chunks.json")
    with open(mitre_processed_path, "w", encoding="utf-8") as f:
        json.dump(mitre_ent_chunks + mitre_ics_chunks, f, indent=2)

    nist_processed_path = os.path.join(PROCESSED_DIR, "nist", "processed_nist_chunks.json")
    with open(nist_processed_path, "w", encoding="utf-8") as f:
        json.dump(nist_chunks, f, indent=2)

    cisa_processed_path = os.path.join(PROCESSED_DIR, "cisa", "processed_cisa_chunks.json")
    with open(cisa_processed_path, "w", encoding="utf-8") as f:
        json.dump(cisa_chunks, f, indent=2)

    # Build Indices
    build_vector_and_bm25_indices(all_chunks)

    print("============================================================")
    print("KNOWLEDGE BASE BUILD COMPLETE")
    print("============================================================")


if __name__ == "__main__":
    main()

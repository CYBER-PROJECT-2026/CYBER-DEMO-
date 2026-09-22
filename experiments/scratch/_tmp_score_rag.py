import sys
sys.path.insert(0, r"d:\-AI-Driven-Intrusion-Detection-System-for-IoT-Networks")
from backend.rag.adaptive_hybrid_rag import AdaptiveHybridRAG

rag = AdaptiveHybridRAG()
print("loaded", rag.loaded, "chunks", len(rag.chunks))

for atk in ["DDoS", "DoS", "Spoofing"]:
    q = rag.build_adaptive_query(
        atk,
        shap_evidence=[{"feature": "Rate", "shap_value": 1.0, "direction": "POSITIVE"}],
    )
    docs = rag.retrieve(q, top_k=6)
    bundle = rag.build_evidence_bundle(atk, docs)
    print("\n====", atk, "====")
    print("QUERY:", q)
    for d in bundle["all_scored_docs"]:
        line = (
            f"  {d['evidence_tier']:16s} acs={d['attack_context_score']:.3f} "
            f"km={d['keyword_match']:.3f} mm={d['mitre_tech_match']:.2f} "
            f"sa={d['source_adjustment']:.2f} | {d['chunk_id']} | {d['title'][:70]}"
        )
        print(line)
    print("excluded:", [d["chunk_id"] for d in bundle["excluded_evidence"]])
    print("mitre:", [d["chunk_id"] for d in bundle["mitre_context"]])
    print("specific:", [d["chunk_id"] for d in bundle["attack_specific_evidence"]])
    print("general:", [d["chunk_id"] for d in bundle["general_security_evidence"]])

import re
import math
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
from .kb_store import CYBERSECURITY_KNOWLEDGE_BASE

class AdaptiveHybridRetriever:
    def __init__(self, kb: List[Dict[str, Any]] = CYBERSECURITY_KNOWLEDGE_BASE):
        self.kb = kb
        self.corpus_docs = []
        self.tokenized_corpus = []

        # Prepare corpus for retrieval
        for item in self.kb:
            # Flatten document fields into search string
            text_parts = [
                item.get("id", ""),
                item.get("title", ""),
                item.get("description", ""),
                item.get("tactic", ""),
                item.get("mitigation", ""),
                " ".join(item.get("guidelines", [])),
                " ".join(item.get("procedure", [])),
                " ".join(item.get("detection_signatures", [])),
                " ".join(item.get("affected_devices", []))
            ]
            full_text = " ".join([p for p in text_parts if p])
            self.corpus_docs.append({
                "item": item,
                "text": full_text
            })
            self.tokenized_corpus.append(self._tokenize(full_text))

        # Initialize BM25 search index
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _tokenize(self, text: str) -> List[str]:
        # Lowercase and extract alphanumeric tokens
        return re.findall(r'\b[a-zA-Z0-9_\-]+\b', text.lower())

    def _dense_semantic_score(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        """
        Calculates dense n-gram / semantic overlap score between query and document.
        """
        if not query_tokens or not doc_tokens:
            return 0.0
        doc_set = set(doc_tokens)
        overlap = sum(1 for t in query_tokens if t in doc_set)
        # Cosine-like jaccard / length normalization
        score = overlap / math.sqrt(len(query_tokens) * len(doc_set) + 1e-5)
        return float(score)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Performs Hybrid Search:
        1. BM25 Sparse Search
        2. Dense Semantic Overlap Search
        3. Reciprocal Rank Fusion (RRF)
        4. Knowledge Re-ranking & Context Compression
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return [doc["item"] for doc in self.corpus_docs[:top_k]]

        # 1. BM25 Scores & Ranking
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)

        # 2. Dense Semantic Scores & Ranking
        dense_scores = [
            self._dense_semantic_score(query_tokens, self.tokenized_corpus[i])
            for i in range(len(self.corpus_docs))
        ]
        dense_ranked_indices = sorted(range(len(dense_scores)), key=lambda i: dense_scores[i], reverse=True)

        # 3. Reciprocal Rank Fusion (RRF)
        # RRF formula: RRF_score(d) = 1/(k + rank_bm25(d)) + 1/(k + rank_dense(d))
        rrf_k = 60
        rrf_scores = {}

        for rank, idx in enumerate(bm25_ranked_indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank + 1))

        for rank, idx in enumerate(dense_ranked_indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank + 1))

        # 4. Sort by RRF Score
        final_ranked_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)

        results = []
        for idx in final_ranked_indices[:top_k]:
            doc_item = dict(self.corpus_docs[idx]["item"])
            doc_item["_retrieval_score"] = round(rrf_scores[idx], 4)
            results.append(doc_item)

        return results

    def build_context_string(self, query: str, top_k: int = 3) -> str:
        """Compresses and formats retrieved documents into concise context for LLM agents"""
        docs = self.retrieve(query, top_k=top_k)
        formatted_blocks = []
        for d in docs:
            category = d.get("category", "INFO")
            title = d.get("title", d.get("id", "Doc"))
            desc = d.get("description", "")
            mitigation = d.get("mitigation", "")
            guidelines = " | ".join(d.get("guidelines", []))
            procedure = " -> ".join(d.get("procedure", []))
            
            block = f"[{category}] {title}\nSummary: {desc}"
            if mitigation:
                block += f"\nRecommended Mitigation: {mitigation}"
            if guidelines:
                block += f"\nCompliance Guidelines: {guidelines}"
            if procedure:
                block += f"\nSOP Procedure: {procedure}"
            formatted_blocks.append(block)

        return "\n\n".join(formatted_blocks)

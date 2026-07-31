"""Semantic FAQ cache engine for project-specific system queries."""


import json
import re
from pathlib import Path
import numpy as np

from utils.logger_config import logger
from utils.models_and_clients import embedding_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FAQ_FILE = PROJECT_ROOT / "data" / "faq.json"

_MEMORY_FAQS: list[dict] | None = None


def _cosine_similarity(vec1: list | np.ndarray, vec2: list | np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors."""
    v1 = np.array(vec1, dtype=np.float32)
    v2 = np.array(vec2, dtype=np.float32)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def _get_embedding(text: str) -> list[float]:
    """Encode text to embedding vector using BGE-M3 embedding model."""
    if embedding_model is None:
        return []
    try:
        if hasattr(embedding_model, "encode"):
            res = embedding_model.encode(text)
            if isinstance(res, dict):
                emb = res.get("dense_vecs") if "dense_vecs" in res else res.get("dense", [])
            else:
                emb = res

            if hasattr(emb, "tolist"):
                return emb.tolist()
            elif isinstance(emb, (list, np.ndarray)):
                return [float(x) for x in emb]
    except Exception as exc:
        logger.warning(f"[FAQ Agent] Embedding generation failed: {exc}")
    return []


def _normalize_text(text: str) -> str:
    """Normalize text for consistent string matching."""
    if not text:
        return ""
    cleaned = re.sub(r"[^\w\s]", "", text.strip().lower())
    return cleaned.strip()


def load_faqs(force_reload: bool = False) -> list[dict]:
    """Load project FAQ items from data/faq.json and populate in-memory embeddings."""
    global _MEMORY_FAQS
    if _MEMORY_FAQS is not None and not force_reload:
        return _MEMORY_FAQS

    if not FAQ_FILE.exists():
        _MEMORY_FAQS = []
        return _MEMORY_FAQS

    try:
        faqs = json.loads(FAQ_FILE.read_text(encoding="utf-8"))
        for item in faqs:
            if "_memory_emb" not in item:
                item["_memory_emb"] = _get_embedding(item["question"])
        _MEMORY_FAQS = faqs
        return _MEMORY_FAQS
    except Exception as exc:
        logger.warning(f"[FAQ Agent] Failed loading FAQ file: {exc}")
        _MEMORY_FAQS = []
        return _MEMORY_FAQS


def find_faq_match(query: str, threshold: float = 0.82) -> dict | None:
    """Check if user query matches a project FAQ entry via exact match or semantic cosine similarity."""
    if not query or not query.strip():
        return None

    clean_query = _normalize_text(query)
    faqs = load_faqs()

    # Stage 1: Exact / Normalized string match
    for item in faqs:
        if _normalize_text(item.get("question", "")) == clean_query:
            logger.info(f"[FAQ Agent] Exact project FAQ match found for query: '{query}'")
            return {
                "question": item["question"],
                "answer": item["answer"],
                "score": 1.0,
                "match_type": "exact",
            }

    # Stage 2: Semantic Cosine Similarity via BGE-M3 in-memory embeddings
    query_emb = _get_embedding(query)
    if not query_emb:
        return None

    best_item = None
    best_score = 0.0

    for item in faqs:
        cached_emb = item.get("_memory_emb") or item.get("embedding")
        if not cached_emb:
            continue
        sim = _cosine_similarity(query_emb, cached_emb)
        if sim > best_score:
            best_score = sim
            best_item = item

    if best_item and best_score >= threshold:
        logger.info(f"[FAQ Agent] Semantic project FAQ match (score={best_score:.4f}) for query: '{query}'")
        return {
            "question": best_item["question"],
            "answer": best_item["answer"],
            "score": round(best_score, 4),
            "match_type": "semantic",
        }

    return None


def track_query_and_promote(query: str, answer: str, confidence: float = 1.0) -> None:
    """No-op: Auto-promotion removed per system requirements."""
    pass

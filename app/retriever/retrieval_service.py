import re
from pathlib import Path
from datetime import datetime
import json
from typing import List, Any
from qdrant_client.models import SparseVector, Fusion, FusionQuery, Prefetch
from fastembed import SparseTextEmbedding

from utils.models_and_clients import embedding_model, qdrant_client, co
from utils.settings import COHERE_EMBEDDING_MODEL, COLLECTION_NAME, RERANKER_NEEDED
from utils.logger_config import logger

bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")

def cohere_rerank(hits: List[Any], query: str, cohere_client=co) -> List[Any]:
    """If `cohere_client` provided, call its reranker; otherwise return hits unchanged."""
    if cohere_client is None or not RERANKER_NEEDED:
        return hits

    try:

        logger.info(f"Reranking {len(hits)} chunks")

        texts = [
            hit.payload["text"]
            for hit in hits
        ]

        response = cohere_client.rerank(
            model=COHERE_EMBEDDING_MODEL,
            query=query,
            documents=texts,
            top_n=3,
        )

        logger.success("Cohere reranking completed")

        ordered = []

        for rank, result in enumerate(response.results):

            logger.info(
                f"Rank {rank+1}"
                f" | relevance={result.relevance_score:.4f}"
            )
            hit = hits[result.index]

            hit.payload.setdefault("metadata", {})
            hit.payload["metadata"]["rerank_score"] = result.relevance_score

            ordered.append(hit)

        return ordered
    
    except Exception:
        logger.exception("Cohere rerank failed; returning original order")
        return hits


def should_include_chunk_for_scope(chunk: Any, scope: str) -> bool:
    if isinstance(chunk, dict):
        payload = chunk.get("payload", {}) or {}
    else:
        payload = getattr(chunk, "payload", {}) or {}
    metadata = payload.get("metadata", {}) or {}
    chunk_type = (metadata.get("chunk_type") or payload.get("chunk_type") or "document").lower()
    if scope == "faq":
        return chunk_type == "faq"
    return chunk_type != "faq"


def retrieve(
    query: str,
    top_k: int = 5,
    use_sparse: bool = True,
    reranker=RERANKER_NEEDED,
    scope: str = "documents",
) -> tuple[List[Any], List[dict]]:
    """Return top_k Qdrant hits using dense + optional sparse retrieval with RRF fusion.

    - `embedding_model` must provide `.encode(text)` returning a list/ndarray.
    - `reranker` is an optional callable(hits, query) -> reordered_hits
    """
    embeddings = embedding_model.encode(
        query,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False
    )

    dense = embeddings["dense_vecs"].tolist()
    bm25_result = list(bm25_model.query_embed([query]))[0]
    sparse = SparseVector(
        indices=bm25_result.indices.tolist(),
        values=bm25_result.values.tolist(),
    )

    fetch_limit = top_k * 3 if scope == "faq" else top_k * 2

    prefetch = [
        Prefetch(
            query = dense,
            using = "dense",
            limit = fetch_limit,
        )
    ]

    if use_sparse:
        prefetch.append(
            Prefetch(
                query = sparse,
                using = "sparse",
                limit = fetch_limit,
            )
        )

    try:
        resp = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=prefetch,
            query=FusionQuery(fusion=Fusion.RRF),
            limit=fetch_limit,
        )

    except Exception as e:
        logger.exception(f"Qdrant query failed: {e}")
        return [], []

    raw_hits = list(resp.points or [])

    # If query contains a specific arXiv paper ID (e.g. 2606.15207), ensure chunks from that target paper and all its figure descriptions are included
    arxiv_match = re.search(r"\b(\d{4}\.\d{4,5})\b", query)
    if arxiv_match:
        target_id = arxiv_match.group(1)
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            img_filter = Filter(must=[FieldCondition(key="metadata.chunk_type", match=MatchValue(value="image"))])
            scrolled_hits, _ = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=img_filter,
                limit=100,
                with_payload=True,
                with_vectors=False,
            )
            paper_img_hits = [h for h in scrolled_hits if target_id in str((h.payload or {}).get("metadata", {}).get("doc_id", ""))]
            if paper_img_hits:
                raw_hits = paper_img_hits + raw_hits
        except Exception:
            pass

    # Scope filtering
    hits = [hit for hit in raw_hits if should_include_chunk_for_scope(hit, scope)][:8]

    logger.info("="*80)
    logger.success(f"Retrieved {len(hits)} chunks for scope '{scope}' (raw: {len(raw_hits)})")

    retrieval_info = {}
    analysis = []
    for i, hit in enumerate(hits):

        metadata = hit.payload.get("metadata", {})
        text = hit.payload.get("text", {})

        score = float(getattr(hit, "score", 0.5) or 0.5)
        logger.debug(f"[Retrieval] Hit {i+1} | score={score:.4f} | doc={metadata.get('doc_id')} | type={metadata.get('chunk_type')} | page={metadata.get('page_number')}")
        
        chunk_id = metadata.get("chunk_id")
        retrieval_info[chunk_id] = {
            "retrieval_rank": i + 1,
            "retrieval_score": score,
        }
    try:
        if reranker:                    
            hits = cohere_rerank(hits, query)

            logger.info("Reranked Chunks")

            for i, hit in enumerate(hits):
                text = hit.payload.get("text", {})
                metadata = hit.payload.get("metadata", {})

                logger.debug(f"[Rerank] Rank {i+1} | score={metadata.get('rerank_score', 0):.4f} | doc={metadata.get('document_name')} | page={metadata.get('page_number')}")
                chunk_id = metadata.get("chunk_id")
                info = retrieval_info[chunk_id]

                analysis.append({
                    "retrieval_rank": info["retrieval_rank"],
                    "retrieval_score": info["retrieval_score"],
                    "rerank_rank": i + 1,
                    "rerank_score": metadata.get("rerank_score"),
                    "document": metadata.get("document_name"),
                    "page": metadata.get("page_number"),
                    "chunk_type": metadata.get("chunk_type"),
                    "chunk_id": chunk_id,
                    "content": text
                })

    except Exception:
        logger.exception("Reranker failed; returning original hits")
    return hits, analysis

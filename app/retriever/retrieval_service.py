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
    if cohere_client is None:
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


def retrieve(
    query: str,
    top_k: int = 5,
    use_sparse: bool = True,
    reranker=RERANKER_NEEDED
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

    prefetch = [
        Prefetch(
            query = dense,
            using = "dense",
            limit = top_k,
        )
    ]

    if use_sparse:
        prefetch.append(
            Prefetch(
                query = sparse,
                using = "sparse",
                limit = top_k,
            )
        )

    try:
        resp = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=prefetch,
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
        )

    except Exception as e:
        logger.exception(f"Qdrant query failed: {e}")
        return [], []

    hits = list(resp.points or [])

    logger.info("="*80)
    logger.success(f"Retrieved {len(hits)} chunks")

    retrieval_info = {}
    analysis = []
    for i, hit in enumerate(hits):

        metadata = hit.payload.get("metadata", {})
        text = hit.payload.get("text", {})

        logger.debug(f"[Retrieval] Hit {i+1} | score={hit.score:.4f} | doc={metadata.get('doc_id')} | type={metadata.get('chunk_type')} | page={metadata.get('page_number')}")
        
        chunk_id = metadata.get("chunk_id")
        retrieval_info[chunk_id] = {
            "retrieval_rank": i + 1,
            "retrieval_score": hit.score,
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

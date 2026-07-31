import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from utils.logger_config import logger

def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int = 1024) -> None:
    """Create the Qdrant collection if it does not already exist."""
    collections = [item.name for item in client.get_collections().collections]
    if collection_name in collections:
        logger.info(f"Collection {collection_name} already exists")
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(),
        },
    )
    logger.success(f"Collection {collection_name} is ready")


def build_point(chunk: dict[str, Any], dense_embedding: list[float], sparse_embedding: dict[str, list[Any]]) -> PointStruct:
    """Create a Qdrant point with dense and sparse embeddings."""
    payload = {
        "text": chunk["text"],
        "metadata": chunk["metadata"],
    }
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["metadata"]["chunk_id"]))
    return PointStruct(
        id=point_id,
        vector={
            "dense": dense_embedding,
            "sparse": SparseVector(
                indices=sparse_embedding.get("indices", []),
                values=sparse_embedding.get("values", []),
            ),
        },
        payload=payload,
    )


def delete_points(client: QdrantClient, collection_name: str, doc_id: str) -> int:
    """Delete all points for a document from a Qdrant collection.

    Builds a strongly-typed `Filter` so callers can also construct it
    independently (e.g. tests inspecting the filter shape).
    """
    selector = Filter(
        must=[FieldCondition(key="metadata.doc_id", match=MatchValue(value=doc_id))]
    )
    return _delete_with_selector(client, collection_name, doc_id, selector)


def _delete_with_selector(
    client: QdrantClient,
    collection_name: str,
    doc_id: str,
    selector: Filter,
) -> int:
    """Internal: pass a pre-built Filter to `client.delete`."""
    deleted_count = client.delete(
        collection_name=collection_name,
        points_selector=selector,
    )
    logger.info(f"Deleted {deleted_count} points for doc_id={doc_id} from {collection_name}")
    return deleted_count


def delete_points_by_doc_id(client: QdrantClient, collection_name: str, doc_id: str) -> int:
    """Public alias matching the rest of the codebase's call sites."""
    return delete_points(client, collection_name, doc_id)


def delete_doc_from_collection(client: QdrantClient, collection_name: str, doc_id: str) -> bool:
    """Delete a document from collection (wrapper for delete_points_by_doc_id)."""
    try:
        delete_points_by_doc_id(client, collection_name, doc_id)
        return True
    except Exception as e:
        logger.error(f"Failed to delete doc_id={doc_id} from {collection_name}: {e}")
        return False


def upsert_chunks(client: QdrantClient, collection_name: str, chunks: list[dict[str, Any]], dense_embeddings: list[list[float]], sparse_embeddings: list[dict[str, list[Any]]]) -> None:
    """Upload chunks to Qdrant in batches."""
    batch_size = 100
    points: list[PointStruct] = []
    chunks_added = 0

    for chunk, dense_embedding, sparse_embedding in zip(chunks, dense_embeddings, sparse_embeddings, strict=False):
        points.append(build_point(chunk, dense_embedding, sparse_embedding))
        chunks_added += 1

        if len(points) == batch_size:
            client.upsert(collection_name=collection_name, points=points, wait=True, timeout=300)
            logger.success(f"Uploaded batch of {len(points)} points")
            points = []

    if points:
        client.upsert(collection_name=collection_name, points=points, wait=True, timeout=300)
        logger.success(f"Uploaded final batch of {len(points)} points")

    logger.success(f"Uploaded {len(chunks)} chunks to Qdrant collection {collection_name}")

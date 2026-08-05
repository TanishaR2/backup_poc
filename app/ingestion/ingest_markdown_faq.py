"""Ingest system FAQ markdown content (data/faq.md) into Qdrant collection 'InsightDocs' with metadata.chunk_type = 'faq'."""

import uuid
from pathlib import Path
from qdrant_client.models import PointStruct, SparseVector
from fastembed import SparseTextEmbedding

from utils.models_and_clients import embedding_model, qdrant_client
from utils.settings import COLLECTION_NAME
from utils.logger_config import logger

bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")


def parse_faq_markdown(file_path: Path) -> list[dict]:
    content = file_path.read_text(encoding="utf-8")
    sections = content.split("\n## ")
    chunks = []
    
    title = sections[0].strip("# \n")
    
    for sec in sections[1:]:
        lines = sec.strip().split("\n")
        header = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        if not body:
            continue
            
        full_text = f"Section: {header}\n{body}"
        chunks.append({
            "header": header,
            "text": full_text,
            "document_name": "InSightDocs_System_FAQ.md",
            "chunk_type": "faq",
        })
    return chunks


def ingest_faq_chunks():
    faq_path = Path(__file__).resolve().parents[2] / "data" / "faq.md"
    if not faq_path.exists():
        logger.error(f"FAQ file not found at {faq_path}")
        return

    chunks = parse_faq_markdown(faq_path)
    logger.info(f"[FAQ Ingestion] Prepared {len(chunks)} FAQ chunks from {faq_path.name}")

    points = []
    for idx, chunk in enumerate(chunks):
        text = chunk["text"]
        
        # Generate dense vector
        embeddings = embedding_model.encode(
            text,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense_vector = embeddings["dense_vecs"].tolist()

        # Generate sparse BM25 vector
        bm25_result = list(bm25_model.query_embed([text]))[0]
        sparse_vector = SparseVector(
            indices=bm25_result.indices.tolist(),
            values=bm25_result.values.tolist(),
        )

        chunk_id = f"faq_chunk_{idx+1}_{uuid.uuid4().hex[:8]}"
        payload = {
            "text": text,
            "chunk_type": "faq",
            "metadata": {
                "chunk_id": chunk_id,
                "document_name": chunk["document_name"],
                "doc_id": "insightdocs_system_faq",
                "chunk_type": "faq",
                "header": chunk["header"],
                "page_number": 1,
            }
        }

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense": dense_vector,
                "sparse": sparse_vector,
            },
            payload=payload,
        )
        points.append(point)

    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )
    logger.success(f"[FAQ Ingestion] Successfully upserted {len(points)} FAQ points into Qdrant collection '{COLLECTION_NAME}'!")


if __name__ == "__main__":
    ingest_faq_chunks()

from typing import Any


def build_chunk_metadata(doc_id: str, document_name: str, content_item: dict[str, Any], chunk_index: int) -> dict[str, Any]:
    """Build rich metadata for a chunk so retrieval can filter by page and type."""
    return {
        "doc_id": doc_id,
        "document_name": document_name,
        "chunk_id": f"{doc_id}__{content_item['chunk_type']}__{chunk_index}",
        "chunk_type": content_item["chunk_type"],
        "page_number": content_item.get("page_number"),
        "source_type": "pdf",
        "source_path": content_item.get("source_path", ""),
        "description_path": content_item.get("description_path", ""),
        "chunk_index": chunk_index,
    }

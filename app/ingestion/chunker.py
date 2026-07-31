from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.logger_config import logger

from app.ingestion.metadata_builder import build_chunk_metadata


def chunk_content(content_items: list[dict[str, Any]], doc_id: str, document_name: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[dict[str, Any]]:
    """Split content items into parent-child chunks (only returning child chunks carrying parent text/metadata)."""
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=50)
    
    chunks: list[dict[str, Any]] = []
    parent_counter = 0
    child_counter = 0

    for item_index, content_item in enumerate(content_items):
        content = content_item.get("content", "").strip()
        if not content:
            continue

        parent_texts = parent_splitter.split_text(content)
        for parent_text in parent_texts:
            parent_counter += 1
            parent_id = f"{doc_id}__parent__{parent_counter}"

            child_texts = child_splitter.split_text(parent_text)
            for child_text in child_texts:
                child_counter += 1
                metadata = build_chunk_metadata(doc_id, document_name, content_item, child_counter)
                metadata.update({
                    "chunk_role": "child",
                    "parent_id": parent_id,
                    "parent_text": parent_text,
                    "chunking_strategy": "parent_child",
                })
                chunks.append(
                    {
                        "text": child_text,
                        "metadata": metadata,
                        "source_item": content_item,
                    }
                )

    logger.info(f"Prepared Parent-Child chunks: {parent_counter} parents, {child_counter} children for {document_name}")
    return chunks

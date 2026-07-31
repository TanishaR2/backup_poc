import json
from pathlib import Path
from typing import Any

from utils.logger_config import logger


def aggregate_content(
    output_dir: Path,
    doc_id: str,
    document_name: str,
    page_texts: list[dict[str, Any]],
    image_records: list[dict[str, Any]],
    table_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate extracted text, image descriptions and table descriptions into a reusable content bundle."""
    content_dir = output_dir / "content"
    content_dir.mkdir(parents=True, exist_ok=True)
    text_dir = content_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=True)

    content_items: list[dict[str, Any]] = []

    for page in page_texts:
        page_text = page.get("text", "").strip()
        if not page_text:
            continue
        page_file = text_dir / f"page_{page['page_number']}.md"
        page_file.write_text(page_text, encoding="utf-8")
        content_items.append(
            {
                "doc_id": doc_id,
                "document_name": document_name,
                "chunk_type": "text",
                "page_number": page["page_number"],
                "content": page_text,
                "source_path": str(page_file),
                "description_path": "",
            }
        )

    for item in image_records:
        description = item.get("description", "") or ""
        if description:
            content_items.append(
                {
                    "doc_id": doc_id,
                    "document_name": document_name,
                    "chunk_type": "image",
                    "page_number": item.get("page_number"),
                    "content": description,
                    "source_path": item.get("source_path", ""),
                    "description_path": item.get("description_path", ""),
                }
            )

    for item in table_records:
        description = item.get("description", "") or ""
        if description:
            content_items.append(
                {
                    "doc_id": doc_id,
                    "document_name": document_name,
                    "chunk_type": "table",
                    "page_number": item.get("page_number"),
                    "content": description,
                    "source_path": item.get("source_path", ""),
                    "description_path": item.get("description_path", ""),
                }
            )

    combined_text = "\n\n".join(
        f"<!-- {entry['chunk_type']} page={entry.get('page_number')} -->\n{entry['content']}"
        for entry in content_items
    )

    content_path = content_dir / "content.md"
    manifest_path = content_dir / "manifest.json"
    content_path.write_text(combined_text, encoding="utf-8")
    manifest_path.write_text(json.dumps(content_items, indent=2), encoding="utf-8")

    logger.success(f"Aggregated content saved to {content_dir}")

    return {
        "content_dir": content_dir,
        "content_path": content_path,
        "manifest_path": manifest_path,
        "content_items": content_items,
    }

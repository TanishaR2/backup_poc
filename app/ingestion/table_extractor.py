from pathlib import Path
from typing import Any

from utils.logger_config import logger


def extract_tables(document: Any, output_dir: Path, doc_id: str) -> list[dict[str, Any]]:
    """Extract tables from a document and save them as markdown files."""
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    saved_tables: list[dict[str, Any]] = []

    if document is None:
        return saved_tables

    for idx, table in enumerate(getattr(document, "tables", []) or [], start=1):
        try:
            prov = table.prov[0] if getattr(table, "prov", None) else None
            page = prov.page_no if prov else "unknown"

            if hasattr(table, "export_to_markdown"):
                try:
                    markdown = table.export_to_markdown(document)
                except TypeError:
                    markdown = table.export_to_markdown()
            else:
                markdown = ""

            output_path = table_dir / f"page_{page}_table_{idx}.md"
            output_path.write_text(markdown or "", encoding="utf-8")

            saved_tables.append(
                {
                    "doc_id": doc_id,
                    "page_number": page,
                    "source_path": str(output_path),
                    "table_name": output_path.name,
                    "description": "",
                    "description_path": "",
                }
            )
            logger.success(f"Saved table -> {output_path}")
        except Exception as exc:
            logger.error(f"Table extraction failed for table {idx}: {exc}")

    return saved_tables

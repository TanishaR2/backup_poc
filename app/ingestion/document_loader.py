from pathlib import Path
from typing import Any

import fitz
from utils.logger_config import logger

try:
    from docling.document_converter import DocumentConverter
except Exception:  # pragma: no cover - optional dependency
    DocumentConverter = None


def load_pdf_document(pdf_path: Path) -> dict[str, Any]:
    """Load a PDF using Docling when available and fall back to PyMuPDF."""
    page_texts: list[dict[str, Any]] = []
    document = None

    if DocumentConverter is not None:
        try:
            converter = DocumentConverter()
            result = converter.convert(str(pdf_path))
            document = getattr(result, "document", None)
            if document is None and hasattr(result, "documents"):
                document = result.documents[0] if result.documents else None
        except Exception as exc:  # pragma: no cover - runtime dependency
            logger.warning(f"Docling failed for {pdf_path.name}: {exc}")

    with fitz.open(pdf_path) as fitz_doc:
        for page_number in range(1, len(fitz_doc) + 1):
            page = fitz_doc[page_number - 1]
            page_text = page.get_text("text").strip()
            page_texts.append({"page_number": page_number, "text": page_text})

    text_content = "\n\n".join(item["text"] for item in page_texts if item["text"])

    return {
        "document": document,
        "page_texts": page_texts,
        "text_content": text_content,
        "page_count": len(page_texts),
    }

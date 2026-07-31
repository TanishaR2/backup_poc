from pathlib import Path
from typing import Any

import fitz
from utils.logger_config import logger


def extract_images(pdf_path: Path, document: Any, output_dir: Path, doc_id: str) -> list[dict[str, Any]]:
    """Extract images from a document and save them into the PDF workspace."""
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    saved_images: list[dict[str, Any]] = []

    if document is None:
        return saved_images

    with fitz.open(pdf_path) as pdf_doc:
        for idx, picture in enumerate(getattr(document, "pictures", []) or [], start=1):
            try:
                prov = picture.prov[0] if getattr(picture, "prov", None) else None
                if not prov:
                    continue

                page_num = prov.page_no
                bbox = prov.bbox
                page = pdf_doc[page_num - 1]
                rect = fitz.Rect(
                    bbox.l,
                    page.rect.height - bbox.t,
                    bbox.r,
                    page.rect.height - bbox.b,
                )
                pix = page.get_pixmap(matrix=fitz.Matrix(4, 4), clip=rect, alpha=False)

                image_path = image_dir / f"page_{page_num}_figure_{idx}.png"
                pix.save(image_path)

                saved_images.append(
                    {
                        "doc_id": doc_id,
                        "page_number": page_num,
                        "source_path": str(image_path),
                        "image_name": image_path.name,
                        "description": "",
                        "description_path": "",
                    }
                )
                logger.success(f"Saved image -> {image_path}")
            except Exception as exc:
                logger.error(f"Image extraction failed for figure {idx}: {exc}")

    return saved_images

"""Multimodal extraction pipeline for parsing PDF text, figures, and tables into document manifests."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from app.ingestion.content_aggregator import aggregate_content
from app.ingestion.document_loader import load_pdf_document
from app.ingestion.image_describer import describe_images
from app.ingestion.image_extractor import extract_images
from app.ingestion.status_tracker import (
    is_completed,
    mark_completed,
    mark_failed,
    mark_processing,
    remove_doc_from_status,
    status_dir_for,
)
from app.ingestion.table_describer import describe_tables
from app.ingestion.table_extractor import extract_tables
from utils.logger_config import logger
from utils.models_and_clients import groq_client
from utils.settings import vlm_provider

load_dotenv()


META_BLOB_NAME = "meta.json"


def _workspace_path(output_dir: Path, collection_name: str, doc_id: str) -> Path:
    return output_dir / "pdfs" / collection_name / doc_id


def _meta_path(output_dir: Path, collection_name: str, doc_id: str) -> Path:
    return _workspace_path(output_dir, collection_name, doc_id) / META_BLOB_NAME


def _is_meta_completed(output_dir: Path, collection_name: str, doc_id: str) -> bool:
    path = _meta_path(output_dir, collection_name, doc_id)
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "completed"
    except Exception:
        return False


def _wipe_workspace(output_dir: Path, collection_name: str, doc_id: str) -> None:
    shutil.rmtree(_workspace_path(output_dir, collection_name, doc_id), ignore_errors=True)


def _force_reset(output_dir: Path, status_dir: Path, collection_name: str, doc_id: str) -> None:
    """Wipe workspace + status entries so the doc is re-extracted from scratch."""
    _wipe_workspace(output_dir, collection_name, doc_id)
    remove_doc_from_status(status_dir, doc_id, collection_name)
    logger.info(f"Force reset {doc_id} in {collection_name}")


def _write_meta(
    output_dir: Path,
    collection_name: str,
    doc_id: str,
    pdf_name: str,
    status: str,
    error: str | None = None,
) -> None:
    meta = {
        "doc_id": doc_id,
        "pdf_name": pdf_name,
        "collection_name": collection_name,
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if error is not None:
        meta["error"] = error
    path = _meta_path(output_dir, collection_name, doc_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _extract_and_aggregate(pdf_path: Path, workspace: Path, doc_id: str) -> int:
    """Run the actual extraction steps; return the count of content items produced."""
    loaded = load_pdf_document(pdf_path)
    document = loaded["document"]
    page_texts = loaded["page_texts"]

    image_records = extract_images(pdf_path, document, workspace, doc_id)
    table_records = extract_tables(document, workspace, doc_id)
    image_records = describe_images(image_records, workspace, vlm_provider)
    table_records = describe_tables(table_records, workspace, groq_client)
    aggregated = aggregate_content(
        output_dir=workspace,
        doc_id=doc_id,
        document_name=pdf_path.name,
        page_texts=page_texts,
        image_records=image_records,
        table_records=table_records,
    )
    return len(aggregated.get("content_items", []))


def process_pdf_content(
    pdf_path: Path,
    output_dir: Path,
    collection_name: str = "default",
    retry_count: int = 0,
    force: bool = False,
) -> int:
    """Process a single PDF and write artifacts to local filesystem.

    Returns the number of content items produced.
    """
    doc_id = pdf_path.stem
    status_dir = status_dir_for(output_dir, collection_name, stage="extraction")

    if force:
        _force_reset(output_dir, status_dir, collection_name, doc_id)

    # Ingestion Safeguards: FileType, FileSize, PageCount, and AI/ML Domain Check
    from app.ingestion.document_validator import validate_document_for_ingestion
    val_res = validate_document_for_ingestion(pdf_path)
    if not val_res["valid"]:
        logger.warning(f"Skipping extraction for {pdf_path.name}: {val_res['reason']}")
        mark_failed(status_dir, doc_id, pdf_path.name, val_res["reason"], _workspace_path(output_dir, collection_name, doc_id), retry_count, collection_name)
        return 0

    if _is_meta_completed(output_dir, collection_name, doc_id):
        logger.debug(f"Skipping extraction for {pdf_path.name}: meta.json already completed in {collection_name}")
        if not is_completed(status_dir, doc_id, collection_name):
            mark_completed(status_dir, doc_id, pdf_path.name, output_dir, collection_name)
        return 0

    workspace = _workspace_path(output_dir, collection_name, doc_id)
    workspace.mkdir(parents=True, exist_ok=True)

    mark_processing(status_dir, doc_id, pdf_path.name, workspace, collection_name)
    _write_meta(output_dir, collection_name, doc_id, pdf_path.name, status="processing")

    try:
        content_count = _extract_and_aggregate(pdf_path, workspace, doc_id)
        _write_meta(output_dir, collection_name, doc_id, pdf_path.name, status="completed")
        mark_completed(status_dir, doc_id, pdf_path.name, workspace, collection_name)
        logger.success(f"Extraction completed for {pdf_path.name} ({content_count} content items)")
        return content_count
    except Exception as exc:
        error_msg = str(exc)
        logger.exception(f"Extraction failed for {pdf_path.name}: {error_msg}")
        mark_failed(status_dir, doc_id, pdf_path.name, error_msg, workspace, retry_count, collection_name)
        _wipe_workspace(output_dir, collection_name, doc_id)
        raise


def run_extractor_pipeline(
    input_dir: Path,
    output_dir: Path,
    collection_name: str = "default",
    *,
    force: bool = False,
) -> dict[str, list[str]]:
    """Run the extractor for every PDF in `input_dir` concurrently."""
    import concurrent.futures

    output_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(input_dir.glob("*.pdf"))
    processed: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []

    logger.info(f"Starting concurrent extraction for {len(pdfs)} PDFs with max_workers=3")

    def process_one(pdf: Path) -> str:
        status_dir = status_dir_for(output_dir, collection_name, stage="extraction")
        if not force and _is_meta_completed(output_dir, collection_name, pdf.stem) and is_completed(status_dir, pdf.stem, collection_name):
            logger.debug(f"Skipping extraction for {pdf.name}: already completed in {collection_name}")
            return "skipped"
        try:
            process_pdf_content(pdf, output_dir, collection_name=collection_name, force=force)
            return "processed"
        except Exception as exc:
            logger.error(f"Extractor pipeline failed for {pdf.name}: {exc}")
            return "failed"

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        future_to_pdf = {executor.submit(process_one, pdf): pdf for pdf in pdfs}
        for future in concurrent.futures.as_completed(future_to_pdf):
            pdf = future_to_pdf[future]
            try:
                res = future.result()
                (processed if res == "processed" else skipped if res == "skipped" else failed).append(pdf.name)
            except Exception as exc:
                logger.error(f"Unhandled exception during concurrent extraction of {pdf.name}: {exc}")
                failed.append(pdf.name)

    return {"processed": processed, "skipped": skipped, "failed": failed}


__all__ = [
    "process_pdf_content",
    "run_extractor_pipeline",
    "_is_meta_completed",
    "_force_reset",
    "_workspace_path",
    "_meta_path",
]

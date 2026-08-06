import asyncio
import base64
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from .schemas import HealthResponse, IngestUploadResponse, QueryResponse
from .services import run_query, save_and_ingest_pdf
from utils.logger_config import logger
from utils.settings import COLLECTION_NAME

router = APIRouter()



@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/ingest", response_model=IngestUploadResponse)
async def ingest(
    file: UploadFile = File(...),
    collection_name: str = Form(default=COLLECTION_NAME),
    force: bool = Form(default=False),
) -> IngestUploadResponse:
    """Upload a PDF and ingest it: save → extract → embed → Qdrant."""
    logger.info(f"[API:Ingest] Received file upload | filename='{file.filename}' | collection='{collection_name}' | force={force}")
    file_bytes = await file.read()
    try:
        import fitz
        _DOMAIN_KEYWORDS = {"neural", "model", "learning", "transformer", "rag", "lora", "fine-tun", "embedding"}
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid PDF file format.")
        first_page_text = doc[0].get_text().lower() if len(doc) > 0 else ""
        if not any(kw in first_page_text for kw in _DOMAIN_KEYWORDS):
            raise HTTPException(status_code=400, detail="PDF does not appear to be a technical or research document.")
    except HTTPException:
        raise

    try:
        result = save_and_ingest_pdf(
            file_bytes=file_bytes,
            filename=file.filename,
            collection_name=collection_name,
            force=force,
        )
    except Exception as exc:
        logger.exception(f"Ingest failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
    return IngestUploadResponse(**result)


@router.post("/query", response_model=QueryResponse)
async def query(
    query: Optional[str] = Form(default=None),
    image: Optional[UploadFile] = File(default=None),
    session_id: str = Form(default="default"),
) -> QueryResponse:
    """Accept a text query and/or an image file. Base64 conversion happens here."""
    MAX_QUERY_LEN = 2000
    if not query and not image:
        raise HTTPException(status_code=422, detail="Provide at least 'query' or 'image'.")
    if query and len(query) > MAX_QUERY_LEN:
        raise HTTPException(status_code=422, detail=f"Query too long (max {MAX_QUERY_LEN} chars).")

    image_base64 = None
    if image:
        image_base64 = base64.b64encode(await image.read()).decode("utf-8")

    query_text = query or ""
    logger.info(f"[API:Query] Processing request | session_id='{session_id}' | query_len={len(query_text)} | has_image={image is not None}")
    try:
        result = await asyncio.to_thread(
            run_query,
            query_text,
            image_base64,
            session_id,
        )
    except Exception as exc:
        logger.exception(f"Query failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
    val = result.get("validation_result") or {}
    return QueryResponse(
        query=result["query"],
        answer=result["answer"],
        latency=result["latency"],
        retrieval_confidence=result.get("retrieval_confidence"),
        validation_score=val.get("score"),
        route=result.get("route"),
        chunks=result.get("chunks"),
        planner_decision=result.get("planner_decision"),
        validation_result=result.get("validation_result"),
        retrieval_analysis=result.get("merged_analysis", {}).get("retrieval_analysis"),
        retrieved_image_path=result.get("retrieved_image_path"),
    )


@router.get("/documents")
def list_documents(collection_name: str = COLLECTION_NAME) -> dict:
    """List unique documents and chunk counts in Qdrant collection."""
    try:
        from utils.models_and_clients import qdrant_client
        batch, _ = qdrant_client.scroll(
            collection_name=collection_name,
            limit=500,
            with_payload=True,
            with_vectors=False,
        )
        docs = {}
        for pt in batch:
            meta = (pt.payload or {}).get("metadata", {}) or {}
            doc_name = meta.get("document_name") or meta.get("doc_id") or "Unknown"
            docs[doc_name] = docs.get(doc_name, 0) + 1
        return {"collection": collection_name, "documents": docs, "total_chunks": len(batch)}
    except Exception as exc:
        logger.error(f"Failed to list documents: {exc}")
        return {"collection": collection_name, "documents": {}, "total_chunks": 0}

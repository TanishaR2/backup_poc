import time
from datetime import datetime
from pathlib import Path

from app.ingestion.extractor_pipeline import run_extractor_pipeline
from app.ingestion.ingestor_pipeline import run_ingestor_pipeline
from app.ingestion.status_tracker import (
    ensure_status_files,
    read_status_file,
)
from app.generation.generation_service import (
    _get_history,
    _add_to_history,
)
from app.generation.query_utils import select_relevant_image_path, should_request_image
from app.agents.graph import app as agent_graph

from utils.logger_config import logger
from utils.settings import COLLECTION_NAME

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_ingestion(
    input_dir: str,
    output_dir: str,
    collection_name: str = COLLECTION_NAME,
    force: bool = False,
) -> dict:
    """Run extractor + ingestor pipeline for all PDFs in ``input_dir``.

    Threaded end-to-end: ``collection_name`` and ``force`` are propagated to
    both pipelines and to the per-collection status tracker.
    """
    logger.info(
        f"Starting ingestion: input={input_dir}, output={output_dir}, "
        f"collection={collection_name}, force={force}"
    )

    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    from app.ingestion.status_tracker import status_dir_for
    status_dir = status_dir_for(out_path, collection_name, stage="extraction")
    completed_path, incomplete_path = ensure_status_files(status_dir)
    completed_before = {
        item["doc_id"]
        for item in read_status_file(completed_path)
        if item.get("collection_name") == collection_name
    }

    pdfs = sorted(in_path.glob("*.pdf"))
    processed: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []

    for pdf in pdfs:
        doc_id = pdf.stem
        if not force and doc_id in completed_before:
            logger.info(f"Skipping {pdf.name}, already completed in {collection_name}")
            skipped.append(pdf.name)
            continue

        try:
            logger.info(f"Extracting: {pdf.name}")
            run_extractor_pipeline(
                input_dir=in_path,
                output_dir=out_path,
                collection_name=collection_name,
                force=force,
            )
            processed.append(pdf.name)
        except Exception as exc:
            logger.error(f"Extractor failed for {pdf.name}: {exc}")
            failed.append(pdf.name)
            continue

    logger.info("Running ingestor pipeline to embed and upload chunks")
    try:
        run_ingestor_pipeline(
            output_dir=out_path,
            collection_name=collection_name,
            force=force,
        )
    except Exception as exc:
        logger.error(f"Ingestor pipeline failed for {collection_name}: {exc}")

    logger.success(
        f"Ingestion done. collection={collection_name}, "
        f"processed={len(processed)}, skipped={len(skipped)}, failed={len(failed)}"
    )

    return {
        "status": "ok",
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "collection_name": collection_name,
        "force": force,
    }


def save_and_ingest_pdf(
    file_bytes: bytes,
    filename: str,
    collection_name: str = COLLECTION_NAME,
    force: bool = False,
) -> dict:
    """Save an uploaded PDF then run the full extraction + ingestion pipeline."""
    doc_id = Path(filename).stem
    single_upload_dir = PROJECT_ROOT / "data" / "uploads" / collection_name / doc_id
    single_upload_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = single_upload_dir / filename
    pdf_path.write_bytes(file_bytes)
    logger.info(f"Saved uploaded PDF → {pdf_path}")

    out_path = PROJECT_ROOT / "data" / "output"
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Running extractor pipeline…")
    run_extractor_pipeline(
        input_dir=single_upload_dir,
        output_dir=out_path,
        collection_name=collection_name,
        force=force,
    )

    logger.info("Running ingestor pipeline…")
    run_ingestor_pipeline(
        output_dir=out_path,
        collection_name=collection_name,
        force=force,
    )

    logger.success(f"Ingestion complete: {filename} → {collection_name}")
    return {
        "status": "completed",
        "doc_id": doc_id,
        "filename": filename,
        "saved_to": str(pdf_path),
        "collection_name": collection_name,
        "force": force,
    }


def run_query(query: str, image_base64: str = None, session_id: str = "default") -> dict:
    logger.info(f"Query received: {query}, session_id={session_id}, image_provided={image_base64 is not None}")
    start = time.time()

    current_history = _get_history(session_id)
    config = {"configurable": {"thread_id": session_id}}

    result = agent_graph.invoke(
        {
            "query": query,
            "image_base64": image_base64,
            "chat_history": current_history,
            "session_id": session_id,
        },
        config=config,
    )

    latency = time.time() - start
    answer = result.get("final_answer", "")
    route = result.get("route", "unknown")

    _add_to_history(session_id, "user", query)
    _add_to_history(session_id, "assistant", answer)

    logger.success(f"Query answered in {latency:.2f}s | route='{route}'")

    # Debug: log agent result keys and content for tracing returned fields (helps verify image path propagation)
    try:
        logger.info(f"[API] Agent result keys: {list(result.keys())}")
        logger.debug(f"[API] Full agent result: {result}")
    except Exception:
        pass

    raw_chunks = result.get("retrieved_docs", []) or []
    retrieval_confidence = result.get("retrieval_confidence")
    val = result.get("validation", {})
    needs_image = result.get("needs_image_in_answer")

    chunks = []
    if route == "rag":
        for c in raw_chunks:
            if isinstance(c, dict):
                chunks.append(c)
            else:
                payload = getattr(c, "payload", {}) or {}
                score = getattr(c, "score", None)
                metadata = payload.get("metadata", {})
                chunks.append({
                    "score": score,
                    "payload": payload,
                    "chunk_type": metadata.get("chunk_type"),
                    "source_path": metadata.get("source_path"),
                })
    else:
        # Support route is independent of RAG — discard chunks and confidence
        raw_chunks = []
        chunks = []
        retrieval_confidence = None

    # Prefer any image path explicitly set by the agent.
    retrieved_image_path = result.get("retrieved_image_path")
    if retrieved_image_path and not Path(retrieved_image_path).exists():
        retrieved_image_path = None

    # Fallback image acquisition if not already set and image is needed (and no image attached)
    effective_query = (result.get("query") or query).strip()
    if not retrieved_image_path and not image_base64 and effective_query and (needs_image is True or should_request_image(effective_query)):
        if route == "rag" and raw_chunks:
            retrieved_image_path = select_relevant_image_path(effective_query, raw_chunks, needs_image=needs_image)

        if not retrieved_image_path:
            from app.agents.web_search_tool import fetch_web_image_for_query
            from app.generation.query_utils import generate_support_diagram_image

            retrieved_image_path = fetch_web_image_for_query(effective_query)
            if not retrieved_image_path:
                retrieved_image_path = generate_support_diagram_image(effective_query)

        if retrieved_image_path and not Path(retrieved_image_path).exists():
            retrieved_image_path = None

    record = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "answer": answer,
        "route": route,
        "validation": val,
        "validation_result": val,
        "planner_decision": {"route": route, "reason": f"Routed to {route} by LangGraph Planner Node"},
        "retrieval_confidence": retrieval_confidence,
        "retrieved_image_path": retrieved_image_path,
        "chunks": chunks,
        "latency": latency,
    }
    logger.info(f"[API] retrieved_image_path before return: {retrieved_image_path}")
    return record




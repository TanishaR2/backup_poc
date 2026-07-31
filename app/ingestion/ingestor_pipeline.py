"""Ingestor pipeline for chunking, embedding, and indexing extracted document manifests into Qdrant."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.ingestion.chunker import chunk_content
from app.ingestion.embedding_service import build_embeddings
from app.ingestion.qdrant_ingestor import (
    delete_points_by_doc_id,
    ensure_collection,
    upsert_chunks,
)
from app.ingestion.recovery import retry_until_done
from app.ingestion.status_tracker import (
    ensure_status_files,
    is_completed,
    mark_completed,
    mark_failed,
    mark_processing,
    read_status_file,
    remove_doc_from_status,
    status_dir_for,
)
from utils.logger_config import logger
from utils.models_and_clients import embedding_model, qdrant_client
from utils.settings import COLLECTION_NAME


META_BLOB_NAME = "meta.json"


def _is_meta_completed(workspace: Path) -> bool:
    path = workspace / META_BLOB_NAME
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("status") == "completed" and "chunk_count" in data
    except Exception:
        return False


def _should_skip_workspace(status_dir: Path, workspace: Path, collection_name: str, doc_id: str) -> bool:
    return is_completed(status_dir, doc_id, collection_name) or _is_meta_completed(workspace)


def _force_reset_workspace(
    qdrant_client: Any,
    collection_name: str,
    workspace: Path,
    status_dir: Path,
) -> None:
    """Wipe the Qdrant points for this doc and clear the status tracker."""
    try:
        ensure_collection(qdrant_client, collection_name)
        delete_points_by_doc_id(qdrant_client, collection_name, workspace.name)
    except Exception as exc:
        logger.warning(f"Pre-force Qdrant cleanup for {workspace.name} in {collection_name} failed: {exc}")
    remove_doc_from_status(status_dir, workspace.name, collection_name)
    logger.info(f"Force reset ingestion for {workspace.name} in {collection_name}")


def _refresh_meta(workspace: Path, doc_id: str, collection_name: str, chunk_count: int) -> None:
    """Write/update meta.json with completed status."""
    path = workspace / META_BLOB_NAME
    try:
        meta = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        meta = {}
    meta.update({
        "doc_id": doc_id,
        "pdf_name": f"{doc_id}.pdf",
        "collection_name": collection_name,
        "status": "completed",
        "chunk_count": chunk_count,
    })
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def ingest_workspace(
    workspace: Path,
    qdrant_client: Any,
    embedding_model: Any,
    collection_name: str,
) -> int:
    """Chunk + embed + upsert one workspace. Returns the chunk count."""
    doc_id = workspace.name
    manifest_path = workspace / "content" / "manifest.json"
    content_items = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Delete pre-existing Qdrant points for a clean upsert.
    try:
        ensure_collection(qdrant_client, collection_name)
        delete_points_by_doc_id(qdrant_client, collection_name, doc_id)
    except Exception as exc:
        logger.warning(f"Pre-ingest Qdrant cleanup for {doc_id} in {collection_name} failed: {exc}")

    chunks = chunk_content(content_items, doc_id, f"{doc_id}.pdf")

    try:
        dense_embeddings: list[list[float]] = []
        sparse_embeddings: list[dict[str, list[Any]]] = []
        for chunk in chunks:
            dense, sparse = build_embeddings(chunk["text"], embedding_model)
            dense_embeddings.append(dense)
            sparse_embeddings.append(sparse)

        ensure_collection(qdrant_client, collection_name)
        upsert_chunks(qdrant_client, collection_name, chunks, dense_embeddings, sparse_embeddings)
    except Exception:
        # On failure: delete whatever we just wrote, then re-raise for retry.
        try:
            delete_points_by_doc_id(qdrant_client, collection_name, doc_id)
        except Exception as cleanup_exc:
            logger.error(f"Qdrant cleanup after failed ingest of {doc_id} failed: {cleanup_exc}")
        raise

    _refresh_meta(workspace, doc_id, collection_name, len(chunks))
    return len(chunks)


def _process_pending_workspace(
    workspace: Path,
    qdrant_client: Any,
    embedding_model: Any,
    collection_name: str,
    status_dir: Path,
) -> None:
    doc_id = workspace.name
    pdf_name = f"{doc_id}.pdf"

    if _should_skip_workspace(status_dir, workspace, collection_name, doc_id):
        logger.info(f"Skipping ingestion for {pdf_name}: already completed in {collection_name}")
        return

    mark_processing(status_dir, doc_id, pdf_name, workspace, collection_name)

    try:
        chunk_count = ingest_workspace(workspace, qdrant_client, embedding_model, collection_name)
        mark_completed(status_dir, doc_id, pdf_name, workspace, collection_name, extra_fields={"chunk_count": chunk_count})
        logger.success(f"{doc_id} ingested ({chunk_count} chunks) into {collection_name}")
    except Exception as exc:
        mark_failed(status_dir, doc_id, pdf_name, str(exc), workspace, 0, collection_name)
        logger.exception(f"Ingestion failed for {doc_id} in {collection_name}: {exc}")


def _iterate_pending_workspaces(output_dir: Path, collection_name: str) -> list[Path]:
    collection_dir = output_dir / "pdfs" / collection_name
    if not collection_dir.exists():
        return []
    return sorted(
        p for p in collection_dir.iterdir()
        if p.is_dir() and (p / "content" / "manifest.json").exists()
    )


def _sync_uncompleted_workspaces_to_status(output_dir: Path, collection_name: str, status_dir: Path) -> None:
    """Populate incomplete.json with any uncompleted workspaces found on disk."""
    workspaces = _iterate_pending_workspaces(output_dir, collection_name)
    for ws in workspaces:
        doc_id = ws.name
        pdf_name = f"{doc_id}.pdf"
        if not _should_skip_workspace(status_dir, ws, collection_name, doc_id):
            mark_processing(status_dir, doc_id, pdf_name, output_dir, collection_name)


def run_ingestor_pipeline(
    output_dir: Path,
    collection_name: str = COLLECTION_NAME,
    *,
    qdrant_client: Any | None = None,
    embedding_model: Any | None = None,
    max_attempts: int = 3,
    force: bool = False,
) -> dict[str, Any]:
    """Run ingestion for a collection, retrying until success or max_attempts."""
    qdrant_client = qdrant_client if qdrant_client is not None else globals()["qdrant_client"]
    embedding_model = embedding_model if embedding_model is not None else globals()["embedding_model"]
    status_dir = status_dir_for(output_dir, collection_name, stage="ingestion")

    def _run_once() -> None:
        workspaces = _iterate_pending_workspaces(output_dir, collection_name)
        if not workspaces:
            logger.info(f"No extracted workspaces under {output_dir / 'pdfs' / collection_name}")
            return
        for workspace in workspaces:
            if force:
                _force_reset_workspace(qdrant_client, collection_name, workspace, status_dir)
            _process_pending_workspace(workspace, qdrant_client, embedding_model, collection_name, status_dir)

    if force:
        for _ in range(max_attempts):
            _run_once()
            _, incomplete_path = ensure_status_files(status_dir)
            if not read_status_file(incomplete_path):
                return {"collection_name": collection_name, "succeeded": True, "max_attempts": max_attempts, "force": force}
        return {"collection_name": collection_name, "succeeded": False, "max_attempts": max_attempts, "force": force}

    # Ensure incomplete.json contains any new/uncompleted workspaces found on disk
    _sync_uncompleted_workspaces_to_status(output_dir, collection_name, status_dir)

    succeeded = retry_until_done(
        _run_once,
        status_dir=status_dir,
        collection=collection_name,
        max_attempts=max_attempts,
        phase="ingestion",
    )
    return {"collection_name": collection_name, "succeeded": succeeded, "max_attempts": max_attempts, "force": force}


__all__ = [
    "ingest_workspace",
    "run_ingestor_pipeline",
    "_process_pending_workspace",
    "_iterate_pending_workspaces",
    "_force_reset_workspace",
]

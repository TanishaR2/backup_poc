"""Per-collection status tracking for the extraction and ingestion pipelines.

Files live at:
- Extraction: ``processing_status/<collection>/extraction/{completed,incomplete}.json``
- Ingestion:  ``processing_status/<collection>/ingestion/{completed,incomplete}.json``

deduped by ``doc_id`` (and ``collection_name`` is stored on every record so the
same ``doc_id`` can exist in multiple collections).
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.logger_config import logger

_status_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def status_dir_for(output_dir: Path, collection_name: str, stage: str | None = "extraction") -> Path:
    """Return (and create) the directory that holds ``{completed,incomplete}.json``
    for a collection stage (e.g. ``stage='extraction'`` or ``stage='ingestion'``).
    """
    if stage:
        status_dir = output_dir / "processing_status" / collection_name / stage
    else:
        status_dir = output_dir / "processing_status" / collection_name
    status_dir.mkdir(parents=True, exist_ok=True)
    return status_dir


def _ensure_status_dir(status_dir: Path) -> Path:
    status_dir.mkdir(parents=True, exist_ok=True)
    return status_dir


def _status_paths(status_dir: Path) -> tuple[Path, Path]:
    _ensure_status_dir(status_dir)
    return status_dir / "completed.json", status_dir / "incomplete.json"


# ---------------------------------------------------------------------------
# Read / write helpers
# ---------------------------------------------------------------------------
def read_status_file(path: Path) -> list[dict[str, Any]]:
    """Read a status JSON file. Returns an empty list when missing or invalid."""
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning(f"Could not read status file {path}; treating it as empty")
        return []


def write_status_file(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def _dedupe(
    records: list[dict[str, Any]], doc_id: str, collection_name: str
) -> list[dict[str, Any]]:
    """Drop any record matching ``(doc_id, collection_name)``."""
    return [
        r for r in records
        if not (r.get("doc_id") == doc_id and r.get("collection_name") == collection_name)
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def ensure_status_files(status_dir: Path) -> tuple[Path, Path]:
    """Create the per-collection stage ``{completed,incomplete}.json`` if missing."""
    completed_path, incomplete_path = _status_paths(status_dir)
    if not completed_path.exists():
        completed_path.write_text("[]", encoding="utf-8")
    if not incomplete_path.exists():
        incomplete_path.write_text("[]", encoding="utf-8")
    return completed_path, incomplete_path


def is_completed(status_dir: Path, doc_id: str, collection_name: str) -> bool:
    """True if ``doc_id`` is already in completed.json for this status_dir."""
    with _status_lock:
        completed_path, _ = ensure_status_files(status_dir)
        records = read_status_file(completed_path)
        return any(
            r.get("doc_id") == doc_id and r.get("collection_name") == collection_name
            for r in records
        )


def mark_processing(
    status_dir: Path,
    doc_id: str,
    pdf_name: str,
    output_dir: Path,
    collection_name: str,
) -> None:
    """Move a doc into incomplete.json with status=processing for this status_dir."""
    with _status_lock:
        _, incomplete_path = ensure_status_files(status_dir)
        records = read_status_file(incomplete_path)
        records = _dedupe(records, doc_id, collection_name)
        records.append(
            {
                "doc_id": doc_id,
                "pdf_name": pdf_name,
                "collection_name": collection_name,
                "status": "processing",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "output_dir": str(output_dir),
            }
        )
        write_status_file(incomplete_path, records)


def mark_completed(
    status_dir: Path,
    doc_id: str,
    pdf_name: str,
    output_dir: Path,
    collection_name: str,
    extra_fields: dict[str, Any] | None = None,
) -> None:
    """Move a doc from incomplete to completed.json with status=completed."""
    with _status_lock:
        completed_path, incomplete_path = ensure_status_files(status_dir)
        completed_records = read_status_file(completed_path)
        incomplete_records = read_status_file(incomplete_path)

        completed_records = _dedupe(completed_records, doc_id, collection_name)
        incomplete_records = _dedupe(incomplete_records, doc_id, collection_name)

        record = {
            "doc_id": doc_id,
            "pdf_name": pdf_name,
            "collection_name": collection_name,
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(output_dir),
        }
        if extra_fields:
            record.update(extra_fields)

        completed_records.append(record)

        write_status_file(completed_path, completed_records)
        write_status_file(incomplete_path, incomplete_records)
        logger.success(f"Completed {pdf_name} in {collection_name} ({status_dir.name})")


def mark_failed(
    status_dir: Path,
    doc_id: str,
    pdf_name: str,
    error: str,
    output_dir: Path,
    retry_count: int,
    collection_name: str,
) -> None:
    """Record a doc as failed in incomplete.json (kept there so retries can pick it up)."""
    with _status_lock:
        _, incomplete_path = ensure_status_files(status_dir)
        incomplete_records = read_status_file(incomplete_path)
        incomplete_records = _dedupe(incomplete_records, doc_id, collection_name)
        incomplete_records.append(
            {
                "doc_id": doc_id,
                "pdf_name": pdf_name,
                "collection_name": collection_name,
                "status": "failed",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error": error,
                "retry_count": retry_count,
                "output_dir": str(output_dir),
            }
        )
        write_status_file(incomplete_path, incomplete_records)
        logger.error(f"Failed {pdf_name} in {collection_name} ({status_dir.name}): {error}")


def remove_doc_from_status(status_dir: Path, doc_id: str, collection_name: str) -> None:
    """Drop a doc from both completed.json and incomplete.json (e.g. before a force reprocess)."""
    with _status_lock:
        completed_path, incomplete_path = ensure_status_files(status_dir)
        completed_records = [
            r for r in read_status_file(completed_path)
            if not (r.get("doc_id") == doc_id and r.get("collection_name") == collection_name)
        ]
        incomplete_records = [
            r for r in read_status_file(incomplete_path)
            if not (r.get("doc_id") == doc_id and r.get("collection_name") == collection_name)
        ]
        write_status_file(completed_path, completed_records)
        write_status_file(incomplete_path, incomplete_records)


def get_pending_documents(status_dir: Path) -> list[dict[str, Any]]:
    """Return every record in incomplete.json (processing or failed)."""
    with _status_lock:
        _, incomplete_path = ensure_status_files(status_dir)
        return [
            item for item in read_status_file(incomplete_path)
            if item.get("status") in {"processing", "failed"}
        ]


def get_pipeline_status(output_dir: Path, collection_name: str) -> dict[str, Any]:
    """Return detailed status report for both extraction and ingestion stages."""
    ext_dir = status_dir_for(output_dir, collection_name, stage="extraction")
    ing_dir = status_dir_for(output_dir, collection_name, stage="ingestion")

    ext_comp, ext_incomp = ensure_status_files(ext_dir)
    ing_comp, ing_incomp = ensure_status_files(ing_dir)

    ext_completed = read_status_file(ext_comp)
    ext_incomplete = read_status_file(ext_incomp)

    ing_completed = read_status_file(ing_comp)
    ing_incomplete = read_status_file(ing_incomp)

    return {
        "collection_name": collection_name,
        "extraction": {
            "completed_count": len(ext_completed),
            "completed_docs": [r.get("doc_id") for r in ext_completed],
            "incomplete_count": len(ext_incomplete),
            "incomplete_docs": [r.get("doc_id") for r in ext_incomplete],
        },
        "ingestion": {
            "completed_count": len(ing_completed),
            "completed_docs": [r.get("doc_id") for r in ing_completed],
            "incomplete_count": len(ing_incomplete),
            "incomplete_docs": [r.get("doc_id") for r in ing_incomplete],
        },
    }


__all__ = [
    "status_dir_for",
    "ensure_status_files",
    "read_status_file",
    "write_status_file",
    "is_completed",
    "mark_processing",
    "mark_completed",
    "mark_failed",
    "remove_doc_from_status",
    "get_pending_documents",
    "get_pipeline_status",
]

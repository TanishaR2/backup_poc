"""Retry helpers for the ingestion pipelines.

`retry_until_done` repeatedly calls a single-document runner until either the
collection's `incomplete.json` is empty (success) or `max_attempts` is reached.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from app.ingestion.status_tracker import (
    ensure_status_files,
    read_status_file,
)
from utils.logger_config import logger


def retry_until_done(
    run_one: Callable[[], Any],
    *,
    status_dir: Path,
    collection: str,
    max_attempts: int = 3,
    phase: str = "ingestion",
) -> bool:
    """Run `run_one` until `incomplete.json` is empty or attempts exhausted.

    Parameters
    ----------
    run_one:
        Callable that processes every still-incomplete document for the
        collection. It should raise to indicate a hard failure (the call site
        is expected to mark each doc failed in incomplete.json before re-raising).
    status_dir:
        Per-collection status directory
        (e.g. `output_dir / "processing_status" / collection`).
    collection:
        Collection name (used for logging only).
    max_attempts:
        Maximum number of retry rounds.
    phase:
        Either ``"extraction"`` or ``"ingestion"`` (used for logging only).

    Returns
    -------
    bool
        True when ``incomplete.json`` ended empty (success), False when the
        retry budget was exhausted with documents still incomplete.
    """
    completed_path, incomplete_path = ensure_status_files(status_dir)

    pending = read_status_file(incomplete_path)
    if not pending:
        logger.success(
            f"[{collection}] {phase}: no pending documents; retry loop done"
        )
        return True

    for attempt in range(1, max_attempts + 1):
        logger.info(
            f"[{collection}] {phase} retry attempt {attempt}/{max_attempts} "
            f"with {len(pending)} pending document(s)"
        )
        try:
            run_one()
        except Exception as exc:
            logger.exception(
                f"[{collection}] {phase} attempt {attempt} raised: {exc}"
            )

        pending = read_status_file(incomplete_path)
        if not pending:
            logger.success(
                f"[{collection}] {phase}: incomplete.json is empty after attempt {attempt}"
            )
            return True

    logger.error(
        f"[{collection}] {phase}: gave up after {max_attempts} attempts; "
        f"{len(pending)} document(s) still incomplete"
    )
    return False


__all__ = ["retry_until_done"]

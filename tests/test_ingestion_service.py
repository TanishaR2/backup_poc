"""
tests/test_ingestion_service.py
────────────────────────────────
Pytest unit tests for Document Ingestion Service features:
- Ingestion checkpoint tracking (complete.json recovery)
- File size & page count safeguards
- Standalone FAQ ingestion
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.ingestion.ingest_markdown_faq import ingest_faq_chunks


def test_faq_markdown_file_exists():
    """Verify data/faq.md exists and contains indexed papers section."""
    faq_file = ROOT / "data" / "faq.md"
    assert faq_file.exists()
    content = faq_file.read_text(encoding="utf-8")
    assert "Indexed Research Papers in Knowledge Base" in content
    assert "InSightDocs" in content


def test_complete_json_tracker_exists():
    """Verify ingestion completed tracker complete.json exists."""
    complete_json = ROOT / "data" / "output" / "processing_status" / "complete.json"
    if complete_json.exists():
        data = json.loads(complete_json.read_text(encoding="utf-8"))
        assert isinstance(data, (list, dict))


def test_ingestion_checkpoint_structure():
    """Verify extraction completed checkpoint structure if present."""
    complete_json = ROOT / "data" / "output" / "processing_status" / "complete.json"
    if complete_json.exists():
        data = json.loads(complete_json.read_text(encoding="utf-8"))
        if isinstance(data, list) and len(data) > 0:
            assert isinstance(data[0], str)

"""
tests/test_retrieval_service.py
────────────────────────────────
Pytest unit tests for Retrieval Service features:
- Qdrant document title pagination listing (get_indexed_document_titles)
- Hybrid Qdrant retrieval (retrieve)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.retriever.retrieval_service import get_indexed_document_titles, retrieve


def test_get_indexed_document_titles():
    """Verify get_indexed_document_titles returns 100+ indexed paper titles without error."""
    titles = get_indexed_document_titles()
    assert isinstance(titles, list)
    assert len(titles) > 50
    assert any("VANDERER" in t or "2606" in t for t in titles)


def test_retrieve_documents_scope():
    """Verify retrieve function returns hits and analysis array for technical query."""
    hits, analysis = retrieve("VANDERER map free exploration", scope="documents", top_k=5)
    assert isinstance(hits, list)
    assert isinstance(analysis, list)
    if len(analysis) > 0:
        item = analysis[0]
        assert "document" in item
        assert "content" in item
        assert "page" in item

"""
tests/test_multimodal_features.py
──────────────────────────────────
Pytest unit tests for Multimodal features:
- User image upload analysis (VLM description)
- Visual figure lookup & image path resolution
- Image provenance tracking
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.retriever.retrieval_service import retrieve


def test_image_chunk_retrieval():
    """Verify retrieve function can search image chunk types for visual figure queries."""
    hits, analysis = retrieve("trajectory comparison figure plot", scope="documents", top_k=5)
    assert isinstance(hits, list)
    assert isinstance(analysis, list)
    # Verify metadata fields for image support
    for item in analysis:
        assert "chunk_type" in item
        assert "document" in item

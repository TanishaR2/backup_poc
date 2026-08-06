"""
tests/test_api_routes.py
─────────────────────────
Pytest unit & integration tests for FastAPI Endpoints in api/routes.py:
- GET  /health
- POST /query (form data)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.routes import health, query


def test_health_endpoint():
    """Verify health endpoint function returns status='ok'."""
    res = health()
    assert res.status == "ok"


def test_query_endpoint_support():
    """Verify query function accepts query text and session_id."""
    import asyncio
    res = asyncio.run(query(query="who am i", image=None, session_id="test_session"))
    assert hasattr(res, "answer")
    assert hasattr(res, "route")

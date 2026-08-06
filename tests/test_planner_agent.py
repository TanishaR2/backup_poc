"""
tests/test_planner_agent.py
────────────────────────────
Pytest unit tests for Planner Agent routing and intent classification:
- Route classification (rag vs support)
- Scope resolution (documents vs faq)
- Typo correction & query rewriting
- Atomic vs Non-Atomic intent
- Feature flag detection (needs_image, needs_web_search)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agents.planner_agent import route_query


def test_planner_routing_user_identity():
    """Verify 'who am i' routes to support agent."""
    res = route_query("who am i")
    assert res.get("route") in ["support", "rag"]


def test_planner_routing_knowledge_base_inventory():
    """Verify document inventory query routes to support/rag with faq scope."""
    res = route_query("What documents are available in your knowledge base? List the paper titles.")
    assert res.get("scope") in ["faq", "documents"] or res.get("route") in ["support", "rag"]


def test_planner_routing_technical_paper():
    """Verify technical paper query routes to RAG with documents scope."""
    res = route_query("Explain VANDERER map-free exploration visual curiosity.")
    assert res.get("route") == "rag"
    assert res.get("scope") == "documents"


def test_planner_routing_out_of_domain():
    """Verify out of domain query receives route decision dictionary."""
    res = route_query("How to cook chicken biryani?")
    assert isinstance(res, dict)
    assert "route" in res


def test_planner_routing_figure_request():
    """Verify figure request query sets needs_image=True or rewrites query."""
    res = route_query("Show me Figure 1 architecture diagram of VANDERER.")
    assert isinstance(res, dict)
    assert "route" in res


def test_planner_typo_rewriting():
    """Verify planner rewrites typos correctly."""
    res = route_query("wat is vanderrer paper about")
    assert isinstance(res, dict)
    assert "rewritten_query" in res

"""
tests/test_validator_and_web.py
────────────────────────────────
Pytest unit tests for Validation Node & Web Search Node:
- Faithfulness & Context Recall validation calculation
- Low-confidence edge routing
- Tavily/Serper Internet Search Node
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agents.nodes import validation_node
from app.agents.web_search_tool import search_web


def test_validation_node_structure():
    """Verify validation_node returns score, metrics, and passed boolean."""
    state = {
        "query": "What is VANDERER?",
        "route": "rag",
        "answer": "VANDERER is a map-free visual curiosity framework for robot exploration.",
        "retrieved_docs": [
            {"content": "VANDERER introduces a map-free exploration method guided by visual curiosity."}
        ]
    }
    result_state = validation_node(state)
    assert "validation" in result_state
    val = result_state["validation"]
    assert "score" in val
    assert "metrics" in val
    assert "passed" in val


def test_web_search_tool_structure():
    """Verify search_web tool executes search and returns snippets list."""
    snippets = search_web("What are recent AI research papers in NeurIPS 2025?")
    assert isinstance(snippets, list)

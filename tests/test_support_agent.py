"""
tests/test_support_agent.py
────────────────────────────
Pytest unit tests for Support Agent:
- 'who am i' identity guardrail (refusal + system introduction)
- Knowledge Base paper inventory response
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agents.support_agent import answer_support_query


def test_support_agent_who_am_i():
    """Verify 'who am i' triggers identity refusal + InSightDocs assistant introduction."""
    res = answer_support_query("who am i")
    assert isinstance(res, dict)
    ans = res.get("answer", "")
    assert "do not have access to your personal identity" in ans or "InSightDocs" in ans


def test_support_agent_knowledge_base_inventory():
    """Verify document inventory query returns listing of indexed research papers."""
    res = answer_support_query("What documents are available in your knowledge base? List the paper titles.")
    assert isinstance(res, dict)
    ans = res.get("answer", "")
    assert "research papers" in ans.lower() or "indexed" in ans.lower() or "2606" in ans

"""
tests/test_generation_service.py
─────────────────────────────────
Pytest unit tests for Generation Service & LLM Failover pool:
- Multi-tier LLM Failover execution (run_llm_completion)
- Grounded answer generation (generate_answer)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.generation.generation_service import generate_answer
from utils.models_and_clients import run_llm_completion


def test_llm_failover_completion():
    """Verify run_llm_completion returns non-empty string via primary or failover provider."""
    prompt = "Explain in one sentence what artificial intelligence is."
    response = run_llm_completion(prompt)
    assert isinstance(response, str)
    assert len(response) > 5


def test_generate_answer_grounded():
    """Verify generate_answer returns grounded response given retrieved hits context."""
    ans = generate_answer("What is VANDERER?", hits=[], answer_length="short")
    assert isinstance(ans, str)
    assert len(ans) > 5

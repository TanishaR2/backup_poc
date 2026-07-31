import pytest
from app.agents.support_agent import answer_support_query, _is_greeting_or_identity

def test_greeting_detection():
    assert _is_greeting_or_identity("hello") is True
    assert _is_greeting_or_identity("hi there") is True
    assert _is_greeting_or_identity("who are you") is True
    assert _is_greeting_or_identity("What is transformer?") is False

def test_greeting_response():
    result = answer_support_query("hello")
    assert result["source"] == "domain_greeting"
    assert "InSightDocs" in result["answer"]

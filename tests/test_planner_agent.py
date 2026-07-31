import pytest
from app.agents.planner_agent import route_query

def test_visual_request_forces_rag():
    decision = route_query("Show me an image of transformer architecture")
    assert decision["route"] == "rag"
    assert decision["needs_image"] is True
    assert decision["answer_length"] == "short"

def test_llm_response_mock_parsing():
    mock_json = '{"route": "support", "needs_image": false, "domain": "greeting", "rewritten_query": "hello"}'
    decision = route_query("hello", llm_response=mock_json)
    assert decision["route"] == "support"
    assert decision["domain"] == "greeting"

def test_fallback_greeting_detection():
    decision = route_query("hello, how are you?")
    assert decision["route"] in ["support", "rag"]
    assert "route" in decision
    assert "rewritten_query" in decision

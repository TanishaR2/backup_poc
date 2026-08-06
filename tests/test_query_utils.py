import pytest
from app.generation.query_utils import (
    normalize_query_text,
    should_request_image,
    compute_retrieval_confidence,
)

def test_normalize_query_text():
    assert normalize_query_text("What is th transformer?") == "What is the transformer?"
    assert normalize_query_text("Show teh image") == "Show the image"

def test_should_request_image():
    # should_request_image is deprecated (always False), relying on Planner LLM needs_image
    assert should_request_image("Show me an image of transformer architecture") is False
    assert should_request_image("Give me the figure of accuracy curve") is False
    assert should_request_image("What is the definition of loss function?") is False

def test_compute_retrieval_confidence():
    hits = [
        {"payload": {"metadata": {"rerank_score": 0.9}}},
        {"payload": {"metadata": {"rerank_score": 0.7}}},
    ]
    confidence = compute_retrieval_confidence(hits)
    assert round(confidence, 2) == 0.80

def test_empty_hits_confidence():
    assert compute_retrieval_confidence([]) == 0.0

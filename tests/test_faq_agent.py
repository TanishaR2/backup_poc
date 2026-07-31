import pytest
from app.agents.faq_agent import load_faqs, find_faq_match, _normalize_text

def test_normalize_text():
    assert _normalize_text("  What is InSightDocs?! ") == "what is insightdocs"

def test_load_faqs():
    faqs = load_faqs()
    assert isinstance(faqs, list)
    assert len(faqs) > 0

def test_faq_match():
    # Attempt match against known FAQ question
    faqs = load_faqs()
    if faqs:
        sample_q = faqs[0]["question"]
        match = find_faq_match(sample_q)
        assert match is not None
        assert "answer" in match

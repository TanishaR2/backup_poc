import pytest
from utils.settings import (
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    RERANKER_NEEDED,
    RETRIEVAL_CONFIDENCE_THRESHOLD,
    VALIDATION_CONFIDENCE_THRESHOLD,
)
from utils.logger_config import logger

def test_settings_values():
    assert COLLECTION_NAME == "InsightDocs"
    assert EMBEDDING_MODEL_NAME == "BAAI/bge-m3"
    assert RERANKER_NEEDED is True
    assert RETRIEVAL_CONFIDENCE_THRESHOLD == 0.50
    assert VALIDATION_CONFIDENCE_THRESHOLD == 0.70

def test_logger_working():
    logger.info("Logger verification check")
    assert logger is not None

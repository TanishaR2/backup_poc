"""Edge / conditional routing functions for InSightDocs LangGraph."""

from utils.logger_config import logger
from utils.settings import (
    RETRIEVAL_CONFIDENCE_THRESHOLD,
    RETRIEVAL_SKIP_VALIDATION_THRESHOLD,
)
from app.agents.state import AgentState


def route_after_retrieval(state: AgentState) -> str:
    """Route after retrieval based on retrieval confidence score and image availability."""
    confidence = state.get("retrieval_confidence", 0.0)
    hits = state.get("retrieved_docs", [])
    retrieved_img = bool(state.get("retrieved_image_path"))

    logger.info(
        f"[Edge: after_retrieval] Confidence={confidence:.2f} | Hits={len(hits)} | RetrievedImage={retrieved_img}"
    )

    # If top ColBERT rerank score is below minimal relevance (< 0.52) and no image retrieved, route to support fallback
    if confidence < 0.52 and not retrieved_img:
        logger.warning(f"[Edge: after_retrieval] Retrieval confidence too low ({confidence:.2f} < 0.52) -> support fallback")
        return "support"

    # If hits retrieved or image found, proceed to generation
    if hits or retrieved_img or confidence >= 0.35:
        logger.info("[Edge: after_retrieval] Context/Image retrieved -> proceeding to generation")
        return "generation"

    logger.warning("[Edge: after_retrieval] No context retrieved and confidence too low -> support agent fallback")
    return "support"


def should_validate(state: AgentState) -> str:
    """Decide whether to run validation or skip it when confidence is high."""
    confidence = state.get("retrieval_confidence", 0.0)
    has_user_uploaded_image = bool(state.get("image_base64"))
    has_retrieved_rag_image = bool(state.get("retrieved_image_path"))

    # Retrieved RAG images MUST execute multimodal image validation
    if has_retrieved_rag_image:
        logger.info("[Edge: should_validate] Retrieved RAG image present -> run multimodal image validation")
        return "validate"

    if confidence >= RETRIEVAL_SKIP_VALIDATION_THRESHOLD or has_user_uploaded_image:
        logger.info(
            f"[Edge: should_validate] Confidence={confidence:.2f} or uploaded_image={has_user_uploaded_image} -> skip validation"
        )
        return "skip"

    logger.info(
        f"[Edge: should_validate] Confidence={confidence:.2f} -> run validation"
    )
    return "validate"


def route_after_validation(state: AgentState) -> str:
    """Route after validation based on whether validation passed."""
    validation = state.get("validation", {})
    passed = validation.get("passed", False)
    score = validation.get("score", 0.0)
    route = state.get("route", "")

    if route == "rag":
        if not passed and score < 0.55:
            logger.warning(f"[Edge: after_validation] RAG route validation FAILED (score={score:.3f} < 0.55) -> routing to support fallback")
            return "support"
        logger.info(f"[Edge: after_validation] RAG route (passed={passed}, score={score:.3f}) -> final")
        return "final"

    if not passed:
        logger.warning(f"[Edge: after_validation] Support Validation FAILED (Score={score:.3f}) -> support agent retry")
        return "support"

    logger.success(f"[Edge: after_validation] Score={score:.3f} PASSED -> final")
    return "final"


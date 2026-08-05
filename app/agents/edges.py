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

    # If hits retrieved or image found, proceed to generation
    if hits or retrieved_img or confidence >= 0.35:
        logger.info("[Edge: after_retrieval] Context/Image retrieved -> proceeding to generation")
        return "generation"

    logger.warning("[Edge: after_retrieval] No context retrieved and confidence too low -> support agent fallback")
    return "support"


def should_validate(state: AgentState) -> str:
    """Decide whether to run validation or skip it when confidence is high."""
    confidence = state.get("retrieval_confidence", 0.0)
    # Skip validation for both user-uploaded images AND retrieved RAG images
    has_image = bool(state.get("image_base64")) or bool(state.get("retrieved_image_path"))

    if confidence >= RETRIEVAL_SKIP_VALIDATION_THRESHOLD or has_image:
        logger.info(
            f"[Edge: should_validate] Confidence={confidence:.2f} or image={has_image} -> skip validation"
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
    answer = state.get("answer", "")

    # Fall back to Support Agent if validation failed and not a grounded page/figure mismatch answer
    if not passed:
        if any(kw in answer.lower() for kw in ["not located on page", "not present on page", "not available on page", "appears on page", "is on page"]):
            logger.info("[Edge: after_validation] Grounded figure/page mismatch response -> final")
            return "final"

        logger.warning(f"[Edge: after_validation] Validation FAILED / Context Missing (Score={score:.3f}) -> support agent fallback")
        return "support"

    logger.success(f"[Edge: after_validation] Score={score:.3f} PASSED -> final")
    return "final"

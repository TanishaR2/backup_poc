"""Edge / conditional routing functions for InSightDocs LangGraph."""

from utils.logger_config import logger
from utils.settings import (
    RETRIEVAL_CONFIDENCE_THRESHOLD,
    RETRIEVAL_SKIP_VALIDATION_THRESHOLD,
)
from app.agents.state import AgentState


def route_after_retrieval(state: AgentState) -> str:
    """Route after retrieval based on retrieval confidence score."""
    confidence = state.get("retrieval_confidence", 0.0)
    has_image = bool(state.get("image_base64"))
    needs_image = bool(state.get("needs_image_in_answer"))
    hits = state.get("retrieved_docs", [])

    logger.info(
        f"[Edge: after_retrieval] Confidence={confidence:.2f} | Hits={len(hits)} | NeedsImage={needs_image} | HasImage={has_image}"
    )
    if needs_image or has_image:
        if hits and confidence >= 0.30:
            logger.info("[Edge: after_retrieval] Context retrieved -> proceeding to generation")
            return "generation"

        else:
            logger.warning("[Edge: after_retrieval] Retrieval confidence too low (< 0.30) -> falling back to support agent")
            return "support"


    else: 
        if confidence >= RETRIEVAL_CONFIDENCE_THRESHOLD:
            logger.info("[Edge: after_retrieval] Confidence sufficient -> generation")
            return "generation"

        else:
            logger.warning("[Edge: after_retrieval] Confidence too low -> support agent")
            return "support"


def should_validate(state: AgentState) -> str:
    """Decide whether to run validation or skip it when confidence is high or image present."""
    confidence = state.get("retrieval_confidence", 0.0)
    has_image = bool(state.get("image_base64"))

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

    if passed:
        logger.success(f"[Edge: after_validation] Score={score:.3f} PASSED -> final")
        return "final"

    logger.warning(f"[Edge: after_validation] Score={score:.3f} FAILED -> support")
    return "support"

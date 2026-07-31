"""Validation agent for evaluating answer faithfulness and context relevancy."""

import json
import re
from typing import Any


from utils.logger_config import logger
from utils.models_and_clients import google_client, groq_client, openai_client
from utils.settings import AZURE_OPENAI_MODEL_NAME, VALIDATION_CONFIDENCE_THRESHOLD


def _extract_json(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
        return None


def validate_answer(query: str, answer: str, context_chunks: list[str], llm_response: str | None = None) -> dict[str, Any]:
    """Score the answer using LLM-as-a-judge (RAGAS framework metrics)."""
    logger.info(f"[Validation Agent] Input: query='{query[:60]}...', answer='{answer[:60]}...'")

    if llm_response:
        parsed = _extract_json(llm_response)
        if parsed:
            faithfulness = float(parsed.get("faithfulness", 0.0) or 0.0)
            answer_relevancy = float(parsed.get("answer_relevancy", 0.0) or 0.0)
            context_recall = float(parsed.get("context_recall", 0.0) or 0.0)
            score = (faithfulness + answer_relevancy + context_recall) / 3.0
            passed = score >= VALIDATION_CONFIDENCE_THRESHOLD
            logger.info(f"[Validation Agent] (Mock) Scores: Faithfulness={faithfulness:.2f}, Relevancy={answer_relevancy:.2f}, Recall={context_recall:.2f}")
            logger.success(f"[Validation Agent] Verdict: {'PASSED' if passed else 'FAILED'} | Score: {score:.3f} (Threshold: {VALIDATION_CONFIDENCE_THRESHOLD})")
            return {
                "score": round(score, 3),
                "metrics": {
                    "faithfulness": faithfulness,
                    "answer_relevancy": answer_relevancy,
                    "context_recall": context_recall,
                },
                "passed": passed,
            }

    context_text = "\n\n".join(context_chunks[:3])
    from prompts.prompts import Validation_prompt

    prompt = Validation_prompt.format(
        query=query,
        context_text=context_text,
        answer=answer,
    )

    from utils.models_and_clients import run_llm_completion
    text = run_llm_completion(prompt=prompt, temperature=0.0)
    if text:
        parsed = _extract_json(text)
        if parsed:
            faithfulness = float(parsed.get("faithfulness", 0.0) or 0.0)
            answer_relevancy = float(parsed.get("answer_relevancy", 0.0) or 0.0)
            context_recall = float(parsed.get("context_recall", 0.0) or 0.0)
            score = (faithfulness + answer_relevancy + context_recall) / 3.0
            passed = score >= VALIDATION_CONFIDENCE_THRESHOLD
            logger.info(f"[Validation Agent] Scores: Faithfulness={faithfulness:.2f}, Relevancy={answer_relevancy:.2f}, Recall={context_recall:.2f}")
            logger.success(f"[Validation Agent] Verdict: {'PASSED' if passed else 'FAILED'} | Score: {score:.3f} (Threshold: {VALIDATION_CONFIDENCE_THRESHOLD})")
            return {
                "score": round(score, 3),
                "metrics": {
                    "faithfulness": faithfulness,
                    "answer_relevancy": answer_relevancy,
                    "context_recall": context_recall,
                },
                "passed": passed,
            }

    logger.warning("[Validation Agent] Verdict: FAILED | Reason: All validation models failed")
    return {
        "score": 0.0,
        "metrics": {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_recall": 0.0},
        "passed": False,
    }

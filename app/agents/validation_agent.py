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


def validate_answer(
    query: str,
    answer: str,
    context_chunks: list[str],
    retrieved_image_path: str | None = None,
    image_descriptions: list[dict] | None = None,
    llm_response: str | None = None,
) -> dict[str, Any]:
    """Score the answer using LLM-as-a-judge (correctness, relevancy, completeness, and optional image validation)."""
    logger.info(f"[Validation Agent] Input: query='{query[:60]}...', answer='{answer[:60]}...', has_image={retrieved_image_path is not None}")

    has_image_validation = bool(retrieved_image_path or image_descriptions)

    if llm_response:
        parsed = _extract_json(llm_response)
        if parsed:
            correctness = float(parsed.get("correctness", parsed.get("faithfulness", 0.0)) or 0.0)
            relevancy = float(parsed.get("relevancy", parsed.get("answer_relevancy", 0.0)) or 0.0)
            completeness = float(parsed.get("completeness", parsed.get("context_recall", 0.0)) or 0.0)
            img_rel = float(parsed.get("image_relevancy", 1.0) if has_image_validation else 1.0)
            img_corr = float(parsed.get("image_correctness", 1.0) if has_image_validation else 1.0)

            if has_image_validation:
                score = (correctness + relevancy + completeness + img_rel + img_corr) / 5.0
            else:
                score = (correctness + relevancy + completeness) / 3.0

            passed = score >= VALIDATION_CONFIDENCE_THRESHOLD
            val_img_path = parsed.get("validated_image_path") or retrieved_image_path
            logger.info(f"[Validation Agent] (Mock) Scores: Correctness={correctness:.2f}, Relevancy={relevancy:.2f}, Completeness={completeness:.2f}" + (f", ImageRel={img_rel:.2f}, ImageCorr={img_corr:.2f}" if has_image_validation else ""))
            logger.success(f"[Validation Agent] Verdict: {'PASSED' if passed else 'FAILED'} | Score: {score:.3f} (Threshold: {VALIDATION_CONFIDENCE_THRESHOLD})")
            return {
                "score": round(score, 3),
                "metrics": {
                    "correctness": correctness,
                    "relevancy": relevancy,
                    "completeness": completeness,
                    "image_relevancy": img_rel,
                    "image_correctness": img_corr,
                },
                "validated_image_path": val_img_path,
                "passed": passed,
            }

    context_text = "\n\n".join(context_chunks[:5])
    from prompts.prompts import Validation_prompt, Image_Validation_prompt

    if has_image_validation:
        desc_lines = []
        if image_descriptions:
            for i, item in enumerate(image_descriptions, 1):
                path = item.get("source_path") or item.get("image_path") or "unknown"
                desc = item.get("description") or item.get("text") or "No description"
                desc_lines.append(f"{i}. [Image Path: {path}]\n   Description: {desc}")
        image_descriptions_text = "\n\n".join(desc_lines) if desc_lines else "None provided."

        prompt = Image_Validation_prompt.format(
            query=query,
            context_text=context_text,
            answer=answer,
            retrieved_image_path=retrieved_image_path or "None selected",
            image_descriptions_text=image_descriptions_text,
        )
    else:
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
            correctness = float(parsed.get("correctness", 0.0) or 0.0)
            relevancy = float(parsed.get("relevancy", 0.0) or 0.0)
            completeness = float(parsed.get("completeness", 0.0) or 0.0)
            img_rel = float(parsed.get("image_relevancy", 1.0) if has_image_validation else 1.0)
            img_corr = float(parsed.get("image_correctness", 1.0) if has_image_validation else 1.0)

            if has_image_validation:
                if img_corr >= 0.70 and img_rel >= 0.60:
                    text_score = (correctness + relevancy + completeness) / 3.0
                    img_score = (img_corr + img_rel) / 2.0
                    score = 0.65 * img_score + 0.35 * max(text_score, 0.75)
                else:
                    score = (correctness + relevancy + completeness + img_rel + img_corr) / 5.0
            else:
                score = (correctness + relevancy + completeness) / 3.0

            passed = score >= VALIDATION_CONFIDENCE_THRESHOLD
            val_img_path = parsed.get("validated_image_path") or retrieved_image_path

            logger.info(f"[Validation Agent] Scores: Correctness={correctness:.2f}, Relevancy={relevancy:.2f}, Completeness={completeness:.2f}" + (f", ImageRel={img_rel:.2f}, ImageCorr={img_corr:.2f}" if has_image_validation else ""))
            logger.success(f"[Validation Agent] Verdict: {'PASSED' if passed else 'FAILED'} | Score: {score:.3f} (Threshold: {VALIDATION_CONFIDENCE_THRESHOLD}) | Validated Image: {val_img_path}")
            return {
                "score": round(score, 3),
                "metrics": {
                    "correctness": correctness,
                    "relevancy": relevancy,
                    "completeness": completeness,
                    "image_relevancy": img_rel,
                    "image_correctness": img_corr,
                },
                "validated_image_path": val_img_path,
                "passed": passed,
            }

    logger.warning("[Validation Agent] Verdict: FAILED | Reason: All validation models failed")
    return {
        "score": 0.0,
        "metrics": {"correctness": 0.0, "relevancy": 0.0, "completeness": 0.0, "image_relevancy": 0.0, "image_correctness": 0.0},
        "validated_image_path": retrieved_image_path,
        "passed": False,
    }

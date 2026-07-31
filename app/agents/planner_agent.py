"""Query classifier and intent router for the multi-agent RAG workflow."""

import json
import re
from typing import Any

from app.generation.query_utils import should_request_image
from utils.logger_config import logger


def _extract_json(text: str) -> dict[str, Any] | None:
    if not text or not text.strip():
        return None
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(cleaned)
    except Exception:
        match = re.search(r"\{.*\}", cleaned, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return None


def route_query(
    query: str,
    chat_history: list = None,
    llm_response: str | None = None,
    has_image: bool = False,
) -> dict[str, Any]:
    """Route user query and perform LLM-based query correction, contextual expansion, and length estimation."""
    logger.info(f"[Planner Agent] Processing query: '{query}' (has_image={has_image})")

    visual_request = (not has_image) and bool(should_request_image(query))

    if llm_response:
        parsed = _extract_json(llm_response)
        if parsed and parsed.get("route") in {"rag", "support", "faq"}:
            return {
                "route": "rag" if visual_request else parsed["route"],
                "needs_image": True if visual_request else parsed.get("needs_image", False),
                "needs_web_search": parsed.get("needs_web_search", False),
                "is_atomic": parsed.get("is_atomic", True),
                "domain": parsed.get("domain", "ai_ml_technical"),
                "rewritten_query": parsed.get("rewritten_query", query.strip()),
                "answer_length": "short" if visual_request else parsed.get("answer_length", "medium"),
                "reason": parsed.get("reason", "mock response"),
            }

    # Format recent conversation history (last 2 pairs / 4 messages)
    history_lines = []
    if chat_history:
        for msg in chat_history[-4:]:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")[:250]
            history_lines.append(f"{role}: {content}")
    history_str = "\n".join(history_lines) if history_lines else "None"

    image_context_prompt = ""
    if has_image:
        image_context_prompt = (
            "\nIMPORTANT MULTIMODAL NOTE:\n"
            "An image IS ATTACHED to this query by the user.\n"
            "DO NOT rewrite the query to ask the user to upload or attach an image!\n"
            "If the user query is empty, brief, or general (e.g. '', 'what is this', 'describe this'), set rewritten_query strictly to 'Describe and analyze the attached image in detail.'\n"
        )

    from prompts.prompts import Planner_prompt

    prompt = Planner_prompt.format(
        image_context_prompt=image_context_prompt,
        history_str=history_str,
        query=query if query.strip() else "Describe and analyze the attached image in detail.",
    )

    from utils.models_and_clients import run_llm_completion
    text = run_llm_completion(prompt=prompt, temperature=0.0)

    if text:
        parsed = _extract_json(text)
        if parsed and parsed.get("route") in {"rag", "support"}:
            route = parsed["route"]
            needs_img = bool(parsed.get("needs_image", False))
            needs_web = bool(parsed.get("needs_web_search", False))
            is_atomic = bool(parsed.get("is_atomic", True))
            domain = parsed.get("domain", "ai_ml_technical")
            rewritten_q = parsed.get("rewritten_query", query.strip()) or query.strip()
            if has_image and (not query.strip() or len(query.strip()) < 5):
                rewritten_q = "Describe and analyze the attached image in detail."
            else:
                # Clean leading chatter if present in rewritten_query
                chatter_pat = r"^(hi|hello|hey|greetings|please|you forgot to give me|can you|could you|please give me|show me|give me|return)\b[\s,]*"
                rewritten_q = re.sub(chatter_pat, "", rewritten_q, flags=re.IGNORECASE).strip()
                rewritten_q = re.sub(r"^(please|give me|show me|return|fetch|get)\b[\s,]*", "", rewritten_q, flags=re.IGNORECASE).strip() or query.strip()
            answer_length = parsed.get("answer_length", "medium")
            if answer_length not in {"short", "medium", "detailed"}:
                answer_length = "medium"

            logger.success(
                f"[Planner Agent] LLM Decision: route='{route}', needs_image={needs_img}, "
                f"needs_web={needs_web}, atomic={is_atomic}, domain='{domain}', length='{answer_length}' | "
                f"Rewritten: '{rewritten_q}' | Reason: {parsed.get('reason', '')}"
            )
            if visual_request:
                route = "rag"
                needs_img = True
                answer_length = "short"

            return {
                "route": route,
                "needs_image": needs_img,
                "needs_web_search": needs_web,
                "is_atomic": is_atomic,
                "domain": domain,
                "rewritten_query": rewritten_q,
                "answer_length": answer_length,
                "reason": parsed.get("reason", ""),
            }

    # Fallback if LLM fails
    lowered = query.lower()
    is_greeting = any(w in lowered for w in ["hello", "hi", "hey", "who are you"])
    is_visual = bool(should_request_image(query))
    is_web = any(
        kw in lowered
        for kw in [
            "sota",
            "latest",
            "recent",
            "release",
            "released",
            "announced",
            "benchmark",
            "2025",
            "2026",
            "anthropic",
            "claude",
        ]
    )
    is_atomic_fb = (lowered.count("?") <= 1)
    # Fallback length heuristic based on query word count
    word_count = len(query.split())
    answer_length_fb = "short" if is_visual or word_count <= 5 else "detailed" if word_count > 15 else "medium"

    route_fb = "support" if is_greeting else "rag"
    domain_fb = "greeting" if is_greeting else "ai_ml_technical"

    logger.success(f"[Planner Agent] Fallback Decision: route='{route_fb}', needs_image={is_visual}, length='{answer_length_fb}'")
    return {
        "route": route_fb,
        "needs_image": is_visual,
        "needs_web_search": is_web,
        "is_atomic": is_atomic_fb,
        "domain": domain_fb,
        "rewritten_query": query.strip(),
        "answer_length": answer_length_fb,
        "reason": "fallback classifier",
    }

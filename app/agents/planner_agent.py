"""Query classifier and intent router for the multi-agent RAG workflow."""

import json
import re
from typing import Any

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

    if llm_response:
        parsed = _extract_json(llm_response)
        if parsed and parsed.get("route") in {"rag", "support", "faq"}:
            raw_route = str(parsed.get("route", "rag")).lower()
            raw_scope = str(parsed.get("scope", "documents")).lower()
            raw_domain = str(parsed.get("domain", "")).lower()
            scope = "faq" if (raw_scope == "faq" or raw_route == "faq" or raw_domain == "project_faq") else "documents"
            route = "support" if raw_route == "support" else "rag"
            return {
                "route": route,
                "scope": scope,
                "needs_image": bool(parsed.get("needs_image", False)),
                "needs_web_search": bool(parsed.get("needs_web_search", False)),
                "is_atomic": bool(parsed.get("is_atomic", True)),
                "domain": parsed.get("domain", "ai_ml_technical"),
                "rewritten_query": parsed.get("rewritten_query", query.strip()),
                "answer_length": parsed.get("answer_length", "medium"),
                "reason": parsed.get("reason", "mock response"),
            }

    # Format extended conversation history (last 5 pairs / 10 messages)
    history_lines = []
    if chat_history:
        for msg in chat_history[-10:]:
            role = msg.get("role", "user").upper()
            content = str(msg.get("content", ""))[:300]
            img_info = f" [Attached Image: {msg.get('image_path')}]" if msg.get("image_path") else ""
            history_lines.append(f"{role}{img_info}: {content}")
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
        if parsed and parsed.get("route") in {"rag", "support", "faq"}:
            raw_route = str(parsed.get("route", "rag")).lower()
            raw_scope = str(parsed.get("scope", "documents")).lower()
            raw_domain = str(parsed.get("domain", "")).lower()

            scope = "faq" if (raw_scope == "faq" or raw_route == "faq" or raw_domain == "project_faq") else "documents"
            route = "support" if raw_route == "support" else "rag"

            needs_img = bool(parsed.get("needs_image", False))
            needs_web = bool(parsed.get("needs_web_search", False))
            is_atomic = bool(parsed.get("is_atomic", True))
            domain = parsed.get("domain", "ai_ml_technical")
            rewritten_q = parsed.get("rewritten_query", query.strip()) or query.strip()
            
            if has_image and (not query.strip() or len(query.strip()) < 5):
                rewritten_q = "Describe and analyze the attached image in detail."

            answer_length = parsed.get("answer_length", "medium")
            if answer_length not in {"short", "medium", "detailed"}:
                answer_length = "medium"

            logger.success(
                f"[Planner Agent] LLM Decision: route='{route}', scope='{scope}', needs_image={needs_img}, "
                f"needs_web={needs_web}, atomic={is_atomic}, domain='{domain}', length='{answer_length}' | "
                f"Rewritten: '{rewritten_q}' | Reason: {parsed.get('reason', '')}"
            )

            return {
                "route": route,
                "scope": scope,
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
    is_faq = any(kw in lowered for kw in ["faq", "insightdocs", "vector db", "how to use", "supported formats"])
    is_web = any(kw in lowered for kw in ["sota", "latest", "recent", "release", "2025", "2026"])
    is_atomic_fb = (lowered.count("?") <= 1)
    word_count = len(query.split())
    answer_length_fb = "short" if word_count <= 5 else "detailed" if word_count > 15 else "medium"

    route_fb = "support" if is_greeting else "rag"
    scope_fb = "faq" if is_faq else "documents"
    domain_fb = "greeting" if is_greeting else "project_faq" if is_faq else "ai_ml_technical"

    logger.success(f"[Planner Agent] Fallback Decision: route='{route_fb}', scope='{scope_fb}', length='{answer_length_fb}'")
    return {
        "route": route_fb,
        "scope": scope_fb,
        "needs_image": False,
        "needs_web_search": is_web,
        "is_atomic": is_atomic_fb,
        "domain": domain_fb,
        "rewritten_query": query.strip(),
        "answer_length": answer_length_fb,
        "reason": "fallback classifier",
    }

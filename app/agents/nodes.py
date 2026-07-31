"""LangGraph Node definitions for InSightDocs.

Each node is a Python function taking AgentState and returning a dict containing
only the modified state fields.
"""

from utils.logger_config import logger
from app.agents.planner_agent import route_query
from app.agents.validation_agent import validate_answer
from app.agents.support_agent import answer_support_query
from app.retriever.retrieval_service import retrieve
from app.generation.query_utils import compute_retrieval_confidence, normalize_query_text, should_request_image
from app.generation.generation_service import generate_answer, _extract_json_response, _clean_answer_text
from app.agents.state import AgentState


def planner_node(state: AgentState) -> dict:
    """Classify incoming query, correct typos, expand context, and decide routing path."""
    query = state.get("query", "")
    image_b64 = state.get("image_base64")
    logger.info(f"[Planner Node] Received query: '{query}' (image_attached={bool(image_b64)})")

    # If image is attached without text, set a clear default query
    if image_b64 and (not query or not query.strip()):
        query = "Describe and analyze this uploaded image in detail."

    normalized = normalize_query_text(query)

    # Check project FAQ match first for InSightDocs system/project questions (only if no image)
    if not image_b64:
        from app.agents.faq_agent import find_faq_match
        faq_match = find_faq_match(normalized, threshold=0.82)
        if faq_match:
            logger.info(f"[Planner Node] Query matched project FAQ: '{faq_match['question']}'")
            return {
                "route": "faq",
                "is_atomic": True,
                "query": normalized,
                "faq_answer": faq_match["answer"],
                "domain": "project_faq",
            }

    chat_hist = state.get("chat_history", [])
    result = route_query(normalized, chat_history=chat_hist, has_image=bool(image_b64))

    route = result["route"]
    is_atomic = result.get("is_atomic", True)
    rewritten = result.get("rewritten_query", query.strip())

    # If an image is attached and the query is an image description request or general image inquiry,
    # route to 'support' so Support Agent's vision LLM handles the image directly.
    if image_b64:
        low_q = normalized.lower()
        is_general_image_req = not state.get("query") or any(
            phrase in low_q for phrase in [
                "describe and analyze", "describe this image", "analyze this image",
                "what is this image", "explain this image", "what does this image show",
                "what is shown in this", "describe the visual"
            ]
        )
        if is_general_image_req:
            logger.info("[Planner Node] General image inquiry detected — routing directly to support")
            route = "support"

    # Non-atomic (compound) queries bypass both RAG and FAQ — send to support
    if route in {"rag", "faq"} and not is_atomic:
        logger.info(f"[Planner Node] Non-atomic query detected — overriding route '{route}' → 'support'")
        route = "support"

    # Guardrail against query rewrites asking for image upload when image is already present
    if image_b64:
        low = (rewritten or "").lower()
        if any(p in low for p in ["please upload", "i don't have the image", "i don't see an uploaded", "no image", "please attach"]):
            logger.info("[Planner Node] LLM rewrite requested image but image already provided — using original query")
            rewritten = query.strip()

    logger.success(f"[Planner Node] Route: '{route}' | Rewritten Query: '{rewritten}' | Atomic: {is_atomic}")

    ret = {
        "route": route,
        "is_atomic": is_atomic,
        "query": rewritten,
        "answer_length": result.get("answer_length", "medium"),
        "retrieved_image_path": None,
        "retrieved_docs": [],
        "web_snippets": [],
        "answer": "",
        "final_answer": "",
    }
    if "faq_answer" in result and route == "faq":
        ret["faq_answer"] = result["faq_answer"]
    ret["domain"] = result.get("domain", "ai_ml_technical")
    if image_b64:
        ret["needs_image_in_answer"] = True
    elif "needs_image" in result:
        ret["needs_image_in_answer"] = bool(result["needs_image"])
    else:
        ret["needs_image_in_answer"] = should_request_image(rewritten)
    return ret


def faq_node(state: AgentState) -> dict:
    """Serve sub-millisecond answer from Semantic FAQ Cache."""
    query = state["query"]
    logger.info(f"[FAQ Node] Serving instant cached answer for query: '{query[:60]}...'")

    answer = state.get("faq_answer", "")
    if not answer:
        from app.agents.faq_agent import find_faq_match
        match = find_faq_match(query, threshold=0.90)
        if match:
            answer = match["answer"]
        else:
            answer = "InSightDocs supports multimodal document intelligence over technical research papers."

    logger.success(f"[FAQ Node] Instant answer served: '{answer[:60]}...'")

    updated_history = list(state.get("chat_history", [])) + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer},
    ]

    result = {
        "final_answer": answer,
        "chat_history": updated_history,
    }
    if state.get("needs_image_in_answer") and state.get("image_base64"):
        result["image_base64"] = state.get("image_base64")
        if state.get("image_description"):
            result["image_description"] = state.get("image_description")

    return result


def retrieval_node(state: AgentState) -> dict:
    """Fetch relevant document chunks from Qdrant using hybrid search and reranking (+ parallel web search if flagged)."""
    query = state["query"]
    logger.info(f"[Retrieval Node] Fetching chunks for query: '{query}'")

    web_snippets = []
    if state.get("needs_web_search"):
        logger.info(f"[Retrieval Node] Executing parallel web search for query: '{query[:50]}...'")
        from concurrent.futures import ThreadPoolExecutor
        from app.agents.web_search_tool import search_web
        with ThreadPoolExecutor() as executor:
            future_qdrant = executor.submit(retrieve, query)
            future_web = executor.submit(search_web, query)
            hits, analysis = future_qdrant.result()
            web_snippets = future_web.result()
    else:
        hits, analysis = retrieve(query)

    confidence = compute_retrieval_confidence(hits)

    logger.success(f"[Retrieval Node] Retrieved {len(hits)} chunks | Web snippets: {len(web_snippets)} | Confidence: {confidence:.2f}")
    return {
        "retrieved_docs": hits,
        "retrieval_confidence": confidence,
        "web_snippets": web_snippets,
    }


def generation_node(state: AgentState) -> dict:
    """Call LLM provider loop to generate answer from retrieved chunks."""
    query = state["query"]
    answer_length = state.get("answer_length", "medium")
    logger.info(f"[Generation Node] Generating answer for query: '{query}' | length='{answer_length}'")

    answer = generate_answer(
        query=query,
        hits=state.get("retrieved_docs", []),
        chat_history=state.get("chat_history", []),
        image_base64=state.get("image_base64"),
        web_snippets=state.get("web_snippets"),
        answer_length=answer_length,
    )

    if answer.startswith("Error:"):
        logger.error("[Generation Node] All generators failed")
    else:
        logger.success("[Generation Node] Answer generated successfully")

    return {"answer": answer}


def validation_node(state: AgentState) -> dict:
    """Evaluate generated answer using RAGAS-like metrics (faithfulness, relevancy, recall)."""
    query = state["query"]
    answer = state.get("answer", "")
    logger.info(f"[Validation Node] Scoring answer for query: '{query[:60]}...'")

    context_texts = []
    for hit in state.get("retrieved_docs", []):
        if isinstance(hit, dict):
            payload = hit.get("payload", {})
        else:
            payload = getattr(hit, "payload", {}) or {}
        metadata = payload.get("metadata", {})
        text = metadata.get("parent_text") or payload.get("text", "")
        if text:
            context_texts.append(text)

    result = validate_answer(query, answer, context_texts)

    verdict = "PASSED" if result.get("passed") else "FAILED"
    score = result.get("score", 0.0)
    logger.success(f"[Validation Node] Verdict: {verdict} | Score: {score:.3f}")
    return {"validation": result}


def support_node(state: AgentState) -> dict:
    """Answer non-RAG queries or fallback when retrieval/validation failed."""
    query = state["query"]
    logger.info(f"[Support Node] Handling support query: '{query[:60]}...'")

    result = answer_support_query(
        query,
        chat_history=state.get("chat_history", []),
        web_snippets=state.get("web_snippets", []),
        image_base64=state.get("image_base64"),
        needs_image=state.get("needs_image_in_answer"),
        answer_length=state.get("answer_length", "medium"),
    )

    answer = result.get("answer", "")
    source = result.get("source", "unknown")
    logger.success(f"[Support Node] Answered via '{source}': '{answer[:60]}...'")

    updated_history = list(state.get("chat_history", [])) + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer},
    ]

    ret = {
        "final_answer": answer,
        "chat_history": updated_history,
        "route": "support",
        "retrieved_image_path": result.get("retrieved_image_path"),
    }
    return ret


def final_node(state: AgentState) -> dict:
    """Finalize RAG answer by cleaning text, extracting citations, and saving chat history."""
    logger.info("[Final Node] Finalizing RAG answer")

    raw_answer = state.get("answer", "")
    parsed = _extract_json_response(raw_answer)
    clean_answer = _clean_answer_text(parsed.get("answer", raw_answer))

    logger.success(f"[Final Node] Answer finalized: '{clean_answer[:60]}...'")

    updated_history = list(state.get("chat_history", [])) + [
        {"role": "user", "content": state["query"]},
        {"role": "assistant", "content": clean_answer},
    ]

    return {
        "final_answer": clean_answer,
        "chat_history": updated_history,
    }
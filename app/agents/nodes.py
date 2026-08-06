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

    chat_hist = state.get("chat_history", [])
    result = route_query(normalized, chat_history=chat_hist, has_image=bool(image_b64))

    route = result["route"]
    scope = result.get("scope", "documents")
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

    logger.success(f"[Planner Node] Route: '{route}' | Scope: '{scope}' | Rewritten Query: '{rewritten}' | Atomic: {is_atomic}")

    ret = {
        "route": route,
        "scope": scope,
        "is_atomic": is_atomic,
        "query": rewritten,
        "answer_length": result.get("answer_length", "medium"),
        "retrieved_image_path": None,
        "retrieved_docs": [],
        "web_snippets": [],
        "answer": "",
        "final_answer": "",
    }
    ret["domain"] = result.get("domain", "ai_ml_technical")
    if image_b64:
        ret["needs_image_in_answer"] = True
    elif "needs_image" in result:
        ret["needs_image_in_answer"] = bool(result["needs_image"])
    else:
        ret["needs_image_in_answer"] = False
    return ret


def faq_node(state: AgentState) -> dict:
    """Legacy FAQ node stub (Routing now uses unified Qdrant retrieval via retrieval_node)."""
    return planner_node(state)


def retrieval_node(state: AgentState) -> dict:
    """Fetch relevant document chunks from Qdrant using hybrid search and reranking (+ parallel web search if flagged)."""
    query = state["query"]
    scope = state.get("scope", "documents")
    logger.info(f"[Retrieval Node] Fetching chunks for query: '{query}' | Scope: '{scope}'")

    web_snippets = []
    if state.get("needs_web_search"):
        logger.info(f"[Retrieval Node] Executing parallel web search for query: '{query[:50]}...'")
        from concurrent.futures import ThreadPoolExecutor
        from app.agents.web_search_tool import search_web
        with ThreadPoolExecutor() as executor:
            future_qdrant = executor.submit(retrieve, query, scope=scope)
            future_web = executor.submit(search_web, query)
            hits, analysis = future_qdrant.result()
            web_snippets = future_web.result()
    else:
        hits, analysis = retrieve(query, scope=scope)

    confidence = compute_retrieval_confidence(hits)

    needs_img = bool(state.get("needs_image_in_answer")) or any(
        w in query.lower() for w in ["figure", "diagram", "image", "illustration", "architecture", "plot", "schematic", "page"]
    )
    selected_img_path = None
    if needs_img:
        from app.generation.query_utils import select_relevant_image_path
        selected_img_path = select_relevant_image_path(query=query, hits=hits, needs_image=True)
        if selected_img_path:
            logger.info(f"[Retrieval Node] Selected image path: '{selected_img_path}'")

    logger.success(f"[Retrieval Node] Retrieved {len(hits)} chunks | Scope: '{scope}' | Web snippets: {len(web_snippets)} | Confidence: {confidence:.2f} | Image: {selected_img_path}")
    return {
        "retrieved_docs": hits,
        "retrieval_confidence": confidence,
        "web_snippets": web_snippets,
        "retrieved_image_path": selected_img_path,
    }


def generation_node(state: AgentState) -> dict:
    """Call LLM provider loop to generate answer from retrieved chunks."""
    import base64
    from pathlib import Path as _Path

    query = state["query"]
    answer_length = state.get("answer_length", "medium")
    logger.info(f"[Generation Node] Generating answer for query: '{query}' | length='{answer_length}'")

    # Use user-uploaded image first; fall back to retrieved RAG image so the
    # LLM can actually see (and describe) the retrieved figure.
    image_b64 = state.get("image_base64")
    if not image_b64:
        from app.generation.query_utils import resolve_existing_image_path
        rag_img_path = resolve_existing_image_path(state.get("retrieved_image_path"))
        if rag_img_path:
            try:
                image_b64 = base64.b64encode(_Path(rag_img_path).read_bytes()).decode("utf-8")
                logger.info(f"[Generation Node] Loaded retrieved RAG image as base64: {rag_img_path}")
            except Exception as _e:
                logger.warning(f"[Generation Node] Could not encode retrieved image: {_e}")

    answer = generate_answer(
        query=query,
        hits=state.get("retrieved_docs", []),
        chat_history=state.get("chat_history", []),
        image_base64=image_b64,
        web_snippets=state.get("web_snippets"),
        answer_length=answer_length,
    )

    if answer.startswith("Error:"):
        logger.error("[Generation Node] All generators failed")
    else:
        logger.success("[Generation Node] Answer generated successfully")

    return {"answer": answer}


def validation_node(state: AgentState) -> dict:
    """Evaluate generated answer using RAGAS-like metrics (correctness, relevancy, completeness, image relevancy)."""
    query = state["query"]
    answer = state.get("answer", "")
    retrieved_image_path = state.get("retrieved_image_path")
    logger.info(f"[Validation Node] Scoring answer for query: '{query[:60]}...' (retrieved_image_path={retrieved_image_path})")

    context_texts = []
    image_descriptions = []

    for hit in state.get("retrieved_docs", []):
        if isinstance(hit, dict):
            payload = hit.get("payload", {})
        else:
            payload = getattr(hit, "payload", {}) or {}
        metadata = payload.get("metadata", {}) or {}
        text = metadata.get("parent_text") or payload.get("text", "")
        ctype = (metadata.get("chunk_type") or "").lower()
        src = metadata.get("source_path") or ""

        if ctype == "image" and src:
            image_descriptions.append({
                "source_path": src,
                "description": text or metadata.get("caption") or "No description",
            })
        elif text:
            context_texts.append(text)

    # If an image was retrieved but no image chunks were in top-5 text context, fetch all image candidate descriptions from Qdrant
    if retrieved_image_path and not image_descriptions:
        try:
            from utils.models_and_clients import qdrant_client
            from utils.settings import COLLECTION_NAME
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            from pathlib import Path

            doc_folder = Path(retrieved_image_path).parent.parent.name
            img_filter = Filter(must=[FieldCondition(key="metadata.chunk_type", match=MatchValue(value="image"))])
            batch, _ = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=img_filter,
                limit=50,
                with_payload=True,
                with_vectors=False,
            )
            for pt in batch:
                meta = (pt.payload or {}).get("metadata", {}) or {}
                src = meta.get("source_path") or ""
                text = (pt.payload or {}).get("text") or meta.get("parent_text") or ""
                doc_id = str(meta.get("doc_id") or meta.get("document_name") or "")
                if src and (doc_folder in src or doc_folder in doc_id):
                    image_descriptions.append({
                        "source_path": src,
                        "description": text or f"Figure on page {meta.get('page_number', 'unknown')}",
                    })
        except Exception as exc:
            logger.warning(f"[Validation Node] Could not fetch image descriptions from Qdrant: {exc}")

    result = validate_answer(
        query=query,
        answer=answer,
        context_chunks=context_texts,
        retrieved_image_path=retrieved_image_path,
        image_descriptions=image_descriptions,
    )

    verdict = "PASSED" if result.get("passed") else "FAILED"
    score = result.get("score", 0.0)
    val_img = result.get("validated_image_path") or retrieved_image_path

    logger.success(f"[Validation Node] Verdict: {verdict} | Score: {score:.3f} | Validated Image: {val_img}")
    return {"validation": result, "retrieved_image_path": val_img}


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
        retrieved_docs=state.get("retrieved_docs", []),
    )

    answer = result.get("answer", "")
    source = result.get("source", "unknown")
    img_path = result.get("retrieved_image_path")
    logger.success(f"[Support Node] Answered via '{source}': '{answer[:60]}...'")

    assistant_msg = {"role": "assistant", "content": answer}
    if img_path:
        assistant_msg["image_path"] = img_path

    updated_history = list(state.get("chat_history", [])) + [
        {"role": "user", "content": query},
        assistant_msg,
    ]

    ret = {
        "final_answer": answer,
        "chat_history": updated_history,
        "route": "support",
        "retrieved_image_path": img_path,
    }
    return ret


def final_node(state: AgentState) -> dict:
    """Finalize RAG answer by cleaning text, extracting citations, and saving chat history."""
    logger.info("[Final Node] Finalizing RAG answer")

    raw_answer = state.get("answer", "")
    parsed = _extract_json_response(raw_answer)
    clean_answer = _clean_answer_text(parsed.get("answer", raw_answer))

    # Extract structured citations from retrieved docs
    citations = []
    seen = set()
    for hit in state.get("retrieved_docs", []):
        if isinstance(hit, dict):
            payload = hit.get("payload", {}) or {}
        else:
            payload = getattr(hit, "payload", {}) or {}
        meta = payload.get("metadata", {}) or {}
        doc_name = meta.get("document_name") or meta.get("doc_id")
        page_num = meta.get("page_number")
        chunk_id = meta.get("chunk_id")
        if doc_name and chunk_id and chunk_id not in seen:
            seen.add(chunk_id)
            citations.append({
                "document": doc_name,
                "page": page_num,
                "chunk_id": chunk_id,
            })

    logger.success(f"[Final Node] Answer finalized: '{clean_answer[:60]}...' | Citations: {len(citations)}")

    assistant_msg = {"role": "assistant", "content": clean_answer, "citations": citations}
    if state.get("retrieved_image_path"):
        assistant_msg["image_path"] = state.get("retrieved_image_path")

    updated_history = list(state.get("chat_history", [])) + [
        {"role": "user", "content": state["query"]},
        assistant_msg,
    ]

    return {
        "final_answer": clean_answer,
        "citations": citations,
        "chat_history": updated_history,
    }
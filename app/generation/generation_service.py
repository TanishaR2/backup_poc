import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from app.generation.query_utils import compute_retrieval_confidence
from utils.logger_config import logger
from utils.models_and_clients import (
    google_client,
    groq_client,
    openai_client,
)
from utils.settings import (
    AZURE_OPENAI_MODEL_NAME_5_4,
    RETRIEVAL_CONFIDENCE_THRESHOLD,
    RETRIEVAL_SKIP_VALIDATION_THRESHOLD,
    VALIDATION_CONFIDENCE_THRESHOLD,
)

from prompts.prompts import Generation_prompt

import threading

SESSION_MAX_TURNS = 10
_SESSION_STORE: dict = {}
_SESSION_LOCK = threading.Lock()

def _get_history(session_id: str) -> list:
    """Thread-safe read; returns a copy so callers cannot mutate internal store."""
    with _SESSION_LOCK:
        return list(_SESSION_STORE.get(session_id, []))

def _add_to_history(session_id: str, role: str, content: str) -> None:
    """Thread-safe append; enforces SESSION_MAX_TURNS cap by evicting oldest pairs."""
    with _SESSION_LOCK:
        history = _SESSION_STORE.setdefault(session_id, [])
        history.append({"role": role, "content": content})
        max_msgs = SESSION_MAX_TURNS * 2
        if len(history) > max_msgs:
            _SESSION_STORE[session_id] = history[-max_msgs:]

SESSION_HISTORIES = _SESSION_STORE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
def _ensure_chat_dir() -> Path:
    base = Path(OUTPUT_DIR) / "chat_conversations"
    date_dir = base / datetime.now().strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    return date_dir

def _image_only_vision(image_base64: str) -> str:
    """Send image directly to vision model using central LLM failover chain."""
    from prompts.prompts import Vision_standalone_prompt
    vision_prompt = Vision_standalone_prompt

    from utils.models_and_clients import run_llm_completion
    res = run_llm_completion(prompt=vision_prompt, image_base64=image_base64, temperature=0.0)
    if res and res.strip():
        return res.strip()

    return "Error: Could not process the image."


def _extract_json_response(text: str) -> dict:
    """Robustly extract JSON from LLM response that may be wrapped in
    markdown code fences or contain extra text around the JSON."""
    cleaned = text.strip()

    # Strip markdown code fences: ```json ... ``` or ``` ... ```
    fence_pattern = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
    match = fence_pattern.search(cleaned)
    if match:
        cleaned = match.group(1).strip()

    # Try direct parse first
    try:
        res = json.loads(cleaned)
        if isinstance(res, dict):
            return res
    except Exception:
        pass

    # Try to find a JSON object { ... } in the text
    brace_start = cleaned.find("{")
    brace_end = cleaned.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        json_candidate = cleaned[brace_start:brace_end + 1]
        try:
            res = json.loads(json_candidate)
            if isinstance(res, dict):
                return res
        except Exception:
            pass

    # All parsing failed — treat the entire text as the answer
    return {"answer": text.strip(), "citations": []}


def _clean_answer_text(answer: str) -> str:
    """Strip any Source Paths, Description Paths, and trailing Citations
    that the LLM may have embedded in the answer text despite instructions."""
    # Remove lines starting with "Source Path:" or "Description Path:"
    answer = re.sub(r'(?:Source Path|Description Path):\s*\S+.*', '', answer)
    # Remove trailing "Citations:" block (bullet lists of chunk IDs)
    answer = re.sub(r'\n*Citations?:?\s*\n(\s*[\*\-•]\s*.*\n?)+', '', answer, flags=re.IGNORECASE)
    # Remove standalone chunk_id-like references
    answer = re.sub(r'\n*\*\s+\S+__(?:image|text|table)__\d+.*', '', answer)
    # Clean up excessive whitespace at end
    return answer.strip()


def generate_answer(
    query: str,
    hits: list,
    chat_history: list = None,
    image_base64: str = None,
    web_snippets: list = None,
    answer_length: str = "medium",
) -> str:
    """Build context from retrieved hits and call the LLM to generate an answer.

    Tries providers in order: openai -> groq -> google.
    Supports multimodal queries via image_base64.
    Returns the raw LLM text (not yet cleaned).
    """
    # Build the answer-length instruction string for the prompt
    _length_instructions = {
        "short": (
            "This is a SHORT answer request. Give a concise, direct response in 1-3 sentences. "
            "Skip all preamble, elaborate introductions, and filler content."
        ),
        "medium": (
            "This is a MEDIUM-LENGTH answer request. Provide a well-organized response with clear structure. "
            "Cover the key points in sufficient depth without going into exhaustive detail."
        ),
        "detailed": (
            "This is a DETAILED answer request. Provide a comprehensive, thorough response with all relevant sub-points, "
            "technical specifics, numerical values, equations, and inter-chunk synthesis. "
            "Do not skip important context that is available in the retrieved chunks."
        ),
    }
    answer_length_instruction = _length_instructions.get(answer_length, _length_instructions["medium"])

    context_parts = []
    for rank, hit in enumerate(hits, start=1):
        metadata = hit.payload.get("metadata", {})
        context_parts.append(
            f"""
            ==============================
            Retrieved Chunk {rank}

            Relevance Rank : {rank}
            Relevance Score: {metadata.get("rerank_score")}

            Document   : {metadata.get("document_name")}
            Doc ID     : {metadata.get("doc_id")}
            Chunk ID   : {metadata.get("chunk_id")}
            Chunk Type : {metadata.get("chunk_type")}
            Page       : {metadata.get("page_number")}
            Source Path: {metadata.get("source_path", "")}
            Description Path: {metadata.get("description_path", "")}

            Content:
            {metadata.get("parent_text") or hit.payload.get("text", "")}
            """
        )

    context = "\n\n-------------------------\n\n".join(context_parts)
    if web_snippets:
        web_text = "\n".join([f"- {s}" for s in web_snippets])
        context += f"\n\n-------------------------\n\nLive Internet Search Validation Context:\n{web_text}"

    prompt = Generation_prompt.format(
        query=query,
        context=context,
        answer_length_instruction=answer_length_instruction,
    )

    if chat_history:
        history_text = "\n".join(
            [f"{msg['role'].upper()}: {msg['content'][:200]}" for msg in chat_history[-5:]]
        )
        prompt = (
            "PRIOR CONVERSATION (resolve pronouns/references only — NOT a fact source):\n"
            f"{history_text}\n\n"
            "IMPORTANT: Do NOT use any prior answer as evidence. Use ONLY the Retrieved Context below.\n\n"
        ) + prompt

    logger.info(f"[generate_answer] Generating answer for query: '{query[:60]}' | length='{answer_length}'")

    from utils.models_and_clients import run_llm_completion
    response_text = run_llm_completion(prompt=prompt, image_base64=image_base64, temperature=0.0)

    if not response_text:
        logger.error("[generate_answer] All models failed in MODEL_FALLBACK_CHAIN")
        return "Error: All models failed to generate a response."

    parsed = _extract_json_response(response_text)
    clean_answer = _clean_answer_text(parsed.get("answer", response_text.strip()))
    return clean_answer



def qna(query: str, rag_mode = True, image_base64: str = None, session_id: str = "default") -> dict:
    """Run retrieval -> generate answer -> log chat."""
    start = time.time()
    logger.info(f"[Pipeline Start] Received query: '{query}'")
    retrieval_latency_s = None
    normalized_query = normalize_query_text(query)
    history = _get_history(session_id)
    history_text = ""
    if history:
        history_text = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in history[-5:]])

    # Image-only: skip RAG, go directly to vision model
    if (not query or not query.strip()) and image_base64:
        logger.info("Image-only query — using direct vision model (no RAG)")
        response_text = _image_only_vision(image_base64)
        parsed = _extract_json_response(response_text)
        answer = _clean_answer_text(parsed.get("answer", response_text.strip()))
        citations = parsed.get("citations", [])
        latency = time.time() - start
        record = {
            "timestamp": datetime.now().isoformat(),
            "query": "[Image-only query]",
            "answer": answer,
            "citations": citations,
            "latency": latency,
            "retrieval_latency_s": None,
            "chunks": None,
            "route": "support",
            "planner_decision": {"route": "support"},
            "retrieval_confidence": None,
            "validation_result": {"score": 1.0, "passed": True},
        }
        chat_dir = _ensure_chat_dir()
        fname = chat_dir / f"{datetime.now().strftime('%H%M%S')}_chat.json"
        fname.write_text(json.dumps(record, indent=2), encoding="utf-8")
        logger.success(f"[generate_answer] Answer generated ({len(answer)} chars)")
        return {"record": record, "merged_analysis": {}, **record}

    planner_decision = route_query(normalized_query)

    if (not rag_mode or planner_decision.get("route") == "support") and not image_base64:
        web_snippets = []
        if planner_decision.get("needs_web_search"):
            from app.agents.web_search_tool import search_web
            web_snippets = search_web(normalized_query)

        support_result = answer_support_query(
            normalized_query,
            chat_history=history,
            web_snippets=web_snippets,
            answer_length=planner_decision.get("answer_length", "medium"),
        )
        response_text = support_result["answer"]
        chunks = None
        latency = time.time() - start
        record = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "answer": response_text,
            "latency": latency,
            "retrieval_latency_s": retrieval_latency_s,
            "chunks": chunks,
            "route": "support",
            "planner_decision": planner_decision,
        }
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": response_text})
        chat_dir = _ensure_chat_dir()
        fname = chat_dir / f"{datetime.now().strftime('%H%M%S')}_chat.json"
        fname.write_text(json.dumps(record, indent=2), encoding="utf-8")
        logger.success(f"[qna] Answer generated ({len(response_text)} chars)")
        return {"record": record, "merged_analysis": {}, **record}

    retrieval_start = time.time()
    logger.info(f"[Retriever] Fetching chunks for query: '{normalized_query}'")
    ret = retrieve(normalized_query)
    if isinstance(ret, tuple) and len(ret) == 2:
        hits, analysis = ret
    else:
        hits, analysis = ret, []
    retrieval_latency_s = time.time() - retrieval_start

    retrieval_confidence = compute_retrieval_confidence(hits)
    logger.success(f"[Retriever] Fetched {len(hits)} chunks. Retrieval Confidence: {retrieval_confidence:.2f} (Threshold: {RETRIEVAL_CONFIDENCE_THRESHOLD})")

    if retrieval_confidence < RETRIEVAL_CONFIDENCE_THRESHOLD and not image_base64:
        logger.warning(f"[Pipeline Check] Retrieval confidence {retrieval_confidence:.2f} < {RETRIEVAL_CONFIDENCE_THRESHOLD}. Falling back to Support Agent.")
        web_snippets = []
        if planner_decision.get("needs_web_search"):
            from app.agents.web_search_tool import search_web
            web_snippets = search_web(normalized_query)

        support_result = answer_support_query(
            f"The retrieved context was too weak for this query: {normalized_query}",
            chat_history=history,
            web_snippets=web_snippets,
            answer_length=planner_decision.get("answer_length", "medium"),
        )
        response_text = support_result["answer"]
        chunks = None
        latency = time.time() - start
        record = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "answer": response_text,
            "latency": latency,
            "retrieval_latency_s": retrieval_latency_s,
            "chunks": chunks,
            "route": "support",
            "planner_decision": planner_decision,
            "retrieval_confidence": retrieval_confidence,
        }
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": response_text})
        chat_dir = _ensure_chat_dir()
        fname = chat_dir / f"{datetime.now().strftime('%H%M%S')}_chat.json"
        fname.write_text(json.dumps(record, indent=2), encoding="utf-8")
        logger.success(f"[qna] Fallback answer generated ({len(response_text)} chars)")
        return {"record": record, "merged_analysis": {}, **record}

    context_parts = []
    for rank, hit in enumerate(hits, start=1):
        metadata = hit.payload.get("metadata", {})
        context_parts.append(
            f"""
            ==============================
            Retrieved Chunk {rank}

            Relevance Rank : {rank}
            Relevance Score: {metadata.get("rerank_score")}

            Document   : {metadata.get("document_name")}
            Doc ID     : {metadata.get("doc_id")}
            Chunk ID   : {metadata.get("chunk_id")}
            Chunk Type : {metadata.get("chunk_type")}
            Page       : {metadata.get("page_number")}
            Source Path: {metadata.get("source_path", "")}
            Description Path: {metadata.get("description_path", "")}

            Content:
            {metadata.get("parent_text") or hit.payload.get("text", "")}
            """
        )

    context = "\n\n-------------------------\n\n".join(context_parts)
    prompt = Generation_prompt.format(query=normalized_query, context=context)
    if history_text:
        prompt = (
            "PRIOR CONVERSATION (resolve pronouns/references only — NOT a fact source):\n"
            f"{history_text}\n\n"
            "IMPORTANT: Do NOT use any prior answer as evidence. Use ONLY the Retrieved Context below.\n\n"
        ) + prompt

    response_text = None
    logger.info(f"[qna] Generating answer for query: '{query[:60]}...'")
    for name in ["openai", "groq", "google"]:
        client = generator_clients.get(name)
        if client is None:
            continue

        try:
            logger.info(f"Trying generator: {name}")
            if name == "groq":
                if image_base64:
                    logger.info("Groq does not support multimodal vision queries; skipping.")
                    continue
                resp = client.chat.completions.create(model=MODELS[0], messages=[{"role":"user","content":prompt}], temperature=0)
                response_text = resp.choices[0].message.content
            elif name == "google":
                if image_base64:
                    import base64
                    from google.genai import types
                    image_bytes = base64.b64decode(image_base64)
                    resp = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=[
                            prompt,
                            types.Part.from_bytes(data=image_bytes, mime_type="image/png")
                        ]
                    )
                else:
                    resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
                response_text = getattr(resp, "text", None) or resp["text"]
            elif name == "openai":
                if image_base64:
                    resp = client.chat.completions.create(
                        model=AZURE_OPENAI_MODEL_NAME_5_4,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{image_base64}"
                                        }
                                    }
                                ]
                            }
                        ]
                    )
                else:
                    resp = client.chat.completions.create(model=AZURE_OPENAI_MODEL_NAME_5_4, messages=[{"role":"user","content":prompt}])
                response_text = resp.choices[0].message.content
                usage = resp.usage

                logger.info(
                    {
                        "provider": "azure_openai",
                        "model": AZURE_OPENAI_MODEL_NAME_5_4,
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens,
                        "reasoning_tokens": usage.completion_tokens_details.reasoning_tokens,
                        "latency_ms": usage.latency_checkpoint["total_duration_ms"],
                        "ttft_ms": usage.latency_checkpoint["service_ttft_ms"],
                    }
                )

            if response_text:
                break

        except Exception as e:
            logger.error(f"{name} failed: {e}")

    if response_text is None:
        logger.error("All generators failed")
        response_text = "Error: All models failed to generate a response."



    validation_result = None
    route = "rag"
    if retrieval_confidence >= RETRIEVAL_SKIP_VALIDATION_THRESHOLD or image_base64:
        logger.info(f"[Validation Check] Skipping validation. Confidence {retrieval_confidence:.2f} >= Skip Threshold {RETRIEVAL_SKIP_VALIDATION_THRESHOLD} or Image provided.")
        validation_result = {"score": 1.0, "passed": True, "metrics": {"faithfulness": 1.0, "answer_relevancy": 1.0, "context_recall": 1.0}}
    else:
        logger.info("[Validation Check] Retrieval confidence below skip threshold. Invoking Validation Agent...")
        validation_result = validate_answer(query, response_text, [hit.payload.get("metadata", {}).get("parent_text") or hit.payload.get("text", "") for hit in hits])
        if not validation_result.get("passed", False):
            logger.warning(f"[Validation Check] Generated answer failed validation. Score {validation_result.get('score')} < {VALIDATION_CONFIDENCE_THRESHOLD}. Falling back to Support Agent.")
            support_result = answer_support_query(
                f"The generated answer was below the validation threshold for: {normalized_query}",
                chat_history=history,
            )
            response_text = support_result["answer"]
            route = "support"


    parsed_response = _extract_json_response(response_text)
    answer = _clean_answer_text(parsed_response.get("answer", response_text.strip()))
    _add_to_history(session_id, "user", query)
    _add_to_history(session_id, "assistant", answer)
    citations = parsed_response.get("citations", [])
    chunks = [
        {
            "score": h.score,
            "text": h.payload.get("metadata", {}).get("parent_text") or h.payload.get("text"),
            "doc_id": h.payload.get("metadata", {}).get("doc_id"),
            "document_name": h.payload.get("metadata", {}).get("document_name"),
            "chunk_id": h.payload.get("metadata", {}).get("chunk_id"),
            "chunk_type": h.payload.get("metadata", {}).get("chunk_type"),
            "page_number": h.payload.get("metadata", {}).get("page_number"),
            "source_type": h.payload.get("metadata", {}).get("source_type"),
            "source_path": h.payload.get("metadata", {}).get("source_path"),
            "description_path": h.payload.get("metadata", {}).get("description_path"),
        }
        for h in hits
    ]

    latency = time.time() - start
    record = {
        "timestamp": datetime.now().isoformat(),
        "query": normalized_query,
        "answer": answer,
        "citations": citations,
        "latency": latency,
        "retrieval_latency_s": retrieval_latency_s,
        "chunks": chunks,
        "route": route,
        "planner_decision": planner_decision,
        "retrieval_confidence": retrieval_confidence,
        "validation_result": validation_result,
    }

    merged_analysis = {
        "retrieval_analysis": analysis,
        "generated_answer_analysis": {
            "answer": answer,
            "citations": citations
        }
    }
    chat_dir = _ensure_chat_dir()
    fname = chat_dir / f"{datetime.now().strftime('%H%M%S')}_chat.json"
    fname.write_text(json.dumps(record, indent=2), encoding="utf-8")

    logger.success(f"Answer: {answer}")

    return {
        "record": record,
        "merged_analysis": merged_analysis,
        **record
    }

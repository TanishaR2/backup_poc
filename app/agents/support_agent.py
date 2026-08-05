"""Support agent for handling general chat, domain boundaries, and parametric fallback responses."""


import json
import re
from typing import Any
from pathlib import Path
from app.generation.query_utils import should_request_image

from utils.logger_config import logger


def _is_greeting_or_identity(query: str) -> bool:
    lowered = re.sub(r"[^\w\s]", "", query.strip().lower()).strip()
    words = lowered.split()
    if not words:
        return True

    greetings = {
        "hello", "hi", "hey", "greetings", "good morning", "good afternoon",
        "who are you", "what is your name", "what can you do", "help", "hi there", "hello there"
    }
    if lowered in greetings:
        return True

    technical_keywords = {
        "image", "architecture", "cnn", "pooling", "transformer", "bert", "model",
        "loss", "paper", "explain", "show", "give", "diagram", "layer", "attention",
        "resnet", "gan", "diffusion", "rag", "eval", "train", "dataset", "framework",
        "what", "how", "why", "describe"
    }

    if any(w in words for w in technical_keywords):
        return False

    if len(words) <= 4 and words[0] in {"hi", "hello", "hey", "greetings"}:
        return True

    return False


def answer_support_query(
    query: str,
    llm_response: str | None = None,
    chat_history: list = None,
    web_snippets: list[str] | None = None,
    image_base64: str | None = None,
    needs_image: bool | None = None,
    answer_length: str = "medium",
    retrieved_docs: list[Any] | None = None,
) -> dict[str, Any]:
    """Answer support, greeting, or out-of-domain queries with strict boundary enforcement and multimodal vision capabilities."""
    logger.info(f"[Support Agent] Input query: '{query}' (image_attached={bool(image_base64)})")

    # 1. Greeting or Assistant Identity (only if no image is attached and is a pure greeting)
    if not image_base64 and _is_greeting_or_identity(query):
        answer = (
            "Hello! I am InSightDocs, your technical research assistant specialized in "
            "Artificial Intelligence, Machine Learning, Deep Learning, and GenAI research papers.\n\n"
            "* Ask about a specific architecture (e.g. *\"What is the Transformer architecture?\"*)\n"
            "* Ask about model training (e.g. *\"How does LoRA fine-tuning work?\"*)\n"
            "* Upload a research paper PDF using the sidebar to analyze its content!"
        )
        logger.success("[Support Agent] Handled greeting query")
        return {"answer": answer, "source": "domain_greeting"}

    history_text = ""
    if chat_history:
        history_text = "\n".join([f"{msg['role'].upper()}: {msg['content'][:200]}" for msg in chat_history[-5:]])

    from prompts.prompts import Support_image_instruction, Support_text_instruction

    # Check if query references a specific arXiv paper ID (e.g. 2606.15207) directly from retrieved_docs
    arxiv_match = re.search(r"\b(\d{4}\.\d{4,5})\b", query)
    paper_context_note = ""
    if arxiv_match:
        target_arxiv_id = arxiv_match.group(1)
        found_doc_id = None
        if retrieved_docs:
            for doc in retrieved_docs:
                if isinstance(doc, dict):
                    meta = doc.get("metadata", {}) or doc.get("payload", {}).get("metadata", {}) or {}
                else:
                    meta = (getattr(doc, "payload", {}) or {}).get("metadata", {}) or {}
                doc_id = str(meta.get("doc_id") or meta.get("document_name") or "")
                if target_arxiv_id in doc_id:
                    found_doc_id = doc_id
                    break

        if found_doc_id:
            clean_title = found_doc_id.replace("_", " ")
            paper_context_note = (
                f"\n\nNOTE: The research paper '{clean_title}' (arXiv ID: {target_arxiv_id}) IS INDEXED in the system database. "
                f"However, the requested figure or page image was not found on that specific page in the paper. "
                f"Do NOT say 'the paper is not indexed'. Instead, politely state that paper '{clean_title}' is indexed in the system, "
                f"but does not contain that figure on that page, and ask if the user would like details about the paper's methodology or available figures.\n"
            )
        else:
            paper_context_note = (
                f"\n\nNOTE: The research paper (arXiv ID: {target_arxiv_id}) is NOT currently indexed in the vector database. "
                f"Politely clarify that paper {target_arxiv_id} is not indexed and answer based on general AI/ML parametric knowledge.\n"
            )

    if image_base64:
        prompt = Support_image_instruction + paper_context_note + "\n"
    else:
        prompt = Support_text_instruction + paper_context_note + "\n"

    if answer_length == "short":
        prompt += (
            "This is a SHORT answer request. Give a concise, direct response in 1-3 sentences. "
            "Skip preamble, filler, and verbose explanations.\n\n"
        )
    elif answer_length == "detailed":
        prompt += (
            "This is a DETAILED answer request. Provide a comprehensive explanation with all relevant sub-points.\n\n"
        )
    else:
        prompt += (
            "This is a MEDIUM-LENGTH answer request. Provide a clear, well-structured explanation without excessive detail.\n\n"
        )

    if web_snippets:
        prompt += (
            "Use the following live web search snippets only if they are relevant to the user's question. "
            "Answer based on facts from the snippets and do not hallucinate new information.\n\n"
            "Live Web Search Snippets:\n"
        )
        for snippet in web_snippets:
            prompt += f"- {snippet}\n"
        prompt += "\n"

    if history_text:
        prompt += f"Prior Conversation:\n{history_text}\n\n"

    effective_query = query.strip() if (query and query.strip()) else "Describe and analyze this attached image in detail."
    prompt += f"User Question: {effective_query}"

    from utils.models_and_clients import run_llm_completion
    text = run_llm_completion(prompt=prompt, image_base64=image_base64, temperature=0.0)

    if text and text.strip():
        answer = text.strip()
        logger.success(f"[Support Agent] Answered: '{answer[:60]}...'")
        result = {"answer": answer, "source": "llm_knowledge"}

        # Image Acquisition for Support Agent
        if not image_base64 and (needs_image or should_request_image(query)):
            logger.info(f"[Support Agent] Image requested for query '{query}' — attempting web image search first")
            try:
                from app.agents.web_search_tool import fetch_web_image_for_query
                from app.generation.query_utils import generate_support_diagram_image

                web_img = fetch_web_image_for_query(query)
                if web_img and Path(web_img).exists():
                    result["retrieved_image_path"] = web_img
                    logger.success(f"[Support Agent] Successfully fetched web image: {web_img}")
                else:
                    logger.info("[Support Agent] Web image search unavailable — using SVG diagram generator fallback")
                    generated = generate_support_diagram_image(query)
                    if generated and Path(generated).exists():
                        result["retrieved_image_path"] = generated
                        logger.success(f"[Support Agent] Generated fallback diagram image: {generated}")
            except Exception as exc:
                logger.exception(f"[Support Agent] Image acquisition failed: {exc}")

        return result

    logger.warning("[Support Agent] Falling back to standard domain boundary statement")
    return {
        "answer": "I am InSightDocs, an assistant specialized strictly in AI, Machine Learning, and Deep Learning research papers. Please ask an AI/ML-related technical question.",
        "source": "domain_fallback",
    }

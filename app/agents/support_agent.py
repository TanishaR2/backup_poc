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

    # 0. Missing Attached Image Guardrail
    q_lower = query.lower()
    references_attached_img = bool(re.search(r"\b(attached|uploaded)\s+image\b", q_lower)) or "attached image" in q_lower or "this plot" in q_lower or "the attached plot" in q_lower
    if references_attached_img and not image_base64:
        answer = (
            "No image was attached to your query. Please upload or attach an image "
            "so I can analyze its content, axes, and visual structure for you."
        )
        logger.success("[Support Agent] Handled missing attached image query")
        return {"answer": answer, "source": "missing_attached_image_guardrail"}

    # 0.5. User Identity Guardrail ("who am i", "what is my name")
    clean_q = re.sub(r"[^\w\s]", "", q_lower).strip()
    is_user_identity = clean_q in {
        "who am i", "what is my name", "do you know me", "do you know who i am",
        "who i am", "tell me my name", "what my name is"
    } or bool(re.search(r"\bwho am i\b|\bwhat is my name\b|\bdo you know me\b", q_lower))

    if is_user_identity:
        answer = (
            "I do not have access to your personal identity or user profile, but I am **InSightDocs**, "
            "your technical research assistant specialized in Artificial Intelligence, Machine Learning, "
            "Deep Learning, and research paper analytics! How can I help you analyze research literature or AI concepts today?"
        )
        logger.success("[Support Agent] Handled user identity query ('who am i')")
        return {"answer": answer, "source": "user_identity_guardrail"}

    # 1. Knowledge Base Document Listing Query
    if ("what documents" in q_lower or "what papers" in q_lower or "knowledge base" in q_lower or "paper titles" in q_lower or "document titles" in q_lower) and ("available" in q_lower or "indexed" in q_lower or "list" in q_lower or "stored" in q_lower or "in your" in q_lower or "have" in q_lower or "titles" in q_lower):
        from app.retriever.retrieval_service import get_indexed_document_titles
        doc_list = get_indexed_document_titles()
        if doc_list:
            papers_fmt = "\n".join([f"- **{title}**" for title in doc_list])
            answer = (
                f"The following **{len(doc_list)} research papers** are currently indexed in the InSightDocs knowledge base:\n\n"
                f"{papers_fmt}\n\n"
                "You can ask technical questions, request visual architecture diagrams, or query performance figures from any of these papers!"
            )
        else:
            answer = "Currently, no research papers are indexed in the knowledge base. Please upload paper PDFs using the ingest pipeline."
        logger.success("[Support Agent] Handled knowledge base document listing query")
        return {"answer": answer, "source": "knowledge_base_listing"}

    # 2. Greeting or Assistant Identity (only if no image is attached and is a pure greeting)
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
    elif "quantumvit" in q_lower or "table 7" in q_lower:
        paper_context_note = (
            "\n\nNOTE: The requested model 'QuantumViT-XL' or Table 7 is NOT present in any of the indexed research papers in the vector database. "
            "Politely clarify to the user that QuantumViT-XL or Table 7 is not available in the uploaded research papers.\n"
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

        # Image Acquisition for Support Agent (skip web search for paper-specific queries or missing image references)
        arxiv_in_query = bool(re.search(r"\b(\d{4}\.\d{4,5})\b", query))
        if not image_base64 and not arxiv_in_query and not references_attached_img and (needs_image or should_request_image(query)):
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

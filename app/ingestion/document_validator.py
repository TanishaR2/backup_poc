"""Pre-ingestion validation for document format, size, page count, and domain scope."""

from pathlib import Path
import fitz

from utils.logger_config import logger
from utils.models_and_clients import run_llm_completion

MAX_FILE_SIZE_MB = 10.0
MAX_PAGE_COUNT = 100


def validate_document_for_ingestion(pdf_path: Path | str) -> dict:
    """Validate PDF file extension, file size, page count, and AI/ML domain scope."""
    pdf_path = Path(pdf_path)
    logger.info(f"[Document Validator] Validating document: {pdf_path.name}")

    if not pdf_path.exists():
        logger.error(f"[Document Validator] File not found: {pdf_path}")
        return {"valid": False, "reason": f"File '{pdf_path.name}' does not exist."}

    # 1. File Type Safeguard
    if pdf_path.suffix.lower() != ".pdf":
        msg = f"Invalid file type '{pdf_path.suffix}'. Only PDF files (.pdf) are allowed."
        logger.warning(f"[Document Validator] Rejected: {msg}")
        return {"valid": False, "reason": msg}

    # 2. File Size Safeguard
    size_mb = pdf_path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        msg = f"File size ({size_mb:.2f} MB) exceeds maximum allowed limit of {MAX_FILE_SIZE_MB} MB."
        logger.warning(f"[Document Validator] Rejected: {msg}")
        return {"valid": False, "reason": msg}

    # 3. Page Count Safeguard
    try:
        doc = fitz.open(str(pdf_path))
        page_count = len(doc)
    except Exception as exc:
        msg = f"Could not parse PDF pages: {exc}"
        logger.error(f"[Document Validator] Rejected: {msg}")
        return {"valid": False, "reason": msg}

    if page_count > MAX_PAGE_COUNT:
        doc.close()
        msg = f"Page count ({page_count} pages) exceeds maximum allowed limit of {MAX_PAGE_COUNT} pages."
        logger.warning(f"[Document Validator] Rejected: {msg}")
        return {"valid": False, "reason": msg}

    # 4. LLM AI/ML/DL Domain Scope Validation
    sample_text = ""
    for i in range(min(3, page_count)):
        text = doc[i].get_text("text") or ""
        sample_text += text + "\n"

    doc.close()
    sample_snippet = sample_text[:1500].strip()

    if not sample_snippet:
        msg = "PDF text content is empty or unreadable."
        logger.warning(f"[Document Validator] Rejected: {msg}")
        return {"valid": False, "reason": msg}

    from prompts.prompts import Document_domain_prompt

    prompt = Document_domain_prompt.format(sample_snippet=sample_snippet)

    llm_res = run_llm_completion(prompt=prompt, temperature=0.0)

    if llm_res:
        import json, re
        try:
            match = re.search(r"\{.*\}", llm_res, re.S)
            parsed = json.loads(match.group(0)) if match else json.loads(llm_res)
            is_ai = bool(parsed.get("is_ai_domain", True))
            reason = parsed.get("reason", "")

            if not is_ai:
                msg = f"Document content is outside AI/ML research domain. ({reason})"
                logger.warning(f"[Document Validator] Rejected: {msg}")
                return {"valid": False, "reason": msg}
        except Exception:
            pass

    logger.success(f"[Document Validator] Document '{pdf_path.name}' passed all safeguards ({page_count} pages, {size_mb:.2f} MB)")
    return {"valid": True, "page_count": page_count, "size_mb": round(size_mb, 2)}

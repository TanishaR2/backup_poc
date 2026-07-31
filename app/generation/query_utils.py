import re
from pathlib import Path
from typing import Any, Iterable


def normalize_query_text(query: str) -> str:
    """Normalize common typing mistakes."""
    if not query:
        return query
    normalized = query.strip()
    normalized = re.sub(r"\bth\b", "the", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bteh\b", "the", normalized, flags=re.IGNORECASE)
    return normalized.strip()


def compute_retrieval_confidence(hits: Iterable[Any]) -> float:
    """Compute retrieval confidence as the average score of top 2 hits."""
    scores = []
    for hit in hits:
        if isinstance(hit, dict):
            payload = hit.get("payload", {})
            metadata = payload.get("metadata", {})
            score = metadata.get("rerank_score") if metadata.get("rerank_score") is not None else hit.get("score")
        else:
            payload = getattr(hit, "payload", {}) or {}
            metadata = payload.get("metadata", {})
            score = metadata.get("rerank_score") if metadata.get("rerank_score") is not None else getattr(hit, "score", None)

        if score is not None:
            try:
                scores.append(float(score))
            except (TypeError, ValueError):
                pass

    if not scores:
        return 0.0

    top_scores = scores[:2]
    return sum(top_scores) / len(top_scores)


def should_request_image(query: str) -> bool:
    """Return True if the query explicitly asks to view/display/show a visual diagram or figure."""
    if not query or not query.strip():
        return False

    lowered = re.sub(r"[^a-z0-9\s]", " ", query.lower()).strip()
    if not lowered:
        return False

    visual_patterns = [
        r"\b(show|give|display|provide|render|draw|illustrate|visualize|view)\b.*?\b(image|figure|fig|diagram|chart|plot|picture|architecture|cell|model|structure)\b",
        r"\b(image|figure|fig|diagram|chart|plot|picture|architecture|cell|model|structure)\b.*?\b(of|showing|displaying|for)\b",
    ]
    for pattern in visual_patterns:
        if re.search(pattern, lowered):
            return True

    visual_keywords = {"image", "figure", "fig", "diagram", "chart", "plot", "picture", "photo", "architecture", "visualize", "visualization"}
    words = set(lowered.split())
    if words & visual_keywords:
        return True

    return False


def _extract_hit_context(hit: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(hit, dict):
        payload = hit.get("payload", {}) or {}
        text = payload.get("text", "") or hit.get("text", "") or ""
        metadata = payload.get("metadata", {}) or {}
        return text, metadata

    payload = getattr(hit, "payload", {}) or {}
    metadata = payload.get("metadata", {}) or {}
    return payload.get("text", "") or "", metadata


def select_relevant_image_path(query: str, hits: Iterable[Any], needs_image: bool | None = None) -> str | None:
    """Select the most relevant image path for the query.
    
    1. Collects image candidates directly from retrieved hits or Qdrant points.
    2. Matches candidate text/description keywords against query terms.
    3. Returns the matching image source_path if available and valid.
    """
    if not query or not query.strip():
        return None
    if needs_image is False:
        return None
    if needs_image is not True and not should_request_image(query):
        return None

    # Standard English filler words only — do NOT strip domain terms (like network, layer, cell, model, encoder)
    stop_words = {
        "the", "this", "that", "these", "those", "show", "me", "what", "is", "are", "a", "an", "of", "for",
        "can", "you", "display", "illustrate", "visualize", "figure", "fig", "diagram",
        "chart", "plot", "image", "picture", "photo", "in", "on", "at", "to", "with", "and", "or",
        "give", "get", "send", "provide", "fetch", "find", "look", "view", "return", "about",
        "please", "tell", "want", "same", "also", "explain"
    }

    query_terms = {
        w for w in re.findall(r"[a-z0-9]+", query.lower())
        if w not in stop_words and len(w) >= 2
    }

    candidates = []

    # 1. Gather image chunks from retrieved hits (from vector search)
    if hits:
        for hit in hits:
            text, metadata = _extract_hit_context(hit)
            chunk_type = (metadata.get("chunk_type") or "").lower()
            source_path = metadata.get("source_path") or ""
            if chunk_type == "image" and source_path:
                text_terms = set(re.findall(r"[a-z0-9]+", text.lower()))
                overlap = len(query_terms & text_terms) if query_terms else 1

                # If query asked for a specific architecture (e.g. rnn, lstm, transformer, cross, encoder), require model keyword alignment
                specific_models = {"rnn", "lstm", "gru", "transformer", "cross", "bert", "gpt", "resnet", "vgg", "unet", "diffusion"}
                requested_models = query_terms & specific_models
                if requested_models:
                    if not (text_terms & requested_models):
                        overlap = 0

                # Guard against attaching evaluation benchmark plots (MSE, loss, scenario) when an architecture diagram is requested
                is_diag_request = any(w in query.lower() for w in ["diagram", "architecture", "schematic", "structure", "cell", "model", "image of", "architecture image"])
                is_eval_plot = any(w in text.lower() for w in ["mse", "percentage of sequence kept", "ablation", "scenario 1", "accuracy vs"]) 
                # If user explicitly requests an architecture/diagram, require diagram-related keywords in the image text
                if is_diag_request:
                    diag_keywords = {"architecture", "diagram", "schematic", "encoder", "decoder", "stack", "layer", "pipeline", "flow", "block", "figure"}
                    text_terms = set(re.findall(r"[a-z0-9]+", text.lower()))
                    if not (text_terms & diag_keywords):
                        overlap = 0
                # Avoid selecting evaluation plots when an architecture diagram is requested
                if is_diag_request and is_eval_plot:
                    overlap = 0

                score = float(metadata.get("rerank_score") or getattr(hit, "score", 0.0) or 0.0)
                candidates.append({
                    "source_path": source_path,
                    "overlap": overlap,
                    "score": score,
                })

    # 2. If hits have no image chunks, scroll Qdrant for image chunks
    if not candidates:
        try:
            from utils.models_and_clients import qdrant_client
            from utils.settings import COLLECTION_NAME
            res = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                limit=200,
                with_payload=True,
                with_vectors=False,
            )
            scrolled_hits = res[0] if res else []
            for hit in scrolled_hits:
                text, metadata = _extract_hit_context(hit)
                if (metadata.get("chunk_type") or "").lower() == "image":
                    source_path = metadata.get("source_path") or ""
                    if source_path:
                        text_terms = set(re.findall(r"[a-z0-9]+", text.lower()))
                        overlap = len(query_terms & text_terms) if query_terms else 1

                        specific_models = {"rnn", "lstm", "gru", "transformer", "cross", "bert", "gpt", "resnet", "vgg", "unet", "diffusion"}
                        requested_models = query_terms & specific_models
                        if requested_models:
                            if not (text_terms & requested_models):
                                overlap = 0

                        is_diag_request = any(w in query.lower() for w in ["diagram", "architecture", "schematic", "structure", "cell", "model", "image of"])
                        is_eval_plot = any(w in text.lower() for w in ["mse", "percentage of sequence kept", "ablation", "scenario 1", "accuracy vs"])
                        if is_diag_request and is_eval_plot:
                            overlap = 0

                        if overlap > 0:
                            candidates.append({
                                "source_path": source_path,
                                "overlap": overlap,
                                "score": 0.5,
                            })
        except Exception:
            pass

    if not candidates:
        return None

    # Filter to candidates with positive keyword overlap
    valid_candidates = [c for c in candidates if c["overlap"] > 0]
    if not valid_candidates:
        return None

    valid_candidates.sort(key=lambda x: (-x["overlap"], -x["score"]))
    return valid_candidates[0]["source_path"]


def generate_support_diagram_image(query: str) -> str | None:
    """Generate a high-contrast white-background SVG architecture diagram for the query and convert to PNG."""
    if not query or not query.strip():
        return None

    import re
    from pathlib import Path
    try:
        import fitz
    except ImportError:
        return None

    from utils.models_and_clients import run_llm_completion
    from utils.logger_config import logger

    logger.info(f"[Support Diagram Generator] Generating diagram image for query: '{query}'")

    prompt = f"""Generate a clean, high-contrast, professional technical architecture diagram in SVG format representing: '{query}'.
CRITICAL DESIGN REQUIREMENTS:
1. MUST have a crisp WHITE background: <rect width="100%" height="100%" fill="#FFFFFF"/>
2. Use dark high-contrast text colors (#0F172A, #1E293B) so all component labels are bold, crisp, and easily readable.
3. Use modern, soft colored rounded card boxes (e.g. #EFF6FF for inputs, #F0FDF4 for encoder layers, #FEF2F2 for attention, #FAF5FF for outputs) with crisp dark borders (#64748B).
4. Include clear arrows (#334155) connecting boxes showing data flow.
5. Set proper width (e.g. 1000), height (e.g. 650), and viewBox.
6. Return valid SVG inside ```xml <svg ...> ... </svg> ``` block.
7. Return ONLY the SVG code block."""

    try:
        res = run_llm_completion(prompt=prompt, temperature=0.1)
        if res and "<svg" in res.lower():
            match = re.search(r"<svg.*?</svg>", res, re.DOTALL | re.IGNORECASE)
            if match:
                svg_code = match.group(0)
                project_root = Path(__file__).resolve().parents[2]
                out_dir = project_root / "data" / "output" / "generated_images"
                out_dir.mkdir(parents=True, exist_ok=True)

                slug = re.sub(r"[^\w]", "_", query.lower()).strip("_")[:40]
                png_path = out_dir / f"support_{slug}.png"

                doc = fitz.open(stream=svg_code.encode("utf-8"), filetype="svg")
                page = doc[0]
                pix = page.get_pixmap(dpi=150)
                pix.save(str(png_path))

                if png_path.exists() and png_path.stat().st_size > 0:
                    logger.success(f"[Support Diagram Generator] Created image: {png_path}")
                    return str(png_path)
    except Exception as exc:
        logger.warning(f"[Support Diagram Generator] Failed to generate diagram image: {exc}")

    return None

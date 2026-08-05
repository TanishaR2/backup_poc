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


def resolve_existing_image_path(path_str: str | None) -> str | None:
    """Resolve an image file path safely against local workspace disk."""
    if not path_str:
        return None
    p = Path(path_str)
    if p.is_file():
        return str(p)

    if "data/output/" in path_str:
        rel_part = path_str.split("data/output/", 1)[1]
        local_cand = Path(__file__).resolve().parents[2] / "data" / "output" / rel_part
        if local_cand.is_file():
            return str(local_cand)

    return None


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
    """[DEPRECATED] Always return False to rely on Planner LLM's semantic needs_image decision."""
    return False


def _extract_hit_context(hit: Any) -> tuple[str, dict]:
    """Extract text content and metadata dict safely from a Qdrant hit object or dictionary."""
    if isinstance(hit, dict):
        payload = hit.get("payload", {}) or {}
        text = payload.get("text", "") or ""
        metadata = payload.get("metadata", {}) or {}
        return text, metadata
    else:
        payload = getattr(hit, "payload", {}) or {}
        text = payload.get("text", "") or ""
        metadata = payload.get("metadata", {}) or {}
        return text, metadata


def select_relevant_image_path(query: str, hits: Iterable[Any], needs_image: bool | None = None) -> str | None:
    """Select the most relevant image path for the query.

    Strategy (minimal — mirrors web-search image flow):
      A. If arXiv ID found in query → targeted Qdrant scroll for that doc's image chunks.
         Score: page_match +100, wrong_page -10, figure_match +50.
         Returns None when score ≤ 0 (page/figure doesn't exist in doc).
      B. Fallback: keyword overlap over hits (for queries without a paper ID).
    """
    if not query or not query.strip():
        return None
    if needs_image is not True:
        return None

    # Extract signals from query
    arxiv_match   = re.search(r"\b(\d{4}\.\d{4,5})\b", query)
    target_arxiv_id = arxiv_match.group(1) if arxiv_match else None
    page_match    = re.search(r"\bpage\s*(\d+)\b", query, re.IGNORECASE)
    target_page   = int(page_match.group(1)) if page_match else None
    fig_match     = re.search(r"\bfig(?:ure)?\s*(\d+)\b", query, re.IGNORECASE)
    target_fig    = int(fig_match.group(1)) if fig_match else None

    # ── A. Targeted Qdrant scan when paper ID is present ─────────────────────
    if target_arxiv_id:
        try:
            from utils.models_and_clients import qdrant_client
            from utils.settings import COLLECTION_NAME
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            img_candidates: list[dict] = []
            seen_src: set[str] = set()
            offset = None
            scroll_count = 0
            image_filter = Filter(must=[FieldCondition(key="metadata.chunk_type", match=MatchValue(value="image"))])

            while scroll_count < 5:
                scroll_count += 1
                batch, next_off = qdrant_client.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=image_filter,
                    limit=500,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for pt in batch:
                    payload  = pt.payload or {}
                    meta     = payload.get("metadata", {}) or {}
                    doc_id   = str(meta.get("doc_id") or meta.get("document_name") or "")
                    ctype    = (meta.get("chunk_type") or "").lower()
                    src      = meta.get("source_path") or ""
                    page_num = meta.get("page_number")

                    if ctype != "image" or not src or target_arxiv_id not in doc_id:
                        continue
                    if src in seen_src:
                        continue
                    seen_src.add(src)

                    score = 0
                    if target_page is not None:
                        if page_num == target_page:
                            score += 100
                        else:
                            # Page mismatch -> hard reject (never attach image from wrong page)
                            score = -999

                    if target_fig is not None:
                        m = re.search(r"figure[_\s]?(\d+)", Path(src).stem, re.IGNORECASE)
                        fig_in_file = int(m.group(1)) if m else None
                        if fig_in_file == target_fig:
                            score += 50
                        else:
                            # Figure mismatch -> hard reject (never attach wrong figure number)
                            score = -999

                    img_candidates.append({"source_path": src, "score": score, "page": page_num})

                offset = next_off
                if not next_off:
                    break

            if img_candidates:
                img_candidates.sort(key=lambda x: -x["score"])
                best = img_candidates[0]
                # score > 0 means page matched; ≤ 0 means that page doesn't exist in doc
                return resolve_existing_image_path(best["source_path"]) if best["score"] > 0 else None

        except Exception:
            pass

    # ── B. Keyword overlap fallback (no paper ID in query) ────────────────────
    stop_words = {
        "the", "this", "that", "these", "those", "show", "me", "what", "is", "are", "a", "an", "of", "for",
        "can", "you", "display", "illustrate", "visualize", "figure", "fig", "diagram",
        "chart", "plot", "image", "picture", "photo", "in", "on", "at", "to", "with", "and", "or",
        "give", "get", "send", "provide", "fetch", "find", "look", "view", "return", "about",
        "please", "tell", "want", "same", "also", "explain", "page", "paper",
    }
    query_terms = {
        w for w in re.findall(r"[a-z0-9]+", query.lower())
        if w not in stop_words and len(w) >= 2
    }

    candidates: list[dict] = []

    if hits:
        for hit in hits:
            text, metadata = _extract_hit_context(hit)
            if (metadata.get("chunk_type") or "").lower() != "image":
                continue
            source_path = metadata.get("source_path") or ""
            if not source_path:
                continue

            text_terms = set(re.findall(r"[a-z0-9]+", text.lower()))
            overlap    = len(query_terms & text_terms) if query_terms else 1

            specific_models = {"rnn", "lstm", "gru", "transformer", "cross", "bert", "gpt", "resnet", "vgg", "unet", "diffusion"}
            if query_terms & specific_models and not (text_terms & (query_terms & specific_models)):
                overlap = 0

            score = float(metadata.get("rerank_score") or getattr(hit, "score", 0.0) or 0.0)
            candidates.append({"source_path": source_path, "overlap": overlap, "score": score})

    valid = [c for c in candidates if c["overlap"] > 0]
    if not valid:
        return None
    valid.sort(key=lambda x: (-x["overlap"], -x["score"]))
    return valid[0]["source_path"]


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

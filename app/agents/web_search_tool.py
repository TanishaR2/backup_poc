"""
app/agents/web_search_tool.py
==============================
Lightweight zero-dependency Internet Web Search Tool using DuckDuckGo HTML endpoint.
Fetches live web search snippets in parallel for queries needing internet verification.
"""

import re
import requests
from utils.logger_config import logger


def _score_snippet(query: str, snippet: str) -> int:
    if not query or not snippet:
        return 0

    query_terms = {
        term for term in re.findall(r"[a-z0-9]+", query.lower()) if len(term) > 1
    }
    snippet_terms = {
        term for term in re.findall(r"[a-z0-9]+", snippet.lower()) if len(term) > 1
    }
    return len(query_terms & snippet_terms)


def _rerank_snippets(query: str, snippets: list[str], top_k: int = 3) -> list[str]:
    scored = []
    for idx, snippet in enumerate(snippets):
        score = _score_snippet(query, snippet)
        scored.append((score, idx, snippet))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [snippet for _, _, snippet in scored[:top_k]]


def search_web(query: str, max_results: int = 5) -> list[str]:
    """Fetch live web search snippets from DuckDuckGo."""
    if not query or not query.strip():
        return []

    url = "https://html.duckduckgo.com/html/"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }
    payload = {"q": query.strip()}

    try:
        resp = requests.post(url, data=payload, headers=headers, timeout=3.5)
        if resp.status_code != 200:
            logger.warning(f"[Web Search] DuckDuckGo returned status {resp.status_code}")
            return []

        snippets_raw = re.findall(r'<a class="result__snippet"[^>]*>(.*?)</a>', resp.text, re.S)
        results = []
        for s in snippets_raw[:max_results]:
            clean_text = re.sub(r"<.*?>", "", s).strip()
            clean_text = re.sub(r"\s+", " ", clean_text)
            if clean_text:
                results.append(clean_text)

        ranked = _rerank_snippets(query, results, top_k=3)
        logger.success(f"[Web Search] Fetched {len(results)} live web snippets and returned top {len(ranked)} for query '{query[:40]}...'")
        return ranked
    except Exception as exc:
        logger.warning(f"[Web Search] Live search failed or timed out: {exc}")
        return []


def fetch_web_image_for_query(query: str) -> str | None:
    """Search web for relevant images matching the query, download the best candidate locally, and return its file path."""
    import os
    import requests
    from pathlib import Path
    from utils.settings import TAVILY_API_KEY

    if not query or not query.strip():
        return None

    if not TAVILY_API_KEY:
        logger.warning("[Web Image Search] TAVILY_API_KEY not set")
        return None

    logger.info(f"[Web Image Search] Searching web images for query: '{query}'")
    search_payload = {
        "api_key": TAVILY_API_KEY,
        "query": f"{query} architecture diagram visual diagram",
        "include_images": True,
        "max_results": 5,
    }

    try:
        resp = requests.post("https://api.tavily.com/search", json=search_payload, timeout=8.0)
        if resp.status_code != 200:
            logger.warning(f"[Web Image Search] Tavily returned status {resp.status_code}")
            return None

        data = resp.json()
        image_urls = data.get("images", [])
        if not image_urls:
            logger.warning(f"[Web Image Search] No web image URLs returned for '{query}'")
            return None

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        out_dir = Path(__file__).resolve().parents[2] / "data" / "output" / "generated_images"
        out_dir.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^\w]", "_", query.lower()).strip("_")[:40]

        for idx, img_url in enumerate(image_urls[:3]):
            if not isinstance(img_url, str) or not img_url.startswith(("http://", "https://")):
                continue
            try:
                img_resp = requests.get(img_url, headers=headers, timeout=6.0)
                if img_resp.status_code == 200 and len(img_resp.content) > 5000:
                    ext = ".png"
                    ctype = img_resp.headers.get("Content-Type", "").lower()
                    if "jpeg" in ctype or "jpg" in ctype:
                        ext = ".jpg"
                    elif "svg" in ctype:
                        ext = ".svg"

                    img_file = out_dir / f"web_{slug}_{idx+1}{ext}"
                    img_file.write_bytes(img_resp.content)

                    if ext == ".svg":
                        try:
                            import fitz
                            doc = fitz.open(stream=img_resp.content, filetype="svg")
                            png_file = out_dir / f"web_{slug}_{idx+1}.png"
                            pix = doc[0].get_pixmap(dpi=150)
                            pix.save(str(png_file))
                            if png_file.exists() and png_file.stat().st_size > 0:
                                logger.success(f"[Web Image Search] Downloaded & rendered web SVG image: {png_file}")
                                return str(png_file)
                        except Exception:
                            pass

                    if img_file.exists() and img_file.stat().st_size > 0:
                        logger.success(f"[Web Image Search] Successfully downloaded web image: {img_file}")
                        return str(img_file)
            except Exception as e:
                logger.warning(f"[Web Image Search] Failed to download {img_url}: {e}")

    except Exception as exc:
        logger.warning(f"[Web Image Search] Search error: {exc}")

    return None

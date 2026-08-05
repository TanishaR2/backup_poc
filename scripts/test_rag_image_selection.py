"""
test_rag_image_selection.py
───────────────────────────
Standalone test script to verify the improved RAG image selection logic.

Run: PYTHONPATH=. uv run python scripts/test_rag_image_selection.py
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.models_and_clients import qdrant_client
from utils.settings import COLLECTION_NAME
from qdrant_client.models import Filter, FieldCondition, MatchValue


def _extract_query_signals(query: str) -> dict:
    arxiv_match = re.search(r"\b(\d{4}\.\d{4,5})\b", query)
    page_match  = re.search(r"\bpage\s*(\d+)\b", query, re.IGNORECASE)
    fig_match   = re.search(r"\bfig(?:ure)?\s*(\d+)\b", query, re.IGNORECASE)
    return {
        "arxiv_id":      arxiv_match.group(1) if arxiv_match else None,
        "page_number":   int(page_match.group(1)) if page_match else None,
        "figure_number": int(fig_match.group(1)) if fig_match else None,
    }


def select_rag_image_path(query: str, hits=None) -> str | None:
    if not query or not query.strip():
        return None

    signals     = _extract_query_signals(query)
    arxiv_id    = signals["arxiv_id"]
    target_page = signals["page_number"]
    target_fig  = signals["figure_number"]

    if arxiv_id:
        try:
            img_candidates = []
            seen_src = set()
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
                    src      = meta.get("source_path") or ""
                    page_num = meta.get("page_number")

                    if not src or arxiv_id not in doc_id:
                        continue
                    if src in seen_src:
                        continue
                    seen_src.add(src)

                    score = 0
                    if target_page is not None:
                        if page_num == target_page:
                            score += 100
                        else:
                            score -= 10

                    if target_fig is not None:
                        m = re.search(r"figure[_\s]?(\d+)", Path(src).stem, re.IGNORECASE)
                        fig_in_file = int(m.group(1)) if m else None
                        if fig_in_file == target_fig:
                            score += 50
                        elif target_page is not None and page_num == target_page:
                            score = -999

                    img_candidates.append({"src": src, "score": score, "page": page_num})

                offset = next_off
                if not next_off:
                    break

            if img_candidates:
                img_candidates.sort(key=lambda x: -x["score"])
                best = img_candidates[0]
                return best["src"] if best["score"] > 0 else None

        except Exception as exc:
            print(f"  [WARN] Qdrant scroll failed: {exc}")

    return None


TEST_CASES = [
    ("Show Figure 3 on page 4 of paper 2606.15207.",          "page_4_figure_3",   True),
    ("Show Figure 2 on page 3 of paper 2606.15207.",          "page_3_figure_2",   True),
    ("Show Figure 1 on page 2 of paper 2606.15207.",          "page_2_figure_1",   True),
    ("What is shown on page 3 of 2606.15207?",                "page_3",            True),
    ("Show figure 3 of paper 2606.15207 on page 10.",         None,                False),
    ("Show figure 9 on page 4 of paper 2606.15207.",          None,                False),
    ("Show Figure 1 on page 5 of paper 2606.15058.",          "page_5_figure_1",   True),
    ("What is on page 12 of paper 2606.15058?",               "page_12",           True),
    ("Display Figure 8 on page 14 of paper 2606.15058.",      "page_14_figure_8",  True),
    ("Show figure 1 on page 99 of paper 2606.15058.",         None,                False),
]


def run_tests():
    print("=" * 70)
    print("  RAG Image Selection — Logic Validation (10 queries)")
    print("=" * 70)

    passed = 0
    failed = 0

    for i, (query, expected_frag, should_find) in enumerate(TEST_CASES, 1):
        signals = _extract_query_signals(query)
        result = select_rag_image_path(query, hits=[])
        fname = Path(result).name if result else None

        if should_find:
            ok = (result is not None) and (expected_frag in (fname or ""))
        else:
            if result is None:
                ok = True
            else:
                qp = signals.get("page_number")
                if qp and fname:
                    pm = re.search(r"page_(\d+)", fname)
                    ok = not (pm and int(pm.group(1)) == qp)
                else:
                    ok = True

        mark = "[OK]" if ok else "[!!]"
        status = "PASS" if ok else "FAIL"
        passed += ok
        failed += not ok

        print(f"{mark} [{i:02d}] {status}  | Query: {query} -> Got: {fname}")

    print("=" * 70)
    print(f"  TOTAL: {passed}/{len(TEST_CASES)} passed, {failed} failed")
    print("=" * 70)
    return failed


if __name__ == "__main__":
    sys.exit(run_tests())

"""
generate_qdrant_doc_summary.py
──────────────────────────────
Scans Qdrant collection 'InsightDocs' and generates a markdown table listing:
- Document ID / Name
- Total Chunks
- Image Chunks
- Text Chunks
- Image Availability Flag
"""

import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.models_and_clients import qdrant_client
from utils.settings import COLLECTION_NAME


def summarize_qdrant_docs():
    print("=" * 80)
    print(f"  Qdrant Inventory Tool — Collection: '{COLLECTION_NAME}'")
    print("=" * 80)
    
    offset = None
    doc_stats = defaultdict(lambda: {"total_chunks": 0, "image_chunks": 0, "text_chunks": 0})
    total_points = 0

    while True:
        batch, next_off = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for pt in batch:
            total_points += 1
            payload = pt.payload or {}
            meta = payload.get("metadata", {}) or {}
            doc_id = str(meta.get("doc_id") or meta.get("document_name") or "unknown_doc")
            chunk_type = meta.get("chunk_type", "text")

            doc_stats[doc_id]["total_chunks"] += 1
            if chunk_type == "image":
                doc_stats[doc_id]["image_chunks"] += 1
            else:
                doc_stats[doc_id]["text_chunks"] += 1

        offset = next_off
        if not next_off:
            break

    print(f"\nScanned {total_points} total points across {len(doc_stats)} unique document IDs.\n")

    # Generate Markdown Table
    md_lines = [
        f"# Qdrant Collection Inventory Summary (`{COLLECTION_NAME}`)",
        "",
        f"> **Total Points Indexed:** {total_points}  ",
        f"> **Total Unique Document IDs:** {len(doc_stats)}  ",
        "",
        "| # | Document ID / Name | Total Chunks | Image Chunks | Text Chunks | Has Images? |",
        "|---|---|:---:|:---:|:---:|:---:|",
    ]

    sorted_docs = sorted(doc_stats.items(), key=lambda x: (-x[1]["image_chunks"], -x[1]["total_chunks"], x[0]))

    for idx, (doc_id, stats) in enumerate(sorted_docs, 1):
        has_img = "✅ Yes" if stats["image_chunks"] > 0 else "❌ No"
        md_lines.append(
            f"| {idx} | `{doc_id}` | {stats['total_chunks']} | **{stats['image_chunks']}** | {stats['text_chunks']} | {has_img} |"
        )

    md_content = "\n".join(md_lines)
    
    out_file = ROOT / "docs" / "qdrant_document_inventory.md"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(md_content)
    print(f"\nSaved inventory report to {out_file}")


if __name__ == "__main__":
    summarize_qdrant_docs()

"""
inspect_qdrant_doc_chunks.py
────────────────────────────
Script to inspect Qdrant vector database for specific document ID and chunk type.

Constants are declared at the beginning of the script as requested.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.models_and_clients import qdrant_client
from utils.settings import COLLECTION_NAME
from qdrant_client.models import Filter, FieldCondition, MatchValue

# ==============================================================================
# DECLARE TARGET SEARCH CONSTANTS HERE
# ==============================================================================
TARGET_DOC_ID = "2606.14023"
TARGET_CHUNK_TYPE = "image"
# ==============================================================================


def inspect_qdrant():
    print("=" * 80)
    print(f"  Qdrant Inspection Tool — Collection: '{COLLECTION_NAME}'")
    print(f"  Searching for DOC_ID: '{TARGET_DOC_ID}' | CHUNK_TYPE: '{TARGET_CHUNK_TYPE}'")
    print("=" * 80)

    try:
        image_filter = Filter(
            must=[FieldCondition(key="metadata.chunk_type", match=MatchValue(value=TARGET_CHUNK_TYPE))]
        ) if TARGET_CHUNK_TYPE else None

        offset = None
        matching_points = []
        all_doc_ids_seen = set()

        while True:
            batch, next_off = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=image_filter,
                limit=250,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for pt in batch:
                payload = pt.payload or {}
                meta = payload.get("metadata", {}) or {}
                doc_id = str(meta.get("doc_id") or meta.get("document_name") or "")
                
                if doc_id:
                    all_doc_ids_seen.add(doc_id)

                if TARGET_DOC_ID.lower() in doc_id.lower():
                    matching_points.append({
                        "point_id": pt.id,
                        "doc_id": doc_id,
                        "chunk_type": meta.get("chunk_type"),
                        "page_number": meta.get("page_number"),
                        "source_path": meta.get("source_path"),
                        "text_snippet": payload.get("text", "")[:200],
                    })

            offset = next_off
            if not next_off:
                break

        print(f"\n[Qdrant Status]")
        print(f"  • Total unique documents found with chunk_type='{TARGET_CHUNK_TYPE}': {len(all_doc_ids_seen)}")
        print(f"  • Matching points for '{TARGET_DOC_ID}': {len(matching_points)}")

        if matching_points:
            print("\n[Matching Chunks Found in Qdrant]:")
            for i, pt in enumerate(matching_points, 1):
                print("-" * 70)
                print(f"  Hit #{i:02d} | Point ID: {pt['point_id']}")
                print(f"  Doc ID:       {pt['doc_id']}")
                print(f"  Chunk Type:   {pt['chunk_type']}")
                print(f"  Page Number:  {pt['page_number']}")
                print(f"  Source Path:  {pt['source_path']}")
                print(f"  Text Snippet: {pt['text_snippet']}...")
        else:
            print(f"\n[WARNING] No points matching doc_id='{TARGET_DOC_ID}' and chunk_type='{TARGET_CHUNK_TYPE}' found in Qdrant!")
            print("\nSample doc_ids indexed in Qdrant:")
            for d in list(all_doc_ids_seen)[:15]:
                print(f"  - {d}")

    except Exception as exc:
        print(f"[ERROR] Failed to query Qdrant: {exc}")

    print("=" * 80)


if __name__ == "__main__":
    inspect_qdrant()

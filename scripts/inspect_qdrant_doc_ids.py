"""Script to inspect unique doc_ids in Qdrant and detail image chunks for paper 2606.15207."""

from utils.models_and_clients import qdrant_client
from utils.settings import COLLECTION_NAME


def main():
    print(f"Scrolling points from Qdrant collection '{COLLECTION_NAME}'...")

    unique_doc_ids = set()
    chunks_207 = []
    
    offset = None
    total_points = 0

    while True:
        records, next_offset = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            break

        total_points += len(records)
        for point in records:
            payload = point.payload or {}
            metadata = payload.get("metadata", {}) or {}
            
            doc_id = metadata.get("doc_id") or metadata.get("document_name") or "UNKNOWN"
            unique_doc_ids.add(doc_id)

            if "2606.15207" in str(doc_id):
                chunks_207.append({
                    "point_id": point.id,
                    "chunk_type": metadata.get("chunk_type"),
                    "page_number": metadata.get("page_number"),
                    "source_path": metadata.get("source_path"),
                    "chunk_id": metadata.get("chunk_id"),
                    "text_snippet": payload.get("text", "")[:100]
                })

        offset = next_offset
        if next_offset is None:
            break

    print("\n" + "=" * 80)
    print(f"Total points scanned in Qdrant: {total_points}")
    print(f"Total unique document IDs found: {len(unique_doc_ids)}")
    print("=" * 80)

    print("\nUnique Document IDs in Qdrant:")
    for idx, d_id in enumerate(sorted(unique_doc_ids), 1):
        print(f"  {idx}. {d_id}")

    print("\n" + "=" * 80)
    is_207_present = any("2606.15207" in str(d) for d in unique_doc_ids)
    print(f"Is paper '2606.15207' present in Qdrant? -> {is_207_present}")
    print("=" * 80)

    if is_207_present:
        image_chunks_207 = [
            c for c in chunks_207
            if c.get("chunk_type") == "image" or (c.get("source_path") and str(c.get("source_path")).endswith(".png"))
        ]
        
        print(f"\nTotal chunks for 2606.15207: {len(chunks_207)}")
        print(f"Total IMAGE chunks for 2606.15207: {len(image_chunks_207)}")
        print("-" * 80)
        
        if image_chunks_207:
            print("Image Chunk Metadata:")
            for idx, img in enumerate(image_chunks_207, 1):
                print(f"\n  Image #{idx}:")
                print(f"    Point ID    : {img['point_id']}")
                print(f"    Chunk Type  : {img['chunk_type']}")
                print(f"    Page Number : {img['page_number']}")
                print(f"    Source Path : {img['source_path']}")
                print(f"    Chunk ID    : {img['chunk_id']}")
                print(f"    Text Snippet: {img['text_snippet']}")
        else:
            print("No image chunks found for paper 2606.15207 in Qdrant.")


if __name__ == "__main__":
    main()

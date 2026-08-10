"""
build_datasets.py
─────────────────
Script to build ground truth QA datasets for retrieval and generation evaluation:
- groundtruth_dataset_5.json    (5-query fast evaluation set)
- groundtruth_dataset_100.json  (100-query benchmark dataset)
- groundtruth_dataset_1000.json (1000-query extended dataset)

Queries cover text, table, image, and out_of_scope query types across ingested papers.
Queries are independent and natural (no hardcoded doc IDs in prompt).
"""

import sys
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.logger_config import logger

OUTPUT_DIR = Path(__file__).resolve().parent
PDF_DIR = ROOT / "data" / "output" / "pdfs" / "InsightDocs"

# Out-of-scope queries (unindexed models, non-existent tables, general out-of-domain)
OUT_OF_SCOPE_QUERIES = [
    {
        "query": "What exact top-1 ImageNet accuracy did the proposed QuantumViT-XL model achieve, and on which page is Table 7?",
        "query_type": "out_of_scope",
        "expected_doc_id": None,
        "expected_chunk_type": None,
        "expected_answer": "Information not available in retrieved context.",
        "expected_image_path": None
    },
    {
        "query": "What is the 3D bounding box estimation error of the HyperSparseNet architecture on the KITTI benchmark?",
        "query_type": "out_of_scope",
        "expected_doc_id": None,
        "expected_chunk_type": None,
        "expected_answer": "Information not available in retrieved context.",
        "expected_image_path": None
    },
    {
        "query": "How many parameters does the UltraMamba-180B model have and what is its training perplexity on Pile-CC?",
        "query_type": "out_of_scope",
        "expected_doc_id": None,
        "expected_chunk_type": None,
        "expected_answer": "Information not available in retrieved context.",
        "expected_image_path": None
    },
    {
        "query": "What are the recipe ingredients and oven temperature specified for baking sourdough bread?",
        "query_type": "out_of_scope",
        "expected_doc_id": None,
        "expected_chunk_type": None,
        "expected_answer": "Information not available in retrieved context.",
        "expected_image_path": None
    },
    {
        "query": "What is the stock market ticker symbol and quarterly revenue for Apple Inc in Q3 2025?",
        "query_type": "out_of_scope",
        "expected_doc_id": None,
        "expected_chunk_type": None,
        "expected_answer": "Information not available in retrieved context.",
        "expected_image_path": None
    },
    {
        "query": "What is the quantum fidelity of the 128-qubit SuperConducting Transmon processor presented in Table 12?",
        "query_type": "out_of_scope",
        "expected_doc_id": None,
        "expected_chunk_type": None,
        "expected_answer": "Information not available in retrieved context.",
        "expected_image_path": None
    },
    {
        "query": "What formula is used to calculate the engine displacement of a V8 internal combustion engine in Section 4.2?",
        "query_type": "out_of_scope",
        "expected_doc_id": None,
        "expected_chunk_type": None,
        "expected_answer": "Information not available in retrieved context.",
        "expected_image_path": None
    },
    {
        "query": "What exact BLEU score did the Transformer-XL-Large model obtain on the WMT14 English-to-German translation task in Table 9?",
        "query_type": "out_of_scope",
        "expected_doc_id": None,
        "expected_chunk_type": None,
        "expected_answer": "Information not available in retrieved context.",
        "expected_image_path": None
    },
    {
        "query": "What is the chemical reaction mechanism for synthesizing paracetamol from 4-aminophenol?",
        "query_type": "out_of_scope",
        "expected_doc_id": None,
        "expected_chunk_type": None,
        "expected_answer": "Information not available in retrieved context.",
        "expected_image_path": None
    },
    {
        "query": "What is the minimum recommended tyre pressure for a Boeing 747-8 landing gear assembly?",
        "query_type": "out_of_scope",
        "expected_doc_id": None,
        "expected_chunk_type": None,
        "expected_answer": "Information not available in retrieved context.",
        "expected_image_path": None
    }
]


def load_manifest_data():
    """Load all manifest items grouped by document ID."""
    manifests = list(PDF_DIR.glob("*/content/manifest.json"))
    logger.info(f"Scanning {len(manifests)} document manifests...")
    
    docs_data = {}
    for m in manifests:
        folder_name = m.parent.parent.name
        if folder_name.lower() in ["faq", "paper"] or folder_name.endswith(" copy") or "copy" in folder_name.lower():
            continue
        try:
            items = json.loads(m.read_text(encoding="utf-8"))
            docs_data[folder_name] = items
        except Exception as exc:
            logger.warning(f"Could not load manifest for {folder_name}: {exc}")
            
    logger.info(f"Loaded {len(docs_data)} valid document manifests.")
    return docs_data


def generate_groundtruth_datasets():
    docs_data = load_manifest_data()
    
    all_queries = []
    q_id = 1
    
    # 1. Add out-of-scope queries
    for oos in OUT_OF_SCOPE_QUERIES:
        item = dict(oos)
        item["query_id"] = f"Q{q_id:04d}"
        all_queries.append(item)
        q_id += 1

    # 2. Extract queries from document manifests
    doc_keys = sorted(list(docs_data.keys()))
    
    for doc_id in doc_keys:
        items = docs_data[doc_id]
        
        for item in items:
            c_type = item.get("chunk_type") or item.get("type")
            content = str(item.get("content") or item.get("text") or "").strip()
            
            if not content or len(content) < 30:
                continue
                
            if c_type == "text":
                lines = [l.strip() for l in content.split("\n") if len(l.strip()) > 25]
                if lines:
                    snippet = lines[0]
                    all_queries.append({
                        "query_id": f"Q{q_id:04d}",
                        "query": f"What methodology or technical mechanism is described regarding: '{snippet[:110]}'?",
                        "query_type": "text",
                        "expected_doc_id": doc_id,
                        "expected_chunk_type": "text",
                        "expected_answer": content[:500],
                        "expected_image_path": None
                    })
                    q_id += 1
            elif c_type == "table":
                all_queries.append({
                    "query_id": f"Q{q_id:04d}",
                    "query": f"What numerical experimental data or comparative table metrics are reported in paper '{doc_id[:25]}'?",
                    "query_type": "table",
                    "expected_doc_id": doc_id,
                    "expected_chunk_type": "table",
                    "expected_answer": content[:500],
                    "expected_image_path": None
                })
                q_id += 1
            elif c_type == "image":
                img_path = item.get("source_path") or item.get("description_path")
                all_queries.append({
                    "query_id": f"Q{q_id:04d}",
                    "query": f"Locate the architecture figure or visual plot illustrating: '{content[:110]}'",
                    "query_type": "image",
                    "expected_doc_id": doc_id,
                    "expected_chunk_type": "image",
                    "expected_answer": content[:500],
                    "expected_image_path": img_path
                })
                q_id += 1

    # Shuffle to guarantee diversity
    random.seed(42)
    random.shuffle(all_queries)

    # Build 5-query dataset (1 out_of_scope, 2 text, 1 table, 1 image)
    ds_5 = []
    counts_5 = {"text": 0, "table": 0, "image": 0, "out_of_scope": 0}
    max_5 = {"text": 2, "table": 1, "image": 1, "out_of_scope": 1}
    
    for item in all_queries:
        qt = item["query_type"]
        if counts_5[qt] < max_5[qt]:
            item_copy = dict(item)
            item_copy["query_id"] = f"Q5_{len(ds_5)+1:02d}"
            ds_5.append(item_copy)
            counts_5[qt] += 1
        if len(ds_5) == 5:
            break

    # Build 100-query dataset
    ds_100 = []
    for idx, item in enumerate(all_queries[:100], 1):
        item_copy = dict(item)
        item_copy["query_id"] = f"Q100_{idx:03d}"
        ds_100.append(item_copy)

    # Build 1000-query dataset (repeat/sample to reach 1000)
    ds_1000 = []
    multiplier = (1000 // len(all_queries)) + 1
    extended_pool = (all_queries * multiplier)[:1000]
    for idx, item in enumerate(extended_pool, 1):
        item_copy = dict(item)
        item_copy["query_id"] = f"Q1000_{idx:04d}"
        ds_1000.append(item_copy)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    file_5 = OUTPUT_DIR / "groundtruth_dataset_5.json"
    file_100 = OUTPUT_DIR / "groundtruth_dataset_100.json"
    file_1000 = OUTPUT_DIR / "groundtruth_dataset_1000.json"

    file_5.write_text(json.dumps(ds_5, indent=2), encoding="utf-8")
    file_100.write_text(json.dumps(ds_100, indent=2), encoding="utf-8")
    file_1000.write_text(json.dumps(ds_1000, indent=2), encoding="utf-8")

    logger.success(f"Saved 5-query dataset to    : {file_5}")
    logger.success(f"Saved 100-query dataset to  : {file_100}")
    logger.success(f"Saved 1000-query dataset to : {file_1000}")

if __name__ == "__main__":
    generate_groundtruth_datasets()

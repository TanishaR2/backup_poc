"""
evaluate_retrieval.py
─────────────────────
Retrieval Evaluation Script for InSightDocs.
Imports `retrieve()` directly from app.retriever.retrieval_service.
NO LangGraph state graph or edge fallback routing.

Computes:
- MRR (Mean Reciprocal Rank)
- MAP (Mean Average Precision)
- Hit Rate @ 1 & Hit Rate @ 5
- Precision @ 5
- Recall @ 5
- F1 Score
- Chunk Type Match Rate
"""

import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.retriever.retrieval_service import retrieve
from utils.logger_config import logger

DATASET_PATH = "groundtruth_dataset_5.json"
DATASET_PATH = "benchmark_dataset_50.json"
def evaluate_retrieval(dataset_path: Path, output_json_path: Path) -> dict:
    logger.info("=" * 80)
    logger.info(f"  RETRIEVAL EVALUATION — Dataset: '{dataset_path.name}'")
    logger.info("=" * 80)

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    logger.info(f"Loaded {len(dataset)} evaluation queries.")

    results = []

    mrr_sum = 0.0
    map_sum = 0.0
    hit1_count = 0
    hit5_count = 0
    precision_sum = 0.0
    recall_sum = 0.0
    type_match_count = 0
    out_of_scope_correct = 0
    total_evaluable = 0

    for idx, item in enumerate(dataset, 1):
        q_id = item["query_id"]
        query_text = item["query"]
        q_type = item["query_type"]
        expected_doc = item.get("expected_doc_id")
        expected_chunk_type = item.get("expected_chunk_type")

        logger.info(f"[{idx}/{len(dataset)}] Evaluating Query {q_id} ({q_type})...")

        # Call core retrieval directly (NO LangGraph, NO fallback routing)
        hits, analysis = retrieve(query=query_text, scope="documents", top_k=5)

        retrieved_docs = []
        retrieved_types = []
        for a in analysis:
            d_name = str(a.get("document", "")).replace(".pdf", "").strip()
            c_type = a.get("chunk_type", "text")
            if d_name:
                retrieved_docs.append(d_name)
            retrieved_types.append(c_type)

        rank = 0
        hit_1 = False
        hit_5 = False
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        reciprocal_rank = 0.0
        type_match = False

        if q_type == "out_of_scope" or expected_doc is None:
            out_of_scope_correct += 1
            hit_1 = True
            hit_5 = True
            reciprocal_rank = 1.0
            precision = 1.0
            recall = 1.0
            f1 = 1.0
        else:
            total_evaluable += 1
            exp_norm = str(expected_doc).replace(".pdf", "").strip()

            for i, r_doc in enumerate(retrieved_docs, 1):
                if r_doc == exp_norm or r_doc.startswith(exp_norm[:30]) or exp_norm.startswith(r_doc[:30]):
                    rank = i
                    reciprocal_rank = 1.0 / rank
                    break

            if rank == 1:
                hit1_count += 1
                hit_1 = True

            if rank > 0 and rank <= 5:
                hit5_count += 1
                hit_5 = True
                precision = 1.0 / 5.0
                recall = 1.0
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            mrr_sum += reciprocal_rank
            map_sum += reciprocal_rank
            precision_sum += precision
            recall_sum += recall

            if retrieved_types and expected_chunk_type and retrieved_types[0] == expected_chunk_type:
                type_match = True
                type_match_count += 1

        res_entry = {
            "query_id": q_id,
            "query": query_text,
            "query_type": q_type,
            "expected_doc_id": expected_doc,
            "expected_chunk_type": expected_chunk_type,
            "retrieved_doc_ids": retrieved_docs,
            "retrieved_chunk_types": retrieved_types,
            "rank": rank,
            "mrr": reciprocal_rank,
            "hit_at_1": hit_1,
            "hit_at_5": hit_5,
            "precision_at_5": precision,
            "recall_at_5": recall,
            "f1_score": f1,
            "chunk_type_match": type_match
        }
        results.append(res_entry)

    num_total = len(dataset)
    num_eval = total_evaluable if total_evaluable > 0 else 1

    summary_metrics = {
        "total_queries": num_total,
        "evaluable_queries": total_evaluable,
        "out_of_scope_queries": len(dataset) - total_evaluable,
        "mrr": round(mrr_sum / num_eval, 4),
        "map": round(map_sum / num_eval, 4),
        "hit_rate_at_1": round(hit1_count / num_eval, 4),
        "hit_rate_at_5": round(hit5_count / num_eval, 4),
        "avg_precision_at_5": round(precision_sum / num_eval, 4),
        "avg_recall_at_5": round(recall_sum / num_eval, 4),
        "avg_f1_score": round((2 * (precision_sum / num_eval) * (recall_sum / num_eval)) / ((precision_sum / num_eval) + (recall_sum / num_eval)) if (precision_sum + recall_sum) > 0 else 0.0, 4),
        "chunk_type_match_rate": round(type_match_count / num_eval, 4)
    }

    output_data = {
        "summary": summary_metrics,
        "details": results
    }

    output_json_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")

    logger.success("=" * 80)
    logger.success("  RETRIEVAL EVALUATION SUMMARY")
    logger.success(f"  MRR            : {summary_metrics['mrr']}")
    logger.success(f"  MAP            : {summary_metrics['map']}")
    logger.success(f"  Hit Rate @ 1   : {summary_metrics['hit_rate_at_1']}")
    logger.success(f"  Hit Rate @ 5   : {summary_metrics['hit_rate_at_5']}")
    logger.success(f"  Avg Precision  : {summary_metrics['avg_precision_at_5']}")
    logger.success(f"  Avg Recall     : {summary_metrics['avg_recall_at_5']}")
    logger.success(f"  Avg F1-Score   : {summary_metrics['avg_f1_score']}")
    logger.success(f"  Results saved  : {output_json_path}")
    logger.success("=" * 80)

    return summary_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate InSightDocs Retrieval")
    parser.add_argument("--dataset", type=str, default=DATASET_PATH, help="Dataset file name in groundtruth_evaluation/")
    args = parser.parse_args()

    ds_path = ROOT / "groundtruth_evaluation" / args.dataset
    out_path = ROOT / "groundtruth_evaluation" / f"retrieval_results_{args.dataset}"

    evaluate_retrieval(ds_path, out_path)

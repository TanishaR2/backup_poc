"""
evaluate_generation.py
──────────────────────
Generation Evaluation Script for InSightDocs.
Imports `retrieve()` from app.retriever.retrieval_service and `run_llm_completion()` from utils.models_and_clients.
NO LangGraph state machine or fallback edge routing.

Uses LLM-as-Judge to score:
- Relevance (0.0 to 1.0)
- Correctness (0.0 to 1.0)
- Completeness (0.0 to 1.0)
- Overall Score
- Image Path Alignment (for image queries)
"""

import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.retriever.retrieval_service import retrieve
from utils.models_and_clients import run_llm_completion
from utils.logger_config import logger


DATASET_PATH = "benchmark_dataset_50.json"

GENERATE_SYSTEM_PROMPT = """You are a grounded multimodal document intelligence assistant.
Answer the user's question based STRICTLY on the provided context chunks below.
If the context chunks do not contain sufficient information to answer the question, state clearly: "Information not available in retrieved context."

Context Chunks:
{context_text}
"""

JUDGE_PROMPT = """You are an expert AI evaluation judge evaluating a RAG system's generated answer against a ground truth answer.

User Question: {query}
Expected Ground Truth Answer: {expected_answer}
RAG Generated Answer: {generated_answer}

Rate the generated answer on three criteria from 0.0 to 1.0:
1. relevance: How well does the answer address the specific question asked? (0.0 = completely irrelevant, 1.0 = perfectly relevant)
2. correctness: Is the information factually accurate compared to the ground truth? (0.0 = completely wrong/hallucinated, 1.0 = factually accurate)
3. completeness: Does the answer cover all key aspects present in the ground truth? (0.0 = completely missing info, 1.0 = fully comprehensive)

Return strictly valid JSON with this format:
{{
  "relevance": 0.95,
  "correctness": 0.90,
  "completeness": 0.85,
  "explanation": "Brief 1-sentence reasoning"
}}
"""


def _extract_hit_content(h):
    doc_id = ""
    chunk_type = "text"
    content = ""
    src_path = None

    if hasattr(h, "payload") and isinstance(h.payload, dict):
        payload = h.payload
        meta = payload.get("metadata", {})
        doc_id = meta.get("document_name") or meta.get("doc_id") or payload.get("doc_id")
        chunk_type = meta.get("chunk_type") or payload.get("chunk_type") or "text"
        content = payload.get("parent_text") or payload.get("text") or meta.get("parent_text") or meta.get("content") or meta.get("text") or ""
        src_path = meta.get("source_path") or meta.get("description_path")
    elif isinstance(h, dict):
        meta = h.get("metadata", {}) if isinstance(h.get("metadata"), dict) else h
        doc_id = meta.get("document", "") or meta.get("document_name") or meta.get("doc_id") or h.get("doc_id") or h.get("document")
        chunk_type = meta.get("chunk_type") or h.get("chunk_type") or "text"
        content = h.get("content") or h.get("text") or meta.get("parent_text") or meta.get("content") or meta.get("text") or ""
        src_path = meta.get("source_path") or meta.get("description_path")

    clean_doc = str(doc_id).replace(".pdf", "").strip() if doc_id else ""
    return clean_doc, chunk_type, str(content), src_path


def evaluate_generation(dataset_path: Path, output_json_path: Path) -> dict:
    logger.info("=" * 80)
    logger.info(f"  GENERATION EVALUATION — Dataset: '{dataset_path.name}'")
    logger.info("=" * 80)

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    logger.info(f"Loaded {len(dataset)} evaluation queries.")

    results = []

    rel_sum = 0.0
    corr_sum = 0.0
    comp_sum = 0.0
    overall_sum = 0.0
    image_match_count = 0
    image_queries_count = 0

    for idx, item in enumerate(dataset, 1):
        q_id = item["query_id"]
        query_text = item["query"]
        q_type = item["query_type"]
        expected_answer = item.get("expected_answer", "")
        expected_img_path = item.get("expected_image_path")

        logger.info(f"[{idx}/{len(dataset)}] Generating & Judging Query {q_id} ({q_type})...")

        # 1. Retrieve top 5 hits directly (NO LangGraph, NO fallback edges)
        hits, analysis = retrieve(query=query_text, scope="documents", top_k=5)

        context_blocks = []
        gen_img_path = None

        for h_idx, a_item in enumerate(analysis, 1):
            c_type = a_item.get("chunk_type", "text")
            content = a_item.get("content", "")
            context_blocks.append(f"[Chunk {h_idx}] ({c_type}): {content}")

        for h in hits:
            _, c_type, _, src_path = _extract_hit_content(h)
            if c_type == "image" and src_path and not gen_img_path:
                gen_img_path = src_path

        context_str = "\n\n".join(context_blocks)

        # 2. Direct LLM Generation
        gen_prompt = GENERATE_SYSTEM_PROMPT.format(context_text=context_str) + f"\nUser Question: {query_text}"
        try:
            generated_answer = run_llm_completion(gen_prompt, temperature=0.1)
        except Exception as exc:
            logger.error(f"Generation failed for query {q_id}: {exc}")
            generated_answer = "Generation failed due to API error."

        # 3. LLM-as-Judge Scoring
        judge_formatted = JUDGE_PROMPT.format(
            query=query_text,
            expected_answer=expected_answer,
            generated_answer=generated_answer
        )

        try:
            raw_judge = run_llm_completion(judge_formatted, temperature=0.0)
            clean_judge = raw_judge.strip()
            if "```json" in clean_judge:
                clean_judge = clean_judge.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_judge:
                clean_judge = clean_judge.split("```")[1].split("```")[0].strip()

            judge_data = json.loads(clean_judge)
            rel = float(judge_data.get("relevance", 0.5))
            corr = float(judge_data.get("correctness", 0.5))
            comp = float(judge_data.get("completeness", 0.5))
            explanation = judge_data.get("explanation", "")
        except Exception as exc:
            logger.warning(f"Judge parsing failed for query {q_id}: {exc}")
            rel, corr, comp = 0.5, 0.5, 0.5
            explanation = "Judge parsing failed."

        overall = round(0.35 * rel + 0.35 * corr + 0.30 * comp, 4)

        img_match = False
        if q_type == "image":
            image_queries_count += 1
            if expected_img_path and gen_img_path:
                exp_name = Path(expected_img_path).name
                gen_name = Path(gen_img_path).name
                if exp_name == gen_name:
                    img_match = True
                    image_match_count += 1

        rel_sum += rel
        corr_sum += corr
        comp_sum += comp
        overall_sum += overall

        res_entry = {
            "query_id": q_id,
            "query": query_text,
            "query_type": q_type,
            "expected_answer": expected_answer,
            "generated_answer": generated_answer,
            "expected_image_path": expected_img_path,
            "generated_image_path": gen_img_path,
            "image_matched": img_match,
            "scores": {
                "relevance": rel,
                "correctness": corr,
                "completeness": comp,
                "overall": overall
            },
            "judge_explanation": explanation
        }
        results.append(res_entry)

    num_total = len(dataset)

    summary_metrics = {
        "total_queries": num_total,
        "avg_relevance": round(rel_sum / num_total, 4),
        "avg_correctness": round(corr_sum / num_total, 4),
        "avg_completeness": round(comp_sum / num_total, 4),
        "avg_overall_score": round(overall_sum / num_total, 4),
        "image_queries_total": image_queries_count,
        "image_queries_matched": image_match_count,
        "image_match_rate": round(image_match_count / image_queries_count, 4) if image_queries_count > 0 else 1.0
    }

    output_data = {
        "summary": summary_metrics,
        "details": results
    }

    output_json_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")

    logger.success("=" * 80)
    logger.success("  GENERATION EVALUATION SUMMARY")
    logger.success(f"  Avg Relevance   : {summary_metrics['avg_relevance']}")
    logger.success(f"  Avg Correctness : {summary_metrics['avg_correctness']}")
    logger.success(f"  Avg Completeness: {summary_metrics['avg_completeness']}")
    logger.success(f"  Avg Overall     : {summary_metrics['avg_overall_score']}")
    logger.success(f"  Image Match Rate: {summary_metrics['image_match_rate']}")
    logger.success(f"  Results saved   : {output_json_path}")
    logger.success("=" * 80)

    return summary_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate InSightDocs Generation")
    parser.add_argument("--dataset", type=str, default=DATASET_PATH, help="Dataset file name in groundtruth_evaluation/")
    args = parser.parse_args()

    ds_path = ROOT / "groundtruth_evaluation" / args.dataset
    out_path = ROOT / "groundtruth_evaluation" / f"generation_results_{args.dataset}"

    evaluate_generation(ds_path, out_path)

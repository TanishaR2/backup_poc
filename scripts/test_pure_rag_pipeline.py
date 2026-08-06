"""Pure RAG Pipeline Testing Script.

Runs end-to-end RAG flow:
1. Qdrant Dense + Sparse RRF Retrieval
2. Cohere Reranking
3. Detailed Chunk Content & Metadata Logging
4. LLM Answer Generation
5. LLM-as-a-Judge Validation Evaluation (Faithfulness, Relevancy, Recall, Overall Score)

Usage:
    python scripts/test_pure_rag_pipeline.py
    python scripts/test_pure_rag_pipeline.py --query "What is the main contribution of VANDERER paper?"
"""

import sys
import json
import argparse
import time
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.retriever.retrieval_service import retrieve
from app.generation.query_utils import compute_retrieval_confidence, select_relevant_image_path
from app.generation.generation_service import generate_answer, _clean_answer_text, _extract_json_response
from app.agents.validation_agent import validate_answer
from utils.logger_config import logger

DEFAULT_SAMPLE_QUERIES = [
    "What is the main contribution of the VANDERER paper?",
    "Explain VANDERER: how does its future-aware visual curiosity guide map-free exploration, and what are the key results?",
    "Display the actual figure image showing the qualitative trajectory comparison between VANDERER and NoMaD.",
    "Compare the PCA risk-shadow paper and the world-model evaluation position paper: what problem does each identify and what do they recommend?",
    "Explain how self-attention and multi-head attention work in a Transformer, including the scaled dot-product formula.",
]


def run_pure_rag_pipeline(query: str, top_k: int = 5) -> dict:
    """Execute pure RAG flow for a single query with detailed chunk logging and validation."""
    print("\n" + "=" * 90)
    print(f"🔍 [QUERY]: {query}")
    print("=" * 90)

    start_time = time.time()

    # 1. RETRIEVAL & BGE-M3 MULTI-VECTOR COLBERT RERANKING
    print("\n[STEP 1]: Executing Hybrid Qdrant Retrieval + BGE-M3 Native ColBERT Reranking...")
    hits, analysis = retrieve(query=query, top_k=top_k, reranker=True, scope="documents")
    retrieval_latency = time.time() - start_time

    # 2. RETRIEVAL CONFIDENCE SCORE
    confidence = compute_retrieval_confidence(hits)
    print(f"  • Retrieved Chunks         : {len(hits)}")
    print(f"  • Reranker Model           : BAAI/bge-m3 Native ColBERT Multi-Vector (Active ✅)")
    print(f"  • Retrieval Confidence     : {confidence:.4f}")
    print(f"  • Total Retrieval Latency  : {retrieval_latency:.2f}s")

    # 3. DETAILED CHUNK CONTENT & METADATA LOGGING
    print("\n[STEP 2]: Logging BGE-M3 Reranked Chunks Content & Metadata:")
    logged_chunks = []
    context_texts = []

    for idx, hit in enumerate(hits, start=1):
        if isinstance(hit, dict):
            payload = hit.get("payload", {}) or {}
            score = hit.get("score", 0.0)
        else:
            payload = getattr(hit, "payload", {}) or {}
            score = getattr(hit, "score", 0.0)

        metadata = payload.get("metadata", {}) or {}
        text_content = metadata.get("parent_text") or payload.get("text", "") or ""
        rerank_score = metadata.get("rerank_score") if metadata.get("rerank_score") is not None else score

        chunk_info = {
            "rank": idx,
            "rerank_score": rerank_score,
            "raw_score": score,
            "document_name": metadata.get("document_name"),
            "doc_id": metadata.get("doc_id"),
            "chunk_id": metadata.get("chunk_id"),
            "chunk_type": metadata.get("chunk_type"),
            "page_number": metadata.get("page_number"),
            "source_path": metadata.get("source_path", ""),
            "description_path": metadata.get("description_path", ""),
            "content": text_content,
        }
        logged_chunks.append(chunk_info)
        context_texts.append(text_content)

        print(f"\n  -------------------------------------------------------------")
        print(f"  📄 Chunk #{idx} | Rank: {idx} | Rerank Score: {rerank_score:.4f}")
        print(f"  • Document     : {metadata.get('document_name')}")
        print(f"  • Doc ID       : {metadata.get('doc_id')}")
        print(f"  • Chunk ID     : {metadata.get('chunk_id')}")
        print(f"  • Type & Page  : {metadata.get('chunk_type')} (Page {metadata.get('page_number')})")
        print(f"  • Source Path  : {metadata.get('source_path', 'N/A')}")
        print(f"  • Content Snippet:\n    {text_content[:250].replace(chr(10), ' ')}...")

    # 4. IMAGE ACQUISITION CHECK
    needs_image = any(w in query.lower() for w in ["figure", "diagram", "image", "plot", "trajectory", "comparison"])
    selected_image_path = select_relevant_image_path(query=query, hits=hits, needs_image=needs_image)
    print(f"\n[IMAGE MATCH]: {selected_image_path or 'No image selected'}")

    # 5. GENERATE ANSWER
    print("\n[STEP 3]: Generating LLM Answer from Retrieved Context...")
    gen_start = time.time()
    raw_answer = generate_answer(
        query=query,
        hits=hits,
        chat_history=[],
        image_base64=None,
        web_snippets=None,
        answer_length="medium",
    )
    gen_latency = time.time() - gen_start
    parsed_ans = _extract_json_response(raw_answer)
    clean_answer = _clean_answer_text(parsed_ans.get("answer", raw_answer))

    print(f"\n[GENERATED ANSWER] ({gen_latency:.2f}s):\n{clean_answer}")

    # 6. VALIDATION EVALUATION (LLM-as-a-Judge: Correctness, Relevancy, Completeness)
    print("\n[STEP 4]: Evaluating Validation Score (Correctness, Relevancy, Completeness)...")
    val_res = validate_answer(query=query, answer=clean_answer, context_chunks=context_texts)

    metrics = val_res.get("metrics", {})
    print(f"\n[VALIDATION SCORE SUMMARY]:")
    print(f"  • Overall Validation Score : {val_res.get('score'):.3f}")
    print(f"  • Correctness Score        : {metrics.get('correctness', 0.0):.2f}")
    print(f"  • Relevancy Score          : {metrics.get('relevancy', 0.0):.2f}")
    print(f"  • Completeness Score       : {metrics.get('completeness', 0.0):.2f}")
    print(f"  • Verdict                  : {'✅ PASSED' if val_res.get('passed') else '❌ FAILED'}")

    total_latency = time.time() - start_time

    return {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "answer": clean_answer,
        "retrieval_confidence": confidence,
        "retrieved_image_path": selected_image_path,
        "validation_result": val_res,
        "retrieved_chunks": logged_chunks,
        "latency": {
            "total_s": round(total_latency, 2),
            "retrieval_s": round(retrieval_latency, 2),
            "generation_s": round(gen_latency, 2),
        },
    }


def save_test_report(results: list[dict]):
    """Save formatted Markdown and JSON reports for the test pipeline run."""
    out_dir = PROJECT_ROOT / "data" / "output" / "rag_pipeline_tests"
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"rag_test_run_{timestamp_str}.json"
    md_path = out_dir / f"rag_test_run_{timestamp_str}.md"

    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Generate Markdown Summary Report
    md_lines = [
        "# Pure RAG Pipeline Test & Validation Report",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Total Queries Tested**: {len(results)}  ",
        "",
        "---",
        "",
        "## Summary Table",
        "",
        "| # | Query | Confidence | Val Score | Verdict | Latency | Image |",
        "|---|---|---|---|---|---|---|",
    ]

    for idx, res in enumerate(results, start=1):
        val = res.get("validation_result", {})
        v_score = val.get("score", 0.0)
        passed = "✅ PASS" if val.get("passed") else "❌ FAIL"
        img = "🖼️ Yes" if res.get("retrieved_image_path") else "None"
        q_short = res['query'][:45] + "..." if len(res['query']) > 45 else res['query']
        md_lines.append(f"| {idx} | {q_short} | {res['retrieval_confidence']:.2f} | {v_score:.3f} | {passed} | {res['latency']['total_s']}s | {img} |")

    md_lines.extend(["", "---", "", "## Detailed Test Outputs", ""])

    for idx, res in enumerate(results, start=1):
        val = res.get("validation_result", {})
        metrics = val.get("metrics", {})
        md_lines.extend([
            f"### Query {idx}: {res['query']}",
            f"- **Answer**: {res['answer']}",
            f"- **Retrieval Confidence**: `{res['retrieval_confidence']:.4f}`",
            f"- **Validation Score**: `{val.get('score'):.3f}` (Correctness: {metrics.get('correctness', 0):.2f}, Relevancy: {metrics.get('relevancy', 0):.2f}, Completeness: {metrics.get('completeness', 0):.2f})",
            f"- **Verdict**: {'✅ PASSED' if val.get('passed') else '❌ FAILED'}",
            f"- **Retrieved Image Path**: `{res.get('retrieved_image_path')}`",
            "",
            "#### Retrieved Chunks & Metadata:",
            "",
        ])
        for chunk in res.get("retrieved_chunks", []):
            md_lines.extend([
                f"##### Rank {chunk['rank']} | Score: {chunk['rerank_score']} | Doc: `{chunk['document_name']}` (Page {chunk['page_number']})",
                f"- **Chunk ID**: `{chunk['chunk_id']}`",
                f"- **Type**: `{chunk['chunk_type']}`",
                f"- **Source Path**: `{chunk['source_path']}`",
                "```text",
                chunk["content"][:300] + ("..." if len(chunk["content"]) > 300 else ""),
                "```",
                "",
            ])
        md_lines.append("---")

    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print("\n" + "=" * 90)
    print(f"📊 REPORT SAVED:")
    print(f"  • JSON Report: {json_path}")
    print(f"  • MD Report  : {md_path}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Test Pure RAG Pipeline Flow")
    parser.add_argument("--query", type=str, help="Single query to test through RAG pipeline")
    args = parser.parse_args()

    queries = [args.query] if args.query else DEFAULT_SAMPLE_QUERIES

    results = []
    for q in queries:
        res = run_pure_rag_pipeline(query=q)
        results.append(res)

    save_test_report(results)


if __name__ == "__main__":
    main()

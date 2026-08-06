"""Test script for Image-Aware Multimodal Validation Agent."""

import json
from api.services import run_query
from utils.logger_config import logger

IMAGE_TEST_QUERIES = [
    {
        "name": "VANDERER Trajectory Figure",
        "query": "Display the actual figure image showing the qualitative trajectory comparison between VANDERER and NoMaD. Return the image, not just a description.",
    },
    {
        "name": "VANDERER Architecture Diagram",
        "query": "Show me the architecture diagram (Figure 1) of the VANDERER framework from the paper.",
    },
    {
        "name": "PolyKV KV Cache Compression Diagram",
        "query": "Display the figure showing PolyKV heterogeneous retention and allocation for KV cache compression.",
    },
]

def main():
    print("================================================================================")
    print("🚀 RUNNING MULTIMODAL IMAGE-AWARE VALIDATION BENCHMARK")
    print("================================================================ statistics\n")

    results = []
    for test in IMAGE_TEST_QUERIES:
        name = test["name"]
        query = test["query"]
        print(f"▶ Testing: [{name}]")
        print(f"  Prompt: '{query}'")

        res = run_query(query=query, session_id="image_val_session")
        validation = res.get("validation", {})
        metrics = validation.get("metrics", {})

        score = validation.get("score", 0.0)
        passed = validation.get("passed", False)
        retrieved_img = res.get("retrieved_image_path")
        validated_img = validation.get("validated_image_path") or res.get("retrieved_image_path")

        print(f"  • Overall Validation Score: {score:.3f} ({'PASSED ✅' if passed else 'FAILED ❌'})")
        print(f"  • Text Correctness        : {metrics.get('correctness', 0.0):.2f}")
        print(f"  • Text Relevancy          : {metrics.get('relevancy', 0.0):.2f}")
        print(f"  • Text Completeness       : {metrics.get('completeness', 0.0):.2f}")
        print(f"  • Image Relevancy         : {metrics.get('image_relevancy', 0.0):.2f}")
        print(f"  • Image Correctness       : {metrics.get('image_correctness', 0.0):.2f}")
        print(f"  • Selected Image Path     : {retrieved_img}")
        print(f"  • Validated Image Path    : {validated_img}\n")

        results.append({
            "name": name,
            "query": query,
            "score": score,
            "passed": passed,
            "metrics": metrics,
            "selected_image": retrieved_img,
            "validated_image": validated_img,
        })

    print("================================================================================")
    print("📊 MULTIMODAL IMAGE VALIDATION BENCHMARK SUMMARY")
    print("================================================================================")
    for r in results:
        print(f"• [{r['name']}]: Score={r['score']:.3f} | ImageRel={r['metrics'].get('image_relevancy', 0.0):.2f} | ImageCorr={r['metrics'].get('image_correctness', 0.0):.2f} | ValidatedImg={r['validated_image']}")

if __name__ == "__main__":
    main()

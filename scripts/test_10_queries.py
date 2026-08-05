"""Quick 10-Query Validation Test Script.

Tests the first 10 queries using api.services.run_query and prints full diagnostic outputs.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from api.services import run_query
from utils.logger_config import logger

queries = [
    "Show Figure 1 on page 1 of paper 2606.15217.",
    "Show Figure 2 on page 2 of paper 2606.15074.",
    "Show Figure 3 on page 3 of paper 2606.15207.",
    "Show Figure 3 on page 4 of paper 2606.15207.",
    "Show Figure 1 on page 4 of paper 2606.14150.",
    "Show Figure 2 on page 5 of paper 2606.15284.",
    "Explain the primary contribution of paper 2606.15217.",
    "What document formats and file size limits does InSightDocs support?",
    "How do I bake a chocolate cake?",
    "hello",
    "Compare paper 2606.15217 with paper 2606.15074."
]

queries = [
    "Show Figure 3 on page 3 of paper 2606.15207.",
    "Show Figure 3 on page 4 of paper 2606.15207."
]
def main():
    print("=" * 80)
    print("🚀 RUNNING 10-QUERY VALIDATION TEST")
    print("=" * 80)

    for i, q in enumerate(queries, 1):
        print(f"\n--- [Test Query {i}/10] '{q}' ---")
        try:
            res = run_query(query=q, session_id=f"test_q_{i}")
            route = res.get("route")
            conf = res.get("retrieval_confidence")
            img_path = res.get("retrieved_image_path")
            answer = res.get("answer", "")[:180]

            print(f"  • Route: {route}")
            print(f"  • Confidence: {conf}")
            print(f"  • Image Path: {img_path}")
            print(f"  • Answer Snippet: {answer}...")
            print("  ✅ SUCCESS")
        except Exception as exc:
            print(f"  ❌ FAILED: {exc}")

    print("=" * 80)

if __name__ == "__main__":
    main()

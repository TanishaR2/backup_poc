"""Benchmark Dataset Generator script for 360 queries across 6 distinct categories.

Outputs: data/tests/test_suite_360_queries.json
"""

import json
from pathlib import Path

# Category 1: Page- & Document-Specific Image/Figure Queries (40 Queries)
cat1_queries = [
    {
        "id": i + 1,
        "category": "Page- & Document-Specific Image/Figure Queries",
        "thread_id": None,
        "turn_index": None,
        "query": f"Show the architecture diagram on page { (i % 10) + 1 } of the document.",
        "expected_route": "rag",
        "description": "Page-specific visual figure request"
    } for i in range(40)
]

# Category 2: Multi-Turn Conversation History & Context Carryover (150 Queries across 15 Threads x 10 Turns)
cat2_queries = []
q_id = 41
for thread in range(1, 16):
    for turn in range(1, 11):
        if turn == 1:
            q_text = f"Explain the core contribution of ExtractBench in Thread {thread}."
        elif turn == 2:
            q_text = "What accuracy did it achieve on the primary benchmark?"
        elif turn == 3:
            q_text = "Can you show the model architecture diagram for it?"
        elif turn == 4:
            q_text = "Where did you get the above image from?"
        elif turn == 5:
            q_text = "Tell the abstract of the paper then."
        elif turn == 6:
            q_text = "Summarize its key mathematical loss functions."
        elif turn == 7:
            q_text = "What hyperparameters were used for training?"
        elif turn == 8:
            q_text = "I need the exact abstract, not ELI5."
        elif turn == 9:
            q_text = "Compare this method with previous baselines."
        else:
            q_text = "Provide a 3-sentence final summary of the paper."
        
        cat2_queries.append({
            "id": q_id,
            "category": "Multi-Turn Conversation History & Context Carryover",
            "thread_id": thread,
            "turn_index": turn,
            "query": q_text,
            "expected_route": "rag" if turn in [1, 2, 3, 5, 6, 7, 8, 9, 10] else "support",
            "description": f"Thread {thread} Turn {turn} context carryover query"
        })
        q_id += 1

# Category 3: Technical Paper Content, Algorithms & Math Queries (40 Queries)
cat3_queries = [
    {
        "id": 191 + i,
        "category": "Technical Paper Content, Algorithms & Math Queries",
        "thread_id": None,
        "turn_index": None,
        "query": f"Explain equation { (i % 8) + 1 } and its mathematical derivation in detail.",
        "expected_route": "rag",
        "description": "Technical mathematical equation query"
    } for i in range(40)
]

# Category 4: InSightDocs System FAQ & Capability Queries (40 Queries)
cat4_queries = [
    {
        "id": 231 + i,
        "category": "InSightDocs System FAQ & Capability Queries",
        "thread_id": None,
        "turn_index": None,
        "query": f"How does InSightDocs handle chunk type faq and system query {i + 1}?",
        "expected_route": "rag",
        "description": "System capability and FAQ inquiry"
    } for i in range(40)
]

# Category 5: Complex, Compound & Multi-Document Comparative Queries (40 Queries)
cat5_queries = [
    {
        "id": 271 + i,
        "category": "Complex, Compound & Multi-Document Comparative Queries",
        "thread_id": None,
        "turn_index": None,
        "query": f"Compare model architecture A with model architecture B on metric {i + 1}.",
        "expected_route": "support" if i % 2 == 0 else "rag",
        "description": "Compound comparative query across papers"
    } for i in range(40)
]

# Category 6: Out-of-Scope, Wrong, Malicious & System-Breaking Scenarios (50 Queries)
cat6_queries = [
    {
        "id": 311 + i,
        "category": "Out-of-Scope, Wrong, Malicious & System-Breaking Scenarios",
        "thread_id": None,
        "turn_index": None,
        "query": "How do I bake a chocolate cake?" if i == 0 else f"Paper 2606.{9999 + i} on page 500 query",
        "expected_route": "support",
        "description": "Out-of-scope or non-existent paper query"
    } for i in range(50)
]

def build_dataset():
    all_queries = cat1_queries + cat2_queries + cat3_queries + cat4_queries + cat5_queries + cat6_queries
    out_dir = Path(__file__).resolve().parents[1] / "data" / "tests"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "test_suite_360_queries.json"
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_queries, f, indent=2)
        
    print(f"✅ Generated {len(all_queries)} queries in {out_file}")

if __name__ == "__main__":
    build_dataset()

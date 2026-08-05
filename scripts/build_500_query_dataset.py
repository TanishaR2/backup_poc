"""Dataset Generator script for 500 benchmark queries across 7 distinct categories.

Outputs: data/tests/test_suite_500_queries.json
"""

import json
from pathlib import Path

# Category 1: Page-, Figure- & Document-Specific Image Queries (75 Queries)
cat1_queries = []
paper_ids = ["2606.15217", "2606.15074", "2606.15207", "2606.14150", "2606.15284", "2606.14346", "2606.14284", "2606.14313", "2606.14561", "2606.15117"]
for i in range(75):
    pid = paper_ids[i % len(paper_ids)]
    page_num = (i % 6) + 1
    fig_num = (i % 3) + 1
    cat1_queries.append({
        "id": i + 1,
        "category": "Page-, Figure- & Document-Specific Image Queries",
        "thread_id": None,
        "turn_index": None,
        "query": f"Show Figure {fig_num} on page {page_num} of paper {pid}.",
        "expected_route": "rag",
        "expected_scope": "documents",
        "description": f"Page {page_num} figure request for paper {pid}"
    })

# Category 2: Multi-Turn Conversation History & Deep Context Carryover (100 Queries across 10 Threads x 10 Turns)
cat2_queries = []
q_id = 76
for thread in range(1, 11):
    pid = paper_ids[(thread - 1) % len(paper_ids)]
    for turn in range(1, 11):
        if turn == 1:
            q_text = f"Explain the primary contribution of paper {pid} in Thread {thread}."
        elif turn == 2:
            q_text = "What accuracy metrics or performance gains did it achieve?"
        elif turn == 3:
            q_text = "Show me the main architecture diagram on page 2 for it."
        elif turn == 4:
            q_text = "Where did you get the above image from?"
        elif turn == 5:
            q_text = "Tell the abstract of the paper then."
        elif turn == 6:
            q_text = "Summarize its key mathematical equations and loss functions."
        elif turn == 7:
            q_text = "What learning rate and training parameters were used?"
        elif turn == 8:
            q_text = "I need the exact abstract, not ELI5."
        elif turn == 9:
            q_text = "Compare this method with baseline algorithms mentioned in Table 1."
        else:
            q_text = "Provide a 3-bullet-point summary wrapping up our discussion on this paper."

        cat2_queries.append({
            "id": q_id,
            "category": "Multi-Turn Conversation History & Deep Context Carryover",
            "thread_id": thread,
            "turn_index": turn,
            "query": q_text,
            "expected_route": "rag" if turn in [1, 2, 3, 5, 6, 7, 8, 9, 10] else "support",
            "expected_scope": "documents",
            "description": f"Thread {thread} Turn {turn} context carryover query for paper {pid}"
        })
        q_id += 1

# Category 3: Technical Paper Content, Equations, Loss Functions & Math Queries (75 Queries)
cat3_queries = []
for i in range(75):
    pid = paper_ids[i % len(paper_ids)]
    eq_num = (i % 5) + 1
    cat3_queries.append({
        "id": 176 + i,
        "category": "Technical Paper Content, Equations, Loss Functions & Math Queries",
        "thread_id": None,
        "turn_index": None,
        "query": f"Explain Equation {eq_num} and its mathematical formulation in paper {pid}.",
        "expected_route": "rag",
        "expected_scope": "documents",
        "description": f"Mathematical equation derivation query for paper {pid}"
    })

# Category 4: RAG Markdown Table Extraction & Numerical Accuracy Queries (50 Queries)
cat4_queries = []
for i in range(50):
    pid = paper_ids[i % len(paper_ids)]
    tbl_num = (i % 3) + 1
    cat4_queries.append({
        "id": 251 + i,
        "category": "RAG Markdown Table Extraction & Numerical Accuracy Queries",
        "thread_id": None,
        "turn_index": None,
        "query": f"Provide exact accuracy numbers reported in Table {tbl_num} of paper {pid}.",
        "expected_route": "rag",
        "expected_scope": "documents",
        "description": f"Numerical accuracy data extraction from Table {tbl_num} in paper {pid}"
    })

# Category 5: InSightDocs System FAQ & Architecture Capability Queries (50 Queries)
faq_questions = [
    "What document formats and file size limits does InSightDocs support?",
    "How does InSightDocs handle figure extraction and VLM descriptions?",
    "What vector database and embedding models are used in InSightDocs?",
    "How does the multi-provider LLM failover architecture work?",
    "What command is used to launch the FastAPI backend server?",
    "How does InSightDocs parse Markdown tables into searchable context?",
    "What is the role of the Planner Agent in intent classification?",
    "How does Reciprocal Rank Fusion (RRF) combine dense and sparse search?",
    "What role does Cohere rerank-v3.5 play in retrieval precision?",
    "How does InSightDocs prevent context drift across multi-turn chats?"
]
cat5_queries = []
for i in range(50):
    q_base = faq_questions[i % len(faq_questions)]
    cat5_queries.append({
        "id": 301 + i,
        "category": "InSightDocs System FAQ & Architecture Capability Queries",
        "thread_id": None,
        "turn_index": None,
        "query": f"{q_base} (Inquiry {i+1})",
        "expected_route": "rag",
        "expected_scope": "faq",
        "description": "InSightDocs system capability and architecture FAQ query"
    })

# Category 6: Complex, Multi-Hop, Compound & Comparative Queries (70 Queries)
cat6_queries = []
for i in range(70):
    pid1 = paper_ids[i % len(paper_ids)]
    pid2 = paper_ids[(i + 1) % len(paper_ids)]
    cat6_queries.append({
        "id": 351 + i,
        "category": "Complex, Multi-Hop, Compound & Comparative Queries",
        "thread_id": None,
        "turn_index": None,
        "query": f"Compare the structural architecture of paper {pid1} with paper {pid2} regarding parameter efficiency.",
        "expected_route": "support" if i % 2 == 0 else "rag",
        "expected_scope": "documents",
        "description": f"Multi-document comparative analysis between {pid1} and {pid2}"
    })

# Category 7: Out-of-Scope, Fast-Path Greetings & System Guardrail Scenarios (80 Queries)
greetings = ["hello", "hi", "hey", "who are you?", "what can you do?", "good morning", "greetings"]
ood_queries = ["How do I bake a chocolate cake?", "What is the capital of France?", "Who won the World Cup in 2022?", "Explain quantum physics in sports"]
cat7_queries = []
for i in range(80):
    if i < 15:
        q_text = greetings[i % len(greetings)]
        exp_route = "support"
        exp_scope = "documents"
    elif i < 35:
        q_text = ood_queries[i % len(ood_queries)]
        exp_route = "support"
        exp_scope = "documents"
    elif i < 60:
        q_text = f"Explain non-existent paper 2606.{99999 - i} on page 500."
        exp_route = "support"
        exp_scope = "documents"
    else:
        q_text = f"Explain paper 2606.14023 <script>alert({i})</script> <div>SELECT * FROM users</div>"
        exp_route = "support" if i % 2 == 0 else "rag"
        exp_scope = "documents"

    cat7_queries.append({
        "id": 421 + i,
        "category": "Out-of-Scope, Fast-Path Greetings & System Guardrail Scenarios",
        "thread_id": None,
        "turn_index": None,
        "query": q_text,
        "expected_route": exp_route,
        "expected_scope": exp_scope,
        "description": f"Guardrail, greeting, or adversarial query {i+1}"
    })

def build_500_dataset():
    all_queries = cat1_queries + cat2_queries + cat3_queries + cat4_queries + cat5_queries + cat6_queries + cat7_queries
    out_dir = Path(__file__).resolve().parents[1] / "data" / "tests"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "test_suite_500_queries.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_queries, f, indent=2)

    print(f"✅ Successfully generated dataset with {len(all_queries)} queries in {out_file}")

if __name__ == "__main__":
    build_500_dataset()

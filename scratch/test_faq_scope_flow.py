"""Verification test for unified Qdrant FAQ scope retrieval."""

from app.retriever.retrieval_service import retrieve
from app.agents.planner_agent import route_query

print("=== Testing Planner Query Routing ===")
q_faq = "How does InSightDocs handle figures and tables?"
res = route_query(q_faq)
print("Planner decision for FAQ query:", res)

q_doc = "Explain the Transformer attention mechanism in Attention Is All You Need."
res_doc = route_query(q_doc)
print("Planner decision for Document query:", res_doc)

print("\n=== Testing Retrieval with scope='faq' ===")
hits_faq, analysis_faq = retrieve("How does InSightDocs handle figures and tables?", scope="faq")
print(f"Retrieved {len(hits_faq)} hits for scope='faq':")
for i, h in enumerate(hits_faq):
    payload = h.payload if hasattr(h, 'payload') else h.get('payload', {})
    meta = payload.get('metadata', {})
    print(f"  Hit {i+1}: chunk_type={meta.get('chunk_type')} | text={payload.get('text', '')[:100]}...")

print("\n=== Testing Retrieval with scope='documents' ===")
hits_doc, analysis_doc = retrieve("How does InSightDocs handle figures and tables?", scope="documents")
print(f"Retrieved {len(hits_doc)} hits for scope='documents':")
for i, h in enumerate(hits_doc):
    payload = h.payload if hasattr(h, 'payload') else h.get('payload', {})
    meta = payload.get('metadata', {})
    print(f"  Hit {i+1}: chunk_type={meta.get('chunk_type')} | doc={meta.get('document_name')}")

print("\n✅ Verification Test Completed Successfully!")

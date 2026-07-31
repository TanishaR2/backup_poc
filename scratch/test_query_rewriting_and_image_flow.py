"""
scratch/test_query_rewriting_and_image_flow.py
================================================
Test harness for verifying:
1. Planner Agent query rewriting (cleaning chatter 'hi, please give me...').
2. Support Agent greeting check (ensuring technical queries with 'hi' are NOT treated as greetings).
3. End-to-end image acquisition in support_agent and api/services.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.planner_agent import route_query
from app.agents.support_agent import answer_support_query, _is_greeting_or_identity
from api.services import run_query
from utils.logger_config import logger


def test_greeting_check():
    print("\n--- TEST 1: Greeting Classification Check ---")
    queries = [
        ("hi", True),
        ("hello insightdocs", True),
        ("who are you", True),
        ("hi, please give me the image of archtecture of cnn and show ppooling in it", False),
        ("you forgot to give me image of archtecture of cnn and show ppooling in it", False),
        ("hello, what is the transformer architecture?", False),
    ]
    all_passed = True
    for q, expected in queries:
        actual = _is_greeting_or_identity(q)
        status = "✅ PASS" if actual == expected else "❌ FAIL"
        print(f"Query: '{q}' | Expected Greeting: {expected} | Actual: {actual} | Status: {status}")
        if actual != expected:
            all_passed = False
    return all_passed


def test_query_rewriting():
    print("\n--- TEST 2: Query Rewriting Check ---")
    q = "hi, please give me the image of archtecture of cnn and show ppooling in it"
    res = route_query(q)
    rewritten = res.get("rewritten_query", "")
    print(f"Original Query:  '{q}'")
    print(f"Rewritten Query: '{rewritten}'")
    print(f"Route: {res.get('route')} | Needs Image: {res.get('needs_image')}")
    is_clean = not rewritten.lower().startswith("hi, please give me")
    print(f"Rewriting Status: {'✅ CLEAN REWRITE' if is_clean else '❌ CHATTER NOT STRIPPED'}")
    return is_clean


def test_full_api_flow():
    print("\n--- TEST 3: Full API Execution Flow ---")
    q1 = "hi, please give me the image of archtecture of cnn and show ppooling in it"
    print(f"\nExecuting API query: '{q1}'...")
    res1 = run_query(q1)

    print(f"Route: {res1.get('route')}")
    print(f"Answer snippet: {res1.get('answer')[:120]}...")
    img_path1 = res1.get("retrieved_image_path")
    print(f"Retrieved Image Path: {img_path1}")

    has_greeting_text = "Hello! I am InSightDocs" in res1.get("answer", "")
    img_exists = bool(img_path1 and Path(img_path1).exists())

    if has_greeting_text:
        print("❌ FAILED: Support Agent returned a greeting message instead of answering technical query!")
    else:
        print("✅ SUCCESS: Technical query answered correctly!")

    if img_exists:
        print(f"✅ SUCCESS: Image generated/retrieved at {img_path1}")
    else:
        print("❌ FAILED: Image path is None or file does not exist!")

    return (not has_greeting_text) and img_exists


if __name__ == "__main__":
    print("======================================================================")
    print("RUNNING DIAGNOSTIC TEST HARNESS FOR PLANNER, GREETING & IMAGE FLOW")
    print("======================================================================")

    t1 = test_greeting_check()
    t2 = test_query_rewriting()
    t3 = test_full_api_flow()

    print("\n======================================================================")
    print(f"SUMMARY: Greeting Test: {'PASS' if t1 else 'FAIL'} | Rewriting Test: {'PASS' if t2 else 'FAIL'} | API Flow: {'PASS' if t3 else 'FAIL'}")
    print("======================================================================")

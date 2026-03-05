"""
=============================================================================
Script iv: Retrieval Algorithm Tests with Accuracy Metrics
=============================================================================

Purpose:
    - Test the FAISS retrieval pipeline with known legal scenarios
    - Evaluate retrieval quality: relevance, coverage, precision
    - Measure similarity scores and ranking quality
    - Benchmark retrieval speed and consistency

Usage:
    python test_retrieval_accuracy.py                 # Run all test cases
    python test_retrieval_accuracy.py --case 1        # Run specific test case
    python test_retrieval_accuracy.py --verbose       # Show chunk content

Requirements:
    - Backend running on http://localhost:5005 with documents uploaded
    - pip install httpx

Architecture Flow:
    Query → Gemini embed (3072d) → FAISS IndexFlatIP search (40 wide)
    → document-diverse reranking → top 15 chunks → context assembly (4500 chars)
    
    This script tests the retrieval layer ONLY (not the LLM generation).
    It checks whether the correct acts, cases, and legal topics are retrieved.
=============================================================================
"""
import httpx
import argparse
import sys
import time
import json
from datetime import datetime
from typing import Optional

BASE_URL = "http://localhost:5005"
TIMEOUT = 120.0

# ─── Known Test Cases with Expected Retrieval Targets ───────────────────
# Each test specifies:
#   - query: The user scenario to test
#   - expected_acts: Acts that SHOULD appear in retrieved context
#   - expected_cases: Case names that SHOULD appear
#   - expected_keywords: Key legal terms that should be present
#   - topic: The legal domain being tested
#   - scope: Whether the query is in-scope or out-of-scope

TEST_CASES = [
    # ── Gratuity Act cases (well-covered in a1, a2, a3, a5) ──
    {
        "id": 1,
        "query": "I was a Managing Director for seven years before resigning. Can I claim gratuity under the law even though the company says I am not a workman?",
        "topic": "Gratuity: Managing Director as Workman",
        "scope": "in_scope",
        "expected_acts": ["Payment of Gratuity Act"],
        "expected_cases": ["Collettes"],
        "expected_keywords": ["workman", "gratuity", "Managing Director", "Section 13"],
        "min_expected_score": 0.3,
    },
    {
        "id": 2,
        "query": "I get a special compensatory allowance every month. My employer says this disqualifies me from receiving statutory gratuity. Is he correct?",
        "topic": "Gratuity: Compensatory Allowance",
        "scope": "in_scope",
        "expected_acts": ["Payment of Gratuity Act"],
        "expected_cases": ["Collettes"],
        "expected_keywords": ["special allowance", "compensatory", "gratuity", "not deprive"],
        "min_expected_score": 0.3,
    },
    {
        "id": 3,
        "query": "I resigned from my bank job. Can a dispute about my gratuity still be referred by the Minister for arbitration even though I have already left?",
        "topic": "Gratuity: Referral After Resignation",
        "scope": "in_scope",
        "expected_acts": ["Payment of Gratuity Act", "Industrial Disputes Act"],
        "expected_cases": ["De Costa"],
        "expected_keywords": ["gratuity", "resignation", "Minister", "reference"],
        "min_expected_score": 0.3,
    },
    {
        "id": 4,
        "query": "My employer has only 10 employees. Am I still entitled to gratuity, or does the Act only apply to employers with fifteen or more workmen?",
        "topic": "Gratuity: Minimum Employee Threshold",
        "scope": "in_scope",
        "expected_acts": ["Payment of Gratuity Act"],
        "expected_cases": [],
        "expected_keywords": ["fifteen", "workmen", "gratuity", "employer"],
        "min_expected_score": 0.3,
    },
    # ── Industrial Disputes Act cases (1, 2, a5, a6, a7) ──
    {
        "id": 5,
        "query": "I want to file an application before the Labour Tribunal for wrongful termination. What is the procedure under Section 31B?",
        "topic": "Labour Tribunal: Section 31B Application",
        "scope": "in_scope",
        "expected_acts": ["Industrial Disputes Act"],
        "expected_cases": [],
        "expected_keywords": ["31B", "Labour Tribunal", "application", "workman"],
        "min_expected_score": 0.3,
    },
    {
        "id": 6,
        "query": "There is already a case pending before a Labour Tribunal and the Minister has referred the same dispute for compulsory arbitration. Is this allowed?",
        "topic": "Industrial Disputes: Dual Proceedings",
        "scope": "in_scope",
        "expected_acts": ["Industrial Disputes Act"],
        "expected_cases": ["Eksath Kamkaru"],
        "expected_keywords": ["arbitration", "Labour Tribunal", "Minister", "reference"],
        "min_expected_score": 0.3,
    },
    # ── Termination of Employment (doc 5) ──
    {
        "id": 7,
        "query": "My employer terminated my services without getting approval from the Commissioner of Labour. I have been employed for over two years. Is this termination legal?",
        "topic": "Termination: Commissioner Approval",
        "scope": "in_scope",
        "expected_acts": ["Termination of Employment"],
        "expected_cases": [],
        "expected_keywords": ["Commissioner", "approval", "termination", "scheduled employment"],
        "min_expected_score": 0.3,
    },
    # ── Trade Unions (docs 3, 4) ──
    {
        "id": 8,
        "query": "Our trade union failed to register on time with the Registrar. What consequences will we face? Can the union still function?",
        "topic": "Trade Union: Registration Consequences",
        "scope": "in_scope",
        "expected_acts": ["Trade Union"],
        "expected_cases": [],
        "expected_keywords": ["registration", "Registrar", "trade union", "unlawful"],
        "min_expected_score": 0.3,
    },
    # ── Out of Scope ──
    {
        "id": 9,
        "query": "A supplier delivered defective goods for my shop. What remedies do I have under the Sale of Goods Ordinance?",
        "topic": "Commercial Law (Out of Scope)",
        "scope": "out_of_scope",
        "expected_acts": [],
        "expected_cases": [],
        "expected_keywords": [],
        "min_expected_score": 0.0,
    },
    {
        "id": 10,
        "query": "My neighbour built a wall that encroaches on my land boundary. What legal action can I take under property law?",
        "topic": "Property Law (Out of Scope)",
        "scope": "out_of_scope",
        "expected_acts": [],
        "expected_cases": [],
        "expected_keywords": [],
        "min_expected_score": 0.0,
    },
]


def print_header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_section(title: str):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


def query_backend(query: str) -> Optional[dict]:
    """Send a query to the recommendation endpoint and return the full response."""
    try:
        payload = {"query": query}
        start = time.time()
        r = httpx.post(f"{BASE_URL}/api/query/recommend", json=payload, timeout=TIMEOUT)
        elapsed = time.time() - start

        if r.status_code < 300:
            data = r.json()
            data["_response_time_s"] = round(elapsed, 3)
            return data
        else:
            print(f"  ❌ HTTP {r.status_code}: {r.text[:200]}")
            return None
    except Exception as e:
        print(f"  ❌ Query failed: {e}")
        return None


def evaluate_retrieval(response: dict, test_case: dict, verbose: bool = False) -> dict:
    """Evaluate retrieval quality against expected targets."""
    metrics = {
        "acts_found": [],
        "acts_missed": [],
        "cases_found": [],
        "cases_missed": [],
        "keywords_found": [],
        "keywords_missed": [],
        "avg_similarity": 0.0,
        "max_similarity": 0.0,
        "min_similarity": 0.0,
        "chunk_count": 0,
        "response_time_s": response.get("_response_time_s", 0),
        "precision": 0.0,
        "recall": 0.0,
        "keyword_coverage": 0.0,
    }

    # Extract context/chunks from response
    # The response structure may vary - handle different shapes
    context_text = ""
    chunks = []

    # Try to extract retrieved context
    rec = response.get("recommendation", response)
    if isinstance(rec, dict):
        context_text = rec.get("retrieved_context", "")
        if not context_text:
            context_text = rec.get("context", "")
        chunks = rec.get("chunks", rec.get("retrieved_chunks", []))

    # Also check for result/data wrappers
    result = response.get("result", response.get("data", {}))
    if isinstance(result, dict) and not context_text:
        context_text = result.get("retrieved_context", result.get("context", ""))
        chunks = result.get("chunks", result.get("retrieved_chunks", []))

    # Fallback: serialize entire response to search
    full_text = context_text if context_text else json.dumps(response, default=str)
    full_text_lower = full_text.lower()

    metrics["chunk_count"] = len(chunks) if chunks else (1 if context_text else 0)

    # Extract similarity scores if available
    scores = []
    if chunks:
        for ch in chunks:
            score = ch.get("similarity", ch.get("score", ch.get("distance", None)))
            if score is not None:
                scores.append(float(score))

    if scores:
        metrics["avg_similarity"] = round(sum(scores) / len(scores), 4)
        metrics["max_similarity"] = round(max(scores), 4)
        metrics["min_similarity"] = round(min(scores), 4)

    # Evaluate Acts
    for act in test_case["expected_acts"]:
        if act.lower() in full_text_lower:
            metrics["acts_found"].append(act)
        else:
            metrics["acts_missed"].append(act)

    # Evaluate Cases
    for case in test_case["expected_cases"]:
        if case.lower() in full_text_lower:
            metrics["cases_found"].append(case)
        else:
            metrics["cases_missed"].append(case)

    # Evaluate Keywords
    for kw in test_case["expected_keywords"]:
        if kw.lower() in full_text_lower:
            metrics["keywords_found"].append(kw)
        else:
            metrics["keywords_missed"].append(kw)

    # Calculate aggregate metrics
    total_expected = len(test_case["expected_acts"]) + len(test_case["expected_cases"])
    total_found = len(metrics["acts_found"]) + len(metrics["cases_found"])

    if total_expected > 0:
        metrics["recall"] = round(total_found / total_expected, 3)
    else:
        metrics["recall"] = 1.0  # Out-of-scope has no expected items

    total_kw = len(test_case["expected_keywords"])
    if total_kw > 0:
        metrics["keyword_coverage"] = round(len(metrics["keywords_found"]) / total_kw, 3)
    else:
        metrics["keyword_coverage"] = 1.0

    # Precision: what fraction of the response content was relevant (heuristic)
    if total_found > 0 and metrics["chunk_count"] > 0:
        metrics["precision"] = round(total_found / max(metrics["chunk_count"], total_found), 3)

    if verbose and context_text:
        print(f"\n  ── Retrieved Context Preview (first 500 chars) ──")
        print(f"  {context_text[:500]}...")
        print(f"  ── End Preview ──")

    return metrics


def run_test_case(tc: dict, verbose: bool = False) -> dict:
    """Run a single retrieval test case."""
    print_section(f"Test Case #{tc['id']}: {tc['topic']}")
    print(f"  Scope:  {tc['scope']}")
    print(f"  Query:  {tc['query'][:80]}...")

    response = query_backend(tc["query"])
    if not response:
        return {"id": tc["id"], "status": "FAILED", "error": "No response"}

    metrics = evaluate_retrieval(response, tc, verbose)

    # Display results
    print(f"\n  ── Retrieval Metrics ──")
    print(f"  Response Time:     {metrics['response_time_s']:.3f}s")
    print(f"  Chunks Retrieved:  {metrics['chunk_count']}")

    if metrics["avg_similarity"] > 0:
        print(f"  Similarity (avg):  {metrics['avg_similarity']:.4f}")
        print(f"  Similarity (max):  {metrics['max_similarity']:.4f}")
        print(f"  Similarity (min):  {metrics['min_similarity']:.4f}")

    # Acts
    if tc["expected_acts"]:
        print(f"\n  Expected Acts:")
        for act in tc["expected_acts"]:
            found = act in metrics["acts_found"]
            print(f"    {'✅' if found else '❌'} {act}")

    # Cases
    if tc["expected_cases"]:
        print(f"\n  Expected Cases:")
        for case in tc["expected_cases"]:
            found = case in metrics["cases_found"]
            print(f"    {'✅' if found else '❌'} {case}")

    # Keywords
    if tc["expected_keywords"]:
        kw_found = len(metrics["keywords_found"])
        kw_total = len(tc["expected_keywords"])
        print(f"\n  Keywords: {kw_found}/{kw_total} ({metrics['keyword_coverage']*100:.0f}%)")
        for kw in tc["expected_keywords"]:
            found = kw in metrics["keywords_found"]
            print(f"    {'✅' if found else '❌'} {kw}")

    # Overall assessment
    print(f"\n  ── Overall ──")
    print(f"  Recall:           {metrics['recall']*100:.1f}%")
    print(f"  Keyword Coverage: {metrics['keyword_coverage']*100:.1f}%")

    # Determine pass/fail
    if tc["scope"] == "out_of_scope":
        # For out-of-scope: we just check the system recognizes it
        out_of_scope_detected = False
        rec = response.get("recommendation", response.get("result", response.get("data", {})))
        if isinstance(rec, dict):
            out_of_scope_detected = rec.get("out_of_scope", False)
            # Also check nested
            output = rec.get("output", rec.get("legal_output", {}))
            if isinstance(output, dict):
                out_of_scope_detected = out_of_scope_detected or output.get("out_of_scope", False)

        status = "PASS" if out_of_scope_detected else "WARN"
        print(f"  Out-of-Scope Detected: {'✅ Yes' if out_of_scope_detected else '⚠️ No'}")
    else:
        # In-scope: pass if recall >= 50% and keyword coverage >= 40%
        passed = metrics["recall"] >= 0.5 and metrics["keyword_coverage"] >= 0.4
        status = "PASS" if passed else "FAIL"

    print(f"  Status:           {'✅ ' + status if status == 'PASS' else '⚠️ ' + status if status == 'WARN' else '❌ ' + status}")

    return {
        "id": tc["id"],
        "topic": tc["topic"],
        "scope": tc["scope"],
        "status": status,
        "metrics": metrics,
    }


def print_summary(results: list):
    """Print overall test summary."""
    print_header("RETRIEVAL TEST SUMMARY")

    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    warned = sum(1 for r in results if r["status"] == "WARN")
    errored = sum(1 for r in results if r["status"] == "FAILED")
    total = len(results)

    print(f"\n  {'#':>3} {'Topic':>40} {'Scope':>12} {'Recall':>8} {'KW Cov':>8} {'Time':>8} {'Status':>8}")
    print(f"  {'─'*3} {'─'*40} {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for r in results:
        topic = r.get("topic", "N/A")[:40]
        scope = r.get("scope", "-")
        m = r.get("metrics", {})
        recall = f"{m.get('recall', 0)*100:.0f}%" if m else "-"
        kw = f"{m.get('keyword_coverage', 0)*100:.0f}%" if m else "-"
        t = f"{m.get('response_time_s', 0):.1f}s" if m else "-"
        status = r["status"]
        icon = "✅" if status == "PASS" else "⚠️" if status == "WARN" else "❌"
        print(f"  {r['id']:3d} {topic:>40} {scope:>12} {recall:>8} {kw:>8} {t:>8} {icon + ' ' + status:>10}")

    print(f"\n  ────────────────────────────────")
    print(f"  Total:   {total}")
    print(f"  Passed:  {passed}")
    print(f"  Failed:  {failed}")
    print(f"  Warned:  {warned}")
    print(f"  Errors:  {errored}")
    print(f"  Score:   {passed}/{total} ({passed/total*100:.0f}%)" if total > 0 else "")

    # Average response time
    times = [r["metrics"]["response_time_s"] for r in results if r.get("metrics")]
    if times:
        print(f"  Avg Response Time: {sum(times)/len(times):.2f}s")
        print(f"  Max Response Time: {max(times):.2f}s")


def main():
    global BASE_URL
    parser = argparse.ArgumentParser(
        description="Retrieval Algorithm Accuracy Tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_retrieval_accuracy.py                # Run all tests
  python test_retrieval_accuracy.py --case 1       # Run test case 1
  python test_retrieval_accuracy.py --verbose      # Show chunk content
  python test_retrieval_accuracy.py --case 8       # Test out-of-scope detection
        """
    )
    parser.add_argument("--case", type=int, help="Run specific test case number")
    parser.add_argument("--verbose", action="store_true", help="Show retrieved context previews")
    parser.add_argument("--base-url", default=BASE_URL, help=f"Backend URL (default: {BASE_URL})")

    args = parser.parse_args()
    BASE_URL = args.base_url

    print_header("RETRIEVAL ALGORITHM ACCURACY TESTS")
    print(f"  Timestamp:   {datetime.now().isoformat()}")
    print(f"  Target:      {BASE_URL}")
    print(f"  Test Cases:  {len(TEST_CASES)}")
    print(f"  Config:      top_k=15, search_width=40, min_sim=0.30")
    print(f"  Embeddings:  Gemini 3072d → FAISS IndexFlatIP")

    # Select test cases
    if args.case:
        cases = [tc for tc in TEST_CASES if tc["id"] == args.case]
        if not cases:
            print(f"\n  ❌ Test case #{args.case} not found. Available: 1-{len(TEST_CASES)}")
            sys.exit(1)
    else:
        cases = TEST_CASES

    # Run tests
    results = []
    for tc in cases:
        result = run_test_case(tc, verbose=args.verbose)
        results.append(result)
        time.sleep(0.5)  # Slight delay between queries

    # Summary
    print_summary(results)


if __name__ == "__main__":
    main()

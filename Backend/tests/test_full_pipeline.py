"""
=============================================================================
Script vi: Full Pipeline Tests — 5 Comprehensive End-to-End Test Cases
=============================================================================

Purpose:
    - Test the COMPLETE RAG pipeline from user query to final structured answer
    - Flow: Query → Gemini Embed → FAISS Search → Context Assembly → LLM → Parse
    - 5 test cases covering different Sri Lankan labour law scenarios
    - Comprehensive metrics: accuracy, retrieval, completeness, performance, format

Usage:
    python test_full_pipeline.py                  # Run all 5 tests
    python test_full_pipeline.py --case 1         # Run specific test case
    python test_full_pipeline.py --verbose        # Show full responses
    python test_full_pipeline.py --save           # Save results to JSON

Requirements:
    - Backend running on http://localhost:5005 (with docs uploaded)
    - Model server running on localhost:5007 (or 5006)
    - pip install httpx

Full Pipeline Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │  User Query                                                     │
    │    ↓                                                            │
    │  POST /api/query/recommend                                      │
    │    ↓                                                            │
    │  Gemini Embedding (3072d)                                       │
    │    ↓                                                            │
    │  FAISS IndexFlatIP Search (40 wide candidates)                  │
    │    ↓                                                            │
    │  Document-Diverse Reranking → Top 15 chunks                     │
    │    ↓                                                            │
    │  Context Assembly (max 4500 chars)                               │
    │    ↓                                                            │
    │  System Prompt + User Prompt → Model Server (Qwen3-8B)          │
    │    ↓                                                            │
    │  JSON Parse (5-level fallback)                                  │
    │    ↓                                                            │
    │  LegalOutput Schema Validation                                  │
    │    ↓                                                            │
    │  Persist to PostgreSQL + Return Response                        │
    └─────────────────────────────────────────────────────────────────┘
=============================================================================
"""
import httpx
import argparse
import sys
import time
import json
import re
from datetime import datetime
from typing import Optional

BASE_URL = "http://localhost:5005"
TIMEOUT = 300.0  # Full pipeline can take 1-3 minutes

# ── 5 Comprehensive Test Cases ───────────────────────────────────────
# Aligned with the ACTUAL indexed corpus (558 chunks, 14 documents):
#   - Payment of Gratuity Act (a1, a2, a3, a5)
#   - Industrial Disputes Act (1, 2, a5, a6, a7)
#   - Termination of Employment of Workmen Act (2, 5)
#   - Trade Unions Ordinance (3, 4)
#   - Employees' Councils Act (6, 7)
# Cases in corpus: Collettes, De Costa v ANZ Grindlays, Perera v Standard Chartered,
#   Eksath Kamkaru v Upali Newspapers, Piyadasa v Bata Shoe, etc.

PIPELINE_TESTS = [
    {
        "id": 1,
        "title": "Gratuity: Managing Director as Workman",
        "query": "I was a Managing Director for seven years before resigning. My company says I am not a workman entitled to gratuity. Is this legally correct under the Payment of Gratuity Act?",
        "expected": {
            "out_of_scope": False,
            "scope_category": "labour_employment_law",
            "must_contain_acts": ["Payment of Gratuity Act"],
            "must_contain_cases": ["Collettes"],
            "must_contain_keywords": ["workman", "gratuity", "Managing Director"],
            "expected_violation_type": "Denial of Gratuity",
            "min_confidence": 0.7,
        },
        "grading": {
            "retrieval_weight": 0.25,
            "act_accuracy_weight": 0.20,
            "case_accuracy_weight": 0.15,
            "schema_weight": 0.15,
            "reasoning_weight": 0.15,
            "performance_weight": 0.10,
        },
    },
    {
        "id": 2,
        "title": "Gratuity: Referral After Resignation",
        "query": "I resigned from my bank and later claimed gratuity. The bank says that since I already left, the Minister cannot refer this dispute for arbitration. Is this correct under the Industrial Disputes Act?",
        "expected": {
            "out_of_scope": False,
            "scope_category": "labour_employment_law",
            "must_contain_acts": ["Industrial Disputes Act"],
            "must_contain_cases": ["De Costa"],
            "must_contain_keywords": ["gratuity", "resignation", "Minister", "reference"],
            "expected_violation_type": None,
            "min_confidence": 0.7,
        },
        "grading": {
            "retrieval_weight": 0.25,
            "act_accuracy_weight": 0.20,
            "case_accuracy_weight": 0.15,
            "schema_weight": 0.15,
            "reasoning_weight": 0.15,
            "performance_weight": 0.10,
        },
    },
    {
        "id": 3,
        "title": "Termination: Commissioner Approval Required",
        "query": "My employer terminated my services without getting approval from the Commissioner of Labour. I have been employed for over two years in a factory. Is this termination legal?",
        "expected": {
            "out_of_scope": False,
            "scope_category": "labour_employment_law",
            "must_contain_acts": ["Termination of Employment"],
            "must_contain_cases": [],
            "must_contain_keywords": ["Commissioner", "approval", "termination", "scheduled employment"],
            "expected_violation_type": None,
            "min_confidence": 0.6,
        },
        "grading": {
            "retrieval_weight": 0.25,
            "act_accuracy_weight": 0.25,
            "case_accuracy_weight": 0.05,
            "schema_weight": 0.15,
            "reasoning_weight": 0.20,
            "performance_weight": 0.10,
        },
    },
    {
        "id": 4,
        "title": "Compensatory Allowance vs Gratuity",
        "query": "My employer says I can't get gratuity because I was already paid a special compensatory allowance every month. Does receiving such allowance deprive me of gratuity under the law?",
        "expected": {
            "out_of_scope": False,
            "scope_category": "labour_employment_law",
            "must_contain_acts": ["Payment of Gratuity Act"],
            "must_contain_cases": ["Collettes"],
            "must_contain_keywords": ["special allowance", "compensatory", "gratuity", "not deprive"],
            "expected_violation_type": "Denial of Gratuity",
            "min_confidence": 0.7,
        },
        "grading": {
            "retrieval_weight": 0.25,
            "act_accuracy_weight": 0.20,
            "case_accuracy_weight": 0.15,
            "schema_weight": 0.15,
            "reasoning_weight": 0.15,
            "performance_weight": 0.10,
        },
    },
    {
        "id": 5,
        "title": "Industrial Disputes: Dual Proceedings",
        "query": "I already filed an application at the Labour Tribunal. Now the Minister has referred the same dispute for compulsory arbitration. Can both proceedings continue at the same time?",
        "expected": {
            "out_of_scope": False,
            "scope_category": "labour_employment_law",
            "must_contain_acts": ["Industrial Disputes"],
            "must_contain_cases": ["Eksath Kamkaru"],
            "must_contain_keywords": ["arbitration", "Labour Tribunal", "Minister"],
            "expected_violation_type": None,
            "min_confidence": 0.7,
        },
        "grading": {
            "retrieval_weight": 0.25,
            "act_accuracy_weight": 0.20,
            "case_accuracy_weight": 0.15,
            "schema_weight": 0.15,
            "reasoning_weight": 0.15,
            "performance_weight": 0.10,
        },
    },
]


def print_header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def query_full_pipeline(query: str) -> Optional[dict]:
    """Send a query through the full backend pipeline."""
    try:
        payload = {"query": query}
        start = time.time()
        r = httpx.post(f"{BASE_URL}/api/query/recommend", json=payload, timeout=TIMEOUT)
        elapsed = time.time() - start

        result = {
            "status_code": r.status_code,
            "response_time_s": round(elapsed, 3),
        }

        if r.status_code < 300:
            data = r.json()
            result["success"] = True
            result["data"] = data
        else:
            result["success"] = False
            result["error"] = r.text[:500]

        return result

    except httpx.TimeoutException:
        return {"success": False, "error": f"Timeout after {TIMEOUT}s", "response_time_s": TIMEOUT}
    except Exception as e:
        return {"success": False, "error": str(e), "response_time_s": 0}


def extract_legal_output(data: dict) -> Optional[dict]:
    """Extract the LegalOutput JSON from the pipeline response."""
    # The backend response structure: { recommendation: { ..., legal_output: {...} } }
    # or { result: { ..., output: {...} } }
    # or the output may be at the top level

    # Try common response shapes
    for path in [
        lambda d: d.get("recommendation", {}).get("legal_output"),
        lambda d: d.get("recommendation", {}).get("output"),
        lambda d: d.get("recommendation"),
        lambda d: d.get("result", {}).get("legal_output"),
        lambda d: d.get("result", {}).get("output"),
        lambda d: d.get("result"),
        lambda d: d.get("data", {}).get("legal_output"),
        lambda d: d.get("legal_output"),
        lambda d: d.get("output"),
        lambda d: d,
    ]:
        try:
            obj = path(data)
            if isinstance(obj, dict) and ("out_of_scope" in obj or "primary_violations" in obj or "scope_category" in obj):
                return obj
        except Exception:
            continue

    # If nothing found, try parsing any string field as JSON
    for key in ["text", "response", "answer", "content"]:
        val = data.get(key)
        if isinstance(val, str) and len(val) > 10:
            try:
                parsed = json.loads(val)
                if isinstance(parsed, dict) and "out_of_scope" in parsed:
                    return parsed
            except json.JSONDecodeError:
                # Try extracting JSON from the string
                match = re.search(r'\{[\s\S]*\}', val)
                if match:
                    try:
                        parsed = json.loads(match.group(0))
                        if "out_of_scope" in parsed:
                            return parsed
                    except json.JSONDecodeError:
                        pass

    return None


def grade_test(response: dict, tc: dict, verbose: bool = False) -> dict:
    """Grade a pipeline test result comprehensively."""
    expected = tc["expected"]
    weights = tc["grading"]
    scores = {}

    data = response.get("data", {})
    legal = extract_legal_output(data)
    resp_time = response.get("response_time_s", 0)

    if legal is None:
        # Try to display what we got
        if verbose:
            print(f"  Raw response (first 500): {json.dumps(data, default=str)[:500]}")
        return {
            "total_score": 0,
            "max_score": 100,
            "grade": "F",
            "status": "PARSE_FAIL",
            "details": {"error": "Could not extract LegalOutput from response"},
            "response_time_s": resp_time,
        }

    full_text = json.dumps(legal, default=str).lower()

    # ── 1. Retrieval Score (based on whether correct acts/cases appear) ──
    retrieval_hits = 0
    retrieval_total = max(len(expected["must_contain_acts"]) + len(expected["must_contain_cases"]), 1)

    for act in expected["must_contain_acts"]:
        if act.lower() in full_text:
            retrieval_hits += 1

    for case in expected["must_contain_cases"]:
        if case.lower() in full_text:
            retrieval_hits += 1

    retrieval_score = (retrieval_hits / retrieval_total) * 100 if retrieval_total > 0 else 100
    scores["retrieval"] = retrieval_score

    # ── 2. Act Accuracy Score ──
    violations = legal.get("primary_violations", [])
    act_score = 0

    if expected["must_contain_acts"]:
        acts_found = 0
        for expected_act in expected["must_contain_acts"]:
            for v in violations:
                if expected_act.lower() in v.get("act_name", "").lower():
                    acts_found += 1
                    break
            else:
                # Check full text as fallback
                if expected_act.lower() in full_text:
                    acts_found += 0.5
        act_score = (acts_found / len(expected["must_contain_acts"])) * 100
    else:
        # Out-of-scope: should have empty violations
        act_score = 100 if len(violations) == 0 else 50

    scores["act_accuracy"] = act_score

    # ── 3. Case Accuracy Score ──
    cases = legal.get("supporting_cases", [])
    case_score = 0

    if expected["must_contain_cases"]:
        cases_found = 0
        for expected_case in expected["must_contain_cases"]:
            for c in cases:
                if expected_case.lower() in c.get("case_name", "").lower():
                    cases_found += 1
                    break
            else:
                if expected_case.lower() in full_text:
                    cases_found += 0.5
        case_score = (cases_found / len(expected["must_contain_cases"])) * 100
    else:
        case_score = 100  # No case expected

    scores["case_accuracy"] = case_score

    # ── 4. Schema Completeness Score ──
    required_fields = [
        "out_of_scope", "scope_category", "summary", "primary_violations",
        "supporting_cases", "legal_reasoning", "recommended_action", "limits", "confidence"
    ]
    fields_present = sum(1 for f in required_fields if f in legal)
    schema_score = (fields_present / len(required_fields)) * 100

    # Check out_of_scope correctness
    if legal.get("out_of_scope") == expected["out_of_scope"]:
        schema_score = min(schema_score + 10, 100)  # Bonus for correct scope
    else:
        schema_score = max(schema_score - 30, 0)  # Penalty for wrong scope

    # Check scope_category
    if not expected["out_of_scope"]:
        if legal.get("scope_category") == "labour_employment_law":
            schema_score = min(schema_score + 5, 100)

    # Check summary consistency
    summary = legal.get("summary", {})
    if summary:
        if summary.get("violation_count") == len(violations):
            schema_score = min(schema_score + 5, 100)
        if summary.get("cases_count") == len(cases):
            schema_score = min(schema_score + 5, 100)

    scores["schema"] = min(schema_score, 100)

    # ── 5. Reasoning Quality Score ──
    reasoning = legal.get("legal_reasoning", "")
    reasoning_score = 0

    if reasoning:
        # Length check (expect 3-4 paragraphs = ~500-2000 chars)
        if len(reasoning) >= 200:
            reasoning_score += 30
        if len(reasoning) >= 500:
            reasoning_score += 20
        if len(reasoning) >= 800:
            reasoning_score += 10

        # Keyword coverage in reasoning
        kw_hits = 0
        kw_total = len(expected["must_contain_keywords"])
        for kw in expected["must_contain_keywords"]:
            if kw.lower() in reasoning.lower():
                kw_hits += 1
        if kw_total > 0:
            reasoning_score += int((kw_hits / kw_total) * 40)
        else:
            reasoning_score += 40  # Out-of-scope

    scores["reasoning"] = min(reasoning_score, 100)

    # ── 6. Performance Score ──
    if resp_time <= 30:
        perf_score = 100
    elif resp_time <= 60:
        perf_score = 80
    elif resp_time <= 120:
        perf_score = 50
    elif resp_time <= 180:
        perf_score = 30
    else:
        perf_score = 10

    scores["performance"] = perf_score

    # ── Weighted Total ──
    total = (
        scores["retrieval"] * weights["retrieval_weight"]
        + scores["act_accuracy"] * weights["act_accuracy_weight"]
        + scores["case_accuracy"] * weights["case_accuracy_weight"]
        + scores["schema"] * weights["schema_weight"]
        + scores["reasoning"] * weights["reasoning_weight"]
        + scores["performance"] * weights["performance_weight"]
    )

    # Grade
    if total >= 85:
        grade = "A"
    elif total >= 70:
        grade = "B"
    elif total >= 55:
        grade = "C"
    elif total >= 40:
        grade = "D"
    else:
        grade = "F"

    status = "PASS" if total >= 55 else "WARN" if total >= 40 else "FAIL"

    return {
        "total_score": round(total, 1),
        "max_score": 100,
        "grade": grade,
        "status": status,
        "scores": scores,
        "response_time_s": resp_time,
        "legal_output": legal if verbose else None,
        "details": {
            "out_of_scope": legal.get("out_of_scope"),
            "scope_category": legal.get("scope_category"),
            "violation_count": len(violations),
            "cases_count": len(cases),
            "reasoning_length": len(reasoning) if reasoning else 0,
            "confidence": legal.get("confidence"),
        },
    }


def run_pipeline_test(tc: dict, verbose: bool = False) -> dict:
    """Execute a single full pipeline test."""
    print_section(f"Pipeline Test #{tc['id']}: {tc['title']}")
    print(f"  Query:  {tc['query'][:80]}...")
    print(f"  Expect: {'OUT-OF-SCOPE' if tc['expected']['out_of_scope'] else 'IN-SCOPE'}")
    if tc["expected"]["must_contain_acts"]:
        print(f"  Acts:   {', '.join(tc['expected']['must_contain_acts'])}")
    if tc["expected"]["must_contain_cases"]:
        print(f"  Cases:  {', '.join(tc['expected']['must_contain_cases'])}")

    # Execute query
    print(f"\n  ⏳ Sending query through full pipeline...")
    response = query_full_pipeline(tc["query"])

    if not response.get("success"):
        print(f"  ❌ Pipeline failed: {response.get('error', 'unknown')}")
        return {
            "id": tc["id"],
            "title": tc["title"],
            "status": "FAILED",
            "error": response.get("error"),
            "response_time_s": response.get("response_time_s", 0),
        }

    print(f"  ✅ Response received ({response['response_time_s']:.1f}s)")

    # Grade
    grade_result = grade_test(response, tc, verbose)

    # Display results
    print(f"\n  ── Scoring Breakdown ──")
    if "scores" in grade_result:
        for dimension, score in grade_result["scores"].items():
            bar_len = int(score / 5)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            label = dimension.replace("_", " ").title()
            print(f"    {label:18s}: {bar} {score:.0f}%")

    print(f"\n  ── Details ──")
    details = grade_result.get("details", {})
    print(f"  Out-of-Scope:     {details.get('out_of_scope')}")
    print(f"  Scope Category:   {details.get('scope_category')}")
    print(f"  Violations Found: {details.get('violation_count')}")
    print(f"  Cases Found:      {details.get('cases_count')}")
    print(f"  Reasoning Chars:  {details.get('reasoning_length')}")
    print(f"  Confidence:       {details.get('confidence')}")
    print(f"  Response Time:    {grade_result['response_time_s']:.2f}s")

    # Show legal output if verbose
    if verbose and grade_result.get("legal_output"):
        print(f"\n  ── Full Legal Output ──")
        print(json.dumps(grade_result["legal_output"], indent=2, default=str)[:2000])

    # Final verdict
    icon = "✅" if grade_result["status"] == "PASS" else "⚠️" if grade_result["status"] == "WARN" else "❌"
    print(f"\n  Total Score:  {grade_result['total_score']:.1f}/100 (Grade: {grade_result['grade']})")
    print(f"  Verdict:      {icon} {grade_result['status']}")

    return {
        "id": tc["id"],
        "title": tc["title"],
        "status": grade_result["status"],
        "grade": grade_result["grade"],
        "total_score": grade_result["total_score"],
        "scores": grade_result.get("scores"),
        "details": grade_result.get("details"),
        "response_time_s": grade_result["response_time_s"],
    }


def print_final_summary(results: list):
    """Print comprehensive final summary."""
    print_header("FULL PIPELINE TEST RESULTS")

    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    warned = sum(1 for r in results if r["status"] == "WARN")
    failed = sum(1 for r in results if r["status"] in ("FAIL", "FAILED"))

    print(f"\n  {'#':>3} {'Title':>42} {'Score':>8} {'Grade':>6} {'Time':>8} {'Status':>8}")
    print(f"  {'─'*3} {'─'*42} {'─'*8} {'─'*6} {'─'*8} {'─'*8}")

    for r in results:
        title = r.get("title", "N/A")[:42]
        score = f"{r.get('total_score', 0):.0f}" if r.get("total_score") is not None else "-"
        grade = r.get("grade", "-")
        t = f"{r.get('response_time_s', 0):.1f}s"
        status = r["status"]
        icon = "✅" if status == "PASS" else "⚠️" if status == "WARN" else "❌"
        print(f"  {r['id']:3d} {title:>42} {score:>8} {grade:>6} {t:>8} {icon + ' ' + status:>10}")

    # Aggregate scores
    valid_results = [r for r in results if r.get("total_score") is not None]
    if valid_results:
        avg_score = sum(r["total_score"] for r in valid_results) / len(valid_results)
        max_score = max(r["total_score"] for r in valid_results)
        min_score = min(r["total_score"] for r in valid_results)

        print(f"\n  ── Aggregate Metrics ──")
        print(f"  Average Score:    {avg_score:.1f}/100")
        print(f"  Best Score:       {max_score:.1f}")
        print(f"  Worst Score:      {min_score:.1f}")

    # Performance metrics
    times = [r["response_time_s"] for r in results if r.get("response_time_s")]
    if times:
        print(f"\n  ── Performance ──")
        print(f"  Total Time:       {sum(times):.1f}s")
        print(f"  Avg Time:         {sum(times)/len(times):.1f}s")
        print(f"  Min Time:         {min(times):.1f}s")
        print(f"  Max Time:         {max(times):.1f}s")

    # Dimension averages
    if valid_results and valid_results[0].get("scores"):
        dimensions = valid_results[0]["scores"].keys()
        print(f"\n  ── Dimension Averages ──")
        for dim in dimensions:
            vals = [r["scores"][dim] for r in valid_results if r.get("scores") and dim in r["scores"]]
            if vals:
                label = dim.replace("_", " ").title()
                avg = sum(vals) / len(vals)
                bar_len = int(avg / 5)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                print(f"    {label:18s}: {bar} {avg:.0f}%")

    # Final verdict
    print(f"\n  ── Final Verdict ──")
    print(f"  Total:     {total} test(s)")
    print(f"  Passed:    {passed}")
    print(f"  Warned:    {warned}")
    print(f"  Failed:    {failed}")
    overall = "PASS" if failed == 0 and warned <= 1 else "WARN" if failed <= 1 else "FAIL"
    icon = "✅" if overall == "PASS" else "⚠️" if overall == "WARN" else "❌"
    print(f"  Overall:   {icon} {overall}")


def save_results(results: list, filepath: str):
    """Save test results to a JSON file."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "base_url": BASE_URL,
        "total_tests": len(results),
        "passed": sum(1 for r in results if r["status"] == "PASS"),
        "failed": sum(1 for r in results if r["status"] in ("FAIL", "FAILED")),
        "results": results,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  📄 Results saved to: {filepath}")


def main():
    global BASE_URL
    parser = argparse.ArgumentParser(
        description="Full Pipeline End-to-End Tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_full_pipeline.py                   # Run all 5 tests
  python test_full_pipeline.py --case 1          # Run test case 1
  python test_full_pipeline.py --verbose         # Show full LLM output
  python test_full_pipeline.py --save            # Save results to JSON
  python test_full_pipeline.py --verbose --save  # Both
        """
    )
    parser.add_argument("--case", type=int, help="Run specific test case (1-5)")
    parser.add_argument("--verbose", action="store_true", help="Show full legal output")
    parser.add_argument("--save", action="store_true", help="Save results to JSON file")
    parser.add_argument("--base-url", default=BASE_URL, help=f"Backend URL (default: {BASE_URL})")
    parser.add_argument("--output-file", default=None, help="Output JSON file path")

    args = parser.parse_args()
    BASE_URL = args.base_url

    print_header("FULL PIPELINE END-TO-END TESTS")
    print(f"  Timestamp:   {datetime.now().isoformat()}")
    print(f"  Backend:     {BASE_URL}")
    print(f"  Test Cases:  {len(PIPELINE_TESTS)}")
    print(f"  Pipeline:    Query → Embed → FAISS → Rerank → Context → LLM → Parse")

    # Pre-check health
    print_section("Pre-flight Health Check")
    try:
        r = httpx.get(f"{BASE_URL}/api/health", timeout=15)
        data = r.json()
        print(f"  Backend Status: {data.get('status', 'unknown')}")
        components = data.get("components", data)
        if isinstance(components, dict):
            for k, v in components.items():
                if k != "status":
                    icon = "✅" if v in (True, "healthy", "connected", "ok") else "❌"
                    print(f"    {icon} {k}: {v}")
    except Exception as e:
        print(f"  ⚠️  Health check failed: {e}")
        print(f"  Proceeding anyway...")

    # Select tests
    if args.case:
        cases = [tc for tc in PIPELINE_TESTS if tc["id"] == args.case]
        if not cases:
            print(f"\n  ❌ Test case #{args.case} not found. Available: 1-{len(PIPELINE_TESTS)}")
            sys.exit(1)
    else:
        cases = PIPELINE_TESTS

    # Run tests
    results = []
    for i, tc in enumerate(cases, 1):
        result = run_pipeline_test(tc, verbose=args.verbose)
        results.append(result)
        if i < len(cases):
            print(f"\n  ⏳ Waiting 2s before next test...")
            time.sleep(2)

    # Summary
    print_final_summary(results)

    # Save
    if args.save:
        output_path = args.output_file or f"pipeline_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_results(results, output_path)


if __name__ == "__main__":
    main()

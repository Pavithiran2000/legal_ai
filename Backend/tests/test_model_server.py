"""
=============================================================================
Script v: Model-Server-Deployed Tests (3 Test Cases)
=============================================================================

Purpose:
    - Test the model-server-deployed (proxy on port 5007 → Modal A10G GPU)
    - Verify /health, /generate, /chat endpoints
    - Run 3 test cases with ~700-word retrieved contexts
    - Validate structured JSON output schema matches LegalOutput format
    - Measure inference latency and output quality

Usage:
    python test_model_server.py                       # Run all 3 tests
    python test_model_server.py --case 1              # Run specific case
    python test_model_server.py --endpoint generate   # Test /generate only
    python test_model_server.py --endpoint chat       # Test /chat only
    python test_model_server.py --port 5006           # Local model server

Requirements:
    - Model server (proxy or local) running
    - pip install httpx

Architecture:
    Proxy Server (localhost:5007) → Modal.com A10G GPU → Ollama → Qwen3-8B GGUF
    
    Endpoints tested:
    ├── GET  /health           → Connectivity + model status
    ├── GET  /model/info       → Model details
    ├── POST /generate         → Direct prompt → response
    ├── POST /chat             → Chat messages → response
    └── GET  /proxy/status     → Proxy → Modal connectivity
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

MODEL_SERVER_URL = "http://localhost:5007"
TIMEOUT = 300.0  # 5 minutes for LLM inference

# ── System Prompt (matches fine-tuning & backend) ─────────────────────

SYSTEM_PROMPT = """You are a legal assistant specialized EXCLUSIVELY in Sri Lankan Labour & Employment Law.

SCOPE DEFINITION:
- IN-SCOPE: Only queries related to "labour_employment_law"
- OUT-OF-SCOPE: ALL other categories

CRITICAL: You MUST return ONLY a single valid JSON object — no markdown, no extra text, no thinking blocks, no code fences.

USE EXACTLY THIS JSON SCHEMA:
{
  "out_of_scope": false,
  "scope_category": "labour_employment_law",
  "confidence": 0.85,
  "summary": { "primary_issue": "...", "violation_count": 1, "acts_count": 1, "cases_count": 1 },
  "primary_violations": [{ "violation_type": "...", "act_name": "...", "act_year": "...", "act_section_number": "...", "act_section_text": "...", "why_relevant": "..." }],
  "supporting_cases": [{ "case_name": "...", "case_year": "...", "case_citation": "...", "case_summary": "...", "why_relevant": "..." }],
  "legal_reasoning": "...",
  "recommended_action": ["..."],
  "limits": ["..."],
  "confidence": 0.85
}

/no_think"""


# ── Test Cases with ~700-word Retrieved Contexts ─────────────────────

TEST_CASES = [
    {
        "id": 1,
        "title": "Managing Director Gratuity Claim",
        "instruction": "I was a Managing Director for seven years before resigning. Can I claim gratuity under the law?",
        "retrieved_context": (
            "The Payment of Gratuity Act No. 12 of 1983 provides the statutory framework for the payment of "
            "gratuity to employees in Sri Lanka. In the case of Collettes Ltd. v. Commissioner of Labour and "
            "Others (1989) 2 Sri LR 6, the Court of Appeal addressed whether a Group Managing Director falls "
            "within the definition of a 'workman' under the Act. The 4th respondent in that case was first "
            "appointed Managing Director of Colombo Paints Company on 1.5.80 and later Group Managing Director "
            "of Collettes Group of Companies until his resignation on 31.5.87. Upon his resignation, he applied "
            "for gratuity, and the Assistant Commissioner of Labour awarded him Rs. 87,500/-.\n\n"
            "The Court held that a Managing Director has a dual capacity of being an employee of the company and "
            "also at the same time takes part in the management of the company. The fact that as Managing Director "
            "or as Group Managing Director he takes part in the management of the affairs of the company does not "
            "deprive him of his other capacity as an employee of the said company. Therefore, the 4th respondent "
            "falls within the definition of a 'workman' set out in the Payment of Gratuity Act.\n\n"
            "Section 8(1) of the Act empowers the Commissioner of Labour to determine whether a workman is "
            "entitled to a gratuity. The Court noted that there is no specific requirement in section 8(1) to "
            "call oral evidence; the Commissioner must simply be satisfied of the relevant matters. Additionally, "
            "Section 7 of the Payment of Gratuity Act defines certain exclusions, but receiving a special allowance "
            "or compensatory allowance does not deprive a person of the right to receive gratuity.\n\n"
            "Section 13 of the Act defines 'workman' in broad terms, which has been the subject of judicial "
            "interpretation. The dual capacity doctrine ensures that even high-level executives who perform "
            "managerial functions are not excluded from the protections afforded to employees under social "
            "security and terminal benefit legislation, provided an employer-employee relationship exists "
            "alongside their directorial duties. The definition of 'workman' in Sri Lankan labour law is "
            "generally broad, encompassing any person who has entered into or works under a contract with "
            "an employer, whether the contract be oral or in writing, express or implied."
        ),
        "expected_fields": {
            "out_of_scope": False,
            "scope_category": "labour_employment_law",
            "expected_act": "Payment of Gratuity Act",
            "expected_case": "Collettes",
            "expected_keywords": ["workman", "dual capacity", "gratuity"],
        },
    },
    {
        "id": 2,
        "title": "Wrongful Dismissal from Bank",
        "instruction": "I was unfairly dismissed from my bank job five years ago. I haven't been able to find work since. How much compensation can I get?",
        "retrieved_context": (
            "In the case of Suranganie Marapana v Bank of Ceylon and others (1997) 3 SLR 156, the Court "
            "addressed the issue of compensation for wrongful dismissal in the banking sector. The applicant's "
            "services were terminated on February 20, 1986, and the applicant remained unemployed for 5 years "
            "and 7 months. The Court observed that the prospects of future employment were damaged by allegations "
            "in the banking sector where confidence is required at the maximum level.\n\n"
            "Held: It is just and fair to award a sum equivalent to seven years' salary earned by the applicant "
            "at the time of dismissal as compensation in lieu of reinstatement. The salary was Rs. 7,822/- per "
            "month, totaling Rs. 657,048/-. The Court emphasized that when computing compensation, the loss of "
            "income due to wrongful dismissal and prospects of future employment are significant factors.\n\n"
            "The Industrial Disputes Act No. 43 of 1950 is the primary legislation governing the resolution of "
            "industrial disputes in Sri Lanka. Section 31B(1) of the Termination of Employment of Workmen "
            "(Special Provisions) Act No. 45 of 1971 provides that no employer shall terminate the services of "
            "a workman without the prior written consent of the workman or the prior written approval of the "
            "Commissioner of Labour. If the termination is found to be wrongful, the workman is entitled to "
            "compensation that takes into account the period of unemployment, loss of future prospects, and "
            "the circumstances of the dismissal.\n\n"
            "The court also noted that while reinstatement is a common remedy, compensation is appropriate "
            "when the trust relationship is irreparably broken. The calculation of compensation considers the "
            "period of unemployment, the loss of future prospects, and the circumstances of the dismissal. If "
            "damaging allegations are made, prospects of re-employment are naturally affected until the applicant "
            "vindicates himself before a judicial body. The remedy by way of Writ of Certiorari cannot be used "
            "to correct errors of fact; judicial review is concerned only with legality."
        ),
        "expected_fields": {
            "out_of_scope": False,
            "scope_category": "labour_employment_law",
            "expected_act": "Industrial Disputes Act",
            "expected_case": "Suranganie Marapana",
            "expected_keywords": ["wrongful dismissal", "compensation", "reinstatement"],
        },
    },
    {
        "id": 3,
        "title": "Out-of-Scope: Property Dispute",
        "instruction": "My landlord is trying to evict me without notice even though I have a written lease. What are my rights?",
        "retrieved_context": (
            "The Rent Act No. 7 of 1972 and its amendments provide the framework for landlord-tenant "
            "relationships in Sri Lanka. This legislation covers aspects of rental agreements, eviction "
            "procedures, and tenant protections. However, this falls outside the scope of labour and "
            "employment law. The legal framework for tenancy and eviction disputes is governed by civil "
            "and property law, including the provisions of the Civil Procedure Code and relevant land "
            "legislation. Tenants who face wrongful eviction should seek remedies through the relevant "
            "magistrate's court or district court."
        ),
        "expected_fields": {
            "out_of_scope": True,
            "scope_category": None,  # Should not be labour_employment_law
            "expected_act": None,
            "expected_case": None,
            "expected_keywords": [],
        },
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


def test_health():
    """Test the /health endpoint."""
    print_section("Health Check")
    try:
        r = httpx.get(f"{MODEL_SERVER_URL}/health", timeout=30)
        data = r.json()
        print(f"  Status Code:      {r.status_code}")
        print(f"  Status:           {data.get('status', 'N/A')}")
        print(f"  Ollama Connected: {data.get('ollama_connected', 'N/A')}")
        print(f"  Active Model:     {data.get('active_model', 'N/A')}")
        models = data.get("available_models", [])
        if models:
            print(f"  Available Models: {', '.join(models)}")
        return r.status_code == 200 and data.get("ollama_connected", False)
    except Exception as e:
        print(f"  ❌ Health check failed: {e}")
        return False


def test_proxy_status():
    """Test the /proxy/status endpoint (proxy-only)."""
    print_section("Proxy Status")
    try:
        r = httpx.get(f"{MODEL_SERVER_URL}/proxy/status", timeout=30)
        if r.status_code == 200:
            data = r.json()
            print(f"  Proxy Running:    {data.get('proxy_running', 'N/A')}")
            print(f"  Modal URL:        {data.get('modal_url', 'N/A')}")
            print(f"  Modal Reachable:  {data.get('modal_reachable', 'N/A')}")
            if data.get("modal_health"):
                mh = data["modal_health"]
                print(f"  Modal Model:      {mh.get('active_model', 'N/A')}")
        elif r.status_code == 404:
            print(f"  (Not a proxy server — likely direct model server)")
        return True
    except Exception as e:
        print(f"  ⚠️  Proxy status not available: {e}")
        return True  # Not critical


def test_model_info():
    """Test the /model/info endpoint."""
    print_section("Model Info")
    try:
        r = httpx.get(f"{MODEL_SERVER_URL}/model/info", timeout=30)
        if r.status_code == 200:
            data = r.json()
            print(f"  Model:        {data.get('active_model', data.get('model', 'N/A'))}")
            print(f"  Parameters:   {data.get('parameters', 'N/A')}")
            print(f"  Quantization: {data.get('quantization', 'N/A')}")
            print(f"  Context Len:  {data.get('context_length', 'N/A')}")
            if data.get("proxy"):
                print(f"  Proxy:        Yes (port {data.get('proxy_port', '?')})")
        else:
            print(f"  Status: {r.status_code}")
        return True
    except Exception as e:
        print(f"  ⚠️  Model info not available: {e}")
        return True


def make_prompt(instruction: str, context: str) -> str:
    """Create user prompt matching fine-tuning format."""
    return (
        f"SCENARIO:\n{instruction}\n\n"
        f"RETRIEVED_CONTEXT:\n{context}\n\n"
        f"TASK: Analyze the scenario using the RETRIEVED_CONTEXT above. "
        f"Extract and list ALL relevant acts, sections, and cases from the context. "
        f"Return ONLY the JSON output object. Be comprehensive."
    )


def parse_json_response(text: str) -> Optional[dict]:
    """Parse JSON from LLM response, handling code fences and extra text."""
    # Strip thinking blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    # Try direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try extracting from code fence
    match = re.search(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding JSON object
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def validate_schema(parsed: dict, expected: dict) -> dict:
    """Validate the parsed JSON against expected LegalOutput schema."""
    results = {
        "schema_valid": True,
        "checks": [],
        "score": 0,
        "max_score": 0,
    }

    # Required top-level fields
    required_fields = [
        "out_of_scope", "scope_category", "summary",
        "primary_violations", "supporting_cases",
        "legal_reasoning", "recommended_action", "limits", "confidence"
    ]

    for field in required_fields:
        results["max_score"] += 1
        if field in parsed:
            results["checks"].append(("✅", f"Field '{field}' present"))
            results["score"] += 1
        else:
            results["checks"].append(("❌", f"Field '{field}' MISSING"))
            results["schema_valid"] = False

    # Check out_of_scope correctness
    results["max_score"] += 1
    if parsed.get("out_of_scope") == expected["out_of_scope"]:
        results["checks"].append(("✅", f"out_of_scope={parsed.get('out_of_scope')} (correct)"))
        results["score"] += 1
    else:
        results["checks"].append(("❌", f"out_of_scope={parsed.get('out_of_scope')} (expected {expected['out_of_scope']})"))

    # Check scope_category
    if not expected["out_of_scope"]:
        results["max_score"] += 1
        if parsed.get("scope_category") == "labour_employment_law":
            results["checks"].append(("✅", "scope_category=labour_employment_law"))
            results["score"] += 1
        else:
            results["checks"].append(("❌", f"scope_category={parsed.get('scope_category')} (expected labour_employment_law)"))

    # Check expected act presence
    if expected.get("expected_act"):
        results["max_score"] += 1
        full_text = json.dumps(parsed).lower()
        if expected["expected_act"].lower() in full_text:
            results["checks"].append(("✅", f"Contains expected act: {expected['expected_act']}"))
            results["score"] += 1
        else:
            results["checks"].append(("❌", f"Missing expected act: {expected['expected_act']}"))

    # Check expected case presence
    if expected.get("expected_case"):
        results["max_score"] += 1
        full_text = json.dumps(parsed).lower()
        if expected["expected_case"].lower() in full_text:
            results["checks"].append(("✅", f"Contains expected case: {expected['expected_case']}"))
            results["score"] += 1
        else:
            results["checks"].append(("❌", f"Missing expected case: {expected['expected_case']}"))

    # Check keywords
    for kw in expected.get("expected_keywords", []):
        results["max_score"] += 1
        full_text = json.dumps(parsed).lower()
        if kw.lower() in full_text:
            results["checks"].append(("✅", f"Keyword found: {kw}"))
            results["score"] += 1
        else:
            results["checks"].append(("⚠️", f"Keyword missing: {kw}"))

    # Validate summary counts consistency
    summary = parsed.get("summary", {})
    violations = parsed.get("primary_violations", [])
    cases = parsed.get("supporting_cases", [])

    if summary:
        results["max_score"] += 1
        if summary.get("violation_count") == len(violations):
            results["checks"].append(("✅", f"violation_count ({len(violations)}) matches array"))
            results["score"] += 1
        else:
            results["checks"].append(("⚠️", f"violation_count ({summary.get('violation_count')}) ≠ array ({len(violations)})"))

        results["max_score"] += 1
        if summary.get("cases_count") == len(cases):
            results["checks"].append(("✅", f"cases_count ({len(cases)}) matches array"))
            results["score"] += 1
        else:
            results["checks"].append(("⚠️", f"cases_count ({summary.get('cases_count')}) ≠ array ({len(cases)})"))

    return results


def run_generate_test(tc: dict) -> dict:
    """Run a test case using POST /generate."""
    prompt = make_prompt(tc["instruction"], tc["retrieved_context"])

    try:
        start = time.time()
        r = httpx.post(
            f"{MODEL_SERVER_URL}/generate",
            json={
                "prompt": prompt,
                "system_prompt": SYSTEM_PROMPT,
                "max_tokens": 6000,
                "temperature": 0.1,
                "top_p": 0.95,
            },
            timeout=TIMEOUT,
        )
        elapsed = time.time() - start

        if r.status_code >= 400:
            return {"success": False, "error": f"HTTP {r.status_code}: {r.text[:200]}", "time_s": elapsed}

        data = r.json()
        text = data.get("text", "")
        usage = data.get("usage", {})
        gen_time = data.get("generation_time_ms", 0)

        return {
            "success": True,
            "text": text,
            "model": data.get("model", "unknown"),
            "usage": usage,
            "generation_time_ms": gen_time,
            "total_time_s": round(elapsed, 2),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }

    except httpx.TimeoutException:
        return {"success": False, "error": f"Timeout after {TIMEOUT}s", "time_s": TIMEOUT}
    except Exception as e:
        return {"success": False, "error": str(e), "time_s": 0}


def run_chat_test(tc: dict) -> dict:
    """Run a test case using POST /chat."""
    user_content = make_prompt(tc["instruction"], tc["retrieved_context"])

    try:
        start = time.time()
        r = httpx.post(
            f"{MODEL_SERVER_URL}/chat",
            json={
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                "max_tokens": 6000,
                "temperature": 0.1,
                "top_p": 0.95,
            },
            timeout=TIMEOUT,
        )
        elapsed = time.time() - start

        if r.status_code >= 400:
            return {"success": False, "error": f"HTTP {r.status_code}: {r.text[:200]}", "time_s": elapsed}

        data = r.json()
        text = data.get("text", "")
        usage = data.get("usage", {})
        gen_time = data.get("generation_time_ms", 0)

        return {
            "success": True,
            "text": text,
            "model": data.get("model", "unknown"),
            "usage": usage,
            "generation_time_ms": gen_time,
            "total_time_s": round(elapsed, 2),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }

    except httpx.TimeoutException:
        return {"success": False, "error": f"Timeout after {TIMEOUT}s", "time_s": TIMEOUT}
    except Exception as e:
        return {"success": False, "error": str(e), "time_s": 0}


def run_test_case(tc: dict, endpoint: str = "generate") -> dict:
    """Run a single test case and validate."""
    print_section(f"Test #{tc['id']}: {tc['title']}")
    print(f"  Endpoint:    POST /{endpoint}")
    print(f"  Instruction: {tc['instruction'][:70]}...")
    print(f"  Context Len: {len(tc['retrieved_context'])} chars")

    # Run inference
    if endpoint == "chat":
        result = run_chat_test(tc)
    else:
        result = run_generate_test(tc)

    if not result["success"]:
        print(f"  ❌ FAILED: {result['error']}")
        return {"id": tc["id"], "title": tc["title"], "status": "FAILED", "error": result["error"]}

    # Display inference stats
    print(f"\n  ── Inference Results ──")
    print(f"  Model:             {result.get('model', 'N/A')}")
    print(f"  Total Time:        {result['total_time_s']:.2f}s")
    print(f"  Generation Time:   {result.get('generation_time_ms', 0)}ms")
    print(f"  Prompt Tokens:     {result.get('prompt_tokens', 0)}")
    print(f"  Completion Tokens: {result.get('completion_tokens', 0)}")
    print(f"  Output Length:     {len(result['text'])} chars")

    # Parse JSON
    parsed = parse_json_response(result["text"])
    if parsed is None:
        print(f"\n  ❌ FAILED: Could not parse JSON from response")
        print(f"  Raw (first 300 chars): {result['text'][:300]}")
        return {"id": tc["id"], "title": tc["title"], "status": "PARSE_FAIL", "raw_output": result["text"][:500]}

    print(f"\n  ✅ JSON parsed successfully")

    # Validate schema
    validation = validate_schema(parsed, tc["expected_fields"])

    print(f"\n  ── Schema Validation ──")
    for icon, msg in validation["checks"]:
        print(f"    {icon} {msg}")

    score_pct = (validation["score"] / validation["max_score"] * 100) if validation["max_score"] > 0 else 0
    print(f"\n  Validation Score: {validation['score']}/{validation['max_score']} ({score_pct:.0f}%)")

    # Determine pass/fail
    if score_pct >= 70:
        status = "PASS"
    elif score_pct >= 50:
        status = "WARN"
    else:
        status = "FAIL"

    icon = "✅" if status == "PASS" else "⚠️" if status == "WARN" else "❌"
    print(f"  Status: {icon} {status}")

    return {
        "id": tc["id"],
        "title": tc["title"],
        "status": status,
        "score": validation["score"],
        "max_score": validation["max_score"],
        "score_pct": round(score_pct, 1),
        "time_s": result["total_time_s"],
        "gen_time_ms": result.get("generation_time_ms", 0),
        "prompt_tokens": result.get("prompt_tokens", 0),
        "completion_tokens": result.get("completion_tokens", 0),
    }


def print_summary(results: list):
    """Print overall test summary."""
    print_header("MODEL SERVER TEST SUMMARY")

    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    warned = sum(1 for r in results if r["status"] == "WARN")
    failed = sum(1 for r in results if r["status"] in ("FAIL", "FAILED", "PARSE_FAIL"))

    print(f"\n  {'#':>3} {'Title':>40} {'Score':>10} {'Time':>8} {'Status':>8}")
    print(f"  {'─'*3} {'─'*40} {'─'*10} {'─'*8} {'─'*8}")

    for r in results:
        title = r.get("title", "N/A")[:40]
        score_str = f"{r.get('score', '-')}/{r.get('max_score', '-')}" if r.get("max_score") else "-"
        t = f"{r.get('time_s', 0):.1f}s" if r.get("time_s") else "-"
        status = r["status"]
        icon = "✅" if status == "PASS" else "⚠️" if status == "WARN" else "❌"
        print(f"  {r['id']:3d} {title:>40} {score_str:>10} {t:>8} {icon + ' ' + status:>10}")

    print(f"\n  Total:  {total}")
    print(f"  Passed: {passed}")
    print(f"  Warned: {warned}")
    print(f"  Failed: {failed}")

    times = [r["time_s"] for r in results if r.get("time_s")]
    if times:
        print(f"\n  Avg Inference Time: {sum(times)/len(times):.2f}s")
        print(f"  Max Inference Time: {max(times):.2f}s")


def main():
    global MODEL_SERVER_URL
    parser = argparse.ArgumentParser(
        description="Model Server Deployed Tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_model_server.py                        # Run all tests via /generate
  python test_model_server.py --case 1               # Run test case 1 only
  python test_model_server.py --endpoint chat         # Use /chat instead of /generate
  python test_model_server.py --port 5006             # Test local model server
        """
    )
    parser.add_argument("--case", type=int, help="Run specific test case (1-3)")
    parser.add_argument("--endpoint", choices=["generate", "chat"], default="generate", help="Endpoint to test")
    parser.add_argument("--port", type=int, default=5007, help="Model server port (default: 5007)")
    parser.add_argument("--base-url", type=str, default=None, help="Full base URL override")

    args = parser.parse_args()
    if args.base_url:
        MODEL_SERVER_URL = args.base_url
    else:
        MODEL_SERVER_URL = f"http://localhost:{args.port}"

    print_header("MODEL SERVER DEPLOYED TESTS")
    print(f"  Timestamp:  {datetime.now().isoformat()}")
    print(f"  Target:     {MODEL_SERVER_URL}")
    print(f"  Endpoint:   POST /{args.endpoint}")
    print(f"  Test Cases: {len(TEST_CASES)}")

    # Pre-flight checks
    healthy = test_health()
    test_proxy_status()
    test_model_info()

    if not healthy:
        print("\n  ⚠️  Model server health check failed. Tests may fail.")
        confirm = input("  Continue anyway? (y/N): ").strip().lower()
        if confirm != "y":
            print("  Aborted.")
            sys.exit(1)

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
        result = run_test_case(tc, endpoint=args.endpoint)
        results.append(result)

    # Summary
    print_summary(results)


if __name__ == "__main__":
    main()

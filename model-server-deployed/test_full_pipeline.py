"""
Full End-to-End Test: Backend (5005) -> Proxy (5007) -> Modal (A10G GPU)

Tests all endpoints through the complete pipeline.
"""
import httpx
import json
import time
import sys

BACKEND = "http://127.0.0.1:5005"
PROXY = "http://127.0.0.1:5007"

results = []

def test(name, method, url, json_body=None, timeout=300):
    """Run a test and print results."""
    t0 = time.time()
    try:
        if method == "GET":
            r = httpx.get(url, timeout=timeout)
        else:
            r = httpx.post(url, json=json_body, timeout=timeout)
        elapsed = time.time() - t0
        ok = r.status_code == 200
        data = r.json() if ok else r.text[:200]
        results.append({"name": name, "ok": ok, "status": r.status_code, "time": round(elapsed, 1)})
        print(f"{'PASS' if ok else 'FAIL'} | {name:40s} | {r.status_code} | {elapsed:.1f}s")
        if ok and isinstance(data, dict):
            # Print key info
            for key in ["status", "text", "active_model", "proxy"]:
                if key in data:
                    val = data[key]
                    if isinstance(val, str) and len(val) > 150:
                        val = val[:150] + "..."
                    print(f"       {key}: {val}")
        return data
    except Exception as e:
        elapsed = time.time() - t0
        results.append({"name": name, "ok": False, "status": 0, "time": round(elapsed, 1)})
        print(f"FAIL | {name:40s} | ERROR | {elapsed:.1f}s | {e}")
        return None

print("=" * 80)
print("  FULL PIPELINE TEST")
print(f"  Backend: {BACKEND}")
print(f"  Proxy:   {PROXY}")
print("=" * 80)
print()

# ── Proxy Tests ──
print("--- PROXY SERVER (port 5007) ---")
test("Proxy Root", "GET", f"{PROXY}/")
test("Proxy Health", "GET", f"{PROXY}/health")
test("Proxy Model Info", "GET", f"{PROXY}/model/info")
test("Proxy Status", "GET", f"{PROXY}/proxy/status")
test("Proxy Generate", "POST", f"{PROXY}/generate", {
    "prompt": "What is the Shop and Office Employees Act? /no_think",
    "max_tokens": 200, "temperature": 0.1,
})
test("Proxy Chat", "POST", f"{PROXY}/chat", {
    "messages": [
        {"role": "system", "content": "You are a Sri Lankan legal AI assistant. /no_think"},
        {"role": "user", "content": "What constitutes unfair dismissal?"},
    ],
    "max_tokens": 200, "temperature": 0.1,
})
print()

# ── Backend Tests ──
print("--- BACKEND (port 5005) ---")
test("Backend Root", "GET", f"{BACKEND}/")
test("Backend Health", "GET", f"{BACKEND}/api/health")
test("Backend Health Ready", "GET", f"{BACKEND}/api/health/ready")

# Test the main recommendation endpoint
print()
print("--- RECOMMENDATION (full pipeline) ---")
data = test("Recommend - Termination", "POST", f"{BACKEND}/api/query/recommend", {
    "query": "An employee was terminated after 10 years of service without any notice. What are their rights?"
})
if data and isinstance(data, dict):
    print(f"       out_of_scope: {data.get('out_of_scope')}")
    print(f"       confidence: {data.get('confidence')}")
    legal = data.get("legal_reasoning", "")
    if legal:
        print(f"       legal_reasoning (first 200): {legal[:200]}...")
    violations = data.get("primary_violations", [])
    print(f"       violations count: {len(violations)}")
    acts = data.get("applicable_acts", [])
    print(f"       applicable_acts count: {len(acts)}")

print()

# Summary
print("=" * 80)
print("  SUMMARY")
print("=" * 80)
passed = sum(1 for r in results if r["ok"])
total = len(results)
print(f"  Passed: {passed}/{total}")
for r in results:
    status = "PASS" if r["ok"] else "FAIL"
    print(f"    {status} | {r['name']:40s} | HTTP {r['status']} | {r['time']}s")
print("=" * 80)

if passed < total:
    print(f"\n  {total - passed} test(s) FAILED")
    sys.exit(1)
else:
    print("\n  ALL TESTS PASSED!")

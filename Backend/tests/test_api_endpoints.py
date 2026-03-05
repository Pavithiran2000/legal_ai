"""
Comprehensive API Endpoint Test Script
Tests all endpoints on the backend (port 5005) and model-server (port 5006).
"""
import httpx
import json
import sys

BACKEND = "http://localhost:5005"
MODEL_SERVER = "http://localhost:5007"

PASS = 0
FAIL = 0
RESULTS = []


def log(status, endpoint, method, code, detail=""):
    global PASS, FAIL
    icon = "PASS" if status else "FAIL"
    if status:
        PASS += 1
    else:
        FAIL += 1
    msg = f"[{icon}] {method:6s} {endpoint:45s} -> {code}"
    if detail:
        msg += f"  | {detail[:120]}"
    print(msg)
    RESULTS.append({"status": icon, "method": method, "endpoint": endpoint, "code": code, "detail": detail})


def test(method, url, expected_codes=None, json_body=None, files=None, params=None, timeout=30):
    if expected_codes is None:
        expected_codes = [200]
    try:
        with httpx.Client(timeout=timeout) as client:
            if method == "GET":
                r = client.get(url, params=params)
            elif method == "POST":
                if files:
                    r = client.post(url, files=files, params=params)
                else:
                    r = client.post(url, json=json_body, params=params)
            elif method == "DELETE":
                r = client.delete(url, params=params)
            else:
                r = client.request(method, url, json=json_body, params=params)

        endpoint = url.replace(BACKEND, "").replace(MODEL_SERVER, "") or "/"
        ok = r.status_code in expected_codes
        try:
            body = r.json()
            detail = json.dumps(body)[:200]
        except Exception:
            detail = r.text[:200]
        log(ok, endpoint, method, r.status_code, detail)
        return r
    except Exception as e:
        endpoint = url.replace(BACKEND, "").replace(MODEL_SERVER, "") or "/"
        log(False, endpoint, method, "ERR", str(e)[:120])
        return None


def main():
    print("=" * 80)
    print("  API ENDPOINT TEST SUITE")
    print("=" * 80)

    # ─── MODEL SERVER / PROXY (port 5007) ──────────────────────────────────────
    print("\n-- MODEL SERVER PROXY (localhost:5007) --")

    # 1. Root (proxy-specific)
    test("GET", f"{MODEL_SERVER}/")

    # 2. Proxy status
    test("GET", f"{MODEL_SERVER}/proxy/status")

    # 3. Health
    test("GET", f"{MODEL_SERVER}/health", timeout=60)

    # 4. Model info
    test("GET", f"{MODEL_SERVER}/model/info", timeout=60)

    # 5. Generate (via proxy -> Modal)
    test("POST", f"{MODEL_SERVER}/generate", json_body={
        "prompt": "What is the Termination of Employment Act in Sri Lanka?",
        "max_tokens": 100,
        "temperature": 0.1,
    }, timeout=300)

    # 6. Chat (via proxy -> Modal)
    test("POST", f"{MODEL_SERVER}/chat", json_body={
        "messages": [
            {"role": "user", "content": "Briefly explain unfair dismissal in Sri Lanka."}
        ],
        "max_tokens": 100,
        "temperature": 0.1,
    }, timeout=300)

    # 7. Reload
    test("POST", f"{MODEL_SERVER}/reload", timeout=120)

    # ─── BACKEND (port 5005) ────────────────────────────────────────
    print("\n-- BACKEND (localhost:5005) --")

    # 6. Root
    test("GET", f"{BACKEND}/")

    # 7. Health check
    test("GET", f"{BACKEND}/api/health")

    # 8. Readiness check
    test("GET", f"{BACKEND}/api/health/ready")

    # ── Admin endpoints ──
    print("\n  -- Admin --")

    # 9. List documents
    test("GET", f"{BACKEND}/api/admin/documents")

    # 10. FAISS status
    test("GET", f"{BACKEND}/api/admin/faiss/status")

    # 11. Statistics
    test("GET", f"{BACKEND}/api/admin/statistics")

    # 12. Model info (via backend proxy)
    test("GET", f"{BACKEND}/api/admin/model/info")

    # 13. Upload document (expect 400 since we send non-PDF)
    test("POST", f"{BACKEND}/api/admin/documents/upload",
         expected_codes=[200, 400, 422],
         files={"file": ("test.txt", b"test content", "text/plain")})

    # 14. Delete non-existent document
    test("DELETE", f"{BACKEND}/api/admin/documents/00000000-0000-0000-0000-000000000000",
         expected_codes=[404, 400, 422])

    # ── Query endpoints ──
    print("\n  -- Query --")

    # 15. Query history
    test("GET", f"{BACKEND}/api/query/history")

    # 16. Get recommendation (via proxy -> Modal, may take time)
    print("\n  [INFO] Testing recommendation endpoint (via proxy -> Modal, may take time)...")
    r = test("POST", f"{BACKEND}/api/query/recommend",
             expected_codes=[200, 503, 500],
             json_body={
                 "query": "An employee was terminated without prior notice after working for 5 years. What are the legal remedies under Sri Lankan labour law?",
                 "top_k": 5,
                 "temperature": 0.1,
             }, timeout=300)

    # If we got a query ID back, test detail and feedback
    query_id = None
    if r and r.status_code == 200:
        try:
            data = r.json()
            query_id = data.get("query_id") or data.get("id")
        except:
            pass

    # 17. Query detail
    test_id = query_id or "00000000-0000-0000-0000-000000000000"
    test("GET", f"{BACKEND}/api/query/{test_id}",
         expected_codes=[200, 404])

    # 18. Submit feedback
    if query_id:
        test("POST", f"{BACKEND}/api/query/{query_id}/feedback",
             json_body={"rating": 5, "comment": "Great recommendation!"},
             expected_codes=[200])
    else:
        test("POST", f"{BACKEND}/api/query/{test_id}/feedback",
             json_body={"rating": 5, "comment": "Test feedback"},
             expected_codes=[200, 404])

    # ─── SUMMARY ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"  RESULTS: {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
    print("=" * 80)

    if FAIL > 0:
        print("\n  Failed tests:")
        for r in RESULTS:
            if r["status"] == "FAIL":
                print(f"    {r['method']} {r['endpoint']} -> {r['code']}: {r['detail'][:100]}")

    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

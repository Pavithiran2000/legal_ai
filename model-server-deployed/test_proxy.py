"""Quick test for proxy server endpoints."""
import httpx
import json
import time

BASE = "http://localhost:5007"

# Test generate
print("=== GENERATE ===")
t0 = time.time()
r = httpx.post(BASE + "/generate", json={
    "prompt": "What are employee rights under the SEA? /no_think",
    "system_prompt": "You are a Sri Lankan legal expert.",
    "max_tokens": 300,
    "temperature": 0.1,
}, timeout=300)
elapsed = time.time() - t0
data = r.json()
print(f"Status: {r.status_code} ({elapsed:.1f}s)")
print(f"Model: {data.get('model')}")
print(f"Usage: {data.get('usage')}")
print(f"Text: {data.get('text', '')[:400]}")
print()

# Test chat
print("=== CHAT ===")
t0 = time.time()
r = httpx.post(BASE + "/chat", json={
    "messages": [
        {"role": "system", "content": "You are a Sri Lankan legal AI. /no_think"},
        {"role": "user", "content": "What is wrongful termination?"},
    ],
    "max_tokens": 300,
    "temperature": 0.1,
}, timeout=300)
elapsed = time.time() - t0
data = r.json()
print(f"Status: {r.status_code} ({elapsed:.1f}s)")
print(f"Model: {data.get('model')}")
print(f"Text: {data.get('text', '')[:400]}")
print()

# Test proxy status
print("=== PROXY STATUS ===")
r = httpx.get(BASE + "/proxy/status", timeout=30)
print(json.dumps(r.json(), indent=2))

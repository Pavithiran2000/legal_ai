"""
Local Proxy Server for Modal-Deployed Model Server.

Runs locally on port 5007 and proxies all requests to the Modal.com
deployed Ollama model server (qwen3-legal-model-server).

This lets the backend connect to localhost:5007 as if it were a local
model server, while the actual inference runs on Modal A10G GPU.

Usage:
    python proxy_server.py
    python proxy_server.py --port 5007
    python proxy_server.py --modal-url https://your-app--ollamaserver-serve.modal.run
"""
import os
import sys
import time
import logging
import argparse
from typing import Optional, List
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("proxy-server")

# ── Configuration ────────────────────────────────────────────────────

# The Modal deployment URL — updated after `modal deploy`
# Format: https://<workspace>--qwen3-legal-model-server-ollamaserver-serve.modal.run
MODAL_URL = os.environ.get(
    "MODAL_URL",
    ""  # Will be set after first deploy
)

PROXY_PORT = int(os.environ.get("PROXY_PORT", "5007"))
REQUEST_TIMEOUT = 300.0  # 5 minutes for LLM inference


# ── Pydantic Models (mirrors modal_app.py) ───────────────────────────

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=32000)
    system_prompt: Optional[str] = Field(default=None)
    max_tokens: Optional[int] = Field(default=3000, ge=1, le=8192)
    temperature: Optional[float] = Field(default=0.1, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0)
    model: Optional[str] = Field(default=None)


class ChatRequest(BaseModel):
    messages: List[dict] = Field(..., min_length=1)
    max_tokens: Optional[int] = Field(default=3000, ge=1, le=8192)
    temperature: Optional[float] = Field(default=0.1, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0)
    model: Optional[str] = Field(default=None)


class GenerateResponse(BaseModel):
    text: str
    model: str
    usage: dict
    generation_time_ms: int


class HealthResponse(BaseModel):
    status: str
    ollama_connected: bool
    active_model: str
    available_models: List[str]
    timestamp: str


class SwitchModelRequest(BaseModel):
    model: str


class SwitchModelResponse(BaseModel):
    previous_model: str
    active_model: str
    status: str


# ── HTTP Client ──────────────────────────────────────────────────────

_client: Optional[httpx.AsyncClient] = None


def _get_modal_url() -> str:
    """Get current Modal URL (can be set dynamically)."""
    url = os.environ.get("MODAL_URL", MODAL_URL)
    if not url:
        raise HTTPException(
            status_code=503,
            detail=(
                "MODAL_URL not configured. Deploy first with: "
                "modal deploy modal_app.py, then set MODAL_URL env var "
                "or pass --modal-url to proxy_server.py"
            ),
        )
    return url.rstrip("/")


async def _forward(method: str, path: str, json_body: dict = None) -> dict:
    """Forward a request to the Modal deployment."""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT))

    url = f"{_get_modal_url()}{path}"
    t0 = time.time()

    try:
        if method == "GET":
            resp = await _client.get(url)
        else:
            resp = await _client.post(url, json=json_body)

        elapsed = time.time() - t0
        logger.info(f"{method} {path} → {resp.status_code} ({elapsed:.1f}s)")

        if resp.status_code >= 400:
            detail = resp.text
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                pass
            raise HTTPException(status_code=resp.status_code, detail=detail)

        return resp.json()

    except httpx.TimeoutException:
        elapsed = time.time() - t0
        logger.error(f"Timeout after {elapsed:.1f}s: {method} {path}")
        raise HTTPException(status_code=504, detail="Modal inference timed out")
    except httpx.ConnectError as e:
        logger.error(f"Connection error: {e}")
        raise HTTPException(status_code=502, detail=f"Cannot reach Modal deployment: {e}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Proxy error: {e}")
        raise HTTPException(status_code=502, detail=f"Proxy error: {e}")


# ── Lifespan ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client
    _client = httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT))

    modal_url = os.environ.get("MODAL_URL", MODAL_URL)
    logger.info("=" * 60)
    logger.info(f"  Proxy Server starting on port {PROXY_PORT}")
    logger.info(f"  Modal URL: {modal_url or '(NOT SET)'}")
    logger.info("=" * 60)

    if modal_url:
        # Test connectivity
        try:
            resp = await _client.get(f"{modal_url.rstrip('/')}/health", timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                logger.info(f"  Modal connected! Model: {data.get('active_model')}")
            else:
                logger.warning(f"  Modal health check returned {resp.status_code}")
        except Exception as e:
            logger.warning(f"  Cannot reach Modal (may be cold-starting): {e}")
    else:
        logger.warning("  MODAL_URL not set. Set it after deploying with:")
        logger.warning("    $env:MODAL_URL = 'https://your-url.modal.run'")

    yield

    if _client:
        await _client.aclose()
    logger.info("Proxy server shut down")


# ── FastAPI App ──────────────────────────────────────────────────────

app = FastAPI(
    title="Modal Model Server Proxy",
    description="Local proxy that forwards to Modal-deployed Ollama on A10G GPU",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints (same API as local model-server) ──────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Proxy health — checks both proxy and Modal deployment."""
    try:
        data = await _forward("GET", "/health")
        return HealthResponse(**data)
    except HTTPException:
        from datetime import datetime
        return HealthResponse(
            status="proxy-only",
            ollama_connected=False,
            active_model="unknown",
            available_models=[],
            timestamp=datetime.utcnow().isoformat(),
        )


@app.get("/model/info")
async def model_info():
    """Get model info from Modal deployment."""
    data = await _forward("GET", "/model/info")
    data["proxy"] = True
    data["proxy_port"] = PROXY_PORT
    return data


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Forward generate request to Modal."""
    data = await _forward("POST", "/generate", request.model_dump())
    return GenerateResponse(**data)


@app.post("/chat", response_model=GenerateResponse)
async def chat(request: ChatRequest):
    """Forward chat request to Modal."""
    data = await _forward("POST", "/chat", request.model_dump())
    return GenerateResponse(**data)


@app.post("/switch-model", response_model=SwitchModelResponse)
async def switch_model(request: SwitchModelRequest):
    """Forward model switch to Modal."""
    data = await _forward("POST", "/switch-model", request.model_dump())
    return SwitchModelResponse(**data)


@app.post("/reload")
async def reload_model():
    """Forward reload to Modal."""
    return await _forward("POST", "/reload")


@app.get("/")
async def root():
    """Proxy root — identifies this as a proxy server."""
    return {
        "name": "Modal Model Server Proxy",
        "version": "1.0.0",
        "proxy_port": PROXY_PORT,
        "modal_url": os.environ.get("MODAL_URL", MODAL_URL) or "(not configured)",
        "status": "running",
    }


@app.get("/proxy/status")
async def proxy_status():
    """Detailed proxy status including Modal connectivity."""
    modal_url = os.environ.get("MODAL_URL", MODAL_URL)
    result = {
        "proxy_running": True,
        "proxy_port": PROXY_PORT,
        "modal_url": modal_url or "(not configured)",
        "modal_reachable": False,
        "modal_health": None,
    }
    if modal_url:
        try:
            data = await _forward("GET", "/health")
            result["modal_reachable"] = True
            result["modal_health"] = data
        except Exception as e:
            result["modal_error"] = str(e)
    return result


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Proxy server for Modal deployment")
    parser.add_argument("--port", type=int, default=PROXY_PORT, help="Port (default: 5007)")
    parser.add_argument("--modal-url", type=str, default=None, help="Modal deployment URL")
    args = parser.parse_args()

    if args.modal_url:
        os.environ["MODAL_URL"] = args.modal_url

    import uvicorn
    uvicorn.run("proxy_server:app", host="0.0.0.0", port=args.port, reload=False)

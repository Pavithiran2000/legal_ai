"""
Model Server - Ollama-based LLM Inference API for Sri Lankan Legal AI.

Serves fine-tuned Qwen3 models via Ollama on port 8001.
Supports switching between 4B (default) and 8B models.
"""
import logging
import re
import time
import subprocess
import sys
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Current active model
active_model: str = settings.default_model
ollama_available: bool = False


# ── Pydantic Models ──────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=32000)
    system_prompt: Optional[str] = Field(default=None)
    max_tokens: Optional[int] = Field(default=3000, ge=1, le=8192)
    temperature: Optional[float] = Field(default=0.1, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0)
    model: Optional[str] = Field(default=None, description="Override model name")


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


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen3 output."""
    if not text:
        return text
    # Remove <think>...</think> blocks (greedy to handle nested)
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Also remove unclosed <think> blocks at the start
    cleaned = re.sub(r'^<think>.*$', '', cleaned, flags=re.DOTALL) if '<think>' in cleaned and '</think>' not in cleaned else cleaned
    return cleaned.strip()


class HealthResponse(BaseModel):
    status: str
    ollama_connected: bool
    active_model: str
    available_models: List[str]
    timestamp: str


class SwitchModelRequest(BaseModel):
    model: str = Field(..., description="Model name: sri-legal-4b or sri-legal-8b")


class SwitchModelResponse(BaseModel):
    previous_model: str
    active_model: str
    status: str


# ── Helpers ──────────────────────────────────────────────────────

def check_ollama() -> bool:
    """Check if Ollama is reachable."""
    try:
        import httpx
        r = httpx.get(f"{settings.ollama_host}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def get_ollama_models() -> List[str]:
    """Get list of models available in Ollama."""
    try:
        import httpx
        r = httpx.get(f"{settings.ollama_host}/api/tags", timeout=5)
        if r.status_code == 200:
            data = r.json()
            return [m["name"].split(":")[0] for m in data.get("models", [])]
    except Exception:
        pass
    return []


def ensure_ollama_running():
    """Start Ollama if not running."""
    global ollama_available
    if check_ollama():
        ollama_available = True
        return True

    logger.info("Starting Ollama...")
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        )
        import asyncio
        time.sleep(5)
        ollama_available = check_ollama()
        return ollama_available
    except FileNotFoundError:
        logger.error("Ollama not installed")
        return False


# ── Lifespan ─────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ollama_available, active_model
    logger.info("Starting Model Server...")

    # Ensure Ollama is running
    if ensure_ollama_running():
        logger.info("Ollama connected")
        models = get_ollama_models()
        logger.info(f"Available Ollama models: {models}")

        # Check if our models exist, if not try to create them
        if settings.default_model not in models:
            logger.warning(f"Default model '{settings.default_model}' not found in Ollama. Running setup...")
            try:
                from setup_models import setup
                setup()
                models = get_ollama_models()
            except Exception as e:
                logger.error(f"Setup failed: {e}")

        if settings.default_model in models:
            active_model = settings.default_model
            logger.info(f"Active model: {active_model}")
            # Warm up: pull model into memory
            try:
                import ollama as ollama_client
                ollama_client.chat(model=active_model, messages=[
                    {"role": "user", "content": "hello"}
                ])
                logger.info("Model warmed up")
            except Exception as e:
                logger.warning(f"Warmup failed (non-critical): {e}")
        else:
            logger.warning("No legal models found in Ollama")
    else:
        logger.error("Ollama not available")

    yield
    logger.info("Shutting down Model Server")


# ── FastAPI App ──────────────────────────────────────────────────

app = FastAPI(
    title="Sri Lankan Legal Model Server",
    description="Ollama-based LLM inference for fine-tuned Qwen3 legal models",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    from datetime import datetime
    is_connected = check_ollama()
    models = get_ollama_models() if is_connected else []
    return HealthResponse(
        status="healthy" if is_connected and active_model in models else "degraded",
        ollama_connected=is_connected,
        active_model=active_model,
        available_models=models,
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/model/info")
async def model_info():
    return {
        "active_model": active_model,
        "default_model": settings.default_model,
        "available_models": settings.available_models,
        "ollama_models": get_ollama_models(),
        "context_length": settings.num_ctx,
        "max_tokens": settings.max_tokens,
        "temperature": settings.temperature,
        "ollama_host": settings.ollama_host,
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text using Ollama chat API."""
    if not ollama_available:
        raise HTTPException(status_code=503, detail="Ollama not available")

    model = request.model or active_model
    try:
        import ollama as ollama_client
        start = time.time()

        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        response = ollama_client.chat(
            model=model,
            messages=messages,
            options={
                "temperature": request.temperature,
                "top_p": request.top_p,
                "num_predict": request.max_tokens,
                "num_ctx": settings.num_ctx,
            }
        )

        gen_time = int((time.time() - start) * 1000)
        text = strip_think_blocks(response.message.content or "")

        return GenerateResponse(
            text=text,
            model=model,
            usage={
                "prompt_tokens": response.prompt_eval_count or 0,
                "completion_tokens": response.eval_count or 0,
                "total_tokens": (response.prompt_eval_count or 0) + (response.eval_count or 0),
            },
            generation_time_ms=gen_time
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/chat", response_model=GenerateResponse)
async def chat(request: ChatRequest):
    """Chat completion using Ollama."""
    if not ollama_available:
        raise HTTPException(status_code=503, detail="Ollama not available")

    model = request.model or active_model
    try:
        import ollama as ollama_client
        start = time.time()

        response = ollama_client.chat(
            model=model,
            messages=request.messages,
            options={
                "temperature": request.temperature,
                "top_p": request.top_p,
                "num_predict": request.max_tokens,
                "num_ctx": settings.num_ctx,
            }
        )

        gen_time = int((time.time() - start) * 1000)
        text = strip_think_blocks(response.message.content or "")

        return GenerateResponse(
            text=text,
            model=model,
            usage={
                "prompt_tokens": response.prompt_eval_count or 0,
                "completion_tokens": response.eval_count or 0,
                "total_tokens": (response.prompt_eval_count or 0) + (response.eval_count or 0),
            },
            generation_time_ms=gen_time
        )
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.post("/switch-model", response_model=SwitchModelResponse)
async def switch_model(request: SwitchModelRequest):
    """Switch active model between 4B and 8B."""
    global active_model

    models = get_ollama_models()
    if request.model not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model}' not found. Available: {models}"
        )

    prev = active_model
    active_model = request.model

    # Warm up new model
    try:
        import ollama as ollama_client
        ollama_client.chat(model=active_model, messages=[
            {"role": "user", "content": "test"}
        ])
    except Exception:
        pass

    return SwitchModelResponse(
        previous_model=prev,
        active_model=active_model,
        status="switched"
    )


@app.post("/reload")
async def reload_model():
    """Reload the current model."""
    global ollama_available
    ollama_available = check_ollama()
    if ollama_available:
        try:
            import ollama as ollama_client
            ollama_client.chat(model=active_model, messages=[
                {"role": "user", "content": "test"}
            ])
            return {"status": "reloaded", "model": active_model}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    raise HTTPException(status_code=503, detail="Ollama not available")


# ── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=settings.host, port=settings.port, reload=True)

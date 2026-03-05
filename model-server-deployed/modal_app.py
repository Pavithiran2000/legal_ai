"""
Modal.com Deployed Model Server - Ollama on A10G GPU.

Runs the finetuned Qwen3-8B model via Ollama on Modal.com infrastructure.
GPU: NVIDIA A10G (24 GB VRAM)
Volume: qwen-3-finetuned (persistent model storage)
"""
import os
import re
import subprocess
import time
import logging
from typing import Optional, List

import modal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Modal Setup ──────────────────────────────────────────────────────

app = modal.App("qwen3-legal-model-server")

volume = modal.Volume.from_name("qwen-3-finetuned", create_if_missing=True)

MODEL_DIR = "/model-data"
OLLAMA_MODELS_DIR = "/root/.ollama"
MODEL_NAME = "sri-legal-8b"
GGUF_FILENAME = "qwen3_8b.gguf"

# Build container image with Ollama pre-installed
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "ca-certificates", "procps", "lsof", "zstd")
    .run_commands(
        "curl -fsSL https://ollama.com/install.sh | sh",
        gpu="a10g",  # Install with GPU support
    )
    .pip_install(
        "fastapi>=0.115.0",
        "uvicorn[standard]>=0.30.0",
        "pydantic>=2.0",
        "pydantic-settings>=2.0",
        "httpx>=0.27.0",
    )
)


# ── Pydantic Models (same API as local model-server) ────────────────

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
    model: str = Field(..., description="Model name")


class SwitchModelResponse(BaseModel):
    previous_model: str
    active_model: str
    status: str


# ── Helper Functions ─────────────────────────────────────────────────

def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen3 output."""
    if not text:
        return text
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    if '<think>' in cleaned and '</think>' not in cleaned:
        cleaned = re.sub(r'^<think>.*$', '', cleaned, flags=re.DOTALL)
    return cleaned.strip()


def _merge_chunks_if_needed():
    """Merge chunked uploads into the final GGUF file if needed."""
    gguf_path = os.path.join(MODEL_DIR, GGUF_FILENAME)
    chunks_dir = os.path.join(MODEL_DIR, "chunks")

    # Already have the full file
    if os.path.exists(gguf_path) and os.path.getsize(gguf_path) > 1_000_000_000:
        print(f"[OK] Model file exists: {gguf_path} ({os.path.getsize(gguf_path) / 1e9:.1f} GB)")
        return True

    # Check for chunks
    if not os.path.exists(chunks_dir):
        print(f"[ERROR] No model file and no chunks directory at {chunks_dir}")
        return False

    chunk_files = sorted([
        f for f in os.listdir(chunks_dir)
        if f.startswith("chunk_")
    ])

    if not chunk_files:
        print("[ERROR] No chunk files found")
        return False

    print(f"[MERGE] Found {len(chunk_files)} chunks. Merging into {GGUF_FILENAME}...")
    start = time.time()

    with open(gguf_path, 'wb') as out:
        for i, chunk_name in enumerate(chunk_files):
            chunk_path = os.path.join(chunks_dir, chunk_name)
            chunk_size = os.path.getsize(chunk_path)
            print(f"  [{i+1}/{len(chunk_files)}] {chunk_name} ({chunk_size / 1e6:.1f} MB)")
            with open(chunk_path, 'rb') as f:
                while True:
                    data = f.read(64 * 1024 * 1024)  # 64MB buffer
                    if not data:
                        break
                    out.write(data)

    elapsed = time.time() - start
    final_size = os.path.getsize(gguf_path)
    print(f"[OK] Merged in {elapsed:.1f}s → {final_size / 1e9:.2f} GB")

    # Persist merged file to volume
    volume.commit()
    return True


def _create_modelfile():
    """Create Ollama Modelfile for the GGUF model."""
    gguf_path = os.path.join(MODEL_DIR, GGUF_FILENAME)
    modelfile_path = os.path.join(MODEL_DIR, "Modelfile")

    content = f"""FROM {gguf_path}

PARAMETER temperature 0.1
PARAMETER top_p 0.95
PARAMETER num_ctx 4096
PARAMETER stop <|im_end|>
PARAMETER stop <|endoftext|>

TEMPLATE \"\"\"{{{{- if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{- end }}}}
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
\"\"\"
"""
    with open(modelfile_path, "w") as f:
        f.write(content)
    print(f"[OK] Modelfile created at {modelfile_path}")
    return modelfile_path


def _start_ollama():
    """Start Ollama server and wait for it to be ready."""
    print("[OLLAMA] Starting Ollama server...")
    env = os.environ.copy()
    env["OLLAMA_HOST"] = "0.0.0.0:11434"

    proc = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    # Wait for Ollama to be ready
    import httpx
    for attempt in range(30):
        try:
            r = httpx.get("http://localhost:11434/api/tags", timeout=3)
            if r.status_code == 200:
                print(f"[OK] Ollama ready after {attempt + 1}s")
                return proc
        except Exception:
            pass
        time.sleep(1)

    raise RuntimeError("Ollama failed to start within 30 seconds")


def _create_ollama_model():
    """Create the Ollama model from GGUF."""
    import httpx

    # Check if model already exists
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code == 200:
            models = [m["name"].split(":")[0] for m in r.json().get("models", [])]
            if MODEL_NAME in models:
                print(f"[OK] Model '{MODEL_NAME}' already exists in Ollama")
                return True
    except Exception:
        pass

    # Create Modelfile
    modelfile_path = _create_modelfile()

    # Create model via CLI
    print(f"[OLLAMA] Creating model '{MODEL_NAME}'...")
    result = subprocess.run(
        ["ollama", "create", MODEL_NAME, "-f", modelfile_path],
        capture_output=True, text=True, timeout=600
    )

    if result.returncode == 0:
        print(f"[OK] Model '{MODEL_NAME}' created!")
        return True
    else:
        print(f"[ERROR] ollama create failed: {result.stderr}")
        return False


def _warmup_model():
    """Warm up the model by running a test generation."""
    import httpx
    print(f"[WARMUP] Loading model '{MODEL_NAME}' into GPU memory...")
    try:
        r = httpx.post(
            "http://localhost:11434/api/chat",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
            timeout=120,
        )
        if r.status_code == 200:
            print("[OK] Model warmed up and loaded into GPU memory")
            return True
    except Exception as e:
        print(f"[WARN] Warmup failed: {e}")
    return False


def _get_ollama_models():
    """Get list of available Ollama models."""
    import httpx
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code == 200:
            return [m["name"].split(":")[0] for m in r.json().get("models", [])]
    except Exception:
        pass
    return []


def _check_ollama():
    """Check if Ollama is responding."""
    import httpx
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


# ── Modal Class for GPU Inference ────────────────────────────────────

@app.cls(
    image=image,
    gpu="a10g",
    volumes={MODEL_DIR: volume},
    timeout=600,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=10)
class OllamaServer:
    """Runs Ollama with finetuned Qwen3 on A10G GPU."""

    active_model: str = MODEL_NAME
    ollama_proc = None

    @modal.enter()
    def setup(self):
        """Container startup: merge chunks if needed, start Ollama, create & load model."""
        print("=" * 60)
        print("  Modal Model Server - Container Starting")
        print(f"  GPU: A10G (24 GB VRAM)")
        print("=" * 60)

        # Step 1: Ensure model file exists (merge chunks if needed)
        volume.reload()  # Get latest volume state
        if not _merge_chunks_if_needed():
            raise RuntimeError(
                f"Model file not found at {MODEL_DIR}/{GGUF_FILENAME}. "
                "Upload it first using upload_model.py"
            )

        # Step 2: Start Ollama server
        self.ollama_proc = _start_ollama()

        # Step 3: Create model from GGUF
        if not _create_ollama_model():
            raise RuntimeError("Failed to create Ollama model")

        # Step 4: Warm up (load into GPU memory)
        _warmup_model()

        self.active_model = MODEL_NAME
        print("=" * 60)
        print("  Model Server READY")
        print("=" * 60)

    @modal.asgi_app()
    def serve(self):
        """Serve FastAPI app with same endpoints as local model-server."""
        import httpx as httpx_client
        from datetime import datetime

        web_app = FastAPI(
            title="Sri Lankan Legal Model Server (Modal Deployed)",
            description="Ollama on A10G GPU - Finetuned Qwen3 for Sri Lankan Legal AI",
            version="2.0.0-modal",
        )

        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        server_ref = self  # capture self for closures

        @web_app.get("/health", response_model=HealthResponse)
        async def health_check():
            is_connected = _check_ollama()
            models = _get_ollama_models() if is_connected else []
            return HealthResponse(
                status="healthy" if is_connected and server_ref.active_model in models else "degraded",
                ollama_connected=is_connected,
                active_model=server_ref.active_model,
                available_models=models,
                timestamp=datetime.utcnow().isoformat(),
            )

        @web_app.get("/model/info")
        async def model_info():
            return {
                "active_model": server_ref.active_model,
                "default_model": MODEL_NAME,
                "available_models": _get_ollama_models(),
                "ollama_models": _get_ollama_models(),
                "context_length": 4096,
                "max_tokens": 4000,
                "temperature": 0.1,
                "ollama_host": "http://localhost:11434",
                "deployment": "modal.com",
                "gpu": "NVIDIA A10G (24 GB)",
            }

        @web_app.post("/generate", response_model=GenerateResponse)
        async def generate(request: GenerateRequest):
            model = request.model or server_ref.active_model
            start = time.time()

            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.prompt})

            try:
                async with httpx_client.AsyncClient(timeout=300) as client:
                    resp = await client.post(
                        "http://localhost:11434/api/chat",
                        json={
                            "model": model,
                            "messages": messages,
                            "stream": False,
                            "options": {
                                "temperature": request.temperature,
                                "top_p": request.top_p,
                                "num_predict": request.max_tokens,
                                "num_ctx": 4096,
                            },
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Ollama error: {e}")

            gen_time = int((time.time() - start) * 1000)
            raw_text = data.get("message", {}).get("content", "")
            text = strip_think_blocks(raw_text)

            return GenerateResponse(
                text=text,
                model=model,
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": (data.get("prompt_eval_count", 0) + data.get("eval_count", 0)),
                },
                generation_time_ms=gen_time,
            )

        @web_app.post("/chat", response_model=GenerateResponse)
        async def chat(request: ChatRequest):
            model = request.model or server_ref.active_model
            start = time.time()

            try:
                async with httpx_client.AsyncClient(timeout=300) as client:
                    resp = await client.post(
                        "http://localhost:11434/api/chat",
                        json={
                            "model": model,
                            "messages": request.messages,
                            "stream": False,
                            "options": {
                                "temperature": request.temperature,
                                "top_p": request.top_p,
                                "num_predict": request.max_tokens,
                                "num_ctx": 4096,
                            },
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Ollama error: {e}")

            gen_time = int((time.time() - start) * 1000)
            raw_text = data.get("message", {}).get("content", "")
            text = strip_think_blocks(raw_text)

            return GenerateResponse(
                text=text,
                model=model,
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": (data.get("prompt_eval_count", 0) + data.get("eval_count", 0)),
                },
                generation_time_ms=gen_time,
            )

        @web_app.post("/switch-model", response_model=SwitchModelResponse)
        async def switch_model(request: SwitchModelRequest):
            models = _get_ollama_models()
            if request.model not in models:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model '{request.model}' not found. Available: {models}",
                )
            prev = server_ref.active_model
            server_ref.active_model = request.model
            return SwitchModelResponse(
                previous_model=prev,
                active_model=server_ref.active_model,
                status="switched",
            )

        @web_app.post("/reload")
        async def reload_model():
            if _check_ollama():
                _warmup_model()
                return {"status": "reloaded", "model": server_ref.active_model}
            raise HTTPException(status_code=503, detail="Ollama not available")

        @web_app.get("/")
        async def root():
            return {
                "name": "Sri Lankan Legal Model Server (Modal)",
                "version": "2.0.0-modal",
                "gpu": "NVIDIA A10G (24 GB)",
                "status": "running",
            }

        return web_app


# ── Utility Functions (run remotely) ────────────────────────────────

@app.function(
    image=image,
    volumes={MODEL_DIR: volume},
    timeout=1800,
)
def merge_model_chunks(filename: str = GGUF_FILENAME):
    """Merge uploaded chunks into the final GGUF file on the volume."""
    volume.reload()
    success = _merge_chunks_if_needed()
    if success:
        volume.commit()
        final_path = os.path.join(MODEL_DIR, filename)
        size = os.path.getsize(final_path) if os.path.exists(final_path) else 0
        return {"status": "merged", "file": filename, "size_bytes": size}
    return {"status": "error", "message": "Merge failed"}


@app.function(
    image=image,
    volumes={MODEL_DIR: volume},
    timeout=60,
)
def list_volume_contents():
    """List all files on the model volume."""
    volume.reload()
    result = []
    for root, dirs, files in os.walk(MODEL_DIR):
        for f in files:
            full = os.path.join(root, f)
            rel = os.path.relpath(full, MODEL_DIR)
            size = os.path.getsize(full)
            result.append({"path": rel, "size_mb": round(size / 1e6, 1)})
    return result


@app.function(
    image=image,
    volumes={MODEL_DIR: volume},
    timeout=60,
)
def check_model_ready():
    """Check if the model file is ready on the volume."""
    volume.reload()
    gguf_path = os.path.join(MODEL_DIR, GGUF_FILENAME)
    if os.path.exists(gguf_path):
        size = os.path.getsize(gguf_path)
        return {
            "ready": size > 1_000_000_000,
            "file": GGUF_FILENAME,
            "size_gb": round(size / 1e9, 2),
        }
    # Check for chunks
    chunks_dir = os.path.join(MODEL_DIR, "chunks")
    if os.path.exists(chunks_dir):
        chunks = sorted(os.listdir(chunks_dir))
        total = sum(os.path.getsize(os.path.join(chunks_dir, c)) for c in chunks)
        return {
            "ready": False,
            "chunks_found": len(chunks),
            "total_chunk_size_gb": round(total / 1e9, 2),
            "message": "Run merge_model_chunks() to merge",
        }
    return {"ready": False, "message": "No model file or chunks found"}

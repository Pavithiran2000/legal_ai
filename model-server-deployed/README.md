# Modal-Deployed Model Server

Runs the finetuned Qwen3-8B model on Modal.com with NVIDIA A10G GPU (24 GB VRAM) via Ollama.

## Architecture

```
Backend (port 5005)
    → Proxy Server (port 5007, local)
        → Modal Deployment (A10G GPU, cloud)
            → Ollama + sri-legal-8b model
```

## Files

| File | Purpose |
|---|---|
| `modal_app.py` | Modal deployment — Ollama on A10G GPU with full API |
| `upload_model.py` | Chunked upload of GGUF model to Modal volume |
| `proxy_server.py` | Local proxy on port 5007 → forwards to Modal |
| `requirements.txt` | Python dependencies |

## Setup Steps

### 1. Authenticate with Modal
```bash
modal token set
```

### 2. Deploy the Modal app (creates volume + deploys server)
```bash
modal deploy modal_app.py
```
This will output the deployment URL, e.g.:
`https://<workspace>--qwen3-legal-model-server-ollamaserver-serve.modal.run`

### 3. Upload the model (chunked, 300 MB per chunk)
```bash
# Upload 8B model (~4.7 GB → 16 chunks)
python upload_model.py --source ../model-server/models/qwen3_8b.gguf

# Check volume status
python upload_model.py --check

# Merge chunks if needed
python upload_model.py --merge-only
```

### 4. Start the local proxy (port 5007)
```bash
# Set the Modal URL from step 2
$env:MODAL_URL = "https://your-workspace--qwen3-legal-model-server-ollamaserver-serve.modal.run"

# Start proxy
python proxy_server.py --port 5007
```

### 5. Update Backend config
In `Backend/.env`, set:
```
MODEL_SERVER_URL=http://localhost:5007
```

### 6. Start Backend
```bash
cd ../Backend
uvicorn src.main:app --host 0.0.0.0 --port 5005
```

## Endpoints (same API as local model-server)

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/model/info` | Model information |
| POST | `/generate` | Text generation |
| POST | `/chat` | Chat completion |
| POST | `/switch-model` | Switch active model |
| POST | `/reload` | Reload model |

## GPU Specs
- **GPU**: NVIDIA A10G (24 GB VRAM)
- **Volume**: `qwen-3-finetuned` (persistent model storage)
- **Model**: `sri-legal-8b` (finetuned Qwen3-8B, ~4.7 GB GGUF)

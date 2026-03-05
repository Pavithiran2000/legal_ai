# Model Server

Standalone Ollama-based LLM inference server for Sri Lankan Labour Law Case Recommendation Platform.

## Overview

This is a **standalone** model server that provides inference API using fine-tuned Qwen3 models via Ollama. It runs independently from the main backend — **no `.env` file required**. All configuration has sensible defaults.

## Features

- **Ollama Integration**: Uses Ollama for efficient model serving with GGUF models
- **Dual Model Support**: Switch between 8B (recommended) and 4B models at runtime
- **Think Block Stripping**: Automatically strips `<think>` blocks from Qwen3 output
- **REST API**: FastAPI-based endpoints for chat, generation, model switching
- **Health Checks**: Built-in health monitoring with Ollama connectivity status
- **Standalone**: No `.env` or external config needed — just run it

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running
- 8GB+ RAM (for 8B model)
- GGUF model files in `models/` directory

## Installation

1. **Create virtual environment:**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install Ollama:**
Download from [ollama.com](https://ollama.com/) and ensure it's running.

4. **Place model files:**
Place GGUF model files in the `models/` directory:
```
models/
├── qwen3_8b.gguf     # 8B model (recommended, ~5GB)
└── qwen3-4b.gguf     # 4B model (lighter, ~2.4GB)
```

5. **Create Ollama models from GGUF files:**
```bash
python setup_models.py
```

## Configuration

**No `.env` file needed.** All defaults are hardcoded in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `5006` | Server port |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API URL |
| `DEFAULT_MODEL` | `sri-legal-8b` | Default active model |
| `TEMPERATURE` | `0.1` | Generation temperature |
| `NUM_CTX` | `4096` | Context window size |
| `MAX_TOKENS` | `3000` | Max generation tokens |

Override any setting via environment variables if needed:
```bash
PORT=6000 python server.py
```

## Running the Server

```bash
# Development (default port 5006)
python server.py

# Production with uvicorn
uvicorn server:app --host 0.0.0.0 --port 5006 --workers 1
```

Verify it's running:
```bash
curl http://localhost:5006/health
```

## API Endpoints

### Health Check
```
GET /health
```
Returns server status, active model, and Ollama connectivity.

### Model Info
```
GET /model/info
```
Returns active model, available models, and generation settings.

### Chat (Primary Endpoint)
```
POST /chat
Content-Type: application/json

{
    "messages": [
        {"role": "system", "content": "You are a legal assistant..."},
        {"role": "user", "content": "What is wrongful termination?"}
    ],
    "temperature": 0.1,
    "max_tokens": 3000
}
```
Response:
```json
{
    "text": "Generated legal response...",
    "model": "sri-legal-8b",
    "generation_time_ms": 4500
}
```

### Generate (Simple Prompt)
```
POST /generate
Content-Type: application/json

{
    "prompt": "Your legal question here...",
    "temperature": 0.1,
    "max_tokens": 3000
}
```

### Switch Model
```
POST /switch-model
Content-Type: application/json

{"model": "sri-legal-8b"}
```

### Reload Model
```
POST /reload
```

## Architecture

```
model-server/
├── server.py          # Main FastAPI application
├── config.py          # Standalone configuration (no .env needed)
├── setup_models.py    # Create Ollama models from GGUF files
├── requirements.txt   # Python dependencies
├── Modelfile.8b       # Ollama Modelfile for 8B
├── Modelfile.4b       # Ollama Modelfile for 4B
├── README.md
├── models/
│   ├── qwen3_8b.gguf  # Fine-tuned 8B model
│   └── qwen3-4b.gguf  # Fine-tuned 4B model
└── latest-fintune/
    ├── qwen3-finetune.ipynb   # Finetuning notebook
    └── labour_data.jsonl      # Training data
```

## Integration with Backend

The Backend connects to this server via HTTP:

```python
# In Backend .env
MODEL_SERVER_URL=http://localhost:5006
```

The Backend uses the `/chat` endpoint for all legal query inference.

## Notes

- **8B model is recommended** — the 4B model may produce lower quality outputs
- Ollama must be running before starting this server
- The server automatically strips `<think>...</think>` blocks from Qwen3 output
- Only one worker is recommended since Ollama handles one request at a time

## License

Internal use only — Sri Lankan Labour Law Project

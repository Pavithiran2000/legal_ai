# Sri Lankan Labour Law Case Recommendation — Backend

Advanced RAG-based legal recommendation system for Sri Lankan Labour and Employment Law.

## Architecture Overview

```
User Query → Backend (FastAPI) → Gemini Embedding → FAISS Search → Context Retrieval
                                                                          ↓
     Response ← JSON Parsing ← Model Server (Ollama/Qwen3 8B) ← LLM Prompt with Context
```

- **Embedding**: Google Gemini Embedding API (`models/gemini-embedding-001`, 3072 dimensions)
- **Vector Store**: FAISS (IndexFlatIP with L2 normalization = cosine similarity)
- **LLM Inference**: Ollama-based Model Server with fine-tuned Qwen3 8B
- **Database**: PostgreSQL (async via SQLAlchemy + asyncpg)

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

3. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your settings (API keys, database URL, etc.)
```

4. **Initialize database:**
```bash
alembic upgrade head
```

## Prerequisites Before Running

### 1. Start PostgreSQL
Ensure PostgreSQL is running and the database exists:
```bash
# Default connection
postgresql://postgres:pavi1234@localhost:5432/legal_arise_new
```

### 2. Start Redis (Optional — for rate limiting)
```bash
docker run -d --name legal-redis -p 6379:6379 redis:7-alpine
```

### 3. Start Model Server
The Ollama-based model server must be running on port **5006**:
```bash
cd ../model-server
python server.py
```

Verify:
```bash
curl http://localhost:5006/health
```

### 4. Start Ollama
Ensure Ollama is installed and running (default port 11434):
```bash
ollama serve
```

## Running the Backend

```bash
# Development mode (port 5005)
python -m src.main

# Or with uvicorn
uvicorn src.main:app --reload --port 5005
```

Verify:
```bash
curl http://localhost:5005/api/health
```

## Environment Variables

Key settings in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `5005` | Backend server port |
| `DATABASE_URL` | `postgresql+asyncpg://...` | PostgreSQL connection |
| `MODEL_SERVER_URL` | `http://localhost:5006` | Model server URL |
| `MODEL_NAME` | `sri-legal-8b` | Active model name |
| `GEMINI_API_KEY` | — | Google Gemini API key (required) |
| `EMBEDDING_DIMENSION` | `3072` | Gemini embedding dimension |
| `FAISS_TOP_K` | `15` | Number of similar chunks to retrieve |
| `FAISS_MIN_SIMILARITY` | `0.3` | Minimum similarity threshold |

## Project Structure

```
Backend/
├── alembic/                    # Database migrations
│   └── versions/               # Migration scripts
├── models/
│   ├── faiss_index/            # Main FAISS index
│   │   ├── index.faiss
│   │   └── documents.pkl
│   └── faiss_partitions/       # Partitioned indices
├── scripts/                    # Utility & test scripts
│   ├── test_all_endpoints.py   # Full endpoint test suite
│   ├── test_model.py           # Model response test
│   ├── test_accuracy.py        # 5-case accuracy validation
│   ├── vector_status.py        # FAISS index status report
│   ├── delete_vectors.py       # Delete vector DB
│   ├── test_embedding.py       # Test embedding service
│   ├── upload_docs.py          # Upload PDFs
│   ├── init_db.py              # Initialize database
│   ├── check_db.py             # Check database state
│   └── fix_db.py               # Fix database issues
├── src/
│   ├── main.py                 # FastAPI app with lifespan
│   ├── api/                    # API routes
│   │   ├── deps.py             # Dependency injection
│   │   └── routes/             # Route handlers
│   │       ├── query.py        # Query endpoints
│   │       ├── admin.py        # Admin endpoints
│   │       └── health.py       # Health endpoints
│   ├── core/                   # Core configuration
│   │   ├── config.py           # Settings from .env
│   │   ├── exceptions.py       # Custom exceptions
│   │   └── logging.py         # Structured logging
│   ├── middleware/             # HTTP middleware
│   │   ├── logging_middleware.py
│   │   └── rate_limiter.py
│   ├── models/                 # SQLAlchemy ORM models
│   │   ├── document.py
│   │   ├── chunk.py
│   │   └── query.py
│   ├── repositories/           # Database access layer
│   ├── schemas/                # Pydantic schemas
│   └── services/               # Business logic
│       ├── recommendation_service.py   # Main pipeline orchestrator
│       ├── llm_client.py               # Model server HTTP client
│       ├── embedding_service.py        # Gemini + ST embedding
│       ├── faiss_service.py            # FAISS index management
│       └── document_service.py         # Document processing
├── uploads/                    # Uploaded PDF files
├── .env                        # Environment configuration
├── requirements.txt
├── pyproject.toml
└── alembic.ini
```

## API Endpoints

### Query API
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/query/recommend` | Submit legal query and get recommendation |
| `GET` | `/api/query/history` | Get query history |
| `GET` | `/api/query/{id}` | Get specific query result |
| `POST` | `/api/query/{id}/feedback` | Submit feedback on query |

### Admin API
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/admin/documents/upload` | Upload legal document (PDF/TXT) |
| `GET` | `/api/admin/documents` | List all documents |
| `GET` | `/api/admin/faiss/status` | FAISS index status |
| `GET` | `/api/admin/statistics` | System statistics |
| `GET` | `/api/admin/model/info` | Current model info |
| `POST` | `/api/admin/model/switch` | Switch between models |

### Health API
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Basic health check |
| `GET` | `/api/health/ready` | Readiness check (DB + FAISS + Model) |

## Utility Scripts

```bash
# Test all endpoints (15 tests)
python scripts/test_all_endpoints.py

# Test model server response
python scripts/test_model.py

# Test model accuracy (5 legal scenarios)
python scripts/test_accuracy.py

# Check FAISS vector status
python scripts/vector_status.py

# Test embedding service
python scripts/test_embedding.py

# Delete all vectors
python scripts/delete_vectors.py

# Upload legal documents
python scripts/upload_docs.py

# Initialize database
python scripts/init_db.py
```

## External Dependencies

| Service | Port | Purpose |
|---------|------|---------|
| **Model Server** | `5006` | Ollama-based LLM inference (Qwen3 8B) |
| **PostgreSQL** | `5432` | Document storage, query logging |
| **Redis** | `6379` | Rate limiting (optional) |
| **Ollama** | `11434` | Model runtime (used by Model Server) |
| **Gemini API** | — | Embedding generation (API key required) |

## License

Internal use only — Sri Lankan Labour Law Project

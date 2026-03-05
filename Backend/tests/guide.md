# Test Scripts Guide

A complete reference for all test scripts in the `tests/` directory. Scripts cover every layer of the system — from raw embedding math up to full end-to-end pipeline validation.

---

## Overview Table

| # | Script | Purpose | Key Flags |
|---|--------|---------|-----------|
| 0 | `test_init_faiss.py` | Create FAISS dirs & init index if missing | `--check`, `--force` |
| i | `test_vector_db_reset.py` | Vector DB reset & live statistics | `--reset`, `--stats-only` |
| ii | `test_upload_docs.py` | Upload docs (single / bulk / folder) | `--file`, `--files`, `--folder`, `--partition` |
| iii | `test_list_documents.py` | List docs with chunk relations | `--details`, `--doc-id`, `--verify` |
| iv | `test_retrieval_accuracy.py` | FAISS retrieval accuracy (10 cases) | `--case`, `--verbose` |
| v | `test_model_server.py` | Model-server endpoint tests (3 cases) | `--case`, `--endpoint`, `--port` |
| vi | `test_full_pipeline.py` | Full E2E pipeline tests (5 cases) | `--case`, `--verbose`, `--save` |
| — | `test_embedding.py` | Embedding service unit tests (6 checks) | *(no flags)* |
| — | `test_api_endpoints.py` | All HTTP endpoints smoke test | *(no flags)* |

---

## Prerequisites

### Services that must be running

| Script | Backend (`:5005`) | Model Server (`:5007`) |
|--------|:-----------------:|:----------------------:|
| `test_vector_db_reset.py` | ✅ | — |
| `test_upload_docs.py` | ✅ | — |
| `test_list_documents.py` | ✅ | — |
| `test_retrieval_accuracy.py` | ✅ | — |
| `test_model_server.py` | — | ✅ |
| `test_full_pipeline.py` | ✅ | ✅ |
| `test_embedding.py` | ✅ *(imports src)* | — |
| `test_api_endpoints.py` | ✅ | ✅ |

### Python dependencies

```bash
pip install httpx tabulate numpy faiss-cpu
```

### Working directory

All commands below assume you are inside `New-backend/Backend/`:

```bash
cd f:\Arise\New-backend\Backend
```

---

## Recommended Run Order

Run the scripts in this order when setting up or validating the system from scratch:

```
0. test_init_faiss.py         ← ensure FAISS dirs exist and index is initialised
1. test_embedding.py          ← verify Gemini embeddings work
2. test_vector_db_reset.py    ← check / clean slate the DB
3. test_upload_docs.py        ← load documents into the system
4. test_list_documents.py     ← confirm everything ingested correctly
5. test_retrieval_accuracy.py ← verify FAISS retrieval quality
6. test_model_server.py       ← validate model server is alive
7. test_full_pipeline.py      ← end-to-end validation
8. test_api_endpoints.py      ← final HTTP smoke test
```

---

## Script Reference

---

### 0 — `test_init_faiss.py`

**Purpose:** Ensure the FAISS index directories and files exist, and trigger index initialisation via the API if they are missing. Safe to run at any time — never deletes data.

**Architecture flow:**

```
GET  /api/health                  → Confirm backend is up
GET  /api/admin/faiss/status      → Read current index state
POST /api/admin/faiss/rebuild     → Rebuild / initialise index from stored chunks
GET  /api/admin/faiss/status      → Confirm final state
```

**Local paths created automatically if missing:**

```
models/faiss_index/       ← main index dir
models/faiss_partitions/  ← partitioned indices dir
```

**Usage:**

```bash
# Check status and auto-init if the index is missing (default)
python tests/test_init_faiss.py

# Read-only status check — no changes made
python tests/test_init_faiss.py --check

# Force a full rebuild even if the index already exists
python tests/test_init_faiss.py --force
```

**What it reports:**
- Local filesystem state (dirs and files present / missing)
- FAISS status before and after (vector count, dimension, partitions)
- Exit code `0` on success, `1` on failure

> Run this **first** when setting up a fresh environment. If no documents have been uploaded yet the index will be initialised empty (0 vectors) — that is expected and correct.

---

### i — `test_vector_db_reset.py`

**Purpose:** Inspect and optionally wipe the entire vector database (PostgreSQL documents + FAISS index).

**Architecture flow:**

```
GET  /api/admin/faiss/status      → FAISS index stats
GET  /api/admin/statistics        → System-wide stats
GET  /api/admin/documents         → List all documents
DELETE /api/admin/documents/{id}  → Delete each document
POST /api/admin/faiss/rebuild     → Rebuild FAISS index
GET  /api/health                  → Final health check
```

**Usage:**

```bash
# Safe read-only view of current stats (default)
python tests/test_vector_db_reset.py

# Show detailed statistics only
python tests/test_vector_db_reset.py --stats-only

# ⚠️ DELETE all documents and rebuild FAISS from scratch
python tests/test_vector_db_reset.py --reset
```

**What it reports:**
- Document count before and after
- FAISS vector count per partition
- System health status
- Operation duration

> **Warning:** `--reset` is irreversible. All documents and their embeddings are permanently deleted.

---

### ii — `test_upload_docs.py`

**Purpose:** Upload PDF documents to the backend — one file, multiple files, or an entire folder.

**Upload pipeline per file:**

```
PDF → PyMuPDF extract → paragraph chunking (1000 chars, 150 overlap)
    → Gemini embedding (3072d) → FAISS IndexFlatIP → PostgreSQL persist
```

**Architecture flow:**

```
POST /api/admin/documents/upload  → Upload (multipart/form-data)
GET  /api/admin/documents         → Verify listing
GET  /api/admin/faiss/status      → Check FAISS after upload
GET  /api/admin/statistics        → Check system stats
```

**Usage:**

```bash
# Upload a single file
python tests/test_upload_docs.py --file ./docs_latest/a5.pdf

# Upload multiple specific files
python tests/test_upload_docs.py --files ./docs_latest/a5.pdf ./docs_latest/a6.pdf

# Upload all PDFs in a folder
python tests/test_upload_docs.py --folder ./docs_latest

# Upload folder into a named partition
python tests/test_upload_docs.py --folder ./docs_latest --partition labour_law_v2
```

**What it reports:**
- Per-file upload status (pass / fail)
- Chunk count generated per document
- FAISS vector count after each upload
- Total elapsed time

---

### iii — `test_list_documents.py`

**Purpose:** Inspect all ingested documents with their chunk relationships, and optionally verify PostgreSQL ↔ FAISS data consistency.

**Architecture flow:**

```
GET  /api/admin/documents    → Documents + metadata
GET  /api/admin/faiss/status → Vector counts per partition
GET  /api/admin/statistics   → Overall system stats
GET  /api/health/ready       → Readiness check
```

**Usage:**

```bash
# List all documents (basic metadata)
python tests/test_list_documents.py

# Include chunk content previews
python tests/test_list_documents.py --details

# Inspect a specific document by UUID
python tests/test_list_documents.py --doc-id <uuid>

# Verify DB ↔ FAISS vector count consistency
python tests/test_list_documents.py --verify
```

**What it reports:**
- Document names, IDs, chunk counts, partition assignment
- Chunk preview text (with `--details`)
- Cross-check between PostgreSQL chunk count and FAISS vector count
- Partition distribution table

---

### iv — `test_retrieval_accuracy.py`

**Purpose:** Evaluate FAISS retrieval quality using 10 pre-defined legal scenarios. Tests the retrieval layer **only** — no LLM generation.

**Retrieval flow:**

```
Query → Gemini embed (3072d) → FAISS IndexFlatIP (40 wide)
      → document-diverse reranking → top 15 chunks → context (4500 chars)
```

**Test cases:**

| ID | Topic | Scope |
|----|-------|-------|
| 1 | Gratuity — Managing Director as Workman | In-scope |
| 2 | Gratuity — Compensatory Allowance | In-scope |
| 3 | Gratuity — Continuous Service Break | In-scope |
| 4 | Industrial Disputes — Unfair Termination | In-scope |
| 5 | Industrial Disputes — Trade Union Interference | In-scope |
| 6 | Termination Act — Retrenchment | In-scope |
| 7 | Termination Act — Constructive Dismissal | In-scope |
| 8 | Trade Unions — Victimisation | In-scope |
| 9 | Out-of-scope — Contract Law | Out-of-scope |
| 10 | Out-of-scope — Criminal Law | Out-of-scope |

**Usage:**

```bash
# Run all 10 test cases
python tests/test_retrieval_accuracy.py

# Run a specific test case
python tests/test_retrieval_accuracy.py --case 4

# Show full chunk content in output
python tests/test_retrieval_accuracy.py --verbose
```

**Metrics reported per case:**
- Recall: how many expected acts/cases were found
- Keyword coverage: percentage of expected keywords present
- Top similarity score from FAISS
- Retrieval latency (ms)
- Pass / Fail verdict

---

### v — `test_model_server.py`

**Purpose:** Test the model-server-deployed proxy and its endpoints with ~700-word legal contexts. Validates structured JSON output against the `LegalOutput` schema.

**Architecture:**

```
localhost:5007 (proxy) → Modal.com A10G GPU → Ollama → Qwen3-8B GGUF
```

**Endpoints tested:**

```
GET  /health         → Connectivity + model status
GET  /model/info     → Model name, quantisation, context length
POST /generate       → Prompt → text response
POST /chat           → Chat messages → response
GET  /proxy/status   → Proxy → Modal connectivity
```

**Test cases:**

| ID | Topic |
|----|-------|
| 1 | Gratuity — Managing Director entitlement |
| 2 | Unfair termination — no notice |
| 3 | Trade union victimisation |

**Usage:**

```bash
# Run all 3 test cases
python tests/test_model_server.py

# Run a specific test case
python tests/test_model_server.py --case 2

# Test only the /generate endpoint
python tests/test_model_server.py --endpoint generate

# Test only the /chat endpoint
python tests/test_model_server.py --endpoint chat

# Point at local model server on a different port
python tests/test_model_server.py --port 5006
```

**What it validates:**
- HTTP 200 from all endpoints
- `LegalOutput` JSON schema fields (`out_of_scope`, `primary_violations`, `supporting_cases`, `legal_reasoning`, etc.)
- Inference latency (target < 120 s)
- Token length of generated response

---

### vi — `test_full_pipeline.py`

**Purpose:** End-to-end validation of the complete RAG pipeline through a single API call. Measures accuracy, schema compliance, retrieval quality, and performance together.

**Full pipeline:**

```
User Query
  → POST /api/query/recommend
  → Gemini Embedding (3072d)
  → FAISS IndexFlatIP Search (40 wide candidates)
  → Document-Diverse Reranking → Top 15 chunks
  → Context Assembly (max 4500 chars)
  → System Prompt + User Prompt → Model Server (Qwen3-8B)
  → JSON Parse (5-level fallback)
  → LegalOutput Schema Validation
  → Persist to PostgreSQL → Return Response
```

**Test cases:**

| ID | Title |
|----|-------|
| 1 | Gratuity — Managing Director as Workman |
| 2 | Industrial Disputes — Unfair Termination |
| 3 | Trade Union Ordinance — Victimisation |
| 4 | Termination of Employment — Retrenchment |
| 5 | Out-of-scope — Criminal Assault Query |

**Scoring dimensions (weighted):**

| Dimension | Weight | What is measured |
|-----------|--------|-----------------|
| Retrieval quality | 20 % | Acts, cases, keywords found in context |
| Correct acts cited | 20 % | Expected acts present in response |
| Correct cases cited | 20 % | Expected case names present |
| Schema compliance | 15 % | All required JSON fields present |
| Legal reasoning | 15 % | `legal_reasoning` non-empty, recommended actions present |
| Performance | 10 % | Response time < 120 s |

**Grade scale:**

| Grade | Score |
|-------|-------|
| A | ≥ 90 % |
| B | ≥ 80 % |
| C | ≥ 70 % |
| D | ≥ 60 % |
| F | < 60 % |

**Usage:**

```bash
# Run all 5 test cases
python tests/test_full_pipeline.py

# Run a specific test case
python tests/test_full_pipeline.py --case 3

# Show full JSON responses in output
python tests/test_full_pipeline.py --verbose

# Save results to a timestamped JSON file
python tests/test_full_pipeline.py --save

# Combine flags
python tests/test_full_pipeline.py --verbose --save
```

**Output includes:**
- Per-case score breakdown across all dimensions
- Letter grade
- Total runtime
- JSON export path (when `--save` is used)

---

### — `test_embedding.py`

**Purpose:** Unit-test the `EmbeddingService` directly (bypasses HTTP — imports `src` in-process). Runs 6 checks.

**Tests:**

| # | Check | Pass condition |
|---|-------|---------------|
| 1 | Single text embedding | Returns vector with correct dimension |
| 2 | Batch embedding | All 3 texts produce correct-dimension vectors |
| 3 | Related-text similarity | Cosine similarity > 0.5 |
| 4 | Unrelated-text similarity | Cosine similarity < 0.7 |
| 5 | Dimension consistency | Short / long / one-word texts all same dim |
| 6 | FAISS index integration | Index dimension matches embedding dimension; search returns 5 scores |

**Usage:**

```bash
# Must be run from the Backend root so src/ is importable
cd f:\Arise\New-backend\Backend
python tests/test_embedding.py
```

> This script runs `asyncio.run()` internally — no `--flags` needed.

---

### — `test_api_endpoints.py`

**Purpose:** HTTP smoke test for every endpoint across both the backend (`:5005`) and model-server proxy (`:5007`). Useful as a quick health check after deployment.

**Endpoints covered:**

*Model server (`:5007`):*
- `GET /` — root
- `GET /proxy/status`
- `GET /health`
- `GET /model/info`
- `POST /generate`
- `POST /chat`
- `POST /reload`

*Backend (`:5005`):*
- `GET /` — root
- `GET /api/health`
- `GET /api/health/ready`
- `GET /api/admin/documents`
- `GET /api/admin/faiss/status`
- `GET /api/admin/statistics`
- `GET /api/admin/model/info`
- `POST /api/admin/documents/upload` *(expects 400 for non-PDF)*
- `DELETE /api/admin/documents/{id}` *(expects 404 for unknown ID)*
- `GET /api/query/history`
- `POST /api/query/recommend`
- `GET /api/query/{id}`
- `POST /api/query/{id}/feedback`

**Usage:**

```bash
python tests/test_api_endpoints.py
```

**Output format:**

```
[PASS] GET    /api/health                               -> 200  | {"status":"ok",...}
[FAIL] POST   /api/query/recommend                     -> 503  | model server offline
...
RESULTS: 17 passed, 1 failed, 18 total
```

---

## Common Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `Connection refused` on `:5005` | Backend not running | Start with `uvicorn src.main:app --port 5005` |
| `Connection refused` on `:5007` | Proxy not running | Start with `python model-server-deployed/proxy_server.py` |
| `ModuleNotFoundError: src` | Wrong working directory for `test_embedding.py` | `cd` to `New-backend/Backend` first |
| Retrieval test: 0 results returned | No documents uploaded | Run `test_upload_docs.py` first |
| Full pipeline: schema validation failure | Model returned non-JSON | Check model server logs; try `--verbose` |
| `httpx.TimeoutException` | LLM inference over 5 min | Increase `TIMEOUT` constant at top of script or retry |

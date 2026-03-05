"""
Main FastAPI Application - Sri Lankan Legal AI Backend
Uses ONLY the finetuned Ollama model for LLM inference.
Gemini is used ONLY for embeddings (with sentence-transformers backup).
"""
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.core.config import settings
from src.core.logging import get_logger
from src.core.database import engine, check_db_connection
from src.core.exceptions import LegalAppException, OutOfScopeError, LLMServiceError
from src.services.embedding_service import EmbeddingService
from src.services.faiss_service import FAISSService
from src.services.llm_client import LLMClient
from src.services.document_service import DocumentService
from src.services.recommendation_service import RecommendationService
from src.services.admin_service import AdminService
from src.api import deps

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    logger.info("=" * 60)
    logger.info("  Sri Lankan Legal AI Backend - Starting Up")
    logger.info("=" * 60)

    # 1. Check database
    db_ok = await check_db_connection()
    if not db_ok:
        logger.error("Database connection failed!")
        sys.exit(1)
    logger.info("[OK] Database connected")

    # 2. Initialize Embedding Service
    embedding_svc = EmbeddingService()
    await embedding_svc.initialize()
    if not embedding_svc.backend:
        logger.error("No embedding backend available!")
        sys.exit(1)
    logger.info(f"[OK] Embedding: {embedding_svc.backend} (dim={embedding_svc.dimension})")

    # 3. Initialize FAISS
    faiss_svc = FAISSService(dimension=embedding_svc.dimension)
    faiss_svc.initialize()
    logger.info(f"[OK] FAISS: {faiss_svc.total_vectors} vectors")

    # 4. Initialize LLM Client
    llm = LLMClient()
    await llm.initialize()
    llm_healthy = await llm.health_check()
    if llm_healthy:
        logger.info("[OK] Model server connected")
    else:
        logger.warning("[WARN] Model server not reachable - queries will fail")

    # 5. Initialize remaining services
    doc_svc = DocumentService()
    rec_svc = RecommendationService(embedding_svc, faiss_svc, llm)
    admin_svc = AdminService(doc_svc, embedding_svc, faiss_svc)

    # 6. Store in deps module for DI
    deps.embedding_service = embedding_svc
    deps.faiss_service = faiss_svc
    deps.llm_client = llm
    deps.document_service = doc_svc
    deps.recommendation_service = rec_svc
    deps.admin_service = admin_svc

    logger.info("=" * 60)
    logger.info(f"  Backend ready on port {settings.port}")
    logger.info("=" * 60)

    yield

    # Shutdown
    logger.info("Shutting down...")
    await llm.close()
    await engine.dispose()
    logger.info("Shutdown complete")


# ── Create App ───────────────────────────────────────────────────────

app = FastAPI(
    title="Sri Lankan Legal AI - Backend",
    description="Labour & Employment Law recommendation system using finetuned Qwen3 model",
    version="2.0.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Exception Handlers ──────────────────────────────────────────────

@app.exception_handler(LegalAppException)
async def legal_app_exception_handler(request: Request, exc: LegalAppException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message, "detail": exc.detail},
    )


@app.exception_handler(OutOfScopeError)
async def out_of_scope_handler(request: Request, exc: OutOfScopeError):
    return JSONResponse(
        status_code=200,
        content={
            "out_of_scope": True,
            "scope_category": exc.detail or "unknown",
            "message": str(exc),
        },
    )


@app.exception_handler(LLMServiceError)
async def llm_error_handler(request: Request, exc: LLMServiceError):
    return JSONResponse(
        status_code=503,
        content={"error": "Model service unavailable", "detail": str(exc)},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# ── Routes ───────────────────────────────────────────────────────────

from src.api.routes.query import router as query_router
from src.api.routes.admin import router as admin_router
from src.api.routes.health import router as health_router

app.include_router(query_router, prefix="/api")
app.include_router(admin_router, prefix="/api")
app.include_router(health_router, prefix="/api")


@app.get("/")
async def root():
    return {
        "name": "Sri Lankan Legal AI Backend",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host=settings.host, port=settings.port, reload=True)

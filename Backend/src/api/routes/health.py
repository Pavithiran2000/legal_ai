"""
Health Routes - System health and readiness checks.
"""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import (
    get_embedding_service,
    get_faiss_service,
    get_llm_client,
    get_session,
)
from src.services.embedding_service import EmbeddingService
from src.services.faiss_service import FAISSService
from src.services.llm_client import LLMClient
from src.core.database import check_db_connection
from src.schemas.admin import SystemHealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    faiss_service: FAISSService = Depends(get_faiss_service),
    llm_client: LLMClient = Depends(get_llm_client),
):
    """Comprehensive health check."""
    db_ok = await check_db_connection()
    llm_ok = await llm_client.health_check()
    embedding_ok = embedding_service.backend is not None
    faiss_ok = faiss_service.total_vectors >= 0  # Always true if initialized

    all_ok = db_ok and llm_ok and embedding_ok and faiss_ok

    return {
        "status": "healthy" if all_ok else "degraded",
        "components": {
            "database": "ok" if db_ok else "error",
            "model_server": "ok" if llm_ok else "error",
            "embedding": embedding_service.backend or "not_initialized",
            "faiss": {
                "status": "ok" if faiss_ok else "error",
                "vectors": faiss_service.total_vectors,
            },
        },
    }


@router.get("/health/ready")
async def readiness_check(
    llm_client: LLMClient = Depends(get_llm_client),
):
    """Quick readiness check for load balancers."""
    llm_ok = await llm_client.health_check()
    db_ok = await check_db_connection()
    return {
        "ready": llm_ok and db_ok,
        "database": db_ok,
        "model_server": llm_ok,
    }

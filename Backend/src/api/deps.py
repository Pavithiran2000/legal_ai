"""
API Dependencies - Service singletons and DB session injection.
"""
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import get_db
from src.services.embedding_service import EmbeddingService
from src.services.faiss_service import FAISSService
from src.services.llm_client import LLMClient
from src.services.document_service import DocumentService
from src.services.recommendation_service import RecommendationService
from src.services.admin_service import AdminService

# ── Service Singletons (initialized at app startup) ─────────────────

embedding_service: EmbeddingService = None  # type: ignore
faiss_service: FAISSService = None  # type: ignore
llm_client: LLMClient = None  # type: ignore
document_service: DocumentService = None  # type: ignore
recommendation_service: RecommendationService = None  # type: ignore
admin_service: AdminService = None  # type: ignore


def get_embedding_service() -> EmbeddingService:
    return embedding_service


def get_faiss_service() -> FAISSService:
    return faiss_service


def get_llm_client() -> LLMClient:
    return llm_client


def get_document_service() -> DocumentService:
    return document_service


def get_recommendation_service() -> RecommendationService:
    return recommendation_service


def get_admin_service() -> AdminService:
    return admin_service


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async for session in get_db():
        yield session

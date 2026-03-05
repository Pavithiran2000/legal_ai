"""
Query Routes - Legal recommendation endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_recommendation_service, get_session
from src.services.recommendation_service import RecommendationService
from src.schemas.query import (
    QueryRequest,
    QueryResponse,
    FeedbackRequest,
    QueryHistoryItem,
    QueryDetailResponse,
)
from src.schemas.common import SuccessResponse, ErrorResponse
from src.repositories.query_repo import QueryRepository
from src.core.logging import get_logger
from src.core.exceptions import LLMServiceError

logger = get_logger(__name__)
router = APIRouter(prefix="/query", tags=["Query"])


@router.post(
    "/recommend",
    response_model=QueryResponse,
    responses={500: {"model": ErrorResponse}},
)
async def get_recommendation(
    request: QueryRequest,
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
    db: AsyncSession = Depends(get_session),
):
    """Get a legal recommendation for a labour law scenario."""
    try:
        result = await recommendation_service.get_recommendation(
            query_text=request.query,
            db=db,
            top_k=request.top_k,
            temperature=request.temperature,
        )
        return result
    except LLMServiceError as e:
        logger.error(f"LLM error: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@router.get(
    "/history",
    response_model=list[QueryHistoryItem],
)
async def get_query_history(
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_session),
):
    """Get recent query history."""
    repo = QueryRepository(db)
    queries = await repo.list_recent(limit=limit, offset=offset)
    return [
        QueryHistoryItem(
            id=str(q.id),
            query_text=q.query_text,
            out_of_scope=q.out_of_scope if q.out_of_scope is not None else False,
            scope_category=q.scope_category,
            confidence=q.confidence,
            model_used=q.model_used,
            created_at=q.created_at,
        )
        for q in queries
    ]


@router.get(
    "/{query_id}",
    response_model=QueryDetailResponse,
)
async def get_query_detail(
    query_id: str,
    db: AsyncSession = Depends(get_session),
):
    """Get detailed query result by ID."""
    repo = QueryRepository(db)
    query = await repo.get_by_id(query_id)
    if not query:
        raise HTTPException(status_code=404, detail="Query not found")

    return QueryDetailResponse(
        id=str(query.id),
        query_text=query.query_text,
        response_json=query.response_json,
        out_of_scope=query.out_of_scope if query.out_of_scope is not None else False,
        scope_category=query.scope_category,
        confidence=query.confidence,
        model_used=query.model_used,
        generation_time_ms=query.generation_time_ms,
        feedback_rating=query.feedback_rating,
        feedback_comment=query.feedback_comment,
        created_at=query.created_at,
    )


@router.post(
    "/{query_id}/feedback",
    response_model=SuccessResponse,
)
async def submit_feedback(
    query_id: str,
    request: FeedbackRequest,
    db: AsyncSession = Depends(get_session),
):
    """Submit feedback for a query result."""
    repo = QueryRepository(db)
    success = await repo.update_feedback(
        query_id=query_id,
        rating=request.rating,
        comment=request.comment,
    )
    if not success:
        raise HTTPException(status_code=404, detail="Query not found")

    return SuccessResponse(message="Feedback submitted")

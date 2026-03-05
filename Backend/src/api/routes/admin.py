"""
Admin Routes - Document management and system operations.
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_admin_service, get_faiss_service, get_llm_client, get_session
from src.services.admin_service import AdminService
from src.services.faiss_service import FAISSService
from src.services.llm_client import LLMClient
from src.schemas.admin import (
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentInfo,
    FAISSStatusResponse,
    SystemHealthResponse,
    StatisticsResponse,
)
from src.schemas.common import SuccessResponse, ErrorResponse
from src.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/admin", tags=["Admin"])


@router.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
)
async def upload_document(
    file: UploadFile = File(...),
    admin_service: AdminService = Depends(get_admin_service),
    db: AsyncSession = Depends(get_session),
):
    """Upload and index a PDF document."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    content = await file.read()
    if len(content) > 50 * 1024 * 1024:  # 50MB limit
        raise HTTPException(status_code=400, detail="File too large (max 50MB)")

    result = await admin_service.upload_and_index_document(
        filename=file.filename,
        content=content,
        db=db,
    )

    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])

    return DocumentUploadResponse(**result)


@router.get(
    "/documents",
    response_model=DocumentListResponse,
)
async def list_documents(
    admin_service: AdminService = Depends(get_admin_service),
    db: AsyncSession = Depends(get_session),
):
    """List all indexed documents."""
    docs = await admin_service.list_documents(db)
    return DocumentListResponse(
        documents=[DocumentInfo(**d) for d in docs],
        total=len(docs),
    )


@router.delete(
    "/documents/{document_id}",
    response_model=SuccessResponse,
)
async def delete_document(
    document_id: str,
    admin_service: AdminService = Depends(get_admin_service),
    db: AsyncSession = Depends(get_session),
):
    """Delete a document and its indexed chunks."""
    result = await admin_service.delete_document(document_id, db)
    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result["message"])
    return SuccessResponse(message=result["message"])


@router.post(
    "/faiss/rebuild",
    response_model=SuccessResponse,
)
async def rebuild_faiss_index(
    admin_service: AdminService = Depends(get_admin_service),
    db: AsyncSession = Depends(get_session),
):
    """Rebuild the FAISS index from stored chunks."""
    result = await admin_service.rebuild_faiss_index(db)
    return SuccessResponse(
        message=f"Index rebuilt with {result['total_vectors']} vectors"
    )


@router.get(
    "/faiss/status",
    response_model=FAISSStatusResponse,
)
async def faiss_status(
    faiss_service: FAISSService = Depends(get_faiss_service),
):
    """Get FAISS index status."""
    status = faiss_service.get_status()
    return FAISSStatusResponse(**status)


@router.get(
    "/statistics",
    response_model=StatisticsResponse,
)
async def get_statistics(
    admin_service: AdminService = Depends(get_admin_service),
    db: AsyncSession = Depends(get_session),
):
    """Get system statistics."""
    stats = await admin_service.get_statistics(db)
    return StatisticsResponse(**stats)


@router.post(
    "/model/switch",
    response_model=SuccessResponse,
)
async def switch_model(
    model_name: str,
    llm_client: LLMClient = Depends(get_llm_client),
):
    """Switch the active LLM model (4B/8B)."""
    result = await llm_client.switch_model(model_name)
    return SuccessResponse(message=f"Switched to {result.get('model', model_name)}")


@router.get(
    "/model/info",
)
async def model_info(
    llm_client: LLMClient = Depends(get_llm_client),
):
    """Get current model info from model server."""
    return await llm_client.get_model_info()

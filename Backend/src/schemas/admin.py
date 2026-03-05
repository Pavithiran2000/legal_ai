"""Admin-related schemas."""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class DocumentUploadResponse(BaseModel):
    status: str = "success"
    document_id: str = ""
    filename: str = ""
    chunks_created: int = 0
    partition: str = ""


class DocumentInfo(BaseModel):
    id: str
    filename: str
    doc_type: str = "pdf"
    status: str = "pending"
    chunk_count: int = 0
    partition: Optional[str] = None
    indexed: bool = False
    created_at: Optional[datetime] = None


class DocumentListResponse(BaseModel):
    success: bool = True
    documents: List[DocumentInfo] = []
    total: int = 0


class FAISSStatusResponse(BaseModel):
    total_vectors: int = 0
    dimension: int = 0
    index_type: str = "IndexFlatIP"
    partitions: List[str] = []
    partition_sizes: dict = {}


class SystemHealthResponse(BaseModel):
    overall_status: str = "unknown"
    components: dict = {}


class StatisticsResponse(BaseModel):
    documents: int = 0
    chunks: int = 0
    queries: int = 0
    out_of_scope_queries: int = 0
    avg_confidence: float = 0.0
    faiss_vectors: int = 0
    embedding_backend: Optional[str] = None
    embedding_dimension: int = 0
    out_of_scope_count: int = 0

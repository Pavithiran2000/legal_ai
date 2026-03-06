"""Application configuration loaded from .env."""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List
import json


class Settings(BaseSettings):
    # Application
    app_name: str = Field(default="Sri Lankan Labour Law Platform")
    app_env: str = Field(default="development")
    debug: bool = Field(default=True)
    api_version: str = Field(default="v1")

    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=5005)

    # Database
    database_url: str = Field(default="postgresql+asyncpg://postgres:pavi1234@localhost:5432/legal_arise_new")

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0")
    rate_limit_requests: int = Field(default=30)
    rate_limit_window: int = Field(default=60)

    # CORS
    cors_origins: str = Field(default='["http://localhost:5173","http://localhost:3000"]')

    # Model Server (proxy to Modal-deployed model on port 5007)
    model_server_url: str = Field(default="http://localhost:5007")
    model_name: str = Field(default="sri-legal-8b")
    model_temperature: float = Field(default=0.1)
    model_max_tokens: int = Field(default=6000)

    # Embedding - Gemini (disabled)
    gemini_api_key: str = Field(default="")
    embedding_model: str = Field(default="sentence-transformers")
    embedding_dimension: int = Field(default=1024)
    embedding_batch_size: int = Field(default=100)

    # Embedding - Sentence Transformers (primary)
    use_sentence_transformers_backup: bool = Field(default=False)
    st_embedding_model: str = Field(default="BAAI/bge-large-en-v1.5")
    st_embedding_dimension: int = Field(default=1024)

    # FAISS
    faiss_index_dir: str = Field(default="./models/faiss_index")
    faiss_index_path: str = Field(default="./models/faiss_index/index.faiss")
    faiss_documents_path: str = Field(default="./models/faiss_index/documents.pkl")
    faiss_partitions_path: str = Field(default="./models/faiss_partitions")
    top_k: int = Field(default=15)
    min_similarity: float = Field(default=0.3)

    # RAG
    rag_max_context_length: int = Field(default=4500)
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=150)

    # Upload
    upload_dir: str = Field(default="./uploads")

    # Query
    query_min_length: int = Field(default=10)
    query_max_length: int = Field(default=2000)

    # Document
    max_upload_size_mb: int = Field(default=50)
    allowed_file_types: str = Field(default='["application/pdf","text/plain"]')

    # Logging
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
        protected_namespaces = ()

    @property
    def cors_origins_list(self) -> List[str]:
        try:
            return json.loads(self.cors_origins)
        except Exception:
            return ["http://localhost:5173", "http://localhost:3000"]

    @property
    def allowed_file_types_list(self) -> List[str]:
        try:
            return json.loads(self.allowed_file_types)
        except Exception:
            return ["application/pdf", "text/plain"]

    @property
    def active_embedding_dimension(self) -> int:
        """Return dimension based on which embedding is active."""
        return self.embedding_dimension


settings = Settings()

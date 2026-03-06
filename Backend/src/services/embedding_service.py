"""
Embedding Service - Gemini primary with sentence-transformers backup.
Used ONLY for document/query embedding (NOT for LLM generation).
"""
import asyncio
import numpy as np
from typing import List, Optional
from src.core.config import settings
from src.core.logging import get_logger
from src.core.exceptions import EmbeddingError

logger = get_logger(__name__)


class EmbeddingService:
    """Dual embedding service: Gemini + sentence-transformers fallback."""

    def __init__(self):
        self._genai = None
        self._st_model = None
        self._active_backend = None
        self._dimension = settings.embedding_dimension

    async def initialize(self):
        """Initialize embedding backends."""
        # Use sentence-transformers as primary when embedding_model is "sentence-transformers"
        if settings.embedding_model == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer
                self._st_model = SentenceTransformer(settings.st_embedding_model)
                self._dimension = settings.st_embedding_dimension
                self._active_backend = "sentence-transformers"
                logger.info(f"Sentence-transformers embedding initialized: {settings.st_embedding_model} (dim={self._dimension})")
                return
            except Exception as e:
                logger.warning(f"Sentence-transformers init failed: {e}")

        # Try Gemini if a Gemini model is configured
        if settings.gemini_api_key and settings.embedding_model != "sentence-transformers":
            try:
                import google.generativeai as genai
                genai.configure(api_key=settings.gemini_api_key)
                self._genai = genai
                # Test embedding
                result = await asyncio.to_thread(
                    genai.embed_content,
                    model=settings.embedding_model,
                    content="test",
                    task_type="retrieval_document"
                )
                self._dimension = len(result["embedding"])
                self._active_backend = "gemini"
                logger.info(f"Gemini embedding initialized (dim={self._dimension})")
                return
            except Exception as e:
                logger.warning(f"Gemini embedding init failed: {e}")

            # Fallback to sentence-transformers if Gemini fails
            try:
                from sentence_transformers import SentenceTransformer
                self._st_model = SentenceTransformer(settings.st_embedding_model)
                self._dimension = settings.st_embedding_dimension
                self._active_backend = "sentence-transformers"
                logger.info(f"Falling back to sentence-transformers: {settings.st_embedding_model} (dim={self._dimension})")
                return
            except Exception as e:
                logger.warning(f"Sentence-transformers fallback init failed: {e}")

        logger.error("No embedding backend available!")

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def backend(self) -> Optional[str]:
        return self._active_backend

    async def embed_text(self, text: str, task_type: str = "retrieval_document") -> np.ndarray:
        """Embed a single text."""
        if self._active_backend == "gemini":
            return await self._embed_gemini(text, task_type)
        elif self._active_backend == "sentence-transformers":
            return await self._embed_st(text)
        else:
            raise EmbeddingError("No embedding backend available")

    async def embed_texts(self, texts: List[str], task_type: str = "retrieval_document") -> List[np.ndarray]:
        """Embed multiple texts."""
        if self._active_backend == "gemini":
            return await self._embed_gemini_batch(texts, task_type)
        elif self._active_backend == "sentence-transformers":
            return await self._embed_st_batch(texts)
        else:
            raise EmbeddingError("No embedding backend available")

    async def embed_query(self, query: str) -> np.ndarray:
        """Embed a query (uses retrieval_query task type)."""
        return await self.embed_text(query, task_type="retrieval_query")

    # ── Gemini ───────────────────────────────────────────────────

    async def _embed_gemini(self, text: str, task_type: str) -> np.ndarray:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = await asyncio.to_thread(
                    self._genai.embed_content,
                    model=settings.embedding_model,
                    content=text,
                    task_type=task_type
                )
                return np.array(result["embedding"], dtype=np.float32)
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt  # 1s, 2s
                    logger.warning(f"Gemini embed attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    logger.warning(f"Gemini embed failed after {max_retries} attempts, trying backup: {e}")
                    if self._st_model:
                        return await self._embed_st(text)
                    raise EmbeddingError(f"Embedding failed: {e}")

    async def _embed_gemini_batch(self, texts: List[str], task_type: str) -> List[np.ndarray]:
        results = []
        batch_size = settings.embedding_batch_size
        max_retries = 3
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            for attempt in range(max_retries):
                try:
                    result = await asyncio.to_thread(
                        self._genai.embed_content,
                        model=settings.embedding_model,
                        content=batch,
                        task_type=task_type
                    )
                    embeddings = result.get("embedding", [])
                    if isinstance(embeddings[0], list):
                        results.extend([np.array(e, dtype=np.float32) for e in embeddings])
                    else:
                        results.append(np.array(embeddings, dtype=np.float32))
                    break  # Success, move to next batch
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait = 2 ** attempt
                        logger.warning(f"Gemini batch embed attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
                        await asyncio.sleep(wait)
                    else:
                        logger.warning(f"Gemini batch embed failed after {max_retries} attempts: {e}")
                        if self._st_model:
                            return await self._embed_st_batch(texts)
                        raise EmbeddingError(f"Batch embedding failed: {e}")
        return results

    # ── Sentence Transformers ────────────────────────────────────

    async def _embed_st(self, text: str) -> np.ndarray:
        embedding = await asyncio.to_thread(self._st_model.encode, text)
        return np.array(embedding, dtype=np.float32)

    async def _embed_st_batch(self, texts: List[str]) -> List[np.ndarray]:
        embeddings = await asyncio.to_thread(self._st_model.encode, texts)
        return [np.array(e, dtype=np.float32) for e in embeddings]

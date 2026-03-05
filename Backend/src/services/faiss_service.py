"""
FAISS Service - Vector index management for document retrieval.
"""
import os
import asyncio
import numpy as np
import faiss
from typing import List, Optional, Tuple, Dict
from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


class FAISSService:
    """Manages FAISS indices for vector similarity search."""

    def __init__(self, dimension: int):
        self._dimension = dimension
        self._index: Optional[faiss.IndexFlatIP] = None
        self._chunk_ids: List[str] = []       # chunk UUIDs in order
        self._partitions: Dict[str, faiss.IndexFlatIP] = {}
        self._partition_chunk_ids: Dict[str, List[str]] = {}
        self._index_dir = settings.faiss_index_dir
        os.makedirs(self._index_dir, exist_ok=True)

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal if self._index else 0

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def partitions(self) -> List[str]:
        return list(self._partitions.keys())

    def initialize(self):
        """Create or load main index."""
        index_path = os.path.join(self._index_dir, "index.faiss")
        ids_path = os.path.join(self._index_dir, "chunk_ids.npy")

        if os.path.exists(index_path) and os.path.exists(ids_path):
            try:
                self._index = faiss.read_index(index_path)
                self._chunk_ids = list(np.load(ids_path, allow_pickle=True))
                logger.info(f"Loaded FAISS index: {self._index.ntotal} vectors (dim={self._index.d})")
                # Adapt dimension if needed
                if self._index.d != self._dimension:
                    logger.warning(f"Index dim {self._index.d} != expected {self._dimension}. Rebuilding.")
                    self._create_empty_index()
                return
            except Exception as e:
                logger.warning(f"Failed to load index: {e}")

        self._create_empty_index()

    def _create_empty_index(self):
        """Create empty FAISS index (IndexFlatIP for cosine similarity)."""
        self._index = faiss.IndexFlatIP(self._dimension)
        self._chunk_ids = []
        logger.info(f"Created empty FAISS index (dim={self._dimension})")

    async def add_vectors(self, embeddings: List[np.ndarray], chunk_ids: List[str]):
        """Add vectors to the main index."""
        if not embeddings:
            return
        vectors = np.stack(embeddings).astype(np.float32)
        faiss.normalize_L2(vectors)
        await asyncio.to_thread(self._index.add, vectors)
        self._chunk_ids.extend(chunk_ids)
        logger.info(f"Added {len(embeddings)} vectors. Total: {self._index.ntotal}")

    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = None,
        min_similarity: float = None,
        partition: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """Search for similar vectors. Returns list of (chunk_id, score)."""
        top_k = top_k or settings.top_k
        min_similarity = min_similarity or settings.min_similarity

        if partition and partition in self._partitions:
            index = self._partitions[partition]
            chunk_ids = self._partition_chunk_ids[partition]
        else:
            index = self._index
            chunk_ids = self._chunk_ids

        if index is None or index.ntotal == 0:
            return []

        query = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)

        actual_k = min(top_k, index.ntotal)
        scores, indices = await asyncio.to_thread(index.search, query, actual_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(chunk_ids):
                continue
            if score >= min_similarity:
                results.append((chunk_ids[idx], float(score)))

        return results

    async def remove_vectors_by_ids(self, chunk_ids_to_remove: set):
        """Rebuild index without specified chunk IDs."""
        if not self._index or self._index.ntotal == 0:
            return

        keep_indices = [
            i for i, cid in enumerate(self._chunk_ids)
            if cid not in chunk_ids_to_remove
        ]

        if len(keep_indices) == len(self._chunk_ids):
            return  # Nothing to remove

        if not keep_indices:
            self._create_empty_index()
            return

        all_vectors = faiss.rev_swig_ptr(self._index.get_xb(), self._index.ntotal * self._dimension)
        all_vectors = all_vectors.reshape(self._index.ntotal, self._dimension)
        keep_vectors = all_vectors[keep_indices].copy()
        keep_ids = [self._chunk_ids[i] for i in keep_indices]

        self._create_empty_index()
        await asyncio.to_thread(self._index.add, keep_vectors)
        self._chunk_ids = keep_ids
        logger.info(f"Removed vectors. New total: {self._index.ntotal}")

    async def save(self):
        """Persist index to disk."""
        if self._index:
            index_path = os.path.join(self._index_dir, "index.faiss")
            ids_path = os.path.join(self._index_dir, "chunk_ids.npy")
            await asyncio.to_thread(faiss.write_index, self._index, index_path)
            np.save(ids_path, np.array(self._chunk_ids, dtype=object))
            logger.info(f"Saved FAISS index ({self._index.ntotal} vectors)")

    async def rebuild_from_data(self, embeddings: List[np.ndarray], chunk_ids: List[str]):
        """Fully rebuild the index from scratch."""
        self._create_empty_index()
        if embeddings:
            await self.add_vectors(embeddings, chunk_ids)
        await self.save()
        logger.info(f"Rebuilt index: {self.total_vectors} vectors")

    def get_status(self) -> dict:
        return {
            "total_vectors": self.total_vectors,
            "dimension": self._dimension,
            "index_type": "IndexFlatIP",
            "partitions": self.partitions,
            "partition_sizes": {
                p: idx.ntotal for p, idx in self._partitions.items()
            }
        }

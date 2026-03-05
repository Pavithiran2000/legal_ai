"""
Admin Service - Document management, indexing, and system operations.
"""
import os
import uuid
import numpy as np
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.logging import get_logger
from src.services.document_service import DocumentService
from src.services.embedding_service import EmbeddingService
from src.services.faiss_service import FAISSService
from src.repositories.document_repo import DocumentRepository
from src.repositories.chunk_repo import ChunkRepository
from src.repositories.query_repo import QueryRepository

logger = get_logger(__name__)


class AdminService:
    """Handles document upload, indexing, and admin operations."""

    def __init__(
        self,
        document_service: DocumentService,
        embedding_service: EmbeddingService,
        faiss_service: FAISSService,
    ):
        self._doc_service = document_service
        self._embedding = embedding_service
        self._faiss = faiss_service

    async def upload_and_index_document(
        self,
        filename: str,
        content: bytes,
        db: AsyncSession,
    ) -> dict:
        """Upload, parse, chunk, embed, and index a document."""
        doc_repo = DocumentRepository(db)
        chunk_repo = ChunkRepository(db)

        # 1. Save file
        filepath = await self._doc_service.save_upload(filename, content)

        # 2. Extract text
        text = self._doc_service.extract_text(filepath)
        if not text.strip():
            return {"status": "error", "message": "No text extracted from document"}

        # 3. Detect partition
        partition = self._doc_service.detect_partition(filename, text)

        # 4. Create document record
        doc = await doc_repo.create(
            filename=filename,
            doc_type="pdf",
            content=text,
            status="processing",
            partition=partition,
        )

        # 5. Chunk text
        chunks_text = self._doc_service.chunk_text(text)
        if not chunks_text:
            await doc_repo.update_status(doc.id, "error")
            return {"status": "error", "message": "No chunks created from document"}

        # 6. Embed chunks
        try:
            embeddings = await self._embedding.embed_texts(chunks_text)
        except Exception as e:
            await doc_repo.update_status(doc.id, "error")
            logger.error(f"Embedding failed for {filename}: {e}")
            return {"status": "error", "message": f"Embedding failed: {e}"}

        # 7. Store chunks in DB
        chunk_ids = []
        for i, (chunk_text, embedding) in enumerate(zip(chunks_text, embeddings)):
            chunk = await chunk_repo.create(
                document_id=doc.id,
                content=chunk_text,
                chunk_index=i,
                partition=partition,
                embedding=embedding.tobytes(),
            )
            chunk_ids.append(str(chunk.id))

        # 8. Add to FAISS index
        await self._faiss.add_vectors(embeddings, chunk_ids)
        await self._faiss.save()

        # 9. Update document status
        await doc_repo.update_status(doc.id, "indexed")
        doc.chunk_count = len(chunks_text)
        doc.indexed = True
        await db.commit()

        logger.info(f"Indexed document: {filename} ({len(chunks_text)} chunks, partition={partition})")

        return {
            "status": "success",
            "document_id": str(doc.id),
            "filename": filename,
            "chunks_created": len(chunks_text),
            "partition": partition,
        }

    async def delete_document(self, document_id: str, db: AsyncSession) -> dict:
        """Delete a document and its chunks from DB and FAISS."""
        doc_repo = DocumentRepository(db)
        chunk_repo = ChunkRepository(db)

        doc = await doc_repo.get_by_id(document_id)
        if not doc:
            return {"status": "error", "message": "Document not found"}

        # Get chunk IDs for FAISS removal
        chunks = await chunk_repo.get_by_document(document_id)
        chunk_ids_to_remove = {str(c.id) for c in chunks}

        # Remove from FAISS
        if chunk_ids_to_remove:
            await self._faiss.remove_vectors_by_ids(chunk_ids_to_remove)
            await self._faiss.save()

        # Delete from DB
        await chunk_repo.delete_by_document(document_id)
        await doc_repo.delete(document_id)
        await db.commit()

        # Delete file
        filepath = os.path.join(settings.upload_dir, doc.filename)
        if os.path.exists(filepath):
            os.remove(filepath)

        logger.info(f"Deleted document: {doc.filename}")
        return {"status": "success", "message": f"Deleted {doc.filename}"}

    async def rebuild_faiss_index(self, db: AsyncSession) -> dict:
        """Rebuild FAISS index from all stored chunks."""
        chunk_repo = ChunkRepository(db)

        # Get all chunks with embeddings
        from sqlalchemy import select
        from src.models.chunk import Chunk
        result = await db.execute(select(Chunk).where(Chunk.embedding.isnot(None)))
        chunks = result.scalars().all()

        if not chunks:
            self._faiss._create_empty_index()
            await self._faiss.save()
            return {"status": "success", "total_vectors": 0}

        embeddings = []
        chunk_ids = []
        for chunk in chunks:
            emb = np.frombuffer(chunk.embedding, dtype=np.float32)
            embeddings.append(emb)
            chunk_ids.append(str(chunk.id))

        await self._faiss.rebuild_from_data(embeddings, chunk_ids)

        logger.info(f"Rebuilt FAISS index: {len(embeddings)} vectors")
        return {"status": "success", "total_vectors": len(embeddings)}

    async def get_statistics(self, db: AsyncSession) -> dict:
        """Get system statistics."""
        doc_repo = DocumentRepository(db)
        chunk_repo = ChunkRepository(db)
        query_repo = QueryRepository(db)

        doc_count = await doc_repo.count()
        chunk_count = await chunk_repo.count()
        query_count = await query_repo.count()
        oos_count = await query_repo.count_out_of_scope()
        avg_confidence = await query_repo.avg_confidence()

        return {
            "documents": doc_count,
            "chunks": chunk_count,
            "queries": query_count,
            "out_of_scope_queries": oos_count,
            "avg_confidence": round(avg_confidence, 3) if avg_confidence else 0,
            "faiss_vectors": self._faiss.total_vectors,
            "embedding_backend": self._embedding.backend,
            "embedding_dimension": self._embedding.dimension,
        }

    async def list_documents(self, db: AsyncSession) -> list:
        """List all documents."""
        doc_repo = DocumentRepository(db)
        docs = await doc_repo.list_all()
        return [
            {
                "id": str(d.id),
                "filename": d.filename,
                "doc_type": d.doc_type,
                "status": d.status,
                "chunk_count": d.chunk_count,
                "partition": d.partition,
                "indexed": d.indexed,
                "created_at": d.created_at.isoformat() if d.created_at else None,
            }
            for d in docs
        ]

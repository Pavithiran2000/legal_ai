"""Chunk repository."""
from typing import List, Optional
from sqlalchemy import select, func, delete as sa_delete
from sqlalchemy.ext.asyncio import AsyncSession
from src.models.chunk import Chunk
from src.core.logging import get_logger

logger = get_logger(__name__)


class ChunkRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, **kwargs) -> Chunk:
        chunk = Chunk(**kwargs)
        self.db.add(chunk)
        await self.db.commit()
        await self.db.refresh(chunk)
        return chunk

    async def get_by_id(self, chunk_id: str) -> Optional[Chunk]:
        result = await self.db.execute(
            select(Chunk).where(Chunk.id == chunk_id)
        )
        return result.scalar_one_or_none()

    async def create_many(self, chunks_data: List[dict]) -> List[Chunk]:
        chunks = [Chunk(**d) for d in chunks_data]
        self.db.add_all(chunks)
        await self.db.commit()
        return chunks

    async def get_by_document(self, document_id: str) -> List[Chunk]:
        result = await self.db.execute(
            select(Chunk).where(Chunk.document_id == document_id)
            .order_by(Chunk.chunk_index)
        )
        return list(result.scalars().all())

    async def count(self) -> int:
        result = await self.db.execute(select(func.count(Chunk.id)))
        return result.scalar() or 0

    async def count_by_document(self, document_id: str) -> int:
        result = await self.db.execute(
            select(func.count(Chunk.id)).where(Chunk.document_id == document_id)
        )
        return result.scalar() or 0

    async def delete_by_document(self, document_id: str):
        await self.db.execute(
            sa_delete(Chunk).where(Chunk.document_id == document_id)
        )
        await self.db.commit()

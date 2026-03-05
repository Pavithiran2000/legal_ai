"""Document repository."""
from typing import List, Optional
from sqlalchemy import select, func, update
from sqlalchemy.ext.asyncio import AsyncSession
from src.models.document import Document
from src.core.logging import get_logger

logger = get_logger(__name__)


class DocumentRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, **kwargs) -> Document:
        doc = Document(**kwargs)
        self.db.add(doc)
        await self.db.commit()
        await self.db.refresh(doc)
        return doc

    async def get_by_id(self, doc_id: str) -> Optional[Document]:
        result = await self.db.execute(select(Document).where(Document.id == doc_id))
        return result.scalar_one_or_none()

    async def list_all(self, skip: int = 0, limit: int = 100) -> List[Document]:
        result = await self.db.execute(
            select(Document).where(Document.is_active == True)
            .order_by(Document.created_at.desc())
            .offset(skip).limit(limit)
        )
        return list(result.scalars().all())

    async def count(self) -> int:
        result = await self.db.execute(
            select(func.count(Document.id)).where(Document.is_active == True)
        )
        return result.scalar() or 0

    async def update_status(self, doc_id: str, status: str, **kwargs):
        stmt = update(Document).where(Document.id == doc_id).values(status=status, **kwargs)
        await self.db.execute(stmt)
        await self.db.commit()

    async def delete(self, doc_id: str):
        doc = await self.get_by_id(doc_id)
        if doc:
            await self.db.delete(doc)
            await self.db.commit()

"""Query repository."""
from typing import List, Optional
from sqlalchemy import select, func, update
from sqlalchemy.ext.asyncio import AsyncSession
from src.models.query import Query
from src.core.logging import get_logger

logger = get_logger(__name__)


class QueryRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, **kwargs) -> Query:
        query = Query(**kwargs)
        self.db.add(query)
        await self.db.commit()
        await self.db.refresh(query)
        return query

    async def get_by_id(self, query_id: str) -> Optional[Query]:
        result = await self.db.execute(select(Query).where(Query.id == query_id))
        return result.scalar_one_or_none()

    async def list_recent(self, limit: int = 50, offset: int = 0) -> List[Query]:
        result = await self.db.execute(
            select(Query).order_by(Query.created_at.desc()).offset(offset).limit(limit)
        )
        return list(result.scalars().all())

    async def count(self) -> int:
        result = await self.db.execute(select(func.count(Query.id)))
        return result.scalar() or 0

    async def count_out_of_scope(self) -> int:
        result = await self.db.execute(
            select(func.count(Query.id)).where(Query.out_of_scope == True)
        )
        return result.scalar() or 0

    async def avg_confidence(self) -> float:
        result = await self.db.execute(
            select(func.avg(Query.confidence)).where(Query.confidence.isnot(None))
        )
        return float(result.scalar() or 0.0)

    async def update_feedback(self, query_id: str, rating: int, comment: Optional[str] = None) -> bool:
        # Check if query exists
        query = await self.get_by_id(query_id)
        if not query:
            return False
        stmt = update(Query).where(Query.id == query_id).values(
            feedback_rating=rating, feedback_comment=comment
        )
        await self.db.execute(stmt)
        await self.db.commit()
        return True

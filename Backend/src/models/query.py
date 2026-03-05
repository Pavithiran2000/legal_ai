"""Query model - stores user queries and responses."""
from sqlalchemy import Column, String, Text, Float, Integer, Boolean, JSON
from src.models.base import Base, UUIDMixin, TimestampMixin


class Query(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "queries"

    query_text = Column(Text, nullable=False)
    response_json = Column(JSON, nullable=True)
    out_of_scope = Column(Boolean, default=False)
    scope_category = Column(String(100), nullable=True)
    confidence = Column(Float, nullable=True)
    model_used = Column(String(100), nullable=True)
    generation_time_ms = Column(Integer, nullable=True)
    context_chunks_used = Column(Integer, default=0)
    feedback_rating = Column(Integer, nullable=True)
    feedback_comment = Column(Text, nullable=True)
    error = Column(Text, nullable=True)

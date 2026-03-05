"""Chunk model - document text segments with embeddings."""
from sqlalchemy import Column, String, Integer, Text, Float, ForeignKey, LargeBinary
from sqlalchemy.orm import relationship
from src.models.base import Base, UUIDMixin, TimestampMixin


class Chunk(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "chunks"

    document_id = Column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, default=0)
    start_char = Column(Integer, default=0)
    end_char = Column(Integer, default=0)
    partition = Column(String(100), nullable=True)
    embedding = Column(LargeBinary, nullable=True)  # serialized numpy array
    metadata_json = Column(Text, nullable=True)

    document = relationship("Document", back_populates="chunks")

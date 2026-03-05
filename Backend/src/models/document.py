"""Document model."""
from sqlalchemy import Column, String, Integer, Text, Boolean, Float
from sqlalchemy.orm import relationship
from src.models.base import Base, UUIDMixin, TimestampMixin


class Document(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "documents"

    filename = Column(String(500), nullable=False)
    original_filename = Column(String(500), nullable=True)
    file_type = Column(String(50), default="application/pdf")
    file_size = Column(Integer, default=0)
    doc_type = Column(String(50), default="unknown")  # act, case, regulation
    title = Column(String(1000), nullable=True)
    description = Column(Text, nullable=True)
    content = Column(Text, nullable=True)
    status = Column(String(50), default="uploaded")  # uploaded, indexed, failed
    chunk_count = Column(Integer, default=0)
    partition = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True)
    version = Column(Integer, default=1)
    indexed = Column(Boolean, default=False)
    index_error = Column(Text, nullable=True)

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

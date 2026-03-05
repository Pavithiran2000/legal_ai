"""initial schema

Revision ID: 001
Revises:
Create Date: 2025-01-01
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Documents table
    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("filename", sa.String(500), nullable=False),
        sa.Column("doc_type", sa.String(50), default="pdf"),
        sa.Column("content", sa.Text),
        sa.Column("status", sa.String(50), default="pending"),
        sa.Column("chunk_count", sa.Integer, default=0),
        sa.Column("partition", sa.String(100)),
        sa.Column("indexed", sa.Boolean, default=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index("ix_documents_status", "documents", ["status"])
    op.create_index("ix_documents_partition", "documents", ["partition"])

    # Chunks table
    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("chunk_index", sa.Integer, default=0),
        sa.Column("partition", sa.String(100)),
        sa.Column("embedding", sa.LargeBinary),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.create_index("ix_chunks_partition", "chunks", ["partition"])

    # Queries table
    op.create_table(
        "queries",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("query_text", sa.Text, nullable=False),
        sa.Column("response_json", postgresql.JSON),
        sa.Column("out_of_scope", sa.Boolean),
        sa.Column("scope_category", sa.String(100)),
        sa.Column("confidence", sa.Float),
        sa.Column("model_used", sa.String(100)),
        sa.Column("generation_time_ms", sa.Integer),
        sa.Column("context_chunks_used", sa.Integer, default=0),
        sa.Column("feedback_rating", sa.Integer),
        sa.Column("feedback_comment", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index("ix_queries_out_of_scope", "queries", ["out_of_scope"])
    op.create_index("ix_queries_created_at", "queries", ["created_at"])


def downgrade() -> None:
    op.drop_table("queries")
    op.drop_table("chunks")
    op.drop_table("documents")

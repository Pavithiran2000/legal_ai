"""add missing columns to match ORM models

Revision ID: 002
Revises: 001
Create Date: 2026-03-06
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- documents: add missing columns ---
    op.add_column("documents", sa.Column("original_filename", sa.String(500), nullable=True))
    op.add_column("documents", sa.Column("file_type", sa.String(50), server_default="application/pdf"))
    op.add_column("documents", sa.Column("file_size", sa.Integer, server_default="0"))
    op.add_column("documents", sa.Column("title", sa.String(1000), nullable=True))
    op.add_column("documents", sa.Column("description", sa.Text, nullable=True))
    op.add_column("documents", sa.Column("is_active", sa.Boolean, server_default=sa.text("true")))
    op.add_column("documents", sa.Column("version", sa.Integer, server_default="1"))
    op.add_column("documents", sa.Column("index_error", sa.Text, nullable=True))

    # --- chunks: add missing columns ---
    op.add_column("chunks", sa.Column("start_char", sa.Integer, server_default="0"))
    op.add_column("chunks", sa.Column("end_char", sa.Integer, server_default="0"))
    op.add_column("chunks", sa.Column("metadata_json", sa.Text, nullable=True))

    # --- queries: add missing column ---
    op.add_column("queries", sa.Column("error", sa.Text, nullable=True))


def downgrade() -> None:
    # queries
    op.drop_column("queries", "error")

    # chunks
    op.drop_column("chunks", "metadata_json")
    op.drop_column("chunks", "end_char")
    op.drop_column("chunks", "start_char")

    # documents
    op.drop_column("documents", "index_error")
    op.drop_column("documents", "version")
    op.drop_column("documents", "is_active")
    op.drop_column("documents", "description")
    op.drop_column("documents", "title")
    op.drop_column("documents", "file_size")
    op.drop_column("documents", "file_type")
    op.drop_column("documents", "original_filename")

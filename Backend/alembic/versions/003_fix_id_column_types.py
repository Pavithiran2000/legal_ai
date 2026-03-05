"""fix id column types from UUID to VARCHAR to match ORM models

Revision ID: 003
Revises: 002
Create Date: 2026-03-06
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop foreign key constraint first
    op.drop_constraint("chunks_document_id_fkey", "chunks", type_="foreignkey")

    # Alter UUID columns to VARCHAR(36) with explicit cast
    op.execute("ALTER TABLE documents ALTER COLUMN id TYPE VARCHAR(36) USING id::text")
    op.execute("ALTER TABLE chunks ALTER COLUMN id TYPE VARCHAR(36) USING id::text")
    op.execute("ALTER TABLE chunks ALTER COLUMN document_id TYPE VARCHAR(36) USING document_id::text")
    op.execute("ALTER TABLE queries ALTER COLUMN id TYPE VARCHAR(36) USING id::text")

    # Recreate foreign key constraint
    op.create_foreign_key(
        "chunks_document_id_fkey",
        "chunks",
        "documents",
        ["document_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    op.drop_constraint("chunks_document_id_fkey", "chunks", type_="foreignkey")

    op.execute("ALTER TABLE documents ALTER COLUMN id TYPE UUID USING id::uuid")
    op.execute("ALTER TABLE chunks ALTER COLUMN id TYPE UUID USING id::uuid")
    op.execute("ALTER TABLE chunks ALTER COLUMN document_id TYPE UUID USING document_id::uuid")
    op.execute("ALTER TABLE queries ALTER COLUMN id TYPE UUID USING id::uuid")

    op.create_foreign_key(
        "chunks_document_id_fkey",
        "chunks",
        "documents",
        ["document_id"],
        ["id"],
        ondelete="CASCADE",
    )

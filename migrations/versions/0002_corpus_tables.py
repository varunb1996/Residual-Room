"""Corpus tables: corpus_versions, sources, chunks.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-28
"""

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

# Fixed here on purpose: a migration must not change if the embedding model changes later.
DIM = 384


def upgrade() -> None:
    op.create_table(
        "corpus_versions",
        sa.Column("version", sa.String, primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("embed_model", sa.String, nullable=False),
        sa.Column("manifest_sha256", sa.String, nullable=False),
        sa.Column("chunk_count", sa.Integer, nullable=False),
    )
    op.create_table(
        "sources",
        sa.Column(
            "corpus_version",
            sa.String,
            sa.ForeignKey("corpus_versions.version", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("source_id", sa.String, primary_key=True),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("origin", sa.String, nullable=False),
        sa.Column("kind", sa.String),
        sa.Column("status", sa.String, nullable=False),
        sa.Column("license", sa.Text, nullable=False),
        sa.Column("commit_allowed", sa.Boolean, nullable=False),
        sa.Column("url", sa.Text),
        sa.Column("text_sha256", sa.String, nullable=False),
    )
    op.create_table(
        "chunks",
        sa.Column("corpus_version", sa.String, primary_key=True),
        sa.Column("chunk_id", sa.String, primary_key=True),
        sa.Column("source_id", sa.String, nullable=False),
        sa.Column("seq", sa.Integer, nullable=False),
        sa.Column("status", sa.String, nullable=False),
        sa.Column("license", sa.Text, nullable=False),
        sa.Column("commit_allowed", sa.Boolean, nullable=False),
        sa.Column("heading_path", sa.Text, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("embedding", Vector(DIM), nullable=False),
        sa.ForeignKeyConstraint(
            ["corpus_version", "source_id"],
            ["sources.corpus_version", "sources.source_id"],
            ondelete="CASCADE",
        ),
        sa.CheckConstraint("status IN ('current', 'superseded')", name="chunks_status_valid"),
    )
    # No vector index yet: exact search over a few thousand rows is fast enough.


def downgrade() -> None:
    op.drop_table("chunks")
    op.drop_table("sources")
    op.drop_table("corpus_versions")

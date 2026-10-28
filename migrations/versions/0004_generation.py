"""Generation: structured segments on versions, and generation jobs.

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-28
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("piece_versions", sa.Column("segments", JSONB))
    op.add_column("piece_versions", sa.Column("generation", JSONB))
    op.create_table(
        "generation_jobs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("status", sa.String, nullable=False),
        sa.Column("explainer", sa.String, nullable=False),
        sa.Column("segment", sa.String, nullable=False),
        sa.Column("kind", sa.String, nullable=False),
        sa.Column("issue_date", sa.Date),
        sa.Column("piece_id", sa.Integer, sa.ForeignKey("pieces.id", ondelete="SET NULL")),
        sa.Column("error", sa.Text),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
    )


def downgrade() -> None:
    op.drop_table("generation_jobs")
    op.drop_column("piece_versions", "generation")
    op.drop_column("piece_versions", "segments")

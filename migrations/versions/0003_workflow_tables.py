"""Approval workflow: pieces, piece_versions, check_runs, overrides, audit_log.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-28
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

NOW = sa.func.now()


def upgrade() -> None:
    op.create_table(
        "pieces",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("kind", sa.String, nullable=False),
        sa.Column("segment", sa.String, nullable=False),
        sa.Column("issue_date", sa.Date),
        sa.Column("state", sa.String, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
    )
    op.create_table(
        "piece_versions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("piece_id", sa.Integer, sa.ForeignKey("pieces.id", ondelete="CASCADE"), nullable=False),
        sa.Column("n", sa.Integer, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("source", sa.String, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.UniqueConstraint("piece_id", "n"),
    )
    op.create_table(
        "check_runs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "version_id", sa.Integer, sa.ForeignKey("piece_versions.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("status", sa.String, nullable=False),
        sa.Column("verdict", sa.String),
        sa.Column("report", JSONB),
        sa.Column("judge_model", sa.String),
        sa.Column("corpus_version", sa.String),
        sa.Column("error", sa.Text),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
    )
    op.create_table(
        "overrides",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "check_run_id", sa.Integer, sa.ForeignKey("check_runs.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("hit_index", sa.Integer, nullable=False),
        sa.Column("rule", sa.String, nullable=False),
        sa.Column("reason", sa.Text, nullable=False),
        sa.Column("actor", sa.String, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.UniqueConstraint("check_run_id", "hit_index"),
        sa.CheckConstraint("length(trim(reason)) > 0", name="override_reason_required"),
    )
    op.create_table(
        "audit_log",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("piece_id", sa.Integer, sa.ForeignKey("pieces.id", ondelete="CASCADE"), nullable=False),
        sa.Column("at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.Column("actor", sa.String, nullable=False),
        sa.Column("action", sa.String, nullable=False),
        sa.Column("from_state", sa.String),
        sa.Column("to_state", sa.String),
        sa.Column("detail", JSONB),
    )


def downgrade() -> None:
    for table in ("audit_log", "overrides", "check_runs", "piece_versions", "pieces"):
        op.drop_table(table)

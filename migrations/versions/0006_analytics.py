"""Analytics: metric_rows, narrations, subscribers.unsubscribed_at.

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-28
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("subscribers", sa.Column("unsubscribed_at", sa.DateTime(timezone=True)))
    op.create_table(
        "metric_rows",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("dataset", sa.String, nullable=False),
        sa.Column("week", sa.Date),
        sa.Column("channel", sa.String, nullable=False),
        sa.Column("segment", sa.String, nullable=False),
        sa.Column("metric", sa.String, nullable=False),
        sa.Column("value", sa.Float, nullable=False),
        sa.Column("n", sa.Integer),
        sa.Column("derived", sa.Boolean, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("dataset IN ('real', 'seeded')", name="metric_rows_dataset_valid"),
    )
    op.create_index("ix_metric_rows_dataset", "metric_rows", ["dataset"])
    op.create_table(
        "narrations",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("dataset", sa.String, nullable=False),
        sa.Column("status", sa.String, nullable=False),
        sa.Column("insights", JSONB),
        sa.Column("segments", JSONB),
        sa.Column("text", sa.Text),
        sa.Column("report", JSONB),
        sa.Column("verdict", sa.String),
        sa.Column("model", sa.String),
        sa.Column("error", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
    )


def downgrade() -> None:
    op.drop_table("narrations")
    op.drop_index("ix_metric_rows_dataset", "metric_rows")
    op.drop_table("metric_rows")
    op.drop_column("subscribers", "unsubscribed_at")

"""Channels and campaigns: subscribers, campaigns, deliveries, links, clicks, kv.

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-28
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None

NOW = sa.func.now()


def upgrade() -> None:
    op.create_table(
        "subscribers",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("email", sa.String, unique=True),
        sa.Column("telegram_chat_id", sa.String, unique=True),
        sa.Column("whatsapp", sa.String, unique=True),
        sa.Column("segment", sa.String, nullable=False),
        sa.Column("subscribed", sa.Boolean, nullable=False),
        sa.Column("opted_in_at", sa.DateTime(timezone=True)),
        sa.Column("whatsapp_last_inbound_at", sa.DateTime(timezone=True)),
        sa.Column("token", sa.String, unique=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
    )
    op.create_table(
        "campaigns",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("piece_id", sa.Integer, sa.ForeignKey("pieces.id", ondelete="CASCADE"), nullable=False),
        sa.Column("version_id", sa.Integer, sa.ForeignKey("piece_versions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("segments", JSONB, nullable=False),
        sa.Column("channels", JSONB, nullable=False),
        sa.Column("scheduled_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.Column("sent_at", sa.DateTime(timezone=True)),
    )
    op.create_table(
        "deliveries",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("campaign_id", sa.Integer, sa.ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False),
        sa.Column("channel", sa.String, nullable=False),
        sa.Column("segment", sa.String, nullable=False),
        sa.Column("recipient", sa.String, nullable=False),
        sa.Column("status", sa.String, nullable=False),
        sa.Column("provider_id", sa.String),
        sa.Column("error", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
    )
    op.create_table(
        "links",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("token", sa.String, unique=True, nullable=False),
        sa.Column("campaign_id", sa.Integer, sa.ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False),
        sa.Column("piece_id", sa.Integer, sa.ForeignKey("pieces.id", ondelete="CASCADE"), nullable=False),
        sa.Column("segment", sa.String, nullable=False),
        sa.Column("channel", sa.String, nullable=False),
        sa.Column("label", sa.String, nullable=False),
        sa.Column("target_url", sa.Text, nullable=False),
    )
    op.create_table(
        "clicks",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("link_id", sa.Integer, sa.ForeignKey("links.id", ondelete="CASCADE"), nullable=False),
        sa.Column("at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
    )
    op.create_table(
        "kv",
        sa.Column("key", sa.String, primary_key=True),
        sa.Column("value", sa.Text, nullable=False),
    )


def downgrade() -> None:
    for table in ("kv", "clicks", "links", "deliveries", "campaigns", "subscribers"):
        op.drop_table(table)

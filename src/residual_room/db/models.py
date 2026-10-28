from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy import Float as sa_Float
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from residual_room.corpus.embed import DIM


class Base(DeclarativeBase):
    pass


class CorpusVersion(Base):
    """One ingest run (e.g. corpus-v1). Generated pieces will record which one they used."""

    __tablename__ = "corpus_versions"

    version: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    embed_model: Mapped[str] = mapped_column(String)
    manifest_sha256: Mapped[str] = mapped_column(String)
    chunk_count: Mapped[int] = mapped_column(Integer)


class SourceRow(Base):
    """A snapshot of a manifest entry as it was in this corpus version (status can change later)."""

    __tablename__ = "sources"

    corpus_version: Mapped[str] = mapped_column(
        ForeignKey("corpus_versions.version", ondelete="CASCADE"), primary_key=True
    )
    source_id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    origin: Mapped[str] = mapped_column(String)
    kind: Mapped[str | None] = mapped_column(String)
    status: Mapped[str] = mapped_column(String)
    license: Mapped[str] = mapped_column(Text)
    commit_allowed: Mapped[bool] = mapped_column(Boolean)
    url: Mapped[str | None] = mapped_column(Text)
    text_sha256: Mapped[str] = mapped_column(String)


class ChunkRow(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        ForeignKeyConstraint(
            ["corpus_version", "source_id"],
            ["sources.corpus_version", "sources.source_id"],
            ondelete="CASCADE",
        ),
        CheckConstraint("status IN ('current', 'superseded')", name="chunks_status_valid"),
    )

    corpus_version: Mapped[str] = mapped_column(String, primary_key=True)
    chunk_id: Mapped[str] = mapped_column(String, primary_key=True)  # e.g. ext-mtbench-paper#0012
    source_id: Mapped[str] = mapped_column(String)
    seq: Mapped[int] = mapped_column(Integer)
    # Denormalized from sources so retrieval can filter superseded chunks without a join.
    status: Mapped[str] = mapped_column(String)
    license: Mapped[str] = mapped_column(Text)
    commit_allowed: Mapped[bool] = mapped_column(Boolean)
    heading_path: Mapped[str] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(DIM))


# ---------- approval workflow (Prompt 6) ----------


class Piece(Base):
    """A newsletter issue or explainer summary moving through the approval states."""

    __tablename__ = "pieces"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(String)  # newsletter | explainer_summary
    segment: Mapped[str] = mapped_column(String)  # reader segment
    issue_date: Mapped[date | None] = mapped_column(Date)
    state: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class PieceVersion(Base):
    """Immutable text of one version. Editing a piece always makes a new version."""

    __tablename__ = "piece_versions"
    __table_args__ = (UniqueConstraint("piece_id", "n"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    piece_id: Mapped[int] = mapped_column(ForeignKey("pieces.id", ondelete="CASCADE"))
    n: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String)  # pasted | generated
    # Generated versions keep the generator's structured segments (the checker
    # uses them as-is) and how they were made: model, corpus version, chunks given.
    segments: Mapped[list | None] = mapped_column(JSONB)
    generation: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class CheckRun(Base):
    __tablename__ = "check_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version_id: Mapped[int] = mapped_column(ForeignKey("piece_versions.id", ondelete="CASCADE"))
    status: Mapped[str] = mapped_column(String)  # running | done | failed
    verdict: Mapped[str | None] = mapped_column(String)
    report: Mapped[dict | None] = mapped_column(JSONB)
    judge_model: Mapped[str | None] = mapped_column(String)
    corpus_version: Mapped[str | None] = mapped_column(String)
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Override(Base):
    """A reviewer's decision that one blocking hit is wrong, with the reason.
    Logged so it can be reviewed later as a possible checker error."""

    __tablename__ = "overrides"
    __table_args__ = (
        UniqueConstraint("check_run_id", "hit_index"),
        CheckConstraint("length(trim(reason)) > 0", name="override_reason_required"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    check_run_id: Mapped[int] = mapped_column(ForeignKey("check_runs.id", ondelete="CASCADE"))
    hit_index: Mapped[int] = mapped_column(Integer)  # index into the run's report["hits"]
    rule: Mapped[str] = mapped_column(String)
    reason: Mapped[str] = mapped_column(Text)
    actor: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AuditEvent(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    piece_id: Mapped[int] = mapped_column(ForeignKey("pieces.id", ondelete="CASCADE"))
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    actor: Mapped[str] = mapped_column(String)
    action: Mapped[str] = mapped_column(String)
    from_state: Mapped[str | None] = mapped_column(String)
    to_state: Mapped[str | None] = mapped_column(String)
    detail: Mapped[dict | None] = mapped_column(JSONB)


class GenerationJob(Base):
    """A background generation request. On success it points at the draft piece it created."""

    __tablename__ = "generation_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    status: Mapped[str] = mapped_column(String)  # running | done | failed
    explainer: Mapped[str] = mapped_column(String)
    segment: Mapped[str] = mapped_column(String)
    kind: Mapped[str] = mapped_column(String)
    issue_date: Mapped[date | None] = mapped_column(Date)
    piece_id: Mapped[int | None] = mapped_column(ForeignKey("pieces.id", ondelete="SET NULL"))
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


# ---------- channels and campaigns (Prompt 7) ----------


class Subscriber(Base):
    __tablename__ = "subscribers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str | None] = mapped_column(String, unique=True)
    telegram_chat_id: Mapped[str | None] = mapped_column(String, unique=True)
    whatsapp: Mapped[str | None] = mapped_column(String, unique=True)
    segment: Mapped[str] = mapped_column(String)
    subscribed: Mapped[bool] = mapped_column(Boolean, default=True)
    opted_in_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # WhatsApp (mock): free-form messages only within 24 h of the subscriber's last message.
    whatsapp_last_inbound_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    unsubscribed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    token: Mapped[str] = mapped_column(String, unique=True)  # for the unsubscribe link
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Campaign(Base):
    __tablename__ = "campaigns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    piece_id: Mapped[int] = mapped_column(ForeignKey("pieces.id", ondelete="CASCADE"))
    version_id: Mapped[int] = mapped_column(ForeignKey("piece_versions.id", ondelete="CASCADE"))
    segments: Mapped[list] = mapped_column(JSONB)
    channels: Mapped[list] = mapped_column(JSONB)
    scheduled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String)  # scheduled | sending | sent | cancelled
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Delivery(Base):
    __tablename__ = "deliveries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    campaign_id: Mapped[int] = mapped_column(ForeignKey("campaigns.id", ondelete="CASCADE"))
    channel: Mapped[str] = mapped_column(String)
    segment: Mapped[str] = mapped_column(String)
    recipient: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String)  # sent | failed | refused | rejected
    provider_id: Mapped[str | None] = mapped_column(String)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Link(Base):
    """A tracked outbound link: /r/{token} redirects to target_url and logs a click."""

    __tablename__ = "links"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    token: Mapped[str] = mapped_column(String, unique=True)
    campaign_id: Mapped[int] = mapped_column(ForeignKey("campaigns.id", ondelete="CASCADE"))
    piece_id: Mapped[int] = mapped_column(ForeignKey("pieces.id", ondelete="CASCADE"))
    segment: Mapped[str] = mapped_column(String)
    channel: Mapped[str] = mapped_column(String)
    label: Mapped[str] = mapped_column(String)  # cta | source | read_next
    target_url: Mapped[str] = mapped_column(Text)


class Click(Base):
    """A real metric (CLAUDE.md principle 9). No IP or personal data is stored."""

    __tablename__ = "clicks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    link_id: Mapped[int] = mapped_column(ForeignKey("links.id", ondelete="CASCADE"))
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class KeyValue(Base):
    """Small worker state, e.g. the Telegram getUpdates offset."""

    __tablename__ = "kv"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[str] = mapped_column(Text)


# ---------- analytics (Prompt 8) ----------


class MetricRow(Base):
    """One number. Narration must cite these rows ([m:<id>]) for every figure it states.

    week None = the whole window. dataset keeps real and seeded data apart; they
    are never mixed in one query."""

    __tablename__ = "metric_rows"
    __table_args__ = (CheckConstraint("dataset IN ('real', 'seeded')", name="metric_rows_dataset_valid"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset: Mapped[str] = mapped_column(String, index=True)  # real | seeded
    week: Mapped[date | None] = mapped_column(Date)  # Monday of the ISO week; None = whole window
    channel: Mapped[str] = mapped_column(String)  # email | telegram | whatsapp | all
    segment: Mapped[str] = mapped_column(String)
    metric: Mapped[str] = mapped_column(String)
    value: Mapped[float] = mapped_column(sa_Float)
    n: Mapped[int | None] = mapped_column(Integer)  # sample the value is computed over
    derived: Mapped[bool] = mapped_column(Boolean, default=False)  # computed by an insight rule
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Narration(Base):
    __tablename__ = "narrations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String)  # running | done | failed
    insights: Mapped[list | None] = mapped_column(JSONB)
    segments: Mapped[list | None] = mapped_column(JSONB)
    text: Mapped[str | None] = mapped_column(Text)
    report: Mapped[dict | None] = mapped_column(JSONB)
    verdict: Mapped[str | None] = mapped_column(String)
    model: Mapped[str | None] = mapped_column(String)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

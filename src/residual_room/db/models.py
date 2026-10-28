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

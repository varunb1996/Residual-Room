from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    String,
    Text,
    func,
)
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

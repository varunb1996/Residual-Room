"""Vector search over one corpus version.

Generation retrieves with include_superseded=False (never ground new content in
a stale source). The checker will retrieve with include_superseded=True, so it
can notice a claim that matches a superseded chunk (P5).
"""

from dataclasses import dataclass

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from residual_room.corpus.embed import Embedder
from residual_room.db.models import ChunkRow, CorpusVersion


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    source_id: str
    status: str
    heading_path: str
    text: str
    distance: float  # cosine distance: 0 = identical direction


def latest_version(engine: Engine) -> str:
    with Session(engine) as session:
        version = session.scalar(
            select(CorpusVersion.version).order_by(CorpusVersion.created_at.desc()).limit(1)
        )
    if version is None:
        raise LookupError("no corpus version ingested yet: run `make ingest`")
    return version


def search(
    engine: Engine,
    embedder: Embedder,
    query: str,
    k: int = 8,
    corpus_version: str | None = None,
    include_superseded: bool = False,
    source_ids: list[str] | None = None,
) -> list[Hit]:
    version = corpus_version or latest_version(engine)
    distance = ChunkRow.embedding.cosine_distance(embedder.embed_query(query)).label("distance")
    stmt = select(ChunkRow, distance).where(ChunkRow.corpus_version == version)
    if not include_superseded:
        stmt = stmt.where(ChunkRow.status == "current")
    if source_ids:
        stmt = stmt.where(ChunkRow.source_id.in_(source_ids))
    stmt = stmt.order_by(distance).limit(k)

    with Session(engine) as session:
        return [
            Hit(row.chunk_id, row.source_id, row.status, row.heading_path, row.text, float(d))
            for row, d in session.execute(stmt)
        ]

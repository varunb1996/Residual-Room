"""Evidence for the judge: the passages a claim is checked against.

Check-time retrieval always includes superseded chunks: the checker has to see
stale text to notice a claim that repeats it (P5). Generation never does.
"""

from dataclasses import dataclass

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from residual_room.corpus.embed import Embedder
from residual_room.corpus.manifest import Manifest
from residual_room.corpus.retrieve import search
from residual_room.db.models import ChunkRow


@dataclass(frozen=True)
class Passage:
    chunk_id: str
    source_id: str
    status: str
    heading_path: str
    text: str

    def render(self) -> str:
        return f"[{self.chunk_id}] (status: {self.status}; {self.heading_path})\n{self.text}"


class CorpusRetriever:
    def __init__(
        self,
        engine: Engine,
        embedder: Embedder,
        corpus_version: str,
        manifest: Manifest,
        operational_sources: list[str] = (),
    ):
        self.engine = engine
        self.embedder = embedder
        self.version = corpus_version
        self.manifest = manifest
        self.operational_sources = list(operational_sources)

    def for_citation(self, citation: str, query: str, k: int = 3) -> list[Passage]:
        """The cited chunk itself, or the k chunks of a cited source closest to the claim."""
        if "#" in citation:
            with Session(self.engine) as s:
                row = s.scalar(
                    select(ChunkRow).where(
                        ChunkRow.corpus_version == self.version, ChunkRow.chunk_id == citation
                    )
                )
            return [_passage(row)] if row else []
        source_id = self.manifest.resolve(citation).id if citation in self.manifest.aliases else citation
        hits = search(
            self.engine, self.embedder, query, k=k, corpus_version=self.version,
            include_superseded=True, source_ids=[source_id],
        )
        return [Passage(h.chunk_id, h.source_id, h.status, h.heading_path, h.text) for h in hits]

    def across_corpus(self, query: str, k: int = 4) -> list[Passage]:
        """Every chunk of the operational docs, plus the k best chunks corpus-wide.

        Operational facts (schedule, series, subscription, corrections) live only
        in a few tiny internal docs, which similarity ranks below longer docs on
        the same topic ("Upgrade to Pro" barely resembles "There is no paid
        tier"). They are a few hundred tokens in total, so they always go in.
        """
        with Session(self.engine) as s:
            rows = s.scalars(
                select(ChunkRow)
                .where(
                    ChunkRow.corpus_version == self.version,
                    ChunkRow.source_id.in_(self.operational_sources),
                )
                .order_by(ChunkRow.source_id, ChunkRow.seq)
            ).all()
        passages = [_passage(r) for r in rows]
        hits = search(
            self.engine, self.embedder, query, k=k, corpus_version=self.version, include_superseded=True
        )
        passages += [Passage(h.chunk_id, h.source_id, h.status, h.heading_path, h.text) for h in hits]
        return list({p.chunk_id: p for p in passages}.values())


def _passage(row: ChunkRow) -> Passage:
    return Passage(row.chunk_id, row.source_id, row.status, row.heading_path, row.text)

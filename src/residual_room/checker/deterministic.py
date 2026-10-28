"""Rules that need no judgment: word lists, counting, and status lookups.

  S1  hype word from the avoid-list (per segment)
  S3  more than one CTA in the whole message (document level)
  P5  a citation resolves to a superseded source or chunk (the cited half of P5;
      the semantic half, a claim matching a superseded chunk, needs the judge)
  P1  a claim has no citation, or cites an id that isn't in the corpus (the
      citation half of P1; whether the source supports the claim needs the judge)
"""

from dataclasses import dataclass

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from residual_room.checker.claims import count_ctas
from residual_room.checker.types import RuleHit, Segment
from residual_room.config import PublicationConfig, get_config
from residual_room.corpus.manifest import Manifest


@dataclass(frozen=True)
class StatusLookup:
    """Source status as of one corpus version, with manifest aliases resolved."""

    statuses: dict[str, str]  # source_id -> current | superseded
    aliases: dict[str, str]

    @classmethod
    def from_manifest(cls, manifest: Manifest) -> "StatusLookup":
        return cls({s.id: s.status for s in manifest.sources.values()}, manifest.aliases)

    @classmethod
    def from_corpus(cls, engine: Engine, corpus_version: str, manifest: Manifest) -> "StatusLookup":
        from residual_room.db.models import SourceRow

        with Session(engine) as session:
            rows = session.execute(
                select(SourceRow.source_id, SourceRow.status).where(
                    SourceRow.corpus_version == corpus_version
                )
            ).all()
        return cls(dict(rows), manifest.aliases)

    def status(self, citation: str) -> str | None:
        """Status of a cited source or chunk id; None if it isn't in the corpus."""
        source_id = citation.split("#", 1)[0]  # chunk ids are <source_id>#<seq>
        return self.statuses.get(self.aliases.get(source_id, source_id))


def check_s1(segments: list[Segment], config: PublicationConfig) -> list[RuleHit]:
    hits = []
    for i, seg in enumerate(segments):
        for word in dict.fromkeys(m.lower() for m in config.avoid_regex.findall(seg.text)):
            hits.append(RuleHit("S1", f"hype word from the avoid-list: {word!r}", i, word))
    return hits


def check_s3(text: str, config: PublicationConfig) -> list[RuleHit]:
    n = count_ctas(text, config)
    if n <= 1:
        return []
    return [RuleHit("S3", f"{n} calls to action; the brand guide allows one per message")]


def check_citations(segments: list[Segment], lookup: StatusLookup) -> list[RuleHit]:
    hits = []
    for i, seg in enumerate(segments):
        if seg.kind == "claim" and not seg.citations:
            hits.append(RuleHit("P1", "factual claim without a citation", i))
        for cite in seg.citations:
            status = lookup.status(cite)
            if status is None:
                hits.append(RuleHit("P1", f"cites {cite!r}, which is not in the corpus", i, cite))
            elif status == "superseded":
                hits.append(RuleHit("P5", f"cites superseded source {cite!r}", i, cite))
    return hits


def run_deterministic(
    text: str, segments: list[Segment], lookup: StatusLookup, config: PublicationConfig | None = None
) -> list[RuleHit]:
    config = config or get_config()
    return check_s1(segments, config) + check_s3(text, config) + check_citations(segments, lookup)

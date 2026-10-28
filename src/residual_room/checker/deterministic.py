"""Rules that need no judgment: word lists, counting, and status lookups.

  S1  hype word from the avoid-list (per segment)
  S3  more than one CTA in the whole message (document level)
  P5  a citation resolves to a superseded source or chunk (the cited half of P5;
      the semantic half, a claim matching a superseded chunk, needs the judge)
  P1  a claim has no citation, or cites an id that isn't in the corpus (the
      citation half of P1; whether the source supports the claim needs the judge)
"""

import re
from dataclasses import dataclass

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from residual_room.checker.claims import count_ctas, normalize
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
        for word in dict.fromkeys(m.lower() for m in config.avoid_regex.findall(normalize(seg.text))):
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


# Words that say nothing about which explainer is meant.
_POINTER_NOISE = {"read", "next", "our", "the", "a", "an", "on", "of", "and", "to", "in", "for", "how",
                  "why", "explainer", "series", "end", "et", "al"}


def check_read_next(segments: list[Segment], config: PublicationConfig) -> list[RuleHit]:
    """A read-next pointer must lead to an explainer that exists (P1 otherwise).

    A lookup, not a judgment, so it's code: the LLM judge won't call a missing
    explainer a contradiction ("no passage says it doesn't exist"). The pointer
    matches if it names an explainer id, or shares a topic word with an
    explainer in the series it names (any series if it names none).
    """
    explainers = config.explainers
    series_words = {w for e in explainers.values() for w in _words(e.series)}
    hits = []
    for i, seg in enumerate(segments):
        if seg.exempt_type != "read_next":
            continue
        text = normalize(seg.text).lower()
        if any(eid in text for eid in explainers):
            continue
        named = {e.series for e in explainers.values() if e.series.lower() in text}
        candidates = [e for e in explainers.values() if not named or e.series in named]
        words = _words(text) - series_words
        if not any(words & _words(e.topic) for e in candidates):
            where = f" in {', '.join(sorted(named))}" if named else ""
            hits.append(RuleHit("P1", f"read-next points to an explainer that doesn't exist{where}", i, "int-series"))
    return hits


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())) - _POINTER_NOISE


def run_deterministic(
    text: str, segments: list[Segment], lookup: StatusLookup, config: PublicationConfig | None = None
) -> list[RuleHit]:
    config = config or get_config()
    return (
        check_s1(segments, config)
        + check_s3(text, config)
        + check_citations(segments, lookup)
        + check_read_next(segments, config)
    )

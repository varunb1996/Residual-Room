"""Entry point: text (or pre-segmented generator output) -> CheckReport.

  1. split raw text into sentences (generator output arrives already segmented)
  2. with a judge: label candidates as claims / statements / exempt (extract.py)
  3. deterministic rules: S1, S3, P1 citation, P5 cited (deterministic.py)
  4. with a judge: P1 support, P2, P3, P4, P5 semantic, P6, S2, S4 (judge.py)
  5. document rule: correction notice for dated issues (document.py)

Without a judge only steps 1, 3 and 5 run (used by fast tests and CI).
"""

from datetime import date

from residual_room.checker.claims import split_segments
from residual_room.checker.deterministic import StatusLookup, run_deterministic
from residual_room.checker.document import check_correction_notice
from residual_room.checker.extract import extract
from residual_room.checker.judge import judge_segments
from residual_room.checker.types import CheckReport, RuleHit, Segment
from residual_room.config import PublicationConfig, get_config
from residual_room.llm.client import LLMError


def check(
    text: str,
    lookup: StatusLookup,
    segments: list[Segment] | None = None,
    config: PublicationConfig | None = None,
    *,
    judge=None,
    retriever=None,
    reader_segment: str | None = None,
    issue_date: date | None = None,
) -> CheckReport:
    config = config or get_config()
    reader_segment = reader_segment or config.default_segment
    segments = segments if segments is not None else split_segments(text, config)
    hits: list[RuleHit] = []

    if judge is not None:
        try:
            segments = extract(segments, judge)
        except LLMError as exc:
            hits.append(RuleHit("JUDGE", f"claim extraction failed: {exc}"))

    hits += run_deterministic(text, segments, lookup, config)
    if judge is not None:
        hits += judge_segments(segments, judge, retriever, reader_segment)
    hits += check_correction_notice(text, issue_date, config)
    return CheckReport(segments, hits)


def default_checker(corpus_version: str = "corpus-v1"):
    """Everything check() needs for a real run: status lookup, judge, retriever."""
    from residual_room.checker.evidence import CorpusRetriever
    from residual_room.corpus.embed import LocalEmbedder
    from residual_room.corpus.manifest import load_manifest
    from residual_room.db.session import get_engine
    from residual_room.llm.client import load_role

    engine, manifest = get_engine(), load_manifest()
    retriever = CorpusRetriever(
        engine, LocalEmbedder(), corpus_version, manifest, get_config().operational_sources
    )
    return {
        "lookup": StatusLookup.from_corpus(engine, corpus_version, manifest),
        "judge": load_role("judge"),
        "retriever": retriever,
    }

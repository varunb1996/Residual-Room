"""Deterministic rules against the real eval texts.

Only rules this half can decide are asserted. Verdicts that need the judge
(e.g. POL-01's P3, or P1 on uncited candidates) are left to the eval harness.
"""

import pytest
from eval_cases import case

from residual_room.checker.deterministic import StatusLookup
from residual_room.checker.run import check
from residual_room.checker.types import RuleHit, Segment
from residual_room.corpus.manifest import load_manifest

CLEAN = [f"CLN-{i:02d}" for i in range(1, 9)]


@pytest.fixture(scope="module")
def lookup():
    return StatusLookup.from_manifest(load_manifest())


def rules(case_id, lookup):
    return check(case(case_id)["text"], lookup).rules


def test_pol08_hype_words_warn_s1(lookup):
    report = check(case("POL-08")["text"], lookup)
    assert report.rules == ["S1"] and report.verdict == "warn"
    assert {h.evidence for h in report.hits} == {"game-changer", "magic"}


def test_pol07_no_longer_s1(lookup):
    assert "S1" not in rules("POL-07", lookup)


def test_pol10_three_ctas_warn_s3(lookup):
    report = check(case("POL-10")["text"], lookup)
    assert report.rules == ["S3"] and report.verdict == "warn"
    assert report.hits[0].segment is None  # document-level


@pytest.mark.parametrize("case_id", CLEAN)
def test_clean_cases_trip_no_deterministic_rule(case_id, lookup):
    assert rules(case_id, lookup) == []


def test_stl07_citing_superseded_explainer_blocks_p5(lookup):
    report = check(case("STL-07")["text"], lookup)
    assert report.rules == ["P5"] and report.verdict == "block"
    assert report.hits[0].evidence == "exp-003-v1"


@pytest.mark.parametrize("case_id", ["STL-08", "CLN-05", "STL-05"])
def test_citing_current_sources_is_not_p5(case_id, lookup):
    assert "P5" not in rules(case_id, lookup)


def test_chunk_ids_and_aliases_resolve(lookup):
    assert lookup.status("int-schedule-v0#0000") == "superseded"
    assert lookup.status("ext-litm-for-t3") == "current"


def test_unknown_citation_is_p1(lookup):
    report = check("Chunks of 512 tokens work best [cite:ext-made-up].", lookup)
    assert report.rules == ["P1"] and report.hits[0].evidence == "ext-made-up"


def test_uncited_claim_is_p1_but_uncited_candidate_is_left_to_the_judge(lookup):
    text = "Chunk sizes of 512 tokens work best."
    assert check(text, lookup).rules == []  # candidate: no verdict without judgment
    claim = [Segment(text, "claim")]
    assert check(text, lookup, segments=claim).rules == ["P1"]


@pytest.mark.db
def test_corpus_lookup_agrees_with_manifest(lookup):
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from residual_room.db.models import CorpusVersion
    from residual_room.db.session import get_engine

    engine = get_engine()
    with Session(engine) as s:
        if s.get(CorpusVersion, "corpus-v1") is None:
            pytest.skip("corpus-v1 not ingested: run `make ingest`")
    from_db = StatusLookup.from_corpus(engine, "corpus-v1", load_manifest())
    assert from_db.statuses == lookup.statuses


def test_per_segment_verdicts():
    from residual_room.checker.types import CheckReport

    report = CheckReport(
        [Segment("a", "candidate"), Segment("b", "candidate")],
        [RuleHit("S1", "hype", 0), RuleHit("P5", "stale", 1)],
    )
    assert report.segment_verdict(0) == "warn"
    assert report.segment_verdict(1) == "block"
    assert report.verdict == "block"


def test_unicode_hyphens_do_not_slip_past_word_lists(lookup):
    # U+2011 non-breaking hyphen, as generators often write it.
    report = check("Reranking is a game\u2011changer.", lookup)
    assert report.rules == ["S1"]
    # U+00A0 non-breaking space inside a CTA.
    assert check("Read the\u00a0explainer.", lookup).segments[0].exempt_type == "cta"


@pytest.mark.parametrize(
    "pointer, ok",
    [
        ("Read next: our First Principles explainer on LLM-as-Judge.", False),  # piece 42: no such explainer
        ("Read next: our First Principles explainer on RAG.", True),  # CLN-08
        ("Read next: our Paper to Practice explainer on Lost in the Middle.", True),
        ("Read next: exp-006.", True),
        ("Read next: our Failure Modes explainer on embeddings.", False),  # topic exists, wrong series
    ],
)
def test_read_next_must_point_to_an_existing_explainer(pointer, ok, lookup):
    report = check(pointer, lookup)
    assert report.segments[0].exempt_type == "read_next"
    assert (report.rules == []) == ok
    if not ok:
        assert report.hits[0].rule == "P1" and report.hits[0].evidence == "int-series"

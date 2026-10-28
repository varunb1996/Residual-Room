import math
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import delete
from sqlalchemy.orm import Session

from residual_room.analytics.insights import MIN_N, detect, two_proportion_test
from residual_room.analytics.metrics import Row, campaign_rows, replace_rows
from residual_room.analytics.seed import seeded_rows
from residual_room.db.models import MetricRow
from residual_room.db.session import get_engine


def test_z_test_matches_a_known_value():
    # 60/1000 vs 40/1000: pooled p = 0.05, se = sqrt(0.05*0.95*2/1000) = 0.009747,
    # z = 0.02/0.009747 = 2.052, two-sided p = 0.040
    z, p = two_proportion_test(60, 1000, 40, 1000)
    assert math.isclose(z, 2.052, abs_tol=0.001) and math.isclose(p, 0.040, abs_tol=0.001)
    assert two_proportion_test(5, 100, 5, 100)[1] == pytest.approx(1.0)


def test_campaign_rows_add_window_totals_and_rates():
    w = datetime(2026, 11, 2).date()
    rows = campaign_rows({(w, "email", "practitioner"): {"deliveries": 200, "clicks": 10},
                          (w + timedelta(weeks=1), "email", "practitioner"): {"deliveries": 100, "clicks": 20}})
    total = {r.metric: r for r in rows if r.week is None}
    assert total["deliveries"].value == 300 and total["clicks"].value == 30
    assert total["ctr"].value == pytest.approx(0.1) and total["ctr"].n == 300


def test_seed_is_deterministic():
    a, b = seeded_rows(), seeded_rows()
    assert [(r.metric, r.value) for r in a] == [(r.metric, r.value) for r in b]


@pytest.fixture
def seeded():
    """Seeded rows stored under the real `seeded` dataset label, cleaned up after."""
    with Session(get_engine()) as s, s.begin():
        replace_rows(s, "seeded", seeded_rows())
        insights = detect(s, "seeded")
    yield insights
    with Session(get_engine()) as s, s.begin():  # leave the app's seeded dataset as the app expects it
        from residual_room.analytics.service import refresh

        refresh(s, "seeded")


@pytest.mark.db
def test_rules_find_every_planted_pattern(seeded):
    found = {(i["rule"], i["segment"], i["channel"]) for i in seeded if i["status"] == "insight"}
    assert ("channel_gap", "practitioner", "telegram vs email") in found  # pattern 1
    assert ("trend_drop", "transitioning", "email") in found  # pattern 2
    assert ("unsubscribe_spike", "practitioner", "all") in found  # pattern 3
    tiny = {(i["segment"], i["channel"]) for i in seeded if i["status"] == "not_enough_data"}
    assert tiny == {("practitioner", "whatsapp"), ("transitioning", "whatsapp")}  # pattern 4


@pytest.mark.db
def test_every_insight_cites_rows_that_exist(seeded):
    with Session(get_engine()) as s:
        for i in seeded:
            for rid in i["row_ids"]:
                row = s.get(MetricRow, rid)
                assert row is not None and row.dataset == "seeded"


@pytest.mark.db
def test_small_samples_never_produce_an_insight():
    w = datetime(2026, 11, 2, tzinfo=UTC).date()
    rows = campaign_rows({(w, "email", "practitioner"): {"deliveries": 40, "clicks": 20},
                          (w, "telegram", "practitioner"): {"deliveries": 30, "clicks": 1}})
    with Session(get_engine()) as s:
        replace_rows(s, "real", rows)  # rolled back below: the real dataset is untouched
        out = detect(s, "real")
        s.rollback()
    assert {i["status"] for i in out} == {"not_enough_data"}
    assert all(f"need {MIN_N}" in i["statement"] for i in out)


# ---------- narration ----------

from residual_room.analytics.narrate import MetricRetriever, check_narration, number_problems  # noqa: E402
from residual_room.generation.schema import GenOutput  # noqa: E402


def _rows():
    w = datetime(2026, 11, 30).date()
    return {
        1: MetricRow(id=1, dataset="seeded", week=None, channel="telegram", segment="practitioner",
                     metric="ctr", value=0.0932, n=2028, derived=False),
        2: MetricRow(id=2, dataset="seeded", week=None, channel="email", segment="practitioner",
                     metric="ctr", value=0.0470, n=3194, derived=False),
        3: MetricRow(id=3, dataset="seeded", week=None, channel="telegram/email", segment="practitioner",
                     metric="ctr_ratio", value=2.0, n=None, derived=True),
        4: MetricRow(id=4, dataset="seeded", week=w, channel="all", segment="practitioner",
                     metric="unsubscribe_rate", value=0.046, n=1079, derived=False),
    }


def _out(*segments):
    return GenOutput.model_validate({"segments": list(segments)})


def test_numbers_must_come_from_cited_rows():
    rows = _rows()
    good = _out({"type": "claim", "text": "In seeded data, practitioners clicked Telegram 9.3% of the time vs 4.7% "
                 "for email, about 2x.", "chunk_ids": ["m:1", "m:2", "m:3"]})
    assert number_problems(good, rows, "seeded") == []
    bad = _out({"type": "claim", "text": "In seeded data, Telegram reached 12.5% for practitioners.",
                "chunk_ids": ["m:1"]})
    assert any("12.5%" in p for p in number_problems(bad, rows, "seeded"))
    wrong_row = _out({"type": "claim", "text": "Seeded: 4.7% for email.", "chunk_ids": ["m:1"]})
    assert number_problems(wrong_row, rows, "seeded")  # 4.7% is in m:2, not the cited m:1


def test_dates_and_unknown_rows_and_seeded_label():
    rows = _rows()
    dated = _out({"type": "claim", "text": "Seeded data: unsubscribes hit 4.6% in the week of 2026-11-30 (n=1079).",
                  "chunk_ids": ["m:4"]})
    assert number_problems(dated, rows, "seeded") == []
    unlabelled = _out({"type": "claim", "text": "Unsubscribes hit 4.6%.", "chunk_ids": ["m:4"]})
    assert any("seeded" in p for p in number_problems(unlabelled, rows, "seeded"))
    ghost = _out({"type": "claim", "text": "Seeded: 4.6%.", "chunk_ids": ["m:99"]})
    assert any("don't exist" in p for p in number_problems(ghost, rows, "seeded"))


def test_narration_goes_through_the_same_claim_checker():
    from fakes import FakeJudge

    def script(system, user):
        if "Label these sentences" in user:
            return {"sentences": []}
        if "Check ONE claim" in system:
            return {"supported": True, "numbers": "match", "evidence_chunk": "m:1", "reason": "ok"}
        if "carries no citation" in system:
            return {"contradicts_current": False, "matches_superseded": False, "reason": "ok"}
        return {"hits": []}

    judge = FakeJudge(script)
    out = _out({"type": "claim", "text": "In seeded data, Telegram reached 9.3% for practitioners.",
                "chunk_ids": ["m:1"]})
    report = check_narration(out, "seeded", _rows(), judge)
    assert report.verdict == "pass" and report.support == {0: "m:1"}
    cited_prompt = next(u for s, u in judge.calls if "Check ONE claim" in s)
    assert "[m:1] seeded" in cited_prompt and "ctr = 0.0932 (9.3%)" in cited_prompt

    unlabelled = _out({"type": "claim", "text": "Telegram reached 9.3% for practitioners.", "chunk_ids": ["m:1"]})
    assert check_narration(unlabelled, "seeded", _rows(), judge).verdict == "block"


def test_metric_retriever_returns_rows_as_passages():
    r = MetricRetriever(_rows())
    (p,) = r.for_citation("m:3", "q")
    assert p.chunk_id == "m:3" and "ctr_ratio = 2 " in p.text + " " and "%" not in p.text
    assert r.for_citation("m:99", "q") == [] and len(r.across_corpus("q")) == 4

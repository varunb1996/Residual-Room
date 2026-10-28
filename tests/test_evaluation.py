from datetime import date

from residual_room.evaluation.cases import HARD_CASES, load_cases
from residual_room.evaluation.metrics import CaseResult, Ratio, rule_score, summarize
from residual_room.evaluation.report import render_markdown
from residual_room.evaluation.split import build_split, load_split, stratum


def result(cid, exp_v, exp_r, got_v, got_r, split="test", hard=False):
    return CaseResult(cid, split, hard, exp_v, exp_r, got_v, got_r)


# ---------- cases ----------

def test_fifty_cases_with_defaults_and_fixture():
    cases = load_cases()
    assert len(cases) == 50
    assert cases["STL-01"].segment == "practitioner"  # no segment in the yaml -> config default
    stl10 = cases["STL-10"]
    assert stl10.issue_date == date(2026, 11, 12) and "Failure Modes this week" in stl10.text
    assert {c for c in cases if cases[c].hard} == set(HARD_CASES)


def test_committable_flag_follows_cited_sources():
    cases = load_cases()
    assert not cases["GRD-01"].committable  # MT-Bench: local fetch only
    assert cases["GRD-14"].committable  # Lost in the Middle (TACL, CC BY)
    assert cases["POL-01"].committable  # cites nothing


# ---------- split ----------

def test_split_is_frozen_and_matches_the_builder():
    assert load_split() == build_split(load_cases())


def test_split_properties():
    cases, split = load_cases(), load_split()
    dev, test = set(split["dev"]), set(split["test"])
    assert not dev & test and dev | test == set(cases)
    assert set(HARD_CASES) <= test
    test_strata = {stratum(cases[c]) for c in test}
    all_strata = {stratum(c) for c in cases.values()}
    assert test_strata == all_strata  # every rule is measured on test


# ---------- metrics ----------

def test_lenient_vs_strict_and_missing_vs_extra():
    exact = result("A", "block", ["P2"], "block", ["P2"])
    extra = result("B", "block", ["P2"], "block", ["P1", "P2"])
    missing = result("C", "block", ["P2", "P5"], "block", ["P5"])
    wrong_verdict = result("D", "warn", ["S1"], "block", ["P1", "S1"])
    assert exact.strict_ok and extra.lenient_ok and not extra.strict_ok
    assert extra.extra == ["P1"] and missing.missing == ["P2"] and not missing.lenient_ok
    assert not wrong_verdict.lenient_ok


def test_extra_rules_count_as_false_positives_for_precision():
    rs = [
        result("A", "block", ["P2"], "block", ["P1", "P2"]),
        result("B", "block", ["P1"], "block", ["P1"]),
        result("C", "block", ["P1"], "pass", []),
    ]
    p1 = rule_score("P1", rs)
    assert (p1.tp, p1.fp, p1.fn) == (1, 1, 1)
    assert str(p1.precision) == "1/2 (50%) ⚠ small n"


def test_summary_counts():
    rs = [
        result("A", "pass", [], "pass", []),
        result("B", "pass", [], "warn", ["S2"]),
        result("C", "block", ["P2"], "block", ["P2"], hard=True),
        result("D", "block", ["P1"], "block", ["JUDGE"]),
    ]
    s = summarize(rs)
    assert (s["lenient"].k, s["lenient"].n) == (2, 4)
    assert (s["pass_fp"].k, s["pass_fp"].n) == (1, 2)
    assert (s["hard"].k, s["hard"].n) == (1, 1)
    assert s["judge_errors"] == 1
    assert str(Ratio(0, 0)) == "n/a (0 cases)"


def test_report_lists_disagreements_and_extras():
    rs = [
        result("A", "block", ["P2"], "block", ["P1", "P2"], split="dev"),
        result("B", "pass", [], "warn", ["S2"]),
    ]
    rs[1].hits = [{"rule": "S2", "reason": "anthropomorphizes | 'agreed'", "evidence": None}]
    md = render_markdown(rs, {"started_at": "2026-10-28T12:00:00", "split": "all"})
    assert "## Disagreements with the expected label (1)" in md
    assert "| B | test | pass [] | warn ['S2'] |" in md
    assert "anthropomorphizes / 'agreed'" in md  # pipes escaped
    assert "## Correct, but with extra rules (1)" in md and "| A | dev |" in md
    assert "| dev | test |" in md  # per-split columns in the summary

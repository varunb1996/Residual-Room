from fakes import FakeJudge, FakeRetriever

from residual_room.checker.evidence import Passage
from residual_room.checker.judge import judge_segments
from residual_room.checker.types import Segment
from residual_room.llm.client import LLMError

TABLE3 = Passage("ext-mtbench-paper#0016", "ext-mtbench-paper", "current", "MT-Bench › 3.3",
                 "[TABLE] Table 3 | Failure rate | 91.3% | 91.3% | 8.7% |")
OLD_SCHEDULE = Passage("int-schedule-v0#0000", "int-schedule-v0", "superseded", "Schedule v0",
                       "The newsletter goes out weekly on Tuesdays at 9:00 AM IST.")
SERIES = Passage("int-series#0000", "int-series", "current", "Series",
                 "| exp-003 | Why retrieval fails | Failure Modes |")


def route(cited=None, uncited=None, policy=None):
    """A fake judge that answers by which prompt it was given."""
    def script(system, user):
        if "Check ONE claim" in system:
            return cited
        if "carries no citation" in system:
            return uncited
        return policy or {"hits": []}
    return script


def test_cited_number_mismatch_is_p2_with_evidence_and_reason():
    judge = FakeJudge(route(cited={"supported": True, "numbers": "mismatch",
                                   "evidence_chunk": "ext-mtbench-paper#0016", "reason": "91.3% is Claude-v1's rate"}))
    retriever = FakeRetriever([TABLE3])
    seg = Segment("GPT-4 fell for the attack 91.3% of the time.", "claim", citations=("ext-mtbench-paper",))
    (hit,) = judge_segments([seg], judge, retriever, "practitioner")
    assert (hit.rule, hit.segment, hit.evidence) == ("P2", 0, "ext-mtbench-paper#0016")
    assert "Claude-v1" in hit.reason
    assert ("cite", "ext-mtbench-paper") in retriever.calls
    assert "Table 3" in judge.calls[0][1]  # the evidence text reached the judge


def test_unsupported_cited_claim_is_p1_and_unknown_evidence_id_is_replaced():
    judge = FakeJudge(route(cited={"supported": False, "numbers": "none",
                                   "evidence_chunk": "made-up#9", "reason": "wrong source"}))
    seg = Segment("Models use the start and end of context best.", "claim", citations=("ext-mtbench-paper",))
    (hit,) = judge_segments([seg], judge, FakeRetriever([TABLE3]), "practitioner")
    assert hit.rule == "P1" and hit.evidence == "ext-mtbench-paper#0016"


def test_uncited_stale_claim_is_p5_plus_p2_when_a_date_conflicts():
    judge = FakeJudge(route(uncited={"relation": "matches_superseded", "conflict_is_numeric": True,
                                     "evidence_chunk": "int-schedule-v0#0000", "reason": "Tuesday is v0"}))
    seg = Segment("Issue 1 arrives on Tuesday, 3 November.", "claim")
    retriever = FakeRetriever([OLD_SCHEDULE])
    hits = judge_segments([seg], judge, retriever, "practitioner")
    assert sorted(h.rule for h in hits) == ["P2", "P5"]
    assert retriever.calls == [("all", seg.text)]  # whole corpus, superseded included


def test_framing_contradicting_series_is_p1_but_consistent_framing_passes():
    contradicts = FakeJudge(route(uncited={"relation": "contradicts_current", "conflict_is_numeric": False,
                                           "evidence_chunk": "int-series#0000", "reason": "Failure Modes"}))
    seg = Segment("This week in Paper to Practice: why retrieval fails.", "exempt", "framing")
    (hit,) = judge_segments([seg], contradicts, FakeRetriever([SERIES]), "practitioner")
    assert hit.rule == "P1"
    consistent = FakeJudge(route(uncited={"relation": "consistent", "reason": "ok"}))
    assert judge_segments([seg], consistent, FakeRetriever([SERIES]), "practitioner") == []


def test_policy_hits_mapped_and_invented_rules_dropped():
    judge = FakeJudge(route(policy={"hits": [
        {"index": 0, "rule": "P3", "reason": "job promise"},
        {"index": 0, "rule": "S4", "reason": "not applicable to practitioners"},
        {"index": 7, "rule": "P4", "reason": "index out of range"},
    ]}))
    seg = Segment("You'll be job-ready in 30 days.", "statement")
    hits = judge_segments([seg], judge, FakeRetriever([]), "practitioner")
    assert [(h.rule, h.segment) for h in hits] == [("P3", 0)]
    assert "S4 does not apply" in judge.calls[-1][0]


def test_s4_only_offered_for_transitioning_readers():
    judge = FakeJudge(route())
    seg = [Segment("Add a cross-encoder reranker.", "statement")]
    judge_segments(seg, judge, FakeRetriever([]), "transitioning")
    assert "S4 —" in judge.calls[-1][0]
    judge_segments(seg, judge, FakeRetriever([]), "practitioner")
    assert "S4 —" not in judge.calls[-1][0]


def test_greetings_ctas_signoffs_are_not_sent_to_the_judge():
    judge = FakeJudge(route())
    segs = [Segment("Hi there,", "exempt", "greeting"), Segment("Read the explainer.", "exempt", "cta")]
    assert judge_segments(segs, judge, FakeRetriever([]), "practitioner") == []
    assert judge.calls == []


def test_judge_failure_blocks_instead_of_passing():
    class Broken:
        model, max_concurrency = "broken", 1

        def ask_json(self, *a):
            raise LLMError("HTTP 503")

    seg = Segment("A cited fact.", "claim", citations=("ext-mtbench-paper",))
    hits = judge_segments([seg], Broken(), FakeRetriever([TABLE3]), "practitioner")
    assert {h.rule for h in hits} == {"JUDGE"} and all(h.severity == "block" for h in hits)

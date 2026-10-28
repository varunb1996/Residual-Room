from eval_cases import case
from fakes import FakeJudge, candidate_indices

from residual_room.checker.claims import split_segments
from residual_room.checker.extract import extract
from residual_room.checker.policy import policy_rules


def test_policy_rules_are_read_from_the_policy_doc():
    rules = policy_rules()
    assert sorted(rules) == ["P1", "P2", "P3", "P4", "P5", "P6", "S1", "S2", "S3", "S4"]
    assert "technical or operational factual claim" in rules["P1"]  # v1.1 wording
    assert "Violation:" in rules["P2"] and "128K" in rules["P2"]
    assert "Allowed:" in rules["P4"]


def test_only_candidates_are_sent_and_labels_applied():
    segs = split_segments(case("CLN-02")["text"])
    # [0] "Failure Modes this week: ..." [1] cited sentence with two numbers, [2] cta, [3] sign-off

    def script(system, user):
        asked = candidate_indices(user)
        assert asked == [0, 1]
        return {"sentences": [
            {"index": 0, "kind": "exempt", "exempt_type": "framing"},
            {"index": 1, "kind": "claim", "claims": [
                "Swapping the answer order, Claude-v1 kept its verdict only 23.8% of the time.",
                "Swapping the answer order, GPT-4 kept its verdict 65.0% of the time.",
            ]},
        ]}

    out = extract(segs, FakeJudge(script))
    assert [(s.kind, s.exempt_type) for s in out] == [
        ("exempt", "framing"), ("claim", None), ("claim", None), ("exempt", "cta"), ("exempt", "sign_off"),
    ]
    # Atomic claims keep the sentence's citation and point back at it.
    assert all(s.citations == ("ext-mtbench-paper",) and s.sentence == 1 for s in out[1:3])


def test_cited_sentence_is_never_exempt_and_skipped_index_becomes_claim():
    segs = split_segments("Some framing. A cited fact [cite:ext-litm-paper].")
    out = extract(segs, FakeJudge(lambda s, u: {"sentences": [
        {"index": 1, "kind": "exempt", "exempt_type": "framing"},  # wrong: it's cited
    ]}))  # index 0 skipped entirely
    assert [s.kind for s in out] == ["claim", "claim"]


def test_cta_label_only_stands_for_the_brand_guides_cta_types():
    segs = split_segments("Upgrade to Residual Room Pro for early access to deep dives.")
    out = extract(segs, FakeJudge(lambda s, u: {"sentences": [
        {"index": 0, "kind": "exempt", "exempt_type": "cta"},
    ]}))
    assert out[0].kind == "statement"  # not an allowed CTA, so it gets checked


def test_no_candidates_means_no_llm_call():
    judge = FakeJudge(lambda s, u: {})
    segs = split_segments("Hi there,\nRead the explainer.\n— Residual Room")
    assert extract(segs, judge) == segs and judge.calls == []

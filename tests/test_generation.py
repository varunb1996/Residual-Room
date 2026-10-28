from fakes import FakeJudge

from residual_room.corpus.retrieve import Hit
from residual_room.generation.generate import generate
from residual_room.generation.schema import GenOutput, problems, render_text, to_check_segments
from residual_room.config import get_config

HITS = [Hit("ext-mtbench-paper#0015", "ext-mtbench-paper", "current", "MT-Bench › 3.3", "Table 2 …", 0.1)]


def seg(type_, text, exempt_type=None, chunk_ids=None):
    return {"type": type_, "text": text, "exempt_type": exempt_type, "chunk_ids": chunk_ids}


GOOD = {"segments": [
    seg("exempt", "Failure Modes this week: your judge flips.", "framing"),
    seg("claim", "Swap the answer order and Claude-v1 kept its verdict 23.8% of the time.",
        chunk_ids=["ext-mtbench-paper#0015"]),
    seg("exempt", "Read the explainer.", "cta"),
    seg("exempt", "— Residual Room", "sign_off"),
]}


def check(out, segment="practitioner", kind="newsletter"):
    return problems(GenOutput.model_validate(out), {"ext-mtbench-paper#0015"}, segment, kind, get_config())


def test_good_newsletter_has_no_problems():
    assert check(GOOD) == []


def test_structural_problems_are_found():
    bad = {"segments": [
        seg("claim", "An uncited fact."),
        seg("claim", "An invented id.", chunk_ids=["made-up#0001"]),
        seg("exempt", "Upgrade to Pro.", "cta"),
    ]}
    found = " | ".join(check(bad, segment="transitioning"))
    assert "no chunk_ids" in found and "not provided" in found
    assert "allowed calls to action" in found
    assert "end with the sign-off" in found and "read_next" in found


def test_render_text_and_checker_segments():
    out = GenOutput.model_validate(GOOD)
    assert render_text(out).splitlines()[1].endswith("23.8% of the time. [cite:ext-mtbench-paper#0015]")
    segs = to_check_segments(out)
    assert [(s.kind, s.exempt_type) for s in segs] == [
        ("exempt", "framing"), ("claim", None), ("exempt", "cta"), ("exempt", "sign_off")]
    assert segs[1].citations == ("ext-mtbench-paper#0015",)


def test_invalid_draft_gets_one_corrective_retry():
    bad = {"segments": GOOD["segments"][:1] + [seg("claim", "Uncited fact.")] + GOOD["segments"][2:]}
    replies = iter([bad, GOOD])
    llm = FakeJudge(lambda s, u: next(replies))
    g = generate("exp-004", "practitioner", "newsletter", HITS, llm, "corpus-v1")
    assert g.problems == [] and g.meta["attempts"] == 2
    assert "no chunk_ids" in llm.calls[1][1]  # the retry was told what was wrong
    assert "[ext-mtbench-paper#0015]" in llm.calls[0][1]  # passages reached the prompt


def test_still_invalid_after_retry_is_kept_with_problems_recorded():
    bad = {"segments": [seg("claim", "Uncited fact.")]}
    g = generate("exp-004", "practitioner", "newsletter", HITS, FakeJudge(lambda s, u: bad), "corpus-v1")
    assert g.problems and g.meta["problems"] == g.problems
    assert g.text == "Uncited fact."  # kept, so the checker blocks it on P1


def test_prompt_carries_the_frozen_docs_and_segment_tone():
    llm = FakeJudge(lambda s, u: GOOD)
    generate("exp-004", "transitioning", "newsletter", HITS, llm, "corpus-v1")
    system, user = llm.calls[0]
    assert "Brand Guide" in system and "Content Policy" in system
    assert "Define every technical term" in user and "LLM-as-judge biases" in user

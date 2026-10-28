"""The judgment rules, answered by the judge model (never the generator).

Three kinds of call:
  cited     each cited claim vs passages from what it cites: P1 support, P2 numbers
  uncited   uncited claims, framing and statements vs the whole corpus, superseded
            included: repeats stale text -> P5, contradicts current text -> P2 for
            a number/date/time/day, else P1
  policy    one call per document: P3, P4, P6, S2, and S4 for transitioning readers

Evidence is retrieved first (sequentially), then the LLM calls run in parallel
within the provider's limits. A call that fails becomes a blocking JUDGE hit.
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from pydantic import BaseModel

from residual_room.checker.evidence import Passage
from residual_room.checker.policy import rules_text
from residual_room.checker.types import RuleHit, Segment
from residual_room.llm.client import LLMError

CITED_SYSTEM = """You are a strict fact-checker for an AI-education newsletter. Check ONE claim against evidence passages taken from the source it cites.

Policy:
{rules}

Decide:
- "supported": true only if the passages state or directly imply everything the claim says (judge numbers separately, below). Overstatements are not supported (e.g. "proved" when the source says it cannot confirm; "entirely" when the source reports a drop). A claim that may be true but that these passages do not contain is not supported.
- "numbers": "none" if the claim has no figures or dates; "match" only if every figure matches the passages in value AND in what it describes (same model, same comparison, same setting); otherwise "mismatch". Leaving out detail the passages give (a year, a unit clear from context, extra decimal places) is not a mismatch, as long as what the claim does state agrees with them. If the only problem is a number, set "supported" true and "numbers" "mismatch".
- "evidence_chunk": the id of the passage you relied on most (the id in square brackets).

Reply with only JSON: {{"supported": true|false, "numbers": "none"|"match"|"mismatch", "evidence_chunk": "<id>", "reason": "<one sentence>"}}"""

UNCITED_SYSTEM = """You check a sentence that carries no citation against passages from the publication's own corpus (its schedule, series list, subscription rules and policies, and the sources it covers). Passages with status "superseded" are outdated versions kept on record.

Answer two separate questions; both can be true at once:
- "contradicts_current": does a current passage clearly say otherwise?
- "matches_superseded": does it repeat something a superseded passage says, where the current passages say something different?
Set "conflict_is_numeric" true if a conflict involves a number, date, time or day. Report a conflict only when a passage clearly states it; if the passages agree with the sentence or don't address it, both answers are false.

Reply with only JSON: {"contradicts_current": true|false, "matches_superseded": true|false, "conflict_is_numeric": true|false, "evidence_chunk": "<id or null>", "reason": "<one sentence>"}"""

POLICY_SYSTEM = """You check sentences from an AI-education newsletter against these content-policy rules:
{rules}

The reader segment is "{segment}". {segment_note}
Sentences the publication clearly marks as its own opinion ("We think ...", "We're not convinced ...") may argue and disagree; that is allowed.
Explicitly framed analogies ("like a reader skimming a long doc", "think of it as ...") are not anthropomorphizing (S2): the brand guide asks for analogies to build intuition.
Report only clear violations, one entry per sentence and rule.

Reply with only JSON: {{"hits": [{{"index": <int>, "rule": "<rule id>", "reason": "<one sentence>"}}]}}  (an empty list if none)"""

# Sentences per policy call. The judge reasons about every sentence before its
# JSON; on a 17-sentence piece that ran past the output limit.
POLICY_BATCH = 8

SEGMENT_NOTES = {
    "transitioning": "These readers are engineers new to AI: apply S4 to any technical term used without a definition.",
    "practitioner": "These readers already build with AI: technical jargon is fine, so S4 does not apply.",
}


class CitedVerdict(BaseModel):
    supported: bool
    numbers: Literal["none", "match", "mismatch"]
    evidence_chunk: str | None = None
    reason: str


class UncitedVerdict(BaseModel):
    # Two booleans, not one relation: "Issue 1 arrives on Tuesday, 3 November"
    # both contradicts the current schedule and echoes the superseded one (STL-03).
    contradicts_current: bool
    matches_superseded: bool
    conflict_is_numeric: bool = False
    evidence_chunk: str | None = None
    reason: str


class PolicyHit(BaseModel):
    index: int
    rule: Literal["P3", "P4", "P6", "S2", "S4"]
    reason: str


class PolicyVerdict(BaseModel):
    hits: list[PolicyHit]


def judge_segments(
    segments: list[Segment], judge, retriever, reader_segment: str, support: dict[int, str] | None = None
) -> list[RuleHit]:
    """Judgment hits for the segments. If `support` is given, it is filled with
    {segment index: chunk id} for cited claims the judge found supported, so the
    review screen can link passing claims to their source too."""
    support = support if support is not None else {}
    jobs: list[Callable[[], list[RuleHit]]] = []
    for i, seg in enumerate(segments):
        if seg.kind == "claim" and seg.citations:
            passages = _dedupe(p for c in seg.citations for p in retriever.for_citation(c, seg.text))
            jobs.append(_guard(i, lambda i=i, s=seg, p=passages: _cited(i, s, p, judge, support)))
        elif seg.kind in ("claim", "statement") or seg.exempt_type == "framing":
            # (read_next is checked deterministically against the explainer map:
            # the judge won't treat a missing explainer as a contradiction.)
            passages = retriever.across_corpus(seg.text)
            jobs.append(_guard(i, lambda i=i, s=seg, p=passages: _uncited(i, s, p, judge)))
    units = _policy_units(segments)
    for k in range(0, len(units), POLICY_BATCH):
        batch = units[k : k + POLICY_BATCH]
        jobs.append(_guard(None, lambda b=batch: _policy(segments, b, reader_segment, judge)))

    with ThreadPoolExecutor(max_workers=max(1, judge.max_concurrency)) as pool:
        return [hit for hits in pool.map(lambda job: job(), jobs) for hit in hits]


def _cited(i: int, seg: Segment, passages: list[Passage], judge, support: dict[int, str]) -> list[RuleHit]:
    if not passages:
        return [RuleHit("P1", f"nothing found in the cited source(s) {', '.join(seg.citations)}", i)]
    user = f"Claim: {seg.text}\n\nEvidence passages:\n\n" + "\n\n".join(p.render() for p in passages)
    v = judge.ask_json(CITED_SYSTEM.format(rules=rules_text(["P1", "P2"])), user, CitedVerdict)
    evidence = _valid_id(v.evidence_chunk, passages)
    if v.supported and v.numbers != "mismatch" and evidence:
        support[i] = evidence
    hits = []
    if not v.supported:
        hits.append(RuleHit("P1", f"cited source does not support the claim: {v.reason}", i, evidence))
    if v.numbers == "mismatch":
        hits.append(RuleHit("P2", f"number does not match the source: {v.reason}", i, evidence))
    return hits


def _uncited(i: int, seg: Segment, passages: list[Passage], judge) -> list[RuleHit]:
    if not passages:
        return []
    user = f"Sentence: {seg.text}\n\nCorpus passages:\n\n" + "\n\n".join(p.render() for p in passages)
    v = judge.ask_json(UNCITED_SYSTEM, user, UncitedVerdict)
    evidence = _valid_id(v.evidence_chunk, passages)
    hits = []
    if v.matches_superseded:
        hits.append(RuleHit("P5", f"repeats a superseded source: {v.reason}", i, evidence))
    if v.contradicts_current:
        rule = "P2" if v.conflict_is_numeric else "P1"
        hits.append(RuleHit(rule, f"contradicts a current source: {v.reason}", i, evidence))
    return hits


def _policy_units(segments: list[Segment]) -> list[int]:
    """Segments the policy rules look at, one per original sentence.

    Whole original sentences, not atomic claims: splitting can separate a term
    from its definition ("position bias: some judges favor the first answer").
    Each sentence is sent once, under the index of its first segment."""
    units: dict[int | tuple, int] = {}
    for i, s in enumerate(segments):
        if s.exempt_type in ("greeting", "sign_off", "cta", "read_next"):
            continue
        key = ("s", s.sentence) if s.sentence is not None else i
        units.setdefault(key, i)
    return list(units.values())


def _policy(segments: list[Segment], first: list[int], reader_segment: str, judge) -> list[RuleHit]:
    rule_ids = ["P3", "P4", "P6", "S2"] + (["S4"] if reader_segment == "transitioning" else [])
    system = POLICY_SYSTEM.format(
        rules=rules_text(rule_ids), segment=reader_segment, segment_note=SEGMENT_NOTES.get(reader_segment, "")
    )
    user = "Sentences:\n" + "\n".join(
        f"[{i}] {segments[i].original or segments[i].text}" for i in first
    )
    v = judge.ask_json(system, user, PolicyVerdict)
    return [
        RuleHit(h.rule, h.reason, h.index)
        for h in v.hits
        if h.index in first and h.rule in rule_ids  # drop anything the model invents
    ]


def _guard(i: int | None, job: Callable[[], list[RuleHit]]) -> Callable[[], list[RuleHit]]:
    def run() -> list[RuleHit]:
        try:
            return job()
        except LLMError as exc:
            return [RuleHit("JUDGE", f"judge could not answer: {exc}", i)]

    return run


def _dedupe(passages) -> list[Passage]:
    return list({p.chunk_id: p for p in passages}.values())


def _valid_id(chunk_id: str | None, passages: list[Passage]) -> str | None:
    ids = [p.chunk_id for p in passages]
    return chunk_id if chunk_id in ids else (ids[0] if ids else None)

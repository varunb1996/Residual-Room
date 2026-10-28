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
- "numbers": "none" if the claim has no figures or dates; "match" only if every figure matches the passages in value AND in what it describes (same model, same comparison, same setting); otherwise "mismatch". If the only problem is a number, set "supported" true and "numbers" "mismatch".
- "evidence_chunk": the id of the passage you relied on most (the id in square brackets).

Reply with only JSON: {{"supported": true|false, "numbers": "none"|"match"|"mismatch", "evidence_chunk": "<id>", "reason": "<one sentence>"}}"""

UNCITED_SYSTEM = """You check a sentence that carries no citation against passages from the publication's own corpus (its schedule, series list, subscription rules and policies, and the sources it covers). Passages with status "superseded" are outdated versions kept on record.

Decide "relation":
- "consistent": the current passages agree with it, or it is a reasonable framing they support.
- "contradicts_current": a current passage clearly says otherwise.
- "matches_superseded": it repeats what a superseded passage says, where the current passages say something different.
- "unrelated": the passages do not address it.
Set "conflict_is_numeric" true if the conflict is about a number, date, time or day. Report a conflict only when a passage clearly states it.

Reply with only JSON: {"relation": "consistent"|"contradicts_current"|"matches_superseded"|"unrelated", "conflict_is_numeric": true|false, "evidence_chunk": "<id or null>", "reason": "<one sentence>"}"""

POLICY_SYSTEM = """You check sentences from an AI-education newsletter against these content-policy rules:
{rules}

The reader segment is "{segment}". {segment_note}
Sentences the publication clearly marks as its own opinion ("We think ...", "We're not convinced ...") may argue and disagree; that is allowed.
Report only clear violations, one entry per sentence and rule.

Reply with only JSON: {{"hits": [{{"index": <int>, "rule": "<rule id>", "reason": "<one sentence>"}}]}}  (an empty list if none)"""

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
    relation: Literal["consistent", "contradicts_current", "matches_superseded", "unrelated"]
    conflict_is_numeric: bool = False
    evidence_chunk: str | None = None
    reason: str


class PolicyHit(BaseModel):
    index: int
    rule: Literal["P3", "P4", "P6", "S2", "S4"]
    reason: str


class PolicyVerdict(BaseModel):
    hits: list[PolicyHit]


def judge_segments(segments: list[Segment], judge, retriever, reader_segment: str) -> list[RuleHit]:
    jobs: list[Callable[[], list[RuleHit]]] = []
    for i, seg in enumerate(segments):
        if seg.kind == "claim" and seg.citations:
            passages = _dedupe(p for c in seg.citations for p in retriever.for_citation(c, seg.text))
            jobs.append(_guard(i, lambda i=i, s=seg, p=passages: _cited(i, s, p, judge)))
        elif seg.kind in ("claim", "statement") or seg.exempt_type == "framing":
            passages = retriever.across_corpus(seg.text)
            jobs.append(_guard(i, lambda i=i, s=seg, p=passages: _uncited(i, s, p, judge)))
    policy_ids = [
        i for i, s in enumerate(segments)
        if s.exempt_type not in ("greeting", "sign_off", "cta", "read_next")
    ]
    if policy_ids:
        jobs.append(_guard(None, lambda: _policy(segments, policy_ids, reader_segment, judge)))

    with ThreadPoolExecutor(max_workers=max(1, judge.max_concurrency)) as pool:
        return [hit for hits in pool.map(lambda job: job(), jobs) for hit in hits]


def _cited(i: int, seg: Segment, passages: list[Passage], judge) -> list[RuleHit]:
    if not passages:
        return [RuleHit("P1", f"nothing found in the cited source(s) {', '.join(seg.citations)}", i)]
    user = f"Claim: {seg.text}\n\nEvidence passages:\n\n" + "\n\n".join(p.render() for p in passages)
    v = judge.ask_json(CITED_SYSTEM.format(rules=rules_text(["P1", "P2"])), user, CitedVerdict)
    evidence = _valid_id(v.evidence_chunk, passages)
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
    if v.relation == "matches_superseded":
        hits = [RuleHit("P5", f"repeats a superseded source: {v.reason}", i, evidence)]
        if v.conflict_is_numeric:
            hits.append(RuleHit("P2", f"date or figure conflicts with the current source: {v.reason}", i, evidence))
        return hits
    if v.relation == "contradicts_current":
        rule = "P2" if v.conflict_is_numeric else "P1"
        return [RuleHit(rule, f"contradicts a current source: {v.reason}", i, evidence)]
    return []


def _policy(segments: list[Segment], ids: list[int], reader_segment: str, judge) -> list[RuleHit]:
    rule_ids = ["P3", "P4", "P6", "S2"] + (["S4"] if reader_segment == "transitioning" else [])
    system = POLICY_SYSTEM.format(
        rules=rules_text(rule_ids), segment=reader_segment, segment_note=SEGMENT_NOTES.get(reader_segment, "")
    )
    user = "Sentences:\n" + "\n".join(f"[{i}] {segments[i].text}" for i in ids)
    v = judge.ask_json(system, user, PolicyVerdict)
    return [
        RuleHit(h.rule, h.reason, h.index)
        for h in v.hits
        if h.index in ids and h.rule in rule_ids  # drop anything the model invents
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

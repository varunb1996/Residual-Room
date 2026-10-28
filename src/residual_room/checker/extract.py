"""LLM claim extraction: decide what each `candidate` sentence is.

The judge model labels every candidate as a factual claim (split into atomic
claims), a non-factual statement, or an exempt type. Two guard rails stay in
code: a sentence that carries a citation is never exempt, and a candidate the
model skips is treated as a claim (so it still gets checked, never waved through).
A "cta" label only stands if the sentence matches one of the brand guide's CTA
types; anything else ("Upgrade to Pro") is a statement and gets checked.
"""

from typing import Literal

from pydantic import BaseModel

from residual_room.checker.claims import count_ctas
from residual_room.checker.types import Segment
from residual_room.llm.client import RoleLLM

SYSTEM = """You label sentences from a newsletter or explainer about AI engineering, for a fact-checking pipeline.

Labels:
- "claim": states a fact that could be true or false: about models, papers, benchmarks, numbers, how techniques work, or about the publication's own operations (schedule, pricing, subscription, which series a topic belongs to). Attributed views ("Some researchers argue X") are claims about what was argued. Split a sentence holding several facts into atomic claims, each a full standalone sentence.
- "statement": not a checkable fact: promises, predictions, recommendations or advice, hype, rhetorical questions, instructions to the reader.
- "exempt": needs no citation. Give exempt_type:
  - "framing": a series label, issue hook or subject line that introduces the piece ("This week in First Principles: how RAG works", "Paper to Practice: Lost in the Middle", "Subject: Why your judge flips its verdict").
  - "transition": a sentence that only moves between sections.
  - "opinion": clearly marked as the publication's own view ("We think ...", "We're not convinced ...").
  - "greeting", "sign_off", "cta" (read the explainer / subscribe / read the source / reply with a question), "read_next".

Reply with only JSON: {"sentences": [{"index": <int>, "kind": "claim"|"statement"|"exempt", "exempt_type": <string or null>, "claims": [<atomic claim sentences, only for kind "claim">]}]}
Label every index you are given, exactly once."""


class SentenceLabel(BaseModel):
    index: int
    kind: Literal["claim", "statement", "exempt"]
    exempt_type: Literal[
        "framing", "transition", "opinion", "greeting", "sign_off", "cta", "read_next"
    ] | None = None
    claims: list[str] = []


class Extraction(BaseModel):
    sentences: list[SentenceLabel]


def extract(segments: list[Segment], judge: RoleLLM) -> list[Segment]:
    candidates = [i for i, s in enumerate(segments) if s.kind == "candidate"]
    if not candidates:
        return segments
    numbered = "\n".join(f"[{i}] {s.text}" for i, s in enumerate(segments))
    asked = "\n".join(f"[{i}] {segments[i].text}" for i in candidates)
    user = f"Full text, for context:\n{numbered}\n\nLabel these sentences:\n{asked}"
    labels = {lab.index: lab for lab in judge.ask_json(SYSTEM, user, Extraction).sentences}

    out: list[Segment] = []
    for i, seg in enumerate(segments):
        if seg.kind != "candidate":
            out.append(seg)
            continue
        label = labels.get(i)
        kind = label.kind if label else "claim"
        if kind == "exempt" and seg.citations:
            kind = "claim"  # a cited sentence is making a claim, whatever the model says
        if kind == "exempt" and label.exempt_type == "cta" and not count_ctas(seg.text):
            kind = "statement"  # only the brand guide's four CTA types are exempt
        if kind == "claim":
            parts = (label.claims if label else []) or [seg.text]
            out += [Segment(p, "claim", citations=seg.citations, sentence=i) for p in parts]
        elif kind == "statement":
            out.append(Segment(seg.text, "statement", citations=seg.citations, sentence=i))
        else:
            out.append(Segment(seg.text, "exempt", label.exempt_type or "transition", sentence=i))
    return out

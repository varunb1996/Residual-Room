"""Shapes shared by the splitter, the rules and (later) the LLM judge."""

from dataclasses import dataclass, field
from typing import Literal

# claim:     a factual statement that needs a citation (labelled by the generator
#            or the LLM claim extractor, never guessed by patterns)
# statement: non-factual, needs no citation, but the judgment rules still apply
#            (promises P3, predictions P4, rumours P6, hype, advice)
# exempt:    greeting, sign_off, cta, read_next, transition, opinion, framing
#            (framing = series labels, hooks, subject lines: no citation, but
#            still checked for contradicting the internal docs)
# candidate: not recognized by pattern; the LLM decides what it is
SegmentKind = Literal["claim", "statement", "exempt", "candidate"]
Verdict = Literal["block", "warn", "pass"]


@dataclass(frozen=True)
class Segment:
    text: str
    kind: SegmentKind
    exempt_type: str | None = None
    citations: tuple[str, ...] = ()  # source or chunk ids, e.g. ext-litm-paper, ext-litm-paper#0007
    sentence: int | None = None  # index of the original sentence, when extraction split it


@dataclass(frozen=True)
class RuleHit:
    rule: str  # P1..P6 (hard), S1..S4 (soft), JUDGE (the judge could not answer)
    reason: str
    segment: int | None = None  # index into the segments; None = whole document
    evidence: str | None = None  # the chunk/source id or matched text behind the hit

    @property
    def severity(self) -> Verdict:
        # An unanswered judgment blocks: unchecked content must not pass.
        return "block" if self.rule.startswith("P") or self.rule == "JUDGE" else "warn"


@dataclass
class CheckReport:
    segments: list[Segment]
    hits: list[RuleHit] = field(default_factory=list)

    @property
    def verdict(self) -> Verdict:
        return _worst(h.severity for h in self.hits)

    @property
    def rules(self) -> list[str]:
        return sorted({h.rule for h in self.hits})

    def hits_for(self, segment: int) -> list[RuleHit]:
        return [h for h in self.hits if h.segment == segment]

    def segment_verdict(self, segment: int) -> Verdict:
        return _worst(h.severity for h in self.hits_for(segment))


def _worst(severities) -> Verdict:
    severities = set(severities)
    return "block" if "block" in severities else "warn" if "warn" in severities else "pass"

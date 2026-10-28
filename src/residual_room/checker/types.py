"""Shapes shared by the splitter, the rules and (later) the LLM judge."""

from dataclasses import dataclass, field
from typing import Literal

# claim:     a factual statement that needs a citation (labelled by the generator
#            or the LLM claim extractor, never guessed by patterns)
# exempt:    greeting, sign_off, cta, read_next, transition, opinion
# candidate: not recognized by pattern; the LLM decides claim vs exempt
SegmentKind = Literal["claim", "exempt", "candidate"]
Verdict = Literal["block", "warn", "pass"]


@dataclass(frozen=True)
class Segment:
    text: str
    kind: SegmentKind
    exempt_type: str | None = None
    citations: tuple[str, ...] = ()  # source or chunk ids, e.g. ext-litm-paper, ext-litm-paper#0007


@dataclass(frozen=True)
class RuleHit:
    rule: str  # P1..P6 (hard), S1..S4 (soft)
    reason: str
    segment: int | None = None  # index into the segments; None = whole document
    evidence: str | None = None  # the chunk/source id or matched text behind the hit

    @property
    def severity(self) -> Verdict:
        return "block" if self.rule.startswith("P") else "warn"


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

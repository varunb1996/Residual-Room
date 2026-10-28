"""The generator's structured output, and deterministic checks on its shape.

These checks are about structure (citations present and real, one CTA, sign-off,
read-next for transitioning readers). Whether claims are *true* is the
checker's job, run afterwards like for any other piece.
"""

from typing import Literal

from pydantic import BaseModel

from residual_room.checker.claims import count_ctas, pattern_exempt_type
from residual_room.checker.types import Segment
from residual_room.config import PublicationConfig

ExemptType = Literal["greeting", "framing", "transition", "opinion", "cta", "read_next", "sign_off"]
SIGN_OFF = "— Residual Room"


class GenSegment(BaseModel):
    type: Literal["claim", "exempt"]
    text: str
    exempt_type: ExemptType | None = None
    chunk_ids: list[str] | None = None  # models sometimes send null for "none"


class GenOutput(BaseModel):
    segments: list[GenSegment]


def problems(out: GenOutput, allowed_ids: set[str], segment: str, kind: str,
             config: PublicationConfig) -> list[str]:
    found = []
    if not out.segments:
        return ["no segments"]
    for i, s in enumerate(out.segments):
        if s.type == "claim":
            ids = s.chunk_ids or []
            if not ids:
                found.append(f"segment {i} is a claim with no chunk_ids")
            if unknown := [c for c in ids if c not in allowed_ids]:
                found.append(f"segment {i} cites ids that were not provided: {unknown}")
        elif s.exempt_type is None:
            found.append(f"segment {i} is exempt but has no exempt_type")
        elif s.exempt_type == "cta" and not count_ctas(s.text, config):
            found.append(f"segment {i} is a cta but isn't one of the allowed calls to action")
    ctas = sum(s.exempt_type == "cta" for s in out.segments)
    if kind == "newsletter" and ctas != 1:
        found.append(f"a newsletter needs exactly one cta, found {ctas}")
    if kind != "newsletter" and ctas > 1:
        found.append(f"at most one cta allowed, found {ctas}")
    if kind == "newsletter" and out.segments[-1].text.strip() != SIGN_OFF:
        found.append(f"a newsletter must end with the sign-off {SIGN_OFF!r}")
    if segment == "transitioning" and not any(s.exempt_type == "read_next" for s in out.segments):
        found.append("transitioning pieces must include a read_next pointer")
    return found


def render_text(out: GenOutput) -> str:
    """Plain text with [cite:chunk-id] markers, one segment per line."""
    lines = []
    for s in out.segments:
        cites = "".join(f" [cite:{c}]" for c in (s.chunk_ids or [])) if s.type == "claim" else ""
        lines.append(s.text.strip() + cites)
    return "\n".join(lines)


def to_check_segments(out: GenOutput, config: PublicationConfig | None = None) -> list[Segment]:
    """Generator output for the checker: no re-splitting, but its labels aren't trusted.

    Claims stay claims (they must cite anyway). An exempt label stands only if a
    deterministic pattern confirms the same type (greeting, cta, read_next,
    sign_off, "We think…" opinion). Anything else, including every framing and
    transition, becomes a candidate that the judge labels, so a factual claim
    can't escape its citation by being called "framing" (judge ≠ generator).
    """
    segments = []
    for s in out.segments:
        text = s.text.strip()
        if s.type == "claim":
            segments.append(Segment(text, "claim", citations=tuple(s.chunk_ids or [])))
        elif s.exempt_type and pattern_exempt_type(text, config) == s.exempt_type:
            segments.append(Segment(text, "exempt", s.exempt_type))
        else:
            segments.append(Segment(text, "candidate"))
    return segments

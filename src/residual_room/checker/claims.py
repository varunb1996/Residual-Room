"""Deterministic pre-splitter for raw text (eval cases, ad-hoc checks).

Splits text into sentences, attaches each [cite:id] marker to its sentence, and
labels only what a pattern can safely recognize: greetings, sign-offs, CTAs,
read-next pointers and marked opinions. Everything else is a `candidate`;
deciding whether a candidate is a factual claim needs judgment, which is the
LLM claim extractor's job. Generator output is already segmented and skips this.
"""

import re
from dataclasses import replace

from residual_room.checker.types import Segment
from residual_room.config import PublicationConfig, get_config

CITE = re.compile(r"\s*\[cite:([^\]\s]+)\]")
# A sentence ends at . ! or ? followed by whitespace (so 23.8% and e.g. stay
# intact), and at every line break: lines in emails are separate units.
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_segments(text: str, config: PublicationConfig | None = None) -> list[Segment]:
    config = config or get_config()
    segments = []
    for n, line in enumerate(text.splitlines()):
        for sentence in SENTENCE_END.split(line.strip()):
            if sentence:
                segments.append(replace(_label(sentence, config), line=n))
    return segments


def _label(sentence: str, config: PublicationConfig) -> Segment:
    citations = tuple(CITE.findall(sentence))
    clean = CITE.sub("", sentence).strip()
    # A cited sentence is making a claim, whatever it starts with.
    if not citations:
        for exempt_type, patterns in config.exempt_regexes.items():
            if any(p.search(clean) for p in patterns):
                return Segment(clean, "exempt", exempt_type)
        if count_ctas(clean, config):
            return Segment(clean, "exempt", "cta")
    return Segment(clean, "candidate", citations=citations)


def count_ctas(text: str, config: PublicationConfig | None = None) -> int:
    config = config or get_config()
    return sum(len(p.findall(text)) for ps in config.cta_regexes.values() for p in ps)

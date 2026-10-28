"""The generation prompt. Brand guide and content policy are read from the frozen
docs at run time, so the generator writes to the same rules the checker enforces."""

from residual_room.config import Explainer, PublicationConfig
from residual_room.corpus.manifest import REPO_ROOT
from residual_room.corpus.retrieve import Hit
from residual_room.generation.schema import SIGN_OFF

STRUCTURE = {
    "newsletter": (
        "A newsletter issue of 180–300 words: a one-line hook (framing), the main piece, "
        "a short paragraph framed for the reader segment, exactly one call to action, then the sign-off "
        f"{SIGN_OFF!r} as the last segment."
    ),
    "explainer_summary": (
        "An explainer summary of about 150 words: the problem, the intuition, the mechanism and where it "
        "fails, in that order. At most one call to action."
    ),
}

SEGMENT_TONE = {
    "transitioning": (
        "Readers are engineers moving into AI. Build intuition first, with analogies from software "
        "engineering. Define every technical term the first time you use it. Include a read_next "
        "segment pointing to a related explainer."
    ),
    "practitioner": (
        "Readers already build RAG, agents and evals. Skip the basics; lead with the mechanism or the "
        "failure, then tradeoffs and evidence."
    ),
}

SYSTEM = """You write for {name}, an English AI-education publication. Follow its brand guide and content policy exactly.

=== BRAND GUIDE ===
{brand_guide}

=== CONTENT POLICY ===
{content_policy}

=== HOW TO WRITE THIS PIECE ===
- Use ONLY facts stated in the passages you are given. Never add facts from memory.
- Every factual sentence is a "claim" segment and lists the chunk_ids (the ids in square brackets) of the passages that state it. Cite only ids you were given.
- Numbers must match the passage exactly, in value and in what they describe (same model, same setting).
- Everything else is an "exempt" segment with an exempt_type: greeting, framing (a hook or series label), transition, opinion (must start with "We think" or similar, clearly ours), cta, read_next, sign_off.
- A cta must be one of: read the explainer, subscribe, read the source, reply with a question.
- One sentence per segment.

Reply with only JSON:
{{"segments": [{{"type": "claim", "text": "...", "chunk_ids": ["..."]}}, {{"type": "exempt", "exempt_type": "framing", "text": "..."}}]}}"""


def build_prompt(
    explainer_id: str, explainer: Explainer, segment: str, kind: str, hits: list[Hit], config: PublicationConfig
) -> tuple[str, str]:
    docs = config.docs
    system = SYSTEM.format(
        name=config.name,
        brand_guide=(REPO_ROOT / docs["brand_guide"]).read_text(encoding="utf-8"),
        content_policy=(REPO_ROOT / docs["content_policy"]).read_text(encoding="utf-8"),
    )
    passages = "\n\n".join(f"[{h.chunk_id}] ({h.heading_path})\n{h.text}" for h in hits)
    user = (
        f"Write: {STRUCTURE[kind]}\n"
        f"Topic: {explainer.topic} ({explainer_id}, series: {explainer.series})\n"
        f"Reader segment: {segment}. {SEGMENT_TONE[segment]}\n\n"
        f"Passages:\n\n{passages}"
    )
    return system, user

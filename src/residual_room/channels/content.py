"""Channel-neutral content: an approved version as blocks plus numbered sources.

Built from exactly the version that was checked and approved (generated
segments as stored, or pasted text split the same way the checker split it).
Each channel then renders this in its own format (render.py).
"""

from dataclasses import dataclass, field

from residual_room.checker.claims import split_segments
from residual_room.config import PublicationConfig
from residual_room.corpus.manifest import Manifest


@dataclass
class Source:
    n: int
    source_id: str
    title: str
    url: str | None  # internal docs have no public URL


@dataclass
class Block:
    text: str
    exempt_type: str | None = None  # greeting, cta, read_next, sign_off, …; None for claims/prose
    refs: list[int] = field(default_factory=list)  # source numbers cited by this block


@dataclass
class Content:
    subject: str
    blocks: list[Block]
    sources: list[Source]
    explainer_id: str | None  # for "read the explainer" links


def build_content(title: str, text: str, segments: list[dict] | None, generation: dict | None,
                  manifest: Manifest, config: PublicationConfig) -> Content:
    if segments is not None:
        raw = [(s["text"].strip(), s.get("exempt_type") if s["type"] == "exempt" else None,
                s.get("chunk_ids") or []) for s in segments]
    else:
        raw = [(s.text, s.exempt_type if s.kind == "exempt" else None, list(s.citations))
               for s in split_segments(text, config)]

    sources: dict[str, Source] = {}
    blocks = []
    for text_, exempt_type, citations in raw:
        refs = []
        for cite in citations:
            sid = cite.split("#", 1)[0]
            sid = manifest.resolve(sid).id if (sid in manifest.sources or sid in manifest.aliases) else sid
            if sid not in sources:
                src = manifest.sources.get(sid)
                sources[sid] = Source(len(sources) + 1, sid, src.title if src else sid, src.url if src else None)
            if sources[sid].n not in refs:
                refs.append(sources[sid].n)
        blocks.append(Block(text_, exempt_type, refs))
    return Content(title, blocks, list(sources.values()), (generation or {}).get("explainer"))


def cta_target(block: Block, content: Content, config: PublicationConfig) -> str | None:
    """Where a call to action points (None for "reply with a question")."""
    site, text = config.site, block.text.lower()
    if "explainer" in text:
        if content.explainer_id:
            return site["base_url"] + site["explainer_path"].format(id=content.explainer_id)
        return site["base_url"]
    if "source" in text:
        return next((s.url for s in content.sources if s.url), None)
    if "subscribe" in text:
        return site["base_url"] + site["subscribe_path"]
    return None

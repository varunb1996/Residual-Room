"""Structure-aware chunking.

Split markdown at headings, then pack paragraphs up to ~TARGET_TOKENS. Tables
and code blocks are never split mid-way (an oversized table is split by rows
with its header and caption repeated). No overlap between chunks, so a
sentence lives in exactly one chunk and "which chunk supports this claim" has
one answer. Every chunk carries its heading path, e.g.
"Judging LLM-as-a-Judge … › 3 LLM as a Judge › 3.4 Addressing limitations".
"""

import re
from dataclasses import dataclass

CHARS_PER_TOKEN = 4  # rough estimate; exact counts don't matter at these sizes
TARGET_TOKENS = 400
MAX_TOKENS = 512
MIN_CHARS = 40  # drop fragments shorter than this (stray captions, lone symbols)
PATH_SEP = " › "


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_id: str
    seq: int
    heading_path: str
    text: str

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)

    @property
    def embed_text(self) -> str:
        return f"{self.heading_path}\n\n{self.text}"


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def chunk_markdown(source_id: str, title: str, md: str) -> list[Chunk]:
    if estimate_tokens(md) <= MAX_TOKENS:
        # Small docs (schedule, subscription rules) stay whole: one fact, one chunk.
        return [Chunk(f"{source_id}#0000", source_id, 0, title, md.strip())]
    chunks: list[Chunk] = []
    for path, blocks in _sections(md, title):
        for text in _pack(blocks):
            if len(text) >= MIN_CHARS:
                seq = len(chunks)
                chunks.append(Chunk(f"{source_id}#{seq:04d}", source_id, seq, path, text))
    return chunks


def _sections(md: str, title: str):
    """Yield (heading_path, blocks) per section. Blocks are paragraphs, whole
    code fences, or whole tables (caption line + table lines)."""
    stack: list[tuple[int, str]] = []
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False

    def path() -> str:
        return PATH_SEP.join([title] + [h for _, h in stack])

    def end_block() -> None:
        if current:
            blocks.append("\n".join(current).strip())
            current.clear()

    for line in md.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            current.append(line)
            continue
        if in_fence:
            current.append(line)
            continue
        heading = re.match(r"(#{1,6}) (.+)", line)
        if heading:
            end_block()
            if blocks:
                yield path(), blocks
                blocks = []
            level, text = len(heading.group(1)), heading.group(2).strip()
            if level == 1 and text == title:
                continue  # the document title is already the root of every path
            stack = [(lv, h) for lv, h in stack if lv < level] + [(level, text)]
        elif not line.strip():
            # A blank line between a [TABLE] caption and its rows keeps them together.
            if not (current and current[0].startswith("[TABLE]")):
                end_block()
        elif line.startswith("[TABLE]") or (
            current and current[0].startswith("[TABLE]") and not line.startswith("|")
        ):
            end_block()
            current.append(line)
        else:
            current.append(line)
    end_block()
    if blocks:
        yield path(), blocks


def _pack(blocks: list[str]) -> list[str]:
    out: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            out.append("\n\n".join(buf))
            buf.clear()

    for block in blocks:
        if block.startswith("[TABLE]"):
            flush()
            out.extend(_split_table(block))
            continue
        pieces = [block] if estimate_tokens(block) <= MAX_TOKENS else _split_long(block)
        for piece in pieces:
            if buf and estimate_tokens("\n\n".join(buf + [piece])) > TARGET_TOKENS:
                flush()
            buf.append(piece)
    flush()
    return out


def _split_table(block: str) -> list[str]:
    if estimate_tokens(block) <= MAX_TOKENS:
        return [block]
    lines = block.splitlines()
    caption = [ln for ln in lines if not ln.startswith("|") and ln.strip()]
    table = [ln for ln in lines if ln.startswith("|")]
    head, rows = table[:2], table[2:]
    prefix = "\n".join(caption) + " (continued)\n\n" + "\n".join(head)
    parts, part = [], []
    for row in rows:
        if part and estimate_tokens(prefix + "\n" + "\n".join(part + [row])) > MAX_TOKENS:
            parts.append(prefix + "\n" + "\n".join(part))
            part = []
        part.append(row)
    if part:
        parts.append(prefix + "\n" + "\n".join(part))
    return parts


def _split_long(block: str) -> list[str]:
    """Split an oversized paragraph or code block at sentence/line boundaries."""
    if block.startswith("```"):
        units = block.splitlines()
        joiner = "\n"
    else:
        units = re.split(r"(?<=[.!?])\s+", block)
        joiner = " "
    parts, part = [], []
    for unit in units:
        if part and estimate_tokens(joiner.join(part + [unit])) > TARGET_TOKENS:
            parts.append(joiner.join(part))
            part = []
        part.append(unit)
    if part:
        parts.append(joiner.join(part))
    return parts

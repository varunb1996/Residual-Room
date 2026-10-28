"""Turn fetched files (arXiv HTML, PDF, notebooks, docs) into plain markdown.

Every extractor returns markdown with `#` headings, so the chunker can split
on structure the same way for every source. Tables become markdown tables with
their caption, and spanned cells are filled in, so each row keeps its labels
(P2 checks that a number matches *what it describes*, not just its value).
"""

import io
import json
import re

from bs4 import BeautifulSoup, Tag
from pypdf import PdfReader

# ---------- arXiv HTML (LaTeXML) ----------

_HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 4, "h6": 2}


def arxiv_html_to_markdown(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find("article") or soup
    # References add noise to retrieval and are never what a claim cites.
    for junk in article.select(".ltx_bibliography, .ltx_authors, .ltx_note_outer, script, style"):
        junk.decompose()
    for m in article.find_all("math"):
        m.replace_with(m.get("alttext", ""))

    out: list[str] = []
    for el in article.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "figure"]):
        if el.find_parent("figure") is not None:
            continue  # handled with its figure
        if el.name in _HEADINGS:
            out.append("#" * _HEADINGS[el.name] + " " + _text(el))
        elif el.name == "p":
            if el.find_parent(["h1", "h2", "h3", "h4", "h5", "h6"]) is None:
                out.append(_text(el))
        elif "ltx_table" in (el.get("class") or []):
            out.append(_table_figure(el))
    return "\n\n".join(x for x in out if x.strip())


def _table_figure(fig: Tag) -> str:
    cap = fig.find("figcaption")
    caption = _text(cap) if cap else "Table"
    tables = fig.find_all("table")
    body = "\n\n".join(_table_to_markdown(t) for t in tables if t.find_parent("table") is None)
    return f"[TABLE] {caption}\n\n{body}"


def _table_to_markdown(table: Tag) -> str:
    grid: dict[tuple[int, int], str] = {}
    for r, tr in enumerate(table.find_all("tr")):
        c = 0
        for cell in tr.find_all(["td", "th"]):
            while (r, c) in grid:
                c += 1
            text = _text(cell).replace("|", "/")
            for dr in range(int(cell.get("rowspan", 1) or 1)):
                for dc in range(int(cell.get("colspan", 1) or 1)):
                    grid[(r + dr, c + dc)] = text
            c += int(cell.get("colspan", 1) or 1)
    if not grid:
        return ""
    nrows = max(r for r, _ in grid) + 1
    ncols = max(c for _, c in grid) + 1
    rows = [[grid.get((r, c), "") for c in range(ncols)] for r in range(nrows)]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * ncols]
    lines += ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join(lines)


def _text(el: Tag) -> str:
    return re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()


# ---------- PDF (ACL Anthology) ----------

# Numbered section headings as they come out of ACL papers: "3 Method", "4.2 Results".
_PDF_HEADING = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,2})\s+([A-Z][A-Za-z][^.]{0,70})$")
_PDF_NAMED = re.compile(r"^(Abstract|References|Acknowledg(e)?ments|Limitations|Appendix|Conclusions?)$")


def pdf_to_markdown(data: bytes) -> str:
    """Best-effort: PDF text has no structure, so headings are found by pattern."""
    reader = PdfReader(io.BytesIO(data))
    lines: list[str] = []
    for page in reader.pages:
        lines.extend((page.extract_text() or "").splitlines())

    out: list[str] = []
    para: list[str] = []

    def flush() -> None:
        if para:
            text = " ".join(para)
            text = re.sub(r"(\w)- (\w)", r"\1\2", text)  # rejoin hyphenated line breaks
            out.append(re.sub(r"\s+", " ", text).strip())
            para.clear()

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if _PDF_NAMED.match(line) or _PDF_HEADING.match(line):
            flush()
            if line.startswith("References"):
                break  # stop before the bibliography
            out.append("## " + line)
        else:
            para.append(line)
            if line.endswith((".", ":", "?")) and len(line) < 60:
                flush()  # a short line ending a sentence usually ends a paragraph
    flush()
    return "\n\n".join(out)


# ---------- repo files ----------


def repo_file_to_markdown(path: str, text: str) -> str:
    """Convert one repo file. Its headings are demoted a level so they nest under
    the `# <path>` heading the fetcher puts above each file."""
    if path.endswith(".ipynb"):
        text = notebook_to_markdown(text)
    elif path.endswith(".rst"):
        text = rst_to_markdown(text)
    elif path.endswith(".mdx"):
        text = mdx_to_markdown(text)
    elif path.endswith(".jsonl"):
        text = jsonl_to_markdown(text)
    return demote_headings(text)


def demote_headings(md: str) -> str:
    out, in_fence = [], False
    for line in md.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        elif not in_fence and re.match(r"#{1,5} ", line):
            line = "#" + line
        out.append(line)
    return "\n".join(out)


def notebook_to_markdown(text: str) -> str:
    """Keep markdown and code cells; drop outputs (often megabytes of data)."""
    nb = json.loads(text)
    out = []
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue
        out.append(src if cell.get("cell_type") == "markdown" else f"```python\n{src}\n```")
    return "\n\n".join(out)


_RST_UNDERLINE = re.compile(r"^([=\-~^\"'`#*+])\1{2,}\s*$")


def rst_to_markdown(text: str) -> str:
    lines = text.splitlines()
    levels: list[str] = []
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if line.strip() and _RST_UNDERLINE.match(nxt) and len(nxt.strip()) >= len(line.strip()):
            char = nxt.strip()[0]
            if char not in levels:
                levels.append(char)
            out.append("#" * min(levels.index(char) + 1, 4) + " " + line.strip())
            i += 2
            continue
        if not _RST_UNDERLINE.match(line):
            out.append(line)
        i += 1
    return "\n".join(out)


def mdx_to_markdown(text: str) -> str:
    text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.S)  # front matter
    kept = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(("import ", "export ")):
            continue
        if re.fullmatch(r"</?[A-Z][\w.]*[^>]*>", s):
            continue  # a line that is only a JSX component tag
        kept.append(line)
    return "\n".join(kept)


def jsonl_to_markdown(text: str) -> str:
    out = []
    for i, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        obj = json.loads(line)
        out.append(f"## {obj.get('name', f'record {i + 1}')}")
        out += [f"{k}: {v}" for k, v in obj.items() if isinstance(v, str) and k != "name"]
    return "\n\n".join(out)

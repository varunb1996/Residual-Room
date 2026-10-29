"""Public pages behind the links in sent messages (no login; see security.OPEN_PATHS).

Only what has already gone out is shown: the version a sent campaign delivered,
never a draft, a piece in review, or an approved-but-unsent piece. Sources are
links to the originals, never their text, so license rules are unaffected.
"""

import html

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from residual_room.api.deps import get_session
from residual_room.channels.content import build_content
from residual_room.channels.render import render_web
from residual_room.config import get_config
from residual_room.corpus.manifest import load_manifest
from residual_room.db.models import Campaign, Piece, PieceVersion

router = APIRouter(include_in_schema=False)

READERS = {"transitioning": "For engineers moving into AI", "practitioner": "For practitioners"}


def latest_sent(s: Session, explainer_id: str) -> dict[str, tuple[Piece, PieceVersion, Campaign]]:
    """Per reader segment, the most recent sent campaign of a piece about this explainer."""
    rows = s.execute(
        select(Campaign, Piece, PieceVersion)
        .join(Piece, Piece.id == Campaign.piece_id)
        .join(PieceVersion, PieceVersion.id == Campaign.version_id)
        .where(Campaign.status == "sent", PieceVersion.generation["explainer"].astext == explainer_id)
        .order_by(Campaign.sent_at.desc())
    ).all()
    found: dict[str, tuple[Piece, PieceVersion, Campaign]] = {}
    for campaign, piece, version in rows:
        for segment in campaign.segments:
            found.setdefault(segment, (piece, version, campaign))
    return found


@router.get("/explainers", response_class=HTMLResponse)
def explainers(s: Session = Depends(get_session)):
    cfg = get_config()
    items = []
    for eid, e in cfg.explainers.items():
        sent = bool(latest_sent(s, eid))
        note = "" if sent else ' <span class="muted">No issue sent yet</span>'
        items.append(f'<li><a href="/explainers/{html.escape(eid)}">{html.escape(e.topic)}</a> '
                     f'<span class="muted">{html.escape(e.series)}</span>{note}</li>')
    body = (f"<h1>Explainers</h1><p class=\"lede\">Every factual sentence in {html.escape(cfg.name)} "
            f"cites an approved source, and a person approves each issue before it is sent.</p>"
            f'<ul class="index">{"".join(items)}</ul>')
    return _page("Explainers", body)


@router.get("/explainers/{explainer_id}", response_class=HTMLResponse)
def explainer(explainer_id: str, s: Session = Depends(get_session)):
    cfg, manifest = get_config(), load_manifest()
    e = cfg.explainers.get(explainer_id)
    if e is None:
        return _page("Not found", '<h1>No such explainer</h1><p><a href="/explainers">All explainers</a></p>', 404)
    superseded = {sid for sid, src in manifest.sources.items() if src.status == "superseded"}
    head = (f'<p class="series">{html.escape(e.series)}</p><h1>{html.escape(e.topic)}</h1>')

    sent = latest_sent(s, explainer_id)
    sections = []
    for segment in cfg.segments:
        if segment not in sent:
            continue
        piece, version, campaign = sent[segment]
        content = build_content(piece.title, version.text, version.segments, version.generation, manifest, cfg)
        when = campaign.sent_at.strftime("%d %B %Y").lstrip("0")
        label = READERS.get(segment, segment) if len(sent) > 1 else ""
        # The subject is the reader-facing title; skip it when it only repeats the topic above.
        title = "" if content.subject == e.topic else f"<h2>{html.escape(content.subject)}</h2>"
        sections.append(
            f'<section><p class="meta">{html.escape(label + (", sent " if label else "Sent ") + when)}</p>'
            f"{title}{render_web(content, superseded)}</section>"
        )
    if not sections:
        own = [src for src in manifest.external if src.topic in e.source_topics]
        items = "".join(
            f'<li><a href="{html.escape(src.url)}" rel="noopener">{html.escape(src.title)}</a></li>'
            if src.url else f"<li>{html.escape(src.title)}</li>" for src in own
        )
        sections.append(f'<p class="muted">No issue sent yet. It will draw on these sources:</p><ol class="sources">{items}</ol>')
    return _page(e.topic, head + "".join(sections) + '<p class="back"><a href="/explainers">All explainers</a></p>')


@router.get("/subscribe", response_class=HTMLResponse)
def subscribe():
    body = ("<h1>Subscribe</h1><p class=\"lede\">Residual Room is in a closed test: issues go only to "
            "test readers for now. Public sign-up opens when it leaves the test.</p>"
            '<p><a href="/explainers">Read the explainers</a></p>')
    return _page("Subscribe", body)


def _page(title: str, body: str, status: int = 200) -> HTMLResponse:
    name = html.escape(get_config().name)
    return HTMLResponse(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} · {name}</title><style>{CSS}</style></head>
<body><header><a class="wordmark" href="/explainers">{name}</a></header><main>{body}</main></body></html>""",
                        status_code=status)


# The app's palette (web/src/styles.css), with system serif fonts: no font requests.
CSS = """
:root { --paper:#f6f7f9; --ink:#18202b; --graphite:#5a6473; --rule:#dde1e7; --red:#b3261e; --link:#2f4ba8; color-scheme: light; }
@media (prefers-color-scheme: dark) {
  :root { --paper:#12161c; --ink:#e6eaf0; --graphite:#9aa4b2; --rule:#2b333e; --red:#f0786e; --link:#8ea2f0; color-scheme: dark; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--paper); color: var(--ink); font: 19px/1.62 Georgia, "Times New Roman", serif; }
header { border-bottom: 1px solid var(--rule); padding: 14px 16px; }
.wordmark { font-weight: 600; font-size: 19px; color: var(--ink); text-decoration: none; }
main { max-width: 40rem; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 34px; line-height: 1.15; margin: 0 0 12px; }
h2 { font-size: 24px; line-height: 1.25; margin: 0 0 16px; }
h3 { font: 600 15px/1.4 system-ui, sans-serif; color: var(--graphite); margin: 28px 0 8px; }
p { margin: 0 0 14px; }
a { color: var(--link); }
sup a { font: 600 12px system-ui, sans-serif; text-decoration: none; margin-left: 2px; }
section { border-top: 1px solid var(--rule); padding-top: 24px; margin-top: 28px; }
.series, .meta, .muted, .back { font: 15px/1.5 system-ui, sans-serif; color: var(--graphite); }
.series { margin-bottom: 6px; }
.lede { color: var(--graphite); }
.aside { color: var(--graphite); }
.sources { font: 15px/1.5 system-ui, sans-serif; padding-left: 22px; }
.sources li { margin-bottom: 6px; }
.superseded { color: var(--red); }
.index { list-style: none; padding: 0; }
.index li { padding: 12px 0; border-top: 1px solid var(--rule); }
.index .muted { display: block; font-size: 14px; }
.back { margin-top: 40px; }
"""

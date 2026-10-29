"""Per-channel formatting, after generation and approval (CLAUDE.md principle 8)."""

import html
from dataclasses import dataclass

from residual_room.channels.content import Block, Content, cta_target, site_url
from residual_room.config import PublicationConfig

TELEGRAM_LIMIT = 4096
WHATSAPP_LIMIT = 1024


@dataclass
class EmailMessage:
    subject: str
    html: str
    text: str


def _block_target(block: Block, content: Content, config: PublicationConfig) -> str | None:
    if block.exempt_type == "cta":
        return cta_target(block, content, config)
    if block.exempt_type == "read_next":
        return site_url(config.site["explainers_path"])
    return None


def render_email(content: Content, linker, unsubscribe_url: str, config: PublicationConfig) -> EmailMessage:
    """HTML and plain text. Citations become numbered, tracked source links; every
    email carries an unsubscribe link (subscription rules)."""
    src_url = {s.n: linker.url(s.url, "source") for s in content.sources if s.url}
    parts_html, parts_text = [], []
    for b in content.blocks:
        target = _block_target(b, content, config)
        label = "cta" if b.exempt_type == "cta" else "read_next"
        body = html.escape(b.text)
        if target:
            link = linker.url(target, label)
            body = f'<a href="{html.escape(link)}">{body}</a>'
            parts_text.append(f"{b.text} {link}")
        else:
            parts_text.append(b.text + "".join(f" [{n}]" for n in b.refs))
        refs = "".join(
            f'<sup><a href="{html.escape(src_url[n])}">[{n}]</a></sup>' if n in src_url else f"<sup>[{n}]</sup>"
            for n in b.refs
        )
        parts_html.append(f"<p>{body}{refs}</p>")
    if content.sources:
        items = "".join(
            f'<li><a href="{html.escape(src_url[s.n])}">{html.escape(s.title)}</a></li>' if s.n in src_url
            else f"<li>{html.escape(s.title)}</li>"
            for s in content.sources
        )
        parts_html.append(f'<p style="color:#666;font-size:13px">Sources</p><ol style="color:#666;font-size:13px">{items}</ol>')
        parts_text.append("\nSources\n" + "\n".join(
            f"[{s.n}] {s.title}" + (f" {src_url[s.n]}" if s.n in src_url else "") for s in content.sources
        ))
    parts_html.append(
        f'<p style="color:#888;font-size:12px">You subscribed to Residual Room. '
        f'<a href="{html.escape(unsubscribe_url)}">Unsubscribe</a></p>'
    )
    parts_text.append(f"\nUnsubscribe: {unsubscribe_url}")
    body = "".join(parts_html)
    page = f'<div style="font-family:system-ui,sans-serif;max-width:600px;line-height:1.5">{body}</div>'
    return EmailMessage(content.subject, page, "\n".join(parts_text))


def render_telegram(content: Content, linker, config: PublicationConfig) -> str:
    """Telegram's HTML subset, within its 4096-character limit."""
    src_url = {s.n: linker.url(s.url, "source") for s in content.sources if s.url}
    message = f"<b>{html.escape(content.subject)}</b>\n"
    for b in content.blocks:
        target = _block_target(b, content, config)
        open_, close = "", ""
        if target:
            link = linker.url(target, "cta" if b.exempt_type == "cta" else "read_next")
            open_, close = f'<a href="{html.escape(link)}">', "</a>"
        refs = "".join(f' <a href="{html.escape(src_url[n])}">[{n}]</a>' if n in src_url else f" [{n}]"
                       for n in b.refs)
        line = f"\n{open_}{html.escape(b.text)}{close}{refs}"
        if len(message) + len(line) > TELEGRAM_LIMIT:
            # Cut the plain text, then escape, so a tag or entity is never split.
            budget = TELEGRAM_LIMIT - len(message) - len(open_) - len(close) - 10
            text = b.text
            while text and len(html.escape(text)) > budget:
                # Escaping lengthens text (< becomes &lt;), so shrink in proportion.
                text = text[: int(len(text) * budget / len(html.escape(text))) - 1]
            if text:
                message += f"\n{open_}{html.escape(text)}…{close}"
            break
        message += line
    return message


def render_whatsapp_template(content: Content, linker, config: PublicationConfig) -> tuple[str, list[str]]:
    """The approved `new_issue` template and its parameters (title, tracked link)."""
    cta = next((b for b in content.blocks if b.exempt_type == "cta"), None)
    target = (cta and cta_target(cta, content, config)) or site_url(config.site["explainers_path"])
    return "new_issue", [content.subject[:200], linker.url(target, "cta")]


def render_web(content: Content, superseded: set[str]) -> str:
    """The public explainer page's body (api/public.py): the sent text with numbered
    citations and a source list. Links to sources only, never their text. Calls to
    action and read-next lines are dropped: the reader is already on that page."""
    parts = []
    for b in content.blocks:
        if b.exempt_type in ("cta", "read_next"):
            continue
        refs = "".join(f'<sup><a href="#src-{n}">{n}</a></sup>' for n in b.refs)
        cls = ' class="aside"' if b.exempt_type else ""
        parts.append(f"<p{cls}>{html.escape(b.text)}{refs}</p>")
    if content.sources:
        items = []
        for s in content.sources:
            title = html.escape(s.title)
            name = f'<a href="{html.escape(s.url)}" rel="noopener">{title}</a>' if s.url else title
            flag = ' <span class="superseded">(since superseded)</span>' if s.source_id in superseded else ""
            items.append(f'<li id="src-{s.n}">{name}{flag}</li>')
        parts.append(f'<h3>Sources</h3><ol class="sources">{"".join(items)}</ol>')
    return "".join(parts)


def fill_template(name: str, params: list[str], config: PublicationConfig) -> str:
    text = config.whatsapp_templates[name]
    for i, p in enumerate(params, 1):
        text = text.replace("{{%d}}" % i, p)
    return text[:WHATSAPP_LIMIT]

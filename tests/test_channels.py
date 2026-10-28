import json
from datetime import UTC, datetime, timedelta

import httpx

from residual_room.channels.content import build_content
from residual_room.channels.links import PreviewLinker
from residual_room.channels.render import (
    TELEGRAM_LIMIT,
    fill_template,
    render_email,
    render_telegram,
    render_whatsapp_template,
)
from residual_room.channels.senders import EmailSender, TelegramSender, WhatsAppMock
from residual_room.config import get_config
from residual_room.corpus.manifest import load_manifest
from residual_room.settings import Settings

SEGMENTS = [
    {"type": "exempt", "exempt_type": "framing", "text": "Failure Modes this week: your judge flips."},
    {"type": "claim", "text": "Claude-v1 kept its verdict 23.8% of the time.", "chunk_ids": ["ext-mtbench-paper#0015"]},
    {"type": "claim", "text": "Accuracy drops mid-context.", "chunk_ids": ["ext-litm-for-t3"]},
    {"type": "exempt", "exempt_type": "cta", "text": "Read the explainer."},
    {"type": "exempt", "exempt_type": "sign_off", "text": "— Residual Room"},
]


def content(segments=SEGMENTS, text=""):
    return build_content("Issue 2", text, segments, {"explainer": "exp-004"}, load_manifest(), get_config())


def settings(**kw):
    base = {"database_url": "postgresql://x/y", "test_emails": "me@example.com",
            "resend_api_key": "re_test", "telegram_bot_token": "t", "telegram_test_chat_id": "-100"}
    return Settings(_env_file=None, **{**base, **kw})


class RecordingLinker(PreviewLinker):
    def __init__(self):
        self.seen = []

    def url(self, target, label):
        self.seen.append((label, target))
        return f"https://track/{len(self.seen)}"


def test_sources_are_numbered_once_and_aliases_resolve():
    c = content()
    assert [(s.n, s.source_id) for s in c.sources] == [(1, "ext-mtbench-paper"), (2, "ext-litm-paper")]
    assert c.blocks[1].refs == [1] and c.blocks[2].refs == [2]


def test_pasted_text_is_split_like_the_checker():
    c = build_content("Welcome", "Welcome to Residual Room!\nIt's free [cite:int-subscription].\n— Residual Room",
                      None, None, load_manifest(), get_config())
    assert [b.exempt_type for b in c.blocks] == ["greeting", None, "sign_off"]
    assert c.sources[0].url is None  # internal doc: no public link


def test_email_links_are_tracked_and_has_unsubscribe():
    linker = RecordingLinker()
    msg = render_email(content(), linker, "https://x/u/abc", get_config())
    labels = {label for label, _ in linker.seen}
    assert labels == {"source", "cta"}
    assert ("cta", "https://residualroom.example/explainers/exp-004") in linker.seen
    assert "https://x/u/abc" in msg.html and "Unsubscribe" in msg.text
    assert "https://arxiv.org" not in msg.html  # only tracked links go out


def test_telegram_is_html_escaped_and_within_limit():
    long = SEGMENTS + [{"type": "exempt", "exempt_type": "transition", "text": "<b>x</b> " * 1500}]
    text = render_telegram(content(long), RecordingLinker(), get_config())
    assert len(text) <= TELEGRAM_LIMIT and "&lt;b&gt;" in text


def test_whatsapp_template_fills_title_and_tracked_link():
    name, params = render_whatsapp_template(content(), RecordingLinker(), get_config())
    assert name == "new_issue" and params[0] == "Issue 2"
    assert fill_template(name, params, get_config()).startswith("New from Residual Room: Issue 2.")


def test_email_only_goes_to_the_test_list():
    calls = []
    transport = httpx.MockTransport(lambda r: calls.append(json.loads(r.content)) or httpx.Response(200, json={"id": "em_1"}))
    sender = EmailSender(settings(), transport)
    msg = render_email(content(), RecordingLinker(), "https://x/u/a", get_config())
    assert sender.send("stranger@example.com", msg, "https://x/u/a").status == "refused"
    assert calls == []  # refused before any API call
    result = sender.send("Me@Example.com", msg, "https://x/u/a")
    assert (result.status, result.provider_id) == ("sent", "em_1")
    assert calls[0]["to"] == ["Me@Example.com"] and calls[0]["headers"]["List-Unsubscribe"] == "<https://x/u/a>"


def test_email_without_key_fails_cleanly():
    msg = render_email(content(), RecordingLinker(), "u", get_config())
    assert EmailSender(settings(resend_api_key=None)).send("me@example.com", msg, "u").status == "failed"


def test_telegram_only_posts_to_the_test_chat():
    calls = []
    ok = {"ok": True, "result": {"message_id": 7}}
    transport = httpx.MockTransport(lambda r: calls.append(json.loads(r.content)) or httpx.Response(200, json=ok))
    sender = TelegramSender(settings(), transport)
    assert sender.send("12345", "hi").status == "refused" and calls == []
    assert sender.send("-100", "hi").provider_id == "7" and calls[0]["parse_mode"] == "HTML"


def test_whatsapp_mock_enforces_opt_in_window_and_templates():
    wa = WhatsAppMock(get_config().whatsapp_templates)
    now = datetime(2026, 11, 12, 9, 0, tzinfo=UTC)
    recent, stale = now - timedelta(hours=2), now - timedelta(hours=30)
    assert wa.send(opted_in=False, last_inbound_at=recent, text="hi", now=now).status == "rejected"
    assert wa.send(opted_in=True, last_inbound_at=recent, text="hi", now=now).status == "sent"
    outside = wa.send(opted_in=True, last_inbound_at=stale, text="hi", now=now)
    assert outside.status == "rejected" and "24-hour" in outside.error
    assert wa.send(opted_in=True, last_inbound_at=None, template="new_issue", now=now).status == "sent"
    assert wa.send(opted_in=True, last_inbound_at=None, template="promo", now=now).status == "rejected"

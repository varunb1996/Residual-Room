"""Campaigns end to end through the API, against the real database. Resend and
Telegram are faked at the HTTP layer; the WhatsApp mock is the real mock."""

import json
import uuid
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from residual_room.api.campaigns import get_senders
from residual_room.api.main import app
from residual_room.channels.campaigns import Senders, add_subscriber
from residual_room.channels.senders import EmailSender, TelegramSender, WhatsAppMock
from residual_room.config import get_config
from residual_room.db.models import Click, Piece, Subscriber
from residual_room.db.session import get_engine
from residual_room.settings import Settings, get_settings
from residual_room.workflow.service import create_piece

pytestmark = pytest.mark.db

TEXT = "Failure Modes this week.\nClaude-v1 kept its verdict 23.8% of the time [cite:ext-mtbench-paper].\nRead the explainer.\n— Residual Room"


@pytest.fixture
def env():
    tag = uuid.uuid4().hex[:8]
    me, stranger, wa = f"me-{tag}@example.com", f"stranger-{tag}@example.com", f"+9100{tag}"
    settings = Settings(_env_file=None, database_url=get_settings().database_url, test_emails=me,
                        resend_api_key="re_test", telegram_bot_token="t", telegram_test_chat_id="-100")
    sent = {"email": [], "telegram": []}
    email_t = httpx.MockTransport(lambda r: sent["email"].append(json.loads(r.content)) or httpx.Response(200, json={"id": "em"}))
    tg_t = httpx.MockTransport(lambda r: sent["telegram"].append(json.loads(r.content)) or httpx.Response(
        200, json={"ok": True, "result": {"message_id": 1}}))
    senders = Senders(EmailSender(settings, email_t), TelegramSender(settings, tg_t), WhatsAppMock(get_config().whatsapp_templates))
    app.dependency_overrides[get_senders] = lambda: senders
    # The campaign service reads TELEGRAM_TEST_CHAT_ID from settings too.
    import residual_room.channels.campaigns as campaigns_mod
    original = campaigns_mod.get_settings
    campaigns_mod.get_settings = lambda: settings

    with Session(get_engine()) as s, s.begin():
        piece = create_piece(s, title=f"test {tag}", kind="newsletter", segment="practitioner", text=TEXT)
        piece.state = "approved"  # the approval flow itself is tested in test_review_api
        subs = [add_subscriber(s, segment="practitioner", email=me),
                add_subscriber(s, segment="practitioner", email=stranger),
                add_subscriber(s, segment="practitioner", whatsapp=wa)]
        ids = {"piece": piece.id, "subs": [x.id for x in subs], "me_token": subs[0].token}

    with TestClient(app) as c:
        yield c, ids, sent, wa
    app.dependency_overrides.clear()
    campaigns_mod.get_settings = original
    with Session(get_engine()) as s, s.begin():
        s.execute(delete(Piece).where(Piece.id == ids["piece"]))
        s.execute(delete(Subscriber).where(Subscriber.id.in_(ids["subs"])))


def test_only_approved_pieces_can_be_scheduled(env):
    c, ids, _, _ = env
    with Session(get_engine()) as s, s.begin():
        s.get(Piece, ids["piece"]).state = "draft"
    r = c.post("/api/campaigns", json={"piece_id": ids["piece"], "segments": ["practitioner"], "channels": ["email"]})
    assert r.status_code == 409


def test_send_now_delivers_per_channel_with_test_list_enforced(env):
    c, ids, sent, wa = env
    r = c.post("/api/campaigns", json={"piece_id": ids["piece"], "segments": ["practitioner"],
                                       "channels": ["email", "telegram", "whatsapp"]})
    assert r.status_code == 201
    camp = c.get("/api/campaigns").json()[0]
    assert camp["status"] == "sent"
    by = {(d["channel"], d["status"]) for d in camp["deliveries"] if d["recipient"] != "(no test chat)"}
    assert ("email", "sent") in by and ("email", "refused") in by  # stranger refused, nothing sent
    assert ("telegram", "sent") in by and ("whatsapp", "sent") in by  # whatsapp via approved template
    assert len(sent["email"]) == 1 and "/r/" in sent["email"][0]["html"] and "/u/" in sent["email"][0]["html"]
    assert sent["telegram"][0]["chat_id"] == "-100"
    assert c.get(f"/api/pieces/{ids['piece']}").json()["state"] == "sent"


def test_click_is_logged_and_redirects(env):
    c, ids, sent, _ = env
    c.post("/api/campaigns", json={"piece_id": ids["piece"], "segments": ["practitioner"], "channels": ["email"]})
    html = sent["email"][0]["html"]
    token = html.split("/r/")[1].split('"')[0]
    r = c.get(f"/r/{token}", follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"].startswith("http")
    camp = c.get("/api/campaigns").json()[0]
    assert sum(x["clicks"] for x in camp["clicks"] if x["channel"] == "email") == 1
    assert c.get("/r/not-a-token", follow_redirects=False).status_code == 404


def test_unsubscribe_link_stops_further_email(env):
    c, ids, sent, _ = env
    assert c.get(f"/u/{ids['me_token']}").status_code == 200
    c.post("/api/campaigns", json={"piece_id": ids["piece"], "segments": ["practitioner"], "channels": ["email"]})
    assert sent["email"] == []


def test_future_campaign_waits_for_the_worker_and_can_be_cancelled(env):
    c, ids, sent, _ = env
    later = (datetime.now(UTC) + timedelta(days=1)).isoformat()
    r = c.post("/api/campaigns", json={"piece_id": ids["piece"], "segments": ["practitioner"],
                                       "channels": ["email"], "scheduled_at": later})
    assert r.json()["status"] == "scheduled" and sent["email"] == []
    assert c.get(f"/api/pieces/{ids['piece']}").json()["state"] == "scheduled"

    from residual_room.worker import run_due
    assert r.json()["id"] not in run_due(get_senders())  # not due yet

    assert c.post(f"/api/campaigns/{r.json()['id']}/cancel").json()["status"] == "cancelled"
    assert c.get(f"/api/pieces/{ids['piece']}").json()["state"] == "approved"


def test_whatsapp_mock_rejects_free_form_outside_the_window(env):
    c, _, _, wa = env
    r = c.post("/api/whatsapp/send", json={"whatsapp": wa, "text": "New issue is out!"}).json()
    assert r["status"] == "rejected" and "24-hour" in r["error"]
    assert c.post("/api/whatsapp/send", json={"whatsapp": wa, "template": "new_issue"}).json()["status"] == "sent"
    c.post("/api/whatsapp/inbound", json={"whatsapp": wa, "text": "hi"})
    assert c.post("/api/whatsapp/send", json={"whatsapp": wa, "text": "Thanks!"}).json()["status"] == "sent"


def test_preview_shows_every_channel(env):
    c, ids, _, _ = env
    p = c.get(f"/api/pieces/{ids['piece']}/preview").json()
    assert "Unsubscribe" in p["email_html"] and "<b>" in p["telegram_html"]
    assert p["whatsapp_text"].startswith("New from Residual Room")


def test_telegram_commands_manage_a_subscriber():
    from residual_room.worker import handle_command

    chat = f"chat-{uuid.uuid4().hex[:8]}"
    with Session(get_engine()) as s, s.begin():
        assert "Subscribed" in handle_command(s, chat, "/start")
        assert "transitioning" in handle_command(s, chat, "/segment transitioning")
        assert "Choose one of" in handle_command(s, chat, "/segment nonsense")
        handle_command(s, chat, "/stop")
        sub = s.scalar(select(Subscriber).where(Subscriber.telegram_chat_id == chat))
        assert sub.segment == "transitioning" and not sub.subscribed
        s.delete(sub)

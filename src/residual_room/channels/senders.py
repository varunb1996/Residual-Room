"""Channel senders. Email and Telegram are real (Resend, Bot API); WhatsApp is a
mock that enforces the platform's rules without sending anything.

Safety rules live here, before any network call: email only reaches addresses
in TEST_EMAILS, Telegram only the test chat. A refusal is a delivery status,
not an exception, so it shows up on the campaign screen.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import httpx

from residual_room.channels.render import EmailMessage
from residual_room.settings import Settings

WHATSAPP_WINDOW = timedelta(hours=24)


@dataclass
class SendResult:
    status: str  # sent | failed | refused | rejected
    provider_id: str | None = None
    error: str | None = None


class EmailSender:
    URL = "https://api.resend.com/emails"

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None):
        self.settings = settings
        self.http = httpx.Client(timeout=30, transport=transport)

    def send(self, to: str, message: EmailMessage, unsubscribe_url: str) -> SendResult:
        if to.strip().lower() not in self.settings.test_email_list:
            return SendResult("refused", error=f"{to} is not in TEST_EMAILS; nothing was sent")
        if not self.settings.resend_api_key:
            return SendResult("failed", error="RESEND_API_KEY is not set")
        r = self.http.post(
            self.URL,
            headers={"Authorization": f"Bearer {self.settings.resend_api_key}"},
            json={
                "from": self.settings.email_from,
                "to": [to],
                "subject": message.subject,
                "html": message.html,
                "text": message.text,
                "headers": {"List-Unsubscribe": f"<{unsubscribe_url}>"},
            },
        )
        if r.status_code in (200, 201):
            return SendResult("sent", provider_id=r.json().get("id"))
        return SendResult("failed", error=f"Resend HTTP {r.status_code}: {r.text[:200]}")


class TelegramSender:
    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None):
        self.settings = settings
        self.http = httpx.Client(timeout=30, transport=transport)

    def api(self, method: str) -> str:
        return f"https://api.telegram.org/bot{self.settings.telegram_bot_token}/{method}"

    def send(self, chat_id: str, html_text: str) -> SendResult:
        if str(chat_id) != str(self.settings.telegram_test_chat_id):
            return SendResult("refused", error=f"chat {chat_id} is not TELEGRAM_TEST_CHAT_ID; nothing was sent")
        if not self.settings.telegram_bot_token:
            return SendResult("failed", error="TELEGRAM_BOT_TOKEN is not set")
        r = self.http.post(self.api("sendMessage"), json={
            "chat_id": chat_id, "text": html_text, "parse_mode": "HTML", "disable_web_page_preview": True,
        })
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        if r.status_code == 200 and body.get("ok"):
            return SendResult("sent", provider_id=str(body["result"]["message_id"]))
        return SendResult("failed", error=f"Telegram HTTP {r.status_code}: {body.get('description', r.text[:200])}")


class WhatsAppMock:
    """WhatsApp Business rules, enforced without sending anything:
    - only opted-in subscribers can be messaged;
    - free-form text only within 24 h of the subscriber's last inbound message;
    - outside that window, only an approved template (config whatsapp_templates)."""

    def __init__(self, templates: dict[str, str]):
        self.templates = templates

    def send(self, *, opted_in: bool, last_inbound_at: datetime | None, text: str | None = None,
             template: str | None = None, now: datetime | None = None) -> SendResult:
        now = now or datetime.now(UTC)
        if not opted_in:
            return SendResult("rejected", error="recipient has not opted in to WhatsApp messages")
        if template is not None:
            if template not in self.templates:
                return SendResult("rejected", error=f"template {template!r} is not approved")
            return SendResult("sent", provider_id=f"mock-{template}")
        in_window = last_inbound_at is not None and now - last_inbound_at <= WHATSAPP_WINDOW
        if not in_window:
            return SendResult("rejected", error="outside the 24-hour window: only approved templates may be sent")
        return SendResult("sent", provider_id="mock-freeform")

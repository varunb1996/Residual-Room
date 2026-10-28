"""`make worker`: sends due campaigns and polls the Telegram bot. One loop, no extra deps.

Telegram runs in polling mode (getUpdates), so no public webhook URL is needed:
  /start               subscribe this chat (practitioner by default)
  /segment <name>      switch to transitioning or practitioner
  /stop                unsubscribe
Replies go only to the chat that wrote to the bot; campaigns still post only to
the test channel (senders.TelegramSender).
"""

import sys
import time
from datetime import UTC, datetime

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from residual_room.channels import campaigns as svc
from residual_room.config import get_config
from residual_room.db.models import KeyValue, Subscriber
from residual_room.db.session import get_engine
from residual_room.settings import get_settings

INTERVAL = 30  # seconds between checks for due campaigns


def run_due(senders: svc.Senders) -> list[int]:
    with Session(get_engine()) as s:
        ids = svc.due_campaign_ids(s)
    sent = []
    for cid in ids:
        with Session(get_engine()) as s, s.begin():
            svc.send_campaign(s, cid, senders)
        sent.append(cid)
    return sent


def handle_command(s: Session, chat_id: str, text: str) -> str:
    segments = get_config().segments
    sub = s.scalar(select(Subscriber).where(Subscriber.telegram_chat_id == chat_id))
    cmd, _, arg = text.strip().partition(" ")
    if cmd == "/start":
        if sub is None:
            sub = svc.add_subscriber(s, segment="practitioner", telegram_chat_id=chat_id)
        sub.subscribed = True
        return f"Subscribed to Residual Room ({sub.segment}). Switch with /segment transitioning or /segment practitioner. /stop to leave."
    if cmd == "/segment":
        if sub is None:
            return "Send /start first."
        if arg.strip() not in segments:
            return f"Choose one of: {', '.join(segments)}"
        sub.segment = arg.strip()
        return f"Segment set to {sub.segment}."
    if cmd == "/stop":
        if sub is not None:
            sub.subscribed = False
            sub.unsubscribed_at = datetime.now(UTC)
        return "Unsubscribed. /start to come back."
    return "Commands: /start, /segment <transitioning|practitioner>, /stop"


def poll_telegram(http: httpx.Client) -> int:
    settings = get_settings()
    if not settings.telegram_bot_token:
        return 0
    api = f"https://api.telegram.org/bot{settings.telegram_bot_token}"
    with Session(get_engine()) as s, s.begin():
        kv = s.get(KeyValue, "telegram_offset")
        offset = int(kv.value) if kv else 0
        updates = http.get(f"{api}/getUpdates", params={"offset": offset, "timeout": 0}).json().get("result", [])
        for u in updates:
            offset = max(offset, u["update_id"] + 1)
            msg = u.get("message") or {}
            if msg.get("chat", {}).get("type") == "private" and msg.get("text"):
                chat_id = str(msg["chat"]["id"])
                reply = handle_command(s, chat_id, msg["text"])
                http.post(f"{api}/sendMessage", json={"chat_id": chat_id, "text": reply})
        if kv:
            kv.value = str(offset)
        else:
            s.add(KeyValue(key="telegram_offset", value=str(offset)))
    return len(updates)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    senders = svc.default_senders()
    http = httpx.Client(timeout=30)
    print(f"worker started: due campaigns every {INTERVAL}s, Telegram polling", flush=True)
    while True:
        try:
            if sent := run_due(senders):
                print(f"sent campaigns {sent}", flush=True)
            if n := poll_telegram(http):
                print(f"handled {n} Telegram update(s)", flush=True)
        except Exception as exc:  # keep the worker alive; the error is visible in its log
            print(f"worker error: {type(exc).__name__}: {exc}", flush=True)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    sys.exit(main())

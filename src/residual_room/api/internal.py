"""The tick: one pass of the worker, for hosts without a free background worker.

Render's free tier has no background workers or cron jobs, so a scheduled
GitHub Actions workflow POSTs here every 10 minutes. One tick = what one loop
of `make worker` does: send due campaigns, poll Telegram once. Protected by a
bearer token (TICK_TOKEN), not the owner's login.
"""

import secrets

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException

from residual_room.api.campaigns import get_senders
from residual_room.channels.campaigns import Senders
from residual_room.settings import get_settings
from residual_room.worker import poll_telegram, run_due

router = APIRouter(prefix="/api/internal", tags=["internal"])


@router.post("/tick")
def tick(authorization: str = Header(default=""), senders: Senders = Depends(get_senders)):
    token = get_settings().tick_token
    if not token:
        raise HTTPException(503, "TICK_TOKEN is not set: the tick endpoint is disabled")
    given = authorization.removeprefix("Bearer ").strip()
    if not secrets.compare_digest(given, token):
        raise HTTPException(401, "bad tick token")
    sent = run_due(senders)
    with httpx.Client(timeout=30) as http:
        updates = poll_telegram(http)
    return {"sent_campaigns": sent, "telegram_updates": updates}

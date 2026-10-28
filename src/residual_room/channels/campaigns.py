"""Campaigns: schedule an approved piece to segments × channels, then send it.

Scheduling moves the piece approved -> scheduled; cancelling moves it back.
Sending (by the worker when due, or at once for "send now") formats the
approved version per channel, makes tracked links, records one delivery per
recipient, and moves the piece to sent if anything was delivered.
"""

import secrets
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from residual_room.channels.content import build_content
from residual_room.channels.links import Linker
from residual_room.channels.render import render_email, render_telegram, render_whatsapp_template
from residual_room.channels.senders import EmailSender, SendResult, TelegramSender, WhatsAppMock
from residual_room.config import PublicationConfig, get_config
from residual_room.corpus.manifest import Manifest, load_manifest
from residual_room.db.models import Campaign, Delivery, Piece, PieceVersion, Subscriber
from residual_room.settings import Settings, get_settings
from residual_room.workflow.service import Conflict, NotFound, WorkflowError, _audit, latest_version
from residual_room.workflow.states import Action, State, next_state

CHANNELS = ("email", "telegram", "whatsapp")


@dataclass
class Senders:
    email: EmailSender
    telegram: TelegramSender
    whatsapp: WhatsAppMock


def default_senders(settings: Settings | None = None) -> Senders:
    settings = settings or get_settings()
    return Senders(EmailSender(settings), TelegramSender(settings), WhatsAppMock(get_config().whatsapp_templates))


# ---------- subscribers ----------


def add_subscriber(s: Session, *, segment: str, email: str | None = None, telegram_chat_id: str | None = None,
                   whatsapp: str | None = None) -> Subscriber:
    if segment not in get_config().segments:
        raise WorkflowError(f"unknown segment {segment}")
    sub = Subscriber(segment=segment, email=email, telegram_chat_id=telegram_chat_id, whatsapp=whatsapp,
                     subscribed=True, opted_in_at=datetime.now(UTC), token=secrets.token_urlsafe(12))
    s.add(sub)
    s.flush()
    return sub


def unsubscribe(s: Session, token: str) -> Subscriber:
    sub = s.scalar(select(Subscriber).where(Subscriber.token == token))
    if sub is None:
        raise NotFound("unknown unsubscribe link")
    sub.subscribed = False
    return sub


# ---------- campaigns ----------


def create_campaign(s: Session, piece_id: int, segments: list[str], channels: list[str],
                    scheduled_at: datetime | None = None, actor: str = "owner") -> Campaign:
    piece = s.get(Piece, piece_id, with_for_update=True)
    if piece is None:
        raise NotFound(f"piece {piece_id} not found")
    if unknown := set(channels) - set(CHANNELS):
        raise WorkflowError(f"unknown channels {sorted(unknown)}")
    if unknown := set(segments) - set(get_config().segments):
        raise WorkflowError(f"unknown segments {sorted(unknown)}")
    if not channels or not segments:
        raise WorkflowError("pick at least one segment and one channel")
    before = piece.state
    try:
        piece.state = next_state(State(piece.state), Action.SCHEDULE)
    except ValueError as exc:
        raise Conflict(f"only approved pieces can be scheduled: {exc}") from exc
    campaign = Campaign(piece_id=piece_id, version_id=latest_version(s, piece_id).id, segments=segments,
                        channels=channels, scheduled_at=scheduled_at or datetime.now(UTC), status="scheduled")
    s.add(campaign)
    s.flush()
    _audit(s, piece, actor, "schedule", before, piece.state,
           {"campaign": campaign.id, "segments": segments, "channels": channels,
            "scheduled_at": campaign.scheduled_at.isoformat()})
    return campaign


def cancel_campaign(s: Session, campaign_id: int, actor: str = "owner") -> Campaign:
    campaign = _campaign(s, campaign_id)
    if campaign.status != "scheduled":
        raise Conflict(f"only scheduled campaigns can be cancelled; this one is {campaign.status}")
    piece = s.get(Piece, campaign.piece_id, with_for_update=True)
    before = piece.state
    piece.state = next_state(State(piece.state), Action.CANCEL)
    campaign.status = "cancelled"
    _audit(s, piece, actor, "cancel", before, piece.state, {"campaign": campaign.id})
    return campaign


def due_campaign_ids(s: Session, now: datetime | None = None) -> list[int]:
    now = now or datetime.now(UTC)
    return list(s.scalars(select(Campaign.id).where(Campaign.status == "scheduled", Campaign.scheduled_at <= now)))


def send_campaign(s: Session, campaign_id: int, senders: Senders, settings: Settings | None = None,
                  config: PublicationConfig | None = None, manifest: Manifest | None = None) -> Campaign:
    settings, config, manifest = settings or get_settings(), config or get_config(), manifest or load_manifest()
    campaign = _campaign(s, campaign_id, lock=True)
    if campaign.status != "scheduled":
        raise Conflict(f"campaign {campaign_id} is {campaign.status}")
    campaign.status = "sending"
    piece, version = s.get(Piece, campaign.piece_id), s.get(PieceVersion, campaign.version_id)
    content = build_content(piece.title, version.text, version.segments, version.generation, manifest, config)
    base = settings.public_base_url.rstrip("/")

    for segment in campaign.segments:
        for channel in campaign.channels:
            linker = Linker(s, base, campaign.id, piece.id, segment, channel)
            for recipient, result in _send(s, channel, segment, content, linker, senders, settings, config, base):
                s.add(Delivery(campaign_id=campaign.id, channel=channel, segment=segment, recipient=recipient,
                               status=result.status, provider_id=result.provider_id, error=result.error))
    s.flush()

    statuses = list(s.scalars(select(Delivery.status).where(Delivery.campaign_id == campaign.id)))
    counts = {st: statuses.count(st) for st in set(statuses)}
    campaign.sent_at = datetime.now(UTC)
    piece = s.get(Piece, campaign.piece_id, with_for_update=True)
    if counts.get("sent"):
        campaign.status = "sent"
        before = piece.state
        piece.state = next_state(State(piece.state), Action.SEND)
        _audit(s, piece, "system", "send", before, piece.state, {"campaign": campaign.id, "deliveries": counts})
    else:
        # Nothing delivered: the piece stays scheduled so it can be cancelled and retried.
        campaign.status = "failed"
        _audit(s, piece, "system", "send_failed", piece.state, piece.state,
               {"campaign": campaign.id, "deliveries": counts})
    return campaign


def _send(s, channel, segment, content, linker, senders, settings, config, base):
    subs = list(s.scalars(select(Subscriber).where(Subscriber.subscribed, Subscriber.segment == segment)))
    if channel == "email":
        for sub in (x for x in subs if x.email):
            unsub = f"{base}/u/{sub.token}"
            yield sub.email, senders.email.send(sub.email, render_email(content, linker, unsub, config), unsub)
    elif channel == "telegram":
        # Campaigns post to the test channel only (a broadcast channel, not per subscriber).
        chat = settings.telegram_test_chat_id
        if not chat:
            yield "(no test chat)", SendResult("failed", error="TELEGRAM_TEST_CHAT_ID is not set")
        else:
            yield chat, senders.telegram.send(chat, render_telegram(content, linker, config))
    elif channel == "whatsapp":
        template, params = render_whatsapp_template(content, linker, config)
        for sub in (x for x in subs if x.whatsapp):
            yield sub.whatsapp, senders.whatsapp.send(
                opted_in=sub.opted_in_at is not None, last_inbound_at=sub.whatsapp_last_inbound_at,
                template=template)


def _campaign(s: Session, campaign_id: int, lock: bool = False) -> Campaign:
    campaign = s.get(Campaign, campaign_id, with_for_update=lock)
    if campaign is None:
        raise NotFound(f"campaign {campaign_id} not found")
    return campaign

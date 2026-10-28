"""Campaigns, subscribers, tracked-link redirects, unsubscribe, and the WhatsApp mock."""

from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from residual_room.api.deps import get_session
from residual_room.channels import campaigns as svc
from residual_room.channels.content import build_content
from residual_room.channels.links import PreviewLinker
from residual_room.channels.render import fill_template, render_email, render_telegram, render_whatsapp_template
from residual_room.config import get_config
from residual_room.corpus.manifest import load_manifest
from residual_room.db.models import Campaign, Click, Delivery, Link, Piece, Subscriber
from residual_room.db.session import get_engine
from residual_room.workflow.service import NotFound, latest_version

router = APIRouter(tags=["campaigns"])

Segment = Literal["transitioning", "practitioner"]
Channel = Literal["email", "telegram", "whatsapp"]


def get_senders() -> svc.Senders:
    return svc.default_senders()


# ---------- schemas ----------


class CampaignIn(BaseModel):
    piece_id: int
    segments: list[Segment] = Field(min_length=1)
    channels: list[Channel] = Field(min_length=1)
    scheduled_at: datetime | None = None  # None = send now


class DeliveryOut(BaseModel):
    channel: str
    segment: str
    recipient: str
    status: str
    error: str | None


class ClickCount(BaseModel):
    channel: str
    segment: str
    label: str
    clicks: int


class CampaignOut(BaseModel):
    id: int
    piece_id: int
    piece_title: str
    segments: list[str]
    channels: list[str]
    scheduled_at: datetime
    status: str
    sent_at: datetime | None
    deliveries: list[DeliveryOut]
    clicks: list[ClickCount]


class PreviewOut(BaseModel):
    email_subject: str
    email_html: str
    email_text: str
    telegram_html: str
    whatsapp_text: str


class SubscriberIn(BaseModel):
    segment: Segment
    email: str | None = None
    whatsapp: str | None = None


class SubscriberOut(BaseModel):
    id: int
    segment: str
    email: str | None
    telegram_chat_id: str | None
    whatsapp: str | None
    subscribed: bool
    whatsapp_last_inbound_at: datetime | None


class WhatsAppInbound(BaseModel):
    whatsapp: str
    text: str


class WhatsAppSendIn(BaseModel):
    whatsapp: str
    text: str | None = None
    template: str | None = None


class SendResultOut(BaseModel):
    status: str
    error: str | None
    rendered: str | None


# ---------- campaigns ----------


@router.get("/api/campaigns", response_model=list[CampaignOut])
def list_campaigns(s: Session = Depends(get_session)):
    return [_campaign_out(s, c) for c in s.scalars(select(Campaign).order_by(Campaign.id.desc()))]


@router.post("/api/campaigns", response_model=CampaignOut, status_code=201)
def create_campaign(body: CampaignIn, background: BackgroundTasks, s: Session = Depends(get_session),
                    senders: svc.Senders = Depends(get_senders)):
    campaign = svc.create_campaign(s, body.piece_id, body.segments, body.channels, body.scheduled_at)
    s.commit()
    if campaign.scheduled_at <= datetime.now(UTC):
        background.add_task(_send_now, campaign.id, senders)  # "send now" doesn't wait for the worker
    return _campaign_out(s, campaign)


@router.post("/api/campaigns/{campaign_id}/cancel", response_model=CampaignOut)
def cancel(campaign_id: int, s: Session = Depends(get_session)):
    campaign = svc.cancel_campaign(s, campaign_id)
    s.flush()
    return _campaign_out(s, campaign)


@router.get("/api/pieces/{piece_id}/preview", response_model=PreviewOut)
def preview(piece_id: int, s: Session = Depends(get_session)):
    piece = s.get(Piece, piece_id)
    if piece is None:
        raise NotFound(f"piece {piece_id} not found")
    v, cfg = latest_version(s, piece_id), get_config()
    content = build_content(piece.title, v.text, v.segments, v.generation, load_manifest(), cfg)
    email = render_email(content, PreviewLinker(), "(unsubscribe link)", cfg)
    name, params = render_whatsapp_template(content, PreviewLinker(), cfg)
    return PreviewOut(email_subject=email.subject, email_html=email.html, email_text=email.text,
                      telegram_html=render_telegram(content, PreviewLinker(), cfg),
                      whatsapp_text=fill_template(name, params, cfg))


# ---------- subscribers and the WhatsApp mock ----------


@router.get("/api/subscribers", response_model=list[SubscriberOut])
def list_subscribers(s: Session = Depends(get_session)):
    return [SubscriberOut.model_validate(x, from_attributes=True) for x in s.scalars(select(Subscriber))]


@router.post("/api/subscribers", response_model=SubscriberOut, status_code=201)
def add_subscriber(body: SubscriberIn, s: Session = Depends(get_session)):
    return SubscriberOut.model_validate(svc.add_subscriber(s, **body.model_dump()), from_attributes=True)


@router.post("/api/whatsapp/inbound", response_model=SubscriberOut)
def whatsapp_inbound(body: WhatsAppInbound, s: Session = Depends(get_session)):
    """Mock: the subscriber messaged us, which opens the 24-hour free-form window."""
    sub = _whatsapp_sub(s, body.whatsapp)
    sub.whatsapp_last_inbound_at = datetime.now(UTC)
    s.flush()
    return SubscriberOut.model_validate(sub, from_attributes=True)


@router.post("/api/whatsapp/send", response_model=SendResultOut)
def whatsapp_send(body: WhatsAppSendIn, s: Session = Depends(get_session),
                  senders: svc.Senders = Depends(get_senders)):
    sub, cfg = _whatsapp_sub(s, body.whatsapp), get_config()
    result = senders.whatsapp.send(opted_in=sub.opted_in_at is not None and sub.subscribed,
                                   last_inbound_at=sub.whatsapp_last_inbound_at,
                                   text=body.text, template=body.template)
    rendered = body.text if body.template is None else (
        fill_template(body.template, [], cfg) if body.template in cfg.whatsapp_templates else None)
    return SendResultOut(status=result.status, error=result.error, rendered=rendered)


# ---------- public links (in sent messages) ----------


@router.get("/r/{token}", include_in_schema=False)
def redirect(token: str, s: Session = Depends(get_session)):
    link = s.scalar(select(Link).where(Link.token == token))
    if link is None:
        raise NotFound("unknown link")
    s.add(Click(link_id=link.id))  # campaign, segment and channel come from the link; nothing personal
    return RedirectResponse(link.target_url, status_code=302)


@router.get("/u/{token}", include_in_schema=False, response_class=HTMLResponse)
def unsubscribe(token: str, s: Session = Depends(get_session)):
    svc.unsubscribe(s, token)
    return HTMLResponse("<p style='font-family:system-ui'>You're unsubscribed from Residual Room.</p>")


# ---------- helpers ----------


def _send_now(campaign_id: int, senders: svc.Senders) -> None:
    with Session(get_engine()) as s, s.begin():
        svc.send_campaign(s, campaign_id, senders)


def _whatsapp_sub(s: Session, number: str) -> Subscriber:
    sub = s.scalar(select(Subscriber).where(Subscriber.whatsapp == number))
    if sub is None:
        raise NotFound(f"no subscriber with WhatsApp number {number}")
    return sub


def _campaign_out(s: Session, c: Campaign) -> CampaignOut:
    deliveries = s.scalars(select(Delivery).where(Delivery.campaign_id == c.id).order_by(Delivery.id))
    clicks = s.execute(
        select(Link.channel, Link.segment, Link.label, func.count(Click.id))
        .join(Click, Click.link_id == Link.id).where(Link.campaign_id == c.id)
        .group_by(Link.channel, Link.segment, Link.label)
    ).all()
    return CampaignOut(
        id=c.id, piece_id=c.piece_id, piece_title=s.get(Piece, c.piece_id).title, segments=c.segments,
        channels=c.channels, scheduled_at=c.scheduled_at, status=c.status, sent_at=c.sent_at,
        deliveries=[DeliveryOut.model_validate(d, from_attributes=True) for d in deliveries],
        clicks=[ClickCount(channel=ch, segment=sg, label=lb, clicks=n) for ch, sg, lb, n in clicks],
    )

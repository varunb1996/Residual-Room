"""Tracked links. Every outbound link becomes {PUBLIC_BASE_URL}/r/{token}; the
redirect endpoint logs a click (campaign, segment, channel; no personal data)."""

import secrets

from sqlalchemy.orm import Session

from residual_room.db.models import Link


class Linker:
    """Makes tracked links for one campaign × segment × channel, one per target."""

    def __init__(self, s: Session, base_url: str, campaign_id: int, piece_id: int, segment: str, channel: str):
        self.s, self.base = s, base_url.rstrip("/")
        self.ctx = {"campaign_id": campaign_id, "piece_id": piece_id, "segment": segment, "channel": channel}
        self.made: dict[str, str] = {}

    def url(self, target: str, label: str) -> str:
        if target not in self.made:
            token = secrets.token_urlsafe(9)
            self.s.add(Link(token=token, label=label, target_url=target, **self.ctx))
            self.made[target] = f"{self.base}/r/{token}"
        return self.made[target]


class PreviewLinker:
    """For previews before a campaign exists: shows the real target, tracks nothing."""

    def url(self, target: str, label: str) -> str:
        return target

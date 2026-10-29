"""Public explainer pages: open without the login, sent pieces only, sources as links."""

import base64
import uuid
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

import residual_room.api.public as public
from residual_room.api.main import app
from residual_room.channels.content import build_content
from residual_room.channels.relink import relink
from residual_room.channels.render import render_web
from residual_room.config import get_config
from residual_room.corpus.manifest import load_manifest
from residual_room.db.models import Campaign, Link, Piece, PieceVersion
from residual_room.db.session import get_engine
from residual_room.settings import get_settings

SEGMENTS = [
    {"type": "exempt", "exempt_type": "framing", "text": "Paper to Practice: what MT-Bench measured."},
    {"type": "claim", "text": "Claude-v1 kept its verdict 23.8% of the time.", "chunk_ids": ["ext-mtbench-paper#0015"]},
    {"type": "exempt", "exempt_type": "cta", "text": "Read the explainer."},
]


@pytest.fixture
def secured(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "admin_password", "correct horse")
    monkeypatch.setattr(s, "admin_user", "owner")
    with TestClient(app) as c:
        yield c


def _piece(s, tag, state, campaign_status, segment):
    piece = Piece(title=f"{state} {tag}", kind="newsletter", segment=segment, state=state)
    s.add(piece)
    s.flush()
    version = PieceVersion(piece_id=piece.id, n=1, text=" ".join(x["text"] for x in SEGMENTS), source="generated",
                           segments=SEGMENTS, generation={"explainer": "exp-006"})
    s.add(version)
    s.flush()
    now = datetime.now(UTC)
    campaign = Campaign(piece_id=piece.id, version_id=version.id, segments=[segment], channels=["telegram"],
                        scheduled_at=now, status=campaign_status, sent_at=now if campaign_status == "sent" else None)
    s.add(campaign)
    s.flush()
    return piece, campaign


@pytest.fixture
def pieces():
    """A sent piece and an approved-but-unsent one about exp-006; deleted afterwards (cascades)."""
    tag = uuid.uuid4().hex[:8]
    with Session(get_engine()) as s, s.begin():
        sent, campaign = _piece(s, tag, "sent", "sent", "transitioning")
        unsent, _ = _piece(s, tag, "approved", "scheduled", "practitioner")
        ids = {"sent": sent.id, "unsent": unsent.id, "campaign": campaign.id, "tag": tag}
    yield ids
    with Session(get_engine()) as s, s.begin():
        s.execute(delete(Piece).where(Piece.id.in_([ids["sent"], ids["unsent"]])))


def test_public_pages_need_no_login_while_screens_still_do(secured):
    assert secured.get("/api/pieces").status_code == 401
    for path in ("/explainers", "/explainers/exp-004", "/subscribe"):
        assert secured.get(path).status_code == 200, path


def test_unknown_explainer_is_404(secured):
    assert secured.get("/explainers/exp-999").status_code == 404


def test_only_sent_pieces_appear(secured, pieces):
    page = secured.get("/explainers/exp-006").text
    assert f"sent {pieces['tag']}" in page
    assert f"approved {pieces['tag']}" not in page  # approved and scheduled, not yet sent


def test_sources_are_links_from_the_manifest_never_text(secured, pieces):
    page = secured.get("/explainers/exp-006").text
    assert 'href="https://arxiv.org/abs/2306.05685"' in page
    with Session(get_engine()) as s:
        version = s.scalar(select(PieceVersion).where(PieceVersion.piece_id == pieces["sent"]))
    content = build_content("t", version.text, version.segments, version.generation, load_manifest(), get_config())
    assert "Read the explainer" not in render_web(content, set())  # the reader is already here
    assert "(since superseded)" in render_web(content, {"ext-mtbench-paper"})


def test_explainer_without_a_sent_issue_lists_its_sources(secured, monkeypatch):
    monkeypatch.setattr(public, "latest_sent", lambda s, eid: {})
    page = secured.get("/explainers/exp-006").text
    assert "No issue sent yet" in page and "https://arxiv.org/abs/2306.05685" in page


def test_relink_rewrites_only_placeholder_targets(pieces):
    old = f"https://placeholder-{pieces['tag']}.example"  # unique, so live links are never touched
    with Session(get_engine()) as s, s.begin():
        for token, target in ((f"a{pieces['tag']}", old + "/explainers/exp-006"),
                              (f"b{pieces['tag']}", "https://arxiv.org/abs/2306.05685")):
            s.add(Link(token=token, campaign_id=pieces["campaign"], piece_id=pieces["sent"], segment="transitioning",
                       channel="telegram", label="cta", target_url=target))
    with Session(get_engine()) as s, s.begin():
        assert relink(s, "https://app.example/", old=old) == 1
    with Session(get_engine()) as s:
        targets = set(s.scalars(select(Link.target_url).where(Link.campaign_id == pieces["campaign"])))
    assert targets == {"https://app.example/explainers/exp-006", "https://arxiv.org/abs/2306.05685"}

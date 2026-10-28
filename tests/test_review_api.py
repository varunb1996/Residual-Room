"""The review workflow end to end through the API, against the real database.
The checker is a fake (no LLM calls); the rules under test are the workflow's."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete
from sqlalchemy.orm import Session

from residual_room.api.deps import get_checker
from residual_room.api.main import app
from residual_room.checker.types import CheckReport, RuleHit, Segment
from residual_room.db.models import CorpusVersion, Piece
from residual_room.db.session import get_engine

pytestmark = pytest.mark.db


def fake_checker(text, segment, issue_date):
    if "FAIL" in text:
        raise RuntimeError("judge unreachable")
    segs = [Segment("New issues land every Tuesday.", "claim"), Segment("Read the explainer.", "exempt", "cta")]
    if "BLOCK" in text:
        hits = [RuleHit("P5", "repeats a superseded source", 0, "int-schedule-v0#0000"),
                RuleHit("S1", "hype word", 0, "magic")]
        return CheckReport(segs, hits), "fake-judge", "corpus-v1"
    return CheckReport(segs, [], {0: "int-schedule-v1#0000"}), "fake-judge", "corpus-v1"


@pytest.fixture
def client():
    app.dependency_overrides[get_checker] = lambda: fake_checker
    created: list[int] = []
    with TestClient(app) as c:
        c.created = created
        yield c
    app.dependency_overrides.clear()
    with Session(get_engine()) as s, s.begin():
        s.execute(delete(Piece).where(Piece.id.in_(created)))


def new_piece(client, text="Clean text.", segment="practitioner"):
    r = client.post("/api/pieces", json={"title": "test piece", "segment": segment, "text": text})
    assert r.status_code == 201, r.text
    client.created.append(r.json()["id"])
    return r.json()


def submit(client, pid):
    # TestClient runs the background check before returning.
    assert client.post(f"/api/pieces/{pid}/submit").status_code == 202
    return client.get(f"/api/pieces/{pid}").json()


def test_blocked_piece_needs_an_override_with_reason_before_approval(client):
    pid = new_piece(client, "BLOCK this")["id"]
    piece = submit(client, pid)
    assert piece["state"] == "in_review" and piece["verdict"] == "block"
    assert piece["blockers"] == [0]

    r = client.post(f"/api/pieces/{pid}/approve")
    assert r.status_code == 409 and "P5" in r.json()["detail"]

    assert client.post(f"/api/pieces/{pid}/overrides", json={"hit_index": 0, "reason": ""}).status_code == 422
    assert client.post(f"/api/pieces/{pid}/overrides", json={"hit_index": 0, "reason": "   "}).status_code == 400
    r = client.post(f"/api/pieces/{pid}/overrides", json={"hit_index": 1, "reason": "it's fine"})
    assert r.status_code == 400 and "warning" in r.json()["detail"]  # only blocking hits are overridden

    r = client.post(f"/api/pieces/{pid}/overrides",
                    json={"hit_index": 0, "reason": "Tuesday is a quote from the old schedule, marked as such"})
    assert r.status_code == 200 and r.json()["blockers"] == []
    assert r.json()["check"]["hits"][0]["override_reason"].startswith("Tuesday is a quote")

    r = client.post(f"/api/pieces/{pid}/approve")
    assert r.status_code == 200 and r.json()["state"] == "approved"

    actions = [a["action"] for a in r.json()["audit"]]
    assert actions == ["create", "submit", "checks_done", "approve_refused", "override", "approve"]
    override_event = r.json()["audit"][4]
    assert override_event["detail"]["rule"] == "P5" and "old schedule" in override_event["detail"]["reason"]


def test_clean_piece_approves_and_links_claims_to_chunks(client):
    pid = new_piece(client)["id"]
    piece = submit(client, pid)
    assert piece["verdict"] == "pass" and piece["blockers"] == []
    assert piece["check"]["segments"][0]["support"] == "int-schedule-v1#0000"
    assert client.post(f"/api/pieces/{pid}/approve").json()["state"] == "approved"


def test_cannot_approve_before_checks(client):
    pid = new_piece(client)["id"]
    r = client.post(f"/api/pieces/{pid}/approve")
    assert r.status_code == 409 and "draft" in r.json()["detail"]


def test_editing_makes_a_new_version_and_resets_to_draft(client):
    pid = new_piece(client, "BLOCK v1")["id"]
    submit(client, pid)
    client.post(f"/api/pieces/{pid}/overrides", json={"hit_index": 0, "reason": "ok on v1"})
    r = client.put(f"/api/pieces/{pid}/text", json={"text": "BLOCK v2"})
    assert r.json()["state"] == "draft" and r.json()["version"] == 2
    assert r.json()["check"] is None  # v2 hasn't been checked
    piece = submit(client, pid)
    assert piece["blockers"] == [0]  # the v1 override does not carry over


def test_reject_needs_a_reason_then_edit_returns_to_draft(client):
    pid = new_piece(client)["id"]
    submit(client, pid)
    assert client.post(f"/api/pieces/{pid}/reject", json={"reason": " "}).status_code == 400
    r = client.post(f"/api/pieces/{pid}/reject", json={"reason": "wrong angle for practitioners"})
    assert r.json()["state"] == "rejected"
    assert client.put(f"/api/pieces/{pid}/text", json={"text": "rewritten"}).json()["state"] == "draft"


def test_failed_checks_return_the_piece_to_draft(client):
    pid = new_piece(client, "FAIL please")["id"]
    piece = submit(client, pid)
    assert piece["state"] == "draft" and piece["check"]["status"] == "failed"
    assert "judge unreachable" in piece["check"]["error"]


def test_chunk_endpoint_returns_source_text(client):
    with Session(get_engine()) as s:
        if s.get(CorpusVersion, "corpus-v1") is None:
            pytest.skip("corpus-v1 not ingested")
    r = client.get("/api/chunks/int-schedule-v1#0000".replace("#", "%23"))
    assert r.status_code == 200
    assert r.json()["status"] == "current" and "Thursdays" in r.json()["text"]
    assert client.get("/api/chunks/nope%230000").status_code == 404

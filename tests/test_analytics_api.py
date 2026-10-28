"""Analytics API against the real database, with fake narration models."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from residual_room.api.analytics import get_narration_models
from residual_room.api.main import app
from residual_room.analytics.service import refresh
from residual_room.db.models import Narration
from residual_room.db.session import get_engine

pytestmark = pytest.mark.db


class FakeModel:
    model = "fake"
    max_concurrency = 1

    def __init__(self, fn):
        self.fn = fn

    def ask_json(self, system, user, schema):
        return schema.model_validate(self.fn(system, user))


@pytest.fixture
def client():
    """Tests share the app's real seeded dataset, so afterwards it is regenerated
    (deterministic) and only the narrations a test created are removed."""
    with Session(get_engine()) as s:
        before = s.scalar(select(func.max(Narration.id))) or 0
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    with Session(get_engine()) as s, s.begin():
        s.execute(delete(Narration).where(Narration.id > before))
        refresh(s, "seeded")


def test_seeded_and_real_are_separate_and_labelled(client):
    seeded = client.post("/api/analytics/seeded/refresh").json()
    assert any(i["status"] == "insight" for i in seeded)
    view = client.get("/api/analytics/seeded").json()
    assert view["label"].startswith("Seeded") and {r["id"] for r in view["rows"]}
    real = client.get("/api/analytics/real").json()
    assert real["label"].startswith("Real")
    assert not {r["id"] for r in real["rows"]} & {r["id"] for r in view["rows"]}


def test_real_data_today_is_not_enough_data(client):
    real = client.post("/api/analytics/real/refresh").json()
    assert real and all(i["status"] == "not_enough_data" for i in real)


def test_narration_job_generates_then_checks(client):
    insights = client.post("/api/analytics/seeded/refresh").json()
    gap = next(i for i in insights if i["rule"] == "channel_gap")
    hi, lo, ratio = gap["row_ids"]
    rows = {r["id"]: r for r in client.get("/api/analytics/seeded").json()["rows"]}

    def generate(system, user):
        return {"segments": [{"type": "claim", "chunk_ids": [f"m:{hi}", f"m:{lo}"],
                              "text": f"In seeded data, practitioners clicked Telegram {rows[hi]['value'] * 100:.1f}% "
                                      f"of the time vs {rows[lo]['value'] * 100:.1f}% for email."}]}

    def judge(system, user):
        if "Label these sentences" in user:
            return {"sentences": []}
        if "Check ONE claim" in system:
            return {"supported": True, "numbers": "match", "evidence_chunk": f"m:{hi}", "reason": "ok"}
        return {"hits": []}

    app.dependency_overrides[get_narration_models] = lambda: (FakeModel(generate), FakeModel(judge))
    r = client.post("/api/analytics/seeded/narrate")
    assert r.status_code == 202
    n = client.get(f"/api/analytics/narrations/{r.json()['id']}").json()
    assert n["status"] == "done" and n["verdict"] == "pass", n
    assert f"[cite:m:{hi}]" in n["text"]
    assert client.get("/api/analytics/seeded").json()["narration"]["id"] == n["id"]


def test_narration_missing_the_seeded_label_gets_one_corrective_retry(client):
    insights = client.post("/api/analytics/seeded/refresh").json()
    hi = next(i for i in insights if i["rule"] == "channel_gap")["row_ids"][0]
    rows = {r["id"]: r for r in client.get("/api/analytics/seeded").json()["rows"]}
    pct = f"{rows[hi]['value'] * 100:.1f}%"
    prompts = []

    def generate(system, user):
        prompts.append(user)
        label = "In seeded data, " if len(prompts) > 1 else ""  # first draft forgets the label
        return {"segments": [{"type": "claim", "chunk_ids": [f"m:{hi}"],
                              "text": f"{label}practitioners clicked Telegram {pct} of the time."}]}

    def judge(system, user):
        if "Check ONE claim" in system:
            return {"supported": True, "numbers": "match", "evidence_chunk": f"m:{hi}", "reason": "ok"}
        return {"sentences": []} if "Label these sentences" in user else {"hits": []}

    app.dependency_overrides[get_narration_models] = lambda: (FakeModel(generate), FakeModel(judge))
    n = client.get(f"/api/analytics/narrations/{client.post('/api/analytics/seeded/narrate').json()['id']}").json()
    assert len(prompts) == 2 and "doesn't say it is seeded" in prompts[1]
    assert n["verdict"] == "pass" and "seeded" in n["text"]

"""Login protection and the tick endpoint (no network, no database writes)."""

import base64

import pytest
from fastapi.testclient import TestClient

import residual_room.api.internal as internal
from residual_room.api.main import app
from residual_room.api.security import check_deploy_safety
from residual_room.settings import get_settings


@pytest.fixture
def secured(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "admin_password", "correct horse")
    monkeypatch.setattr(s, "admin_user", "owner")
    monkeypatch.setattr(s, "tick_token", "tick-secret")
    with TestClient(app) as c:
        yield c


def auth(user, password):
    return {"Authorization": "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()}


@pytest.mark.parametrize("path", ["/api/pieces", "/api/campaigns", "/api/analytics/real"])
def test_screens_and_api_need_the_login(secured, path):
    assert secured.get(path).status_code == 401
    assert secured.get(path, headers=auth("owner", "wrong")).status_code == 401
    assert secured.get(path, headers=auth("owner", "correct horse")).status_code != 401


def test_reader_and_machine_paths_stay_open(secured):
    assert secured.get("/health").status_code in (200, 503)
    assert secured.get("/r/unknown-token", follow_redirects=False).status_code == 404  # reached the route
    assert secured.get("/u/unknown-token").status_code == 404


def test_tick_needs_its_own_token(secured, monkeypatch):
    monkeypatch.setattr(internal, "run_due", lambda senders: [7])
    monkeypatch.setattr(internal, "poll_telegram", lambda http: 2)  # never touch the real bot in tests
    assert secured.post("/api/internal/tick").status_code == 401
    assert secured.post("/api/internal/tick", headers={"Authorization": "Bearer nope"}).status_code == 401
    r = secured.post("/api/internal/tick", headers={"Authorization": "Bearer tick-secret"})
    assert r.status_code == 200 and r.json() == {"sent_campaigns": [7], "telegram_updates": 2}


def test_tick_is_disabled_without_a_token(monkeypatch):
    monkeypatch.setattr(get_settings(), "tick_token", None)
    with TestClient(app) as c:
        assert c.post("/api/internal/tick", headers={"Authorization": "Bearer x"}).status_code == 503


def test_refuses_to_start_on_render_without_a_password(monkeypatch):
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setattr(get_settings(), "admin_password", None)
    with pytest.raises(RuntimeError, match="ADMIN_PASSWORD"):
        check_deploy_safety()

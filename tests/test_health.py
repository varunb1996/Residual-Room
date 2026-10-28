import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import create_engine

from residual_room.api.main import app
from residual_room.db.session import get_engine
from residual_room.settings import get_settings


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_health_is_503_when_database_unreachable(client):
    # Port 1 on localhost refuses connections immediately.
    dead = create_engine("postgresql+psycopg://u:p@127.0.0.1:1/x", connect_args={"connect_timeout": 2})
    app.dependency_overrides[get_engine] = lambda: dead

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {"status": "error", "database": "unreachable"}


@pytest.mark.db
def test_health_is_200_against_real_postgres_with_pgvector(client):
    # Fails (not skips) without a database, so `make test` can't pass silently.
    try:
        get_settings()
    except ValidationError:
        pytest.fail("DATABASE_URL is not set: copy .env.example to .env and fill it in")

    response = client.get("/health")

    assert response.status_code == 200, response.json()
    body = response.json()
    assert body["status"] == "ok"
    assert body["pgvector"]  # e.g. "0.8.0"

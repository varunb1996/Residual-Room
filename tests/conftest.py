import pytest

from residual_room.settings import get_settings


@pytest.fixture(autouse=True)
def no_login(monkeypatch):
    """API tests run without the login even if the local .env sets ADMIN_PASSWORD.
    Tests of the login itself set it again (see the `secured` fixtures)."""
    monkeypatch.setattr(get_settings(), "admin_password", None)

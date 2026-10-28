from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str

    # Channels (Prompt 7). All optional: a channel without its key reports
    # "not configured" on send instead of failing at startup.
    resend_api_key: str | None = None
    email_from: str = "Residual Room <onboarding@resend.dev>"
    test_emails: str = ""  # comma-separated; mail never goes anywhere else
    telegram_bot_token: str | None = None
    telegram_test_chat_id: str | None = None
    # Not "localhost": browsers may try IPv6 ::1 first, where uvicorn isn't listening.
    public_base_url: str = "http://127.0.0.1:8000"

    @field_validator("database_url")
    @classmethod
    def use_psycopg3(cls, url: str) -> str:
        # Neon and Render hand out plain postgres:// URLs; SQLAlchemy would pick
        # psycopg2 for those, but we install psycopg 3.
        for prefix in ("postgresql://", "postgres://"):
            if url.startswith(prefix):
                return "postgresql+psycopg://" + url.removeprefix(prefix)
        return url

    @property
    def test_email_list(self) -> set[str]:
        return {e.strip().lower() for e in self.test_emails.split(",") if e.strip()}


@lru_cache
def get_settings() -> Settings:
    # Loaded lazily so the app and unit tests import without a .env present.
    return Settings()

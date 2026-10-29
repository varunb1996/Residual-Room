#!/bin/sh
# Container start (Render): migrate the database, then serve the API and screens.
set -e
uv run --no-sync alembic upgrade head
exec uv run --no-sync uvicorn residual_room.api.main:app --host 0.0.0.0 --port "${PORT:-8000}"

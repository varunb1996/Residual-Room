.PHONY: setup fetch ingest query eval client web migrate run test

setup:
	uv sync

SPLIT ?= all

fetch:
	uv run python -m residual_room.corpus.fetch

ingest:
	uv run python -m residual_room.corpus.ingest

# make query Q="position bias in LLM judges"
query:
	uv run python scripts/query.py "$(Q)"

# make eval SPLIT=test   (needs make fetch + make ingest, and the judge's API key)
eval:
	uv run python -m residual_room.evaluation.run --split $(SPLIT)

# Regenerate the typed TS client from FastAPI's OpenAPI schema.
client:
	uv run python scripts/export_openapi.py
	cd web && npm run gen

# Build the review screen; FastAPI then serves it at /.
web:
	cd web && npm ci && npm run build

migrate:
	uv run alembic upgrade head

run:
	uv run uvicorn residual_room.api.main:app --reload

test:
	uv run pytest

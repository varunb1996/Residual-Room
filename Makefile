.PHONY: setup fetch ingest query scout approve-source generate eval client web migrate run worker test

setup:
	uv sync

SPLIT ?= all

# make fetch [IDS="ext-a ext-b"]  (default: every external source)
fetch:
	uv run python -m residual_room.corpus.fetch $(IDS)

# make ingest [VERSION=corpus-v2]  (a new version; existing versions are never overwritten)
VERSION ?= corpus-v1
ingest:
	uv run python -m residual_room.corpus.ingest --version $(VERSION)

# make scout TOPIC="Rerankers for RAG" SERIES="Failure Modes"  (writes data/proposed_sources.yaml)
scout:
	uv run python -m residual_room.scout.scout --topic "$(TOPIC)" --series "$(SERIES)"

# make approve-source ID=p1  (the only way a scouted source enters the manifest)
approve-source:
	uv run python -m residual_room.scout.approve --id $(ID)

# make query Q="position bias in LLM judges"
query:
	uv run python scripts/query.py "$(Q)"

# make generate EXP=exp-004 SEGMENT=practitioner [KIND=newsletter]  (saves a draft for review)
KIND ?= newsletter
generate:
	uv run python -m residual_room.generation.generate --explainer $(EXP) --segment $(SEGMENT) --kind $(KIND) --save

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

# Sends due campaigns and polls the Telegram bot (keep running alongside `make run`).
worker:
	uv run python -u -m residual_room.worker

test:
	uv run pytest

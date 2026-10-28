.PHONY: setup fetch ingest query eval migrate run test

setup:
	uv sync

# Stubs until their prompts land.
fetch:
	uv run python -m residual_room.corpus.fetch

ingest:
	uv run python -m residual_room.corpus.ingest

# make query Q="position bias in LLM judges"
query:
	uv run python scripts/query.py "$(Q)"

eval:
	@echo make eval: not implemented yet - Prompt 5. Needs make fetch first.

migrate:
	uv run alembic upgrade head

run:
	uv run uvicorn residual_room.api.main:app --reload

test:
	uv run pytest

.PHONY: setup fetch ingest eval migrate run test

setup:
	uv sync

# Stubs until their prompts land.
fetch:
	@echo make fetch: not implemented yet - Prompt 2

ingest:
	@echo make ingest: not implemented yet - Prompt 2

eval:
	@echo make eval: not implemented yet - Prompt 5. Needs make fetch first.

migrate:
	uv run alembic upgrade head

run:
	uv run uvicorn residual_room.api.main:app --reload

test:
	uv run pytest

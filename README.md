# Residual Room

A grounded content and campaign engine for Residual Room, an English AI-education publication. Every factual claim in generated content cites an approved source, passes brand and policy checks, and is approved by a human before it is sent. See `CLAUDE.md` for the architecture and `docs/decisions.md` for the decisions log.

## Setup

Needs [uv](https://docs.astral.sh/uv/), GNU make, and a Postgres database where the pgvector extension can be enabled (e.g. a free Neon project).

```bash
cp .env.example .env   # then set DATABASE_URL
make setup             # install Python 3.12 deps into .venv
make migrate           # enable pgvector
make test
make run               # http://127.0.0.1:8000/health
```

## Make targets

| Target | What it does |
|---|---|
| `setup` | `uv sync` |
| `fetch` | Download external sources (Prompt 2) |
| `ingest` | Chunk, embed and store the corpus (Prompt 2) |
| `eval` | Run the eval cases through the checker (Prompt 5) |
| `migrate` | Apply database migrations |
| `run` | Start the API with reload |
| `test` | Run pytest. Includes a test against the real database, so `DATABASE_URL` must be set |

`make eval` needs `make fetch` first: several eval cases cite sources whose license doesn't allow committing their text, so it is fetched into the gitignored `corpus_local/`. CI will run only the cases whose sources are committable.

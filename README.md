# Residual Room

A grounded content and campaign engine for Residual Room, an English AI-education publication. Every factual claim in generated content cites an approved source, passes brand and policy checks, and is approved by a human before it is sent. See `CLAUDE.md` for the architecture and `docs/decisions.md` for the decisions log.

## Setup

Needs [uv](https://docs.astral.sh/uv/), GNU make, Node 18+ (for the screens), and a Postgres database where the pgvector extension can be enabled (e.g. a free Neon project).

```bash
cp .env.example .env   # then fill in DATABASE_URL and the keys you need
make setup             # install Python 3.12 deps into .venv
make migrate           # create the tables
make fetch             # download external sources (see below)
make ingest            # chunk + embed + store as corpus-v1
make web               # build the screens
make run               # http://127.0.0.1:8000
make test
```

Embeddings run locally (`BAAI/bge-small-en-v1.5` via fastembed); the model (~65 MB) downloads into `.cache/` on first use. The LLMs are free tiers (NVIDIA build, OpenRouter): see `config/models.yaml`.

## Make targets

| Target | What it does |
|---|---|
| `setup` | `uv sync` |
| `fetch` | Download external sources from `data/sources_manifest.yaml` into `corpus/` (committable) or `corpus_local/` (gitignored). `IDS="…"` for specific sources; `GITHUB_TOKEN` lifts GitHub's anonymous rate limit |
| `ingest` | Chunk, embed and store every source as a corpus version (`VERSION=corpus-v2`; versions are never overwritten) |
| `query` | Top matching chunks for `Q="..."` |
| `scout` | Propose new sources for `TOPIC="…" SERIES="…"` into `data/proposed_sources.yaml` |
| `approve-source` | Move proposal `ID=p1` into the manifest (the only way in) |
| `generate` | Generate a draft: `EXP=exp-004 SEGMENT=practitioner` |
| `eval` | Run the 50 eval cases through the checker (`SPLIT=dev|test|all`) |
| `client` | Regenerate the TypeScript client from the OpenAPI schema |
| `web` | Build the React screens (served by FastAPI at `/`) |
| `migrate` | Apply database migrations |
| `run` | Start the API with reload |
| `worker` | Send due campaigns and poll the Telegram bot (local alternative to the tick) |
| `test` | Run pytest. Some tests use the real database, so `DATABASE_URL` must be set |

`make eval` needs `make fetch` first: several eval cases cite sources whose license doesn't allow committing their text, so it is fetched into the gitignored `corpus_local/`.

## Deploy (Render, free tier)

Render's free tier runs web services only (no background workers or cron jobs), so:

- **Web service** (`render.yaml`, Docker): the API, the built screens, migrations on start. Protected by HTTP Basic auth (`ADMIN_USER` / `ADMIN_PASSWORD`); the app refuses to start on Render without a password. Open without login: `/health`, tracked links `/r/…`, unsubscribe links `/u/…`.
- **Scheduler**: a free [cron-job.org](https://cron-job.org) job POSTs `/api/internal/tick` every 10 minutes (sends due campaigns, polls Telegram) with `Authorization: Bearer <TICK_TOKEN>`, so a scheduled send goes out within about 10 minutes. `.github/workflows/tick.yml` is a manual "Run workflow" button for an immediate tick.
- **Database**: the existing Neon Postgres (`DATABASE_URL`). The corpus is already ingested there; nothing is ingested on Render.

Steps:
1. Push this repo to GitHub.
2. In Render: **New → Blueprint**, pick the repo, and fill in the `sync: false` secrets (`DATABASE_URL`, `ADMIN_PASSWORD`, `TICK_TOKEN`, `PUBLIC_BASE_URL` = the service's `https://….onrender.com` URL, the LLM, Resend and Telegram keys, `TEST_EMAILS`).
3. In GitHub: **Settings → Secrets and variables → Actions**, add `APP_URL` (the same `https://….onrender.com`) and `TICK_TOKEN` (the same value as on Render).
4. On cron-job.org (free): a job that POSTs `https://….onrender.com/api/internal/tick` every 10 minutes with the header `Authorization: Bearer <TICK_TOKEN>`. The 10-minute calls also keep the free instance awake, which uses about 744 of Render's 750 free instance hours a month.
4. Stop any local `make worker` (two Telegram pollers would compete for updates).

Paid alternative: an always-on Render background worker (commented out in `render.yaml`).

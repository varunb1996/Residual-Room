# Residual Room: one image with the API and the built React screens.

# 1. Build the screens (TypeScript type-check + Vite).
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# 2. The Python app, installed with uv from the lockfile.
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.8.19 /uv /uvx /bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
# Dependencies first, so code changes don't reinstall them.
COPY pyproject.toml uv.lock README.md .python-version ./
RUN uv sync --frozen --no-dev --no-install-project
COPY . .
RUN uv sync --frozen --no-dev
COPY --from=web /web/dist ./web/dist
# Download the embedding model now, not on every cold start (~65 MB).
RUN uv run --no-sync python -c "from residual_room.corpus.embed import LocalEmbedder; LocalEmbedder().embed_query('warm up')"

CMD ["sh", "scripts/start.sh"]

# Decisions log

## 2026-10-25 — Scoping (from grill session)
- Brief: broad "AI marketing & knowledge orchestration platform" scoped to a grounded content engine with an eval harness. Portfolio project, solo, ~4 weeks at 12–15 hrs/week.
- Organization: Residual Room, the owner's own English AI-education publication (newsletters + explainers).
- Segments: transitioning engineers, practitioners.
- Languages: English only (owner cannot verify technical content in other languages).
- Grounding: claim-level citations, exemption list, LLM judge calibrated on owner's labels, judge model ≠ generator model.
- Corpus: license-filtered manifest + fetch script; internal docs (brand guide, policy, editorial) plus traps (superseded schedule, seeded correction).
- New topics: added through a propose → approve → ingest → generate flow, never fetched at generation time.
- Channels: email live, Telegram live, WhatsApp mock.
- Metrics: tracked clicks and subscriptions; seeded data labelled.
- Approval: deterministic state machine, hard blocks, logged overrides, web review screen.
- Stack: FastAPI, Postgres+pgvector, React (Vite) served by FastAPI, generated TS client, uv, render.yaml.
- Football: week-4 config swap to prove reusability (first to cut).

## 2026-10-28 — Orientation (Prompt 0)

### Layout and build order
- Repo layout approved: `src/residual_room/` package (corpus, llm, generation, checker, evaluation, workflow, channels, analytics, api), `web/`, `config/`, `corpus/` (committed), `corpus_local/` (gitignored), `eval/fixtures/`, `eval/results/`.
- `config/publication.yaml` holds the structured values the deterministic checks need (avoid-list, CTA patterns, exempt types, segments, doc paths). A test asserts its avoid-list matches the brand guide, so the two can't drift. The football swap is a second config file.
- Build order: checker + eval harness come **before** generation. The eval cases are hand-written text, so the never-cut checker gets measured first, and generation is built against a working checker.

### Citations and scoring
- Claims may cite a source ID or a chunk ID. A source-level cite is judged against the best-matching chunks within that source.
- Eval scoring: the verdict must match exactly, and the expected rules must be a subset of the predicted rules (expected ⊆ predicted). Extra predicted rules are reported separately, not counted as errors.
- Dev/test split of the eval cases, stratified by rule. Prompt changes only against dev; test is reported, never tuned on. "Calibrated judge" = per-rule agreement reported on the test split. Raw counts are reported alongside percentages.
- P5 has two parts: deterministic (a cited chunk is superseded) and semantic (a claim matches a superseded chunk better than a current one). Check-time retrieval includes superseded chunks; generation-time retrieval excludes them.
- P5 also labels the document-level rule "first issue after a correction must carry a notice" (STL-10). The rule itself comes from `docs/editorial/corrections.md`, not the content policy; the P5 label is kept to match the eval. STL-10 gets a fixture newsletter in `eval/fixtures/`.
- STL cases have no segment; they default to `practitioner`.

### Frozen-doc amendments (owner-approved)
- Content policy v1.1: P1 covers "technical or operational factual claims". Internal docs are citable for operational facts (schedule, pricing, series, subscription).
- Brand guide v1.1: a "read next" pointer is not a CTA. Added as exempt type `read_next`, and listed in the policy's citation-exempt list for consistency.

### Eval label and text changes (owner-approved)
- POL-07: expected rules `[P6, S1]` → `[P6]` (mislabelled; neither phrase is on the avoid list).
- STL-06: text changed to "…why retrieval fails because of chunking and near-duplicate documents." (old text also fit exp-005). Label unchanged.
- CLN-01: jargon defined for transitioning readers ("dense index (a store of text embeddings)", "pretrained generator (a language model that writes the answer)").
- CLN-06: "change their minds" → "change their verdict" (S2).

### Corpus
- corpus-v1 uses manifest statuses as written; a scripted correction replay (v1 → v2) is optional, later.
- Only exp-003 v1 and v2 enter corpus-v1 among the explainers. Claude drafts both (~300 words; v1 has the wrong claim, v2 the correction with a dated note per corrections.md); the owner reviews and approves. Other explainers are outputs; approved ones may enter a later corpus version.
- Manifest gets a `fetch_url` field (data, not policy) for the Anthology/TACL versions. Each URL is verified to resolve before use. HTML is preferred over PDF where it exists (e.g. arXiv HTML for MT-Bench).
- Each whole-repo source gets an include-list of 2–5 files relevant to the six topics, approved by the owner before fetching.
- Chunks carry the canonical source ID; aliases (e.g. `ext-litm-for-t3`) resolve to it.
- A test asserts key MT-Bench table numbers (65.0, 23.8, 8.7 from Tables 2–3) survive intact in chunks.
- `make eval` requires `make fetch` first (documented in the README). CI runs only cases whose sources are committable.

### Later prompts
- WhatsApp mock follows Prompt 7: template-only outside the 24-hour window, plus opt-in.
- State machine (finalized at Prompt 6): a hard-blocked item moves to `in_review` with a blocked flag; overrides happen only in `in_review`, by the reviewer, with a reason; `rejected → draft`.
- Model choices (generator, judge, embeddings) are asked at Prompts 1–3.

## 2026-10-28 — Skeleton (Prompt 1)
- Python pinned to 3.12 (uv-managed); the system Python 3.14 is newer than some driver wheels are tested on.
- Driver: psycopg 3. Plain `postgres://` URLs from Neon/Render are rewritten to `postgresql+psycopg://` in settings, so `.env` can hold the URL exactly as Neon gives it.
- Migrations: Alembic, hand-written until there are ORM models. `migrations/env.py` reads `DATABASE_URL` from settings, never from `alembic.ini`, so the secret lives only in `.env`.
- `/health` returns 200 only if the database answers *and* pgvector is installed; otherwise 503 with no exception text (it can contain the host).
- The real-database test fails rather than skips when `DATABASE_URL` is missing, so `make test` can't pass silently.
- `.gitattributes` forces LF: make recipes break on CRLF, and deploys run on Linux.
- Dev-machine note: the JioFiber router's DNS refuses every name under `aws.neon.tech` (Neon's database endpoints), while `neon.tech` resolves. Fixed by pointing the PC's Wi-Fi DNS at 1.1.1.1 / 8.8.8.8. Not a code issue.

## 2026-10-28 — Fetch and ingest (Prompt 2)

### Free models only
- Owner constraint: no paid model APIs anywhere (embeddings, generator, judge).
- Embeddings: `BAAI/bge-small-en-v1.5` (384 dims), run locally via `fastembed` (ONNX, no PyTorch). Free, no key, no rate limit, works offline, deterministic (same text → same vector, so eval runs are reproducible against a corpus version). Queries get bge's retrieval instruction prefix; passages don't. Deploy-time memory on Render's free tier to be checked at Prompt 10.

### Manifest (data changes, owner-approved)
- `path` on internal sources; `fetch_url` and `include` on external ones. `url` stays the canonical reference.
- Six Anthology/TACL PDFs; arXiv HTML pinned to a version for MT-Bench (v4), RAG (v4), RAG survey (v5). Every fetch_url checked to resolve with the right title.
- Repos moved: sentence-transformers → `huggingface/`, ragas → `vibrantlabsai/`; LangChain splitter docs → `langchain-ai/docs`. Licenses rechecked: unchanged (Apache-2.0, Apache-2.0, MIT).
- Repo fetches pin the commit SHA of HEAD at fetch time (recorded in `meta.json`).
- `committable` = `commit_allowed AND license_verified`; an unverified license can never be committed.

### Extraction
- Everything is normalized to markdown with `#` headings so one chunker handles all sources.
- arXiv HTML: tables become markdown tables with their caption, and rowspan/colspan cells are filled in, so every row keeps its labels (MT-Bench Table 2's "rename" rows keep their judge name). Bibliographies are dropped.
- PDFs: pypdf (BSD; PyMuPDF avoided because it's AGPL). Headings found by pattern ("3.4 Title"); text after "References" dropped.
- Notebooks keep markdown and code cells, drop outputs. Repo-file headings are demoted one level to nest under their file path.

### Chunking
- Split at headings, then pack paragraphs to ~400 tokens (hard cap ~512, at ~4 chars/token). Docs under the cap stay whole (one fact, one chunk for the small internal docs).
- Tables are one chunk each (caption + rows); oversized tables split by rows with caption and header repeated. Code fences never split mid-block.
- No overlap: a sentence lives in exactly one chunk, so "which chunk supports this claim" has one answer.
- Chunk IDs: `<source_id>#<seq>` (e.g. `ext-mtbench-paper#0012`), unique within a corpus version. Each chunk stores its heading path; embeddings are computed on heading path + text.

### Storage and retrieval
- Tables `corpus_versions`, `sources` (manifest snapshot per version), `chunks` (status, license and commit flag denormalized onto each row). Deleting a version cascades.
- A corpus version is written in one transaction and is immutable: re-ingesting needs `--replace`.
- No vector index yet: exact cosine search over a few thousand rows is fast enough.
- `search(include_superseded=False)` by default (generation); the checker will pass `True`.
- `make fetch` uses one GitHub API call per repo (anonymous limit is 60/hour) and honours an optional `GITHUB_TOKEN`.

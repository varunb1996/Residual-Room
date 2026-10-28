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

## 2026-10-28 — Checker, deterministic half (build-order step W1-d)
- **"Uncited sentence → P1" is not deterministic.** POL-03 ("This series won't make you job-ready on its own") and POL-08 ("Reranking is a game-changer") are uncited but expected pass/warn: they aren't factual claims. Deciding what is a claim is claim extraction, which CLAUDE.md assigns to the LLM.
- So segments have three kinds: `claim` (set only by the generator or the LLM extractor), `exempt` (with a type), and `candidate` (not recognized by pattern; the LLM decides). The deterministic P1 rule fires only on `claim` segments without a citation, or on any citation to an id not in the corpus.
- The pre-splitter labels by pattern only what is safe: greeting, sign_off, cta, read_next, opinion. `transition` needs judgment, so only the LLM assigns it. A sentence carrying a citation is never exempt.
- S3 counts CTA *phrases* across the whole message (POL-10 has three CTAs in two sentences). "unsubscribe" is not a subscribe CTA; "Read next" is not a CTA.
- P5 (cited half) resolves chunk ids (`<source>#<seq>`) and aliases to a source, and reads status from the corpus version's `sources` table (`StatusLookup.from_corpus`), with the manifest as the test fallback. The two agree for corpus-v1 (tested).
- Verdict: any P rule → block, any S rule → warn, else pass; per-segment verdicts come from the same rule.
- `config/publication.yaml` holds the avoid-list, CTA patterns and exempt patterns; drift tests fail if the avoid-list or the CTA types disagree with the brand guide.
- Baseline on all 50 cases with deterministic rules only: they fire on exactly 3 cases (POL-08 S1, POL-10 S3, STL-07 P5), all correct, and fire on none of the 23 expected-pass cases. The other 24 cases need the judge.

## 2026-10-28 — Checker, judge half (build-order step W2-a)

### Models (free only)
- Judge + claim extractor: `nvidia/nemotron-3-ultra-550b-a55b` on NVIDIA build.nvidia.com; fallback: the same model's `:free` endpoint on OpenRouter (~50 req/day on a free key). Generator: `openai/gpt-oss-20b` on NVIDIA. Different families (NVIDIA vs OpenAI) to avoid self-preference bias. Config in `config/models.yaml`; loading fails if judge and generator share a model.
- Ruled out by smoke test (2026-10-28): Nemotron 3 Super and Llama-Nemotron 70B/253B return 404 on this NVIDIA account; gpt-oss-120b has no free endpoint on NVIDIA or OpenRouter; OpenRouter's free Gemma was rate-limited upstream. Gemini isn't free for the owner. Local models don't fit the dev PC (7.2 GB RAM, integrated GPU).
- Observed latency: judge ~2–15 s per call; generator gpt-oss-20b on NVIDIA 74–315 s per call (queueing). Revisit at generation time.

### Client
- One `httpx` client for OpenAI-compatible APIs, no SDK. Per-provider rate limiter shared across roles, retries on 408/409/429/5xx honouring Retry-After, endpoint fallback on any failure.
- Judge at temperature 0; replies must be JSON validated by pydantic, with one corrective retry. A judge call that still fails becomes a blocking `JUDGE` hit: unchecked content never passes.
- Responses cached on disk (`.cache/llm/`, keyed by primary model + prompt + params): eval reruns cost no quota and are reproducible.

### Segment kinds (owner-approved)
- New kind `statement`: non-factual sentences (promises, predictions, advice, hype). No citation needed, but P3/P4/P6/S2/S4 apply.
- New exempt type `framing`: series labels, hooks, subject lines. No citation needed, but checked against the internal docs; a contradiction is P1. This reconciles CLN-01/03/04 (correct series label, pass) with STL-06 (wrong series label, P1).
- Guard rails in code, not left to the model: a cited sentence is never exempt; a candidate the model skips becomes a claim; an LLM "cta" label stands only if the sentence matches one of the brand guide's four CTA types ("Upgrade to Pro" is a statement and gets checked — STL-09).

### Judge checks
- Cited claims: evidence = the cited chunk, or the 3 closest chunks within a cited source (superseded included). The judge rates support (P1) and numbers separately (P2: value AND referent); if only a number is wrong it should say supported + mismatch. Rule text in prompts is read from `docs/content-policy.md`.
- Uncited claims, framing and statements: checked against the corpus, superseded included. matches_superseded → P5 (+P2 if a number/date/day conflicts); contradicts_current → P2 if numeric else P1.
- **Operational docs are always shown in full** (`operational_sources` in the publication config: both schedules, series, subscription, corrections). Found during the live run: similarity search ranked them below longer docs on the same topic (int-subscription was 6th of 9 internal sources for "Upgrade to Residual Room Pro"), so STL-06 and STL-09 passed wrongly. They total a few hundred tokens, so including them always is cheap. A test keeps each one a single chunk.
- Policy rules: one call per document over all non-exempt sentences (plus opinions and framing): P3, P4, P6, S2, and S4 only for transitioning readers. Rules the model invents or indexes out of range are dropped.
- Correction notice (STL-10): deterministic, from `schedule` and `corrections` in the publication config (drift-tested against schedule-v1.md and corrections.md). Fixture: `eval/fixtures/stl-10-issue-2.md`.

### Live spot check (not the eval; that's the next step)
- 15 cases run live: GRD-05, GRD-11, GRD-13, GRD-17, POL-12 (the five hard cases), POL-01, STL-01, STL-04, STL-05, STL-06, STL-09, STL-10, CLN-01, CLN-03, CLN-07. All 15 match their expected verdict and rules (expected ⊆ predicted).
- Extra rules seen: P1 alongside P2 on number errors, S2 on "agreed", S2/S4 on GRD-17. Allowed by the scoring rule, but S2/S4 over-firing may cost precision and false warnings on clean cases. To be measured by the eval harness, not tuned per case.

## 2026-10-28 — Eval harness (Prompt 5)
- Frozen split `eval/split.yaml`: 21 dev / 29 test, stratified by first expected rule; the five hard cases and the single-case rules (P4, S1–S4) are test-only, so every rule is measured on test. Deterministic (hash order); the builder refuses to overwrite it, and a test fails if the file drifts from the builder. Prompt changes only against dev failures.
- Metrics, all with raw counts and a ⚠ flag under 5 cases: lenient case accuracy (the agreed rule: verdict + expected ⊆ predicted), strict (exact rules), per-rule precision/recall (extra rules count as false positives here, so over-firing shows), pass-case false-positive rate, hard-case accuracy, blocking judge errors.
- Results: `eval/results/<datetime>-<split>.md` + `.json`, headed with date, split, judge/generator/embedding models, corpus version, git commit, cache state.
- `--committable-only` runs the 30 cases whose cited sources can be committed (CI subset).
- First full run (2026-10-28, `eval/results/2026-10-28T1310-all.md`, judge Nemotron 3 Ultra, corpus-v1, 1,471 s): lenient 44/50 (88%), strict 32/50 (64%), pass-case false positives 4/23 (17%), hard cases 5/5. Six disagreements: two possibly mislabelled (POL-05, CLN-08) and four checker errors with proposed structural fixes, written up in the report for owner review.

### Fixes after the first run (owner-approved: "all four fixes")
- **Test contamination, on record:** three of the four fixes were found by looking at **test** failures (CLN-08, STL-03, POL-03/POL-09); only the citation-scope fix came from dev (CLN-05). Test scores from 2026-10-28T1351 onward are optimistic. The honest remedy is adding fresh held-out cases later.
- Citation scope: an uncited claim inherits the citations of the next cited claim on the same line. Only claims inherit, never across lines; the judge still checks each claim. Segments record their line.
- Policy rules judge the original sentence once, not each atomic claim, so a term stays next to its definition.
- Uncited check: two booleans (contradicts current, matches superseded) instead of one relation, so a sentence can be both P5 and P2.
- Extraction: "statement" now explicitly covers disclaimers about reader outcomes and descriptions of a model's mind, with a general tie-breaker ("could a source confirm or refute this?").
- Bug found by the rerun: the extractor sometimes sends `"claims": null`; the schema now accepts it (it had turned CLN-02 into a blocking JUDGE error).
- Second full run (`eval/results/2026-10-28T1351-all.md`, 1,695 s): lenient 44/50, strict 32/50, pass-case false positives 5/23, hard 5/5, one judge error (CLN-02, since fixed; rerun alone passes → corrected lenient 45/50). POL-03 and POL-09 fixed; GRD-16 newly flagged S2 on an explicit analogy; CLN-05 swapped its P1 for a P2 on an omitted year.
- Proposed next (dev-motivated, not yet applied): numbers that omit detail (e.g. a year) aren't a P2 mismatch (CLN-05); explicitly framed analogies aren't S2 (GRD-16, and the brand guide encourages analogies).
- Variance: with 50 cases, a ±1–2 case swing between runs is noise. Compare runs on the disagreement lists, not the headline number alone.

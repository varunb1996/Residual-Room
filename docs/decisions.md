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
- Variance: with 50 cases, a ±1–2 case swing between runs is noise. Compare runs on the disagreement lists, not the headline number alone.

### Owner review of the disagreement table (closes Prompt 5)
- **Labels kept as written.** POL-05 and CLN-08 stay `pass` although the review suggested P1 and S4: labels only change the eval score, not the app's behaviour. Both remain known disagreements in future reports.
- Two dev-motivated prompt fixes applied: leaving out detail the source gives (a year, a unit clear from context) is not a P2 mismatch; explicitly framed analogies are not S2.
- Targeted check (`eval/results/2026-10-28T1429-custom.md`, 6 cases, ~2 min): CLN-05 now passes; GRD-16's S2 is gone but the judge now raises S4 on "models"; regression checks GRD-02, GRD-11, GRD-05 (P2 still caught) and POL-09 (S2 still caught) all pass.
- Open: S4 over-fires on everyday AI words ("model", "prompt"). Not fixed yet.
- **Eval cost, owner feedback:** full 50-case runs take ~25–30 min and use the owner's time. From now on: small fixes are checked with targeted reruns (the affected cases plus a few regression cases); full runs only at milestones and when asked. Parallelizing cases (the judge uses ~5 of ~40 allowed requests/minute) would cut a full run to ~8–10 min.
- Fixed: the runner crashed on Windows printing the ⚠ small-sample flag (cp1252 console); stdout is now UTF-8.

## 2026-10-28 — Approval state machine and review screen (Prompt 6)
- Pieces are created from pasted text until generation exists (Prompt 3); generation will call the same create path. Everything after creation is the real flow.
- State machine (`workflow/states.py`, pure code): draft → checks → in_review → approved → scheduled → sent; in_review → rejected; checks → draft on failure; edit from draft / in_review / rejected / approved → draft as a new version. One table lists every allowed move; a test walks all 7×8 state-action pairs.
- A hard-blocked piece reaches in_review with its blocking hits shown; approval is gated separately. **Overrides are per hit**: the reviewer overrides each blocking hit with a written reason (blank refused, warnings can't be overridden). This records *which rule* the reviewer thinks the checker got wrong, for later review as possible checker errors.
- Overrides and approvals belong to one version; an edit is a new version and must be checked again (tested).
- A refused approval is itself written to the audit log (committed even though the request fails with 409).
- Tables (migration 0003): pieces, piece_versions, check_runs (full report as JSONB), overrides (unique per hit; DB check that the reason isn't blank), audit_log. No auth: actions are recorded as "owner"; the checker's own events as "system".
- Checks run in a FastAPI background task; the screen polls while the state is `checks`. A crashed checker returns the piece to draft with the error recorded, never leaves it stuck.
- The checker now records which chunk supports each passing cited claim, so passing claims link to a source on screen too.
- Review screen: React + Vite, plain CSS, light/dark. Typed client generated from FastAPI's OpenAPI schema (`make client` → `web/src/schema.d.ts`, `openapi-fetch`); `tsc` fails if the API and screen drift. FastAPI serves `web/dist` at `/`, API at `/api`.
- TypeScript pinned to 5.x: openapi-typescript needs the TypeScript JS API, which TypeScript 7 (native compiler) doesn't ship. `@types/react`/`@types/react-dom` added as dev dependencies.
- Verified in the browser with the real judge: STL-07 text → block (P5, superseded `exp-003-v1`), claim links to the SUPERSEDED source passage, approve refused, override with reason logged, approve succeeds; CLN-07 text → pass, claims link to `int-schedule-v1#0000` (CURRENT), approved. Audit trail shows every step.
- Found during the walk-through: manifest titles containing a comma inside an unquoted YAML flow mapping are truncated (`int-schedule-v1` parses as "Publishing Schedule v1 (Thursdays 9:00 IST" plus a stray key). Not fixed: the manifest is owner data; needs quoting, then re-ingest.
- Fixed (owner-approved): quoted the `int-schedule-v1` title in the manifest and rebuilt corpus-v1 with `--replace` (still 789 chunks, 8 superseded). A manifest test now fails on any unexpected key, which is how an unquoted comma shows up. Rebuilding a version in place is acceptable only because nothing has shipped against corpus-v1 yet; from the first sent piece on, corpus versions are immutable and fixes go into corpus-v2. Spot check after the rebuild: STL-04 and STL-05 unchanged.

## 2026-10-28 — Grounded generation (Prompt 3)
- What can be generated comes from an `explainers` map in the publication config (drift-tested against series.md): topic, series, the manifest topics whose sources are always eligible, and optional subtopics used as extra retrieval queries (exp-004: position, verbosity, self-enhancement bias; editable).
- Retrieval for generation: current chunks only; several queries per explainer; at most 4 chunks per source and 12 in total; each of the explainer's own manifest sources gets one guaranteed slot. Plain top-k was lopsided (13 of 16 exp-004 chunks from MT-Bench); diversified it is 12 chunks from 5 sources.
- The generator returns JSON segments: `claim` (with chunk_ids) or `exempt` (with a type). Code validates structure deterministically: every claim cites ≥1 chunk, every cited id was actually provided, one CTA from the brand guide's four types (newsletters exactly one), the sign-off ends a newsletter, transitioning pieces include read_next. One corrective retry with the problems listed; if problems remain, the draft is saved with them recorded and the checker blocks it. Nothing is dropped silently.
- Truthfulness is not checked at generation time: generated drafts go through the same checker and review as pasted ones. The checker uses the generator's stored segments as-is (no splitting, no extraction call).
- Each generated version stores its segments plus generator model, corpus version, the chunk ids it was given, attempts and remaining problems (CLAUDE.md principle 3).
- Brand guide and content policy are read from the frozen docs into the prompt at run time, so generator and checker follow the same text.
- Generation runs as a background job (`generation_jobs`, migration 0004); no DB session is held while the model runs. Failures are recorded on the job; no half-made piece.
- Generator: gpt-oss-20b on NVIDIA with a 10-minute timeout (observed 74–315 s per call) and `google/gemma-4-31b-it:free` on OpenRouter as a best-effort fallback (third model family). max_tokens 6000 leaves room for hidden reasoning.
- **Done criterion met (live, 2026-10-28):** exp-004 newsletter for practitioners (piece 41, 4m04s): 4/4 claims cite ≥1 chunk; for transitioning readers (piece 42, 7m26s): 11/11; no structural problems, both on the first attempt.
- Checking piece 42 with the real judge (~3 min) → **block**: P2 on "only 60% … stayed consistent" (source: "more than 60%", Table 2: 65.0%), P1 on an invented claim that calibration mitigates self-reinforcement (G-Eval names the risk, not the fix). Nine passing claims linked to supporting chunks.
- Gaps found, not yet fixed:
  1. **Generator labels can hide claims.** The practitioner draft opens with "LLM judges sometimes prefer longer or earlier answers…" labelled `framing`, so it skips the citation requirement. The checker trusts the generator's labels; it should re-classify the generator's own exempt segments (judge ≠ generator applies to labels too).
  2. **read_next isn't checked.** "Read next: our First Principles explainer on LLM-as-Judge" points to an explainer that doesn't exist; read_next segments are exempt from the uncited check. They should be checked against the explainer map.
  3. **Judge JSON robustness.** The document-level policy call returned no JSON twice on a 17-sentence piece → blocking JUDGE hit (safe, but needs a fix, likely more max_tokens or batching sentences).
  4. **Unicode hyphens.** The generator writes non-breaking hyphens (U+2011, "GPT‑4"); word-list checks like S1 ("game-changer") would miss them. Normalize text before deterministic checks.

### Gap fixes after the first live generation (owner-approved)
1. **Generator labels aren't trusted.** When a generated piece is checked, an exempt label stands only if a deterministic pattern confirms the same type (greeting, cta, read_next, sign_off, "We think…" opinion). Framing and transition have no pattern, so the judge always re-labels them; a factual claim labelled "framing" becomes a claim and needs a citation. Claims stay claims.
2. **read_next is checked deterministically** against the explainer map: it must name an explainer id, or share a topic word with an explainer in the series it names; otherwise P1 (evidence `int-series`). First tried as an LLM uncited check, which failed live: the judge answered "No current passage explicitly states there is no First Principles explainer on LLM-as-Judge" — a missing thing isn't a contradiction to it. Existence is a lookup, so it's code (CLAUDE.md principle 1). The generator prompt also lists the other explainers by topic and series.
3. **Cut-off judge replies.** Cause: Nemotron reasons in plain text (~8,000 characters on a 17-sentence piece) and hit max_tokens=2000 before writing JSON; the cut-off reply had been cached. Now: the client raises on `finish_reason == "length"` (not cached; the fallback endpoint gets a chance), judge max_tokens is 6000, and policy rules are sent in batches of at most 8 sentences.
4. **Unicode look-alikes.** One `normalize()` used only for matching (S1 word list, CTA counting, exempt patterns): U+2010/2011/2012/2212 → "-", U+00A0/202F → space, soft hyphen removed. En/em dashes are left alone (the sign-off uses one). Displayed text is unchanged.
- Live verification (2026-10-28, pieces 41/42 + four eval cases, ~10 min): piece 41's opening "framing" was re-labelled by the judge as two claims → P1 each (fix 1 works). Piece 42 checked with no JUDGE error (fix 3), P2 and P1 caught again; its bad read_next is now P1 via the deterministic lookup (fix 2, verified on the stored piece). POL-01, POL-09, POL-11 unchanged. CLN-08 now warn [S2, S4] (label pass; S4 already a known disagreement, S2 new).
- Open (owner's call, not tuned): S2 fires on everyday verbs ("prefer", "favor", "defend"); S4 fires on "LLM" throughout a transitioning piece that never defines it (arguably right). Both are about how strictly to read S2/S4.

## 2026-10-28 — Channels and campaigns (Prompt 7)
- One content model per approved version (`channels/content.py`): blocks plus numbered sources, built from exactly what was checked (generated segments as stored, pasted text split like the checker). Each channel renders it after approval (CLAUDE.md principle 8): email HTML + text, Telegram's HTML subset (≤4096 chars, cut on plain text then escaped so no tag is split), WhatsApp approved template + parameters.
- Email via Resend's HTTP API (free plan: 100/day, 3,000/month), no SDK. **Hard rule in code: mail only goes to addresses in TEST_EMAILS**; anyone else is a `refused` delivery and no API call is made. Every email has an unsubscribe link (`/u/{token}`) and a List-Unsubscribe header.
- Telegram via the Bot API in polling mode (no public webhook needed). Campaigns post only to TELEGRAM_TEST_CHAT_ID. The worker answers /start, /segment, /stop, replying only to the chat that wrote.
- WhatsApp is a mock enforcing: opt-in required; free-form only within 24 h of the subscriber's last inbound message; otherwise only approved templates (`whatsapp_templates` in the publication config). Campaigns use the `new_issue` template.
- Every outbound link is tracked: `/r/{token}` → `links` (campaign, piece, segment, channel, label, target) → `clicks` (link + time only, no IP or personal data) → 302. Link targets: manifest URLs for sources; a placeholder site URL (config `site`) for explainer/subscribe CTAs until a website exists.
- State machine: new `cancel` (scheduled → approved). Scheduling needs an approved piece; sending moves it to sent if anything was delivered; if nothing was delivered the campaign is `failed` and the piece stays scheduled (cancel and retry).
- `make worker`: one loop, every 30 s: send due campaigns, poll Telegram. "Send now" from the API sends immediately in a background task instead of waiting for the worker.
- Campaign screen (screen 2 of 3): pick an approved piece, segments, channels, send now or at a time; previews of all three channel formats (email in a sandboxed iframe, never executed); deliveries per recipient with status and reason; click counts by channel, segment and link label.
- Known limit: tracked links use PUBLIC_BASE_URL (default http://127.0.0.1:8000), so clicks only register from this PC until the app is deployed (Prompt 10).
- **Done criterion met (live, 2026-10-28):** campaign 9 sent the approved welcome email (piece 8) to the owner's inbox via Resend and to the "ResRoom" Telegram test channel (both arrived); the WhatsApp mock rejected a free-form message outside the 24-hour window and accepted the approved template; campaign 11 (piece 79, CLN-02 text) logged 3 clicks in `clicks` (email/transitioning: 1 cta, 2 source).
- Found during the live run:
  - **Internal-doc citations produce no links.** Piece 8 cited only the schedule and subscription docs, which have no public URL, and had no CTA, so its email had nothing to track. Emails citing only internal docs stay unmeasurable until a website hosts those docs.
  - **localhost vs IPv6.** Links to `http://localhost:8000` failed in the owner's browser ("site can't be reached"): it resolved localhost to IPv6 `::1`, where uvicorn wasn't listening. The default PUBLIC_BASE_URL is now `http://127.0.0.1:8000`. Deployment (Prompt 10) replaces it with the public URL.
- The Telegram test channel's chat id was found from the bot's `my_chat_member` update when the owner made it an admin.

## 2026-10-28 — Analytics (Prompt 8)
- **One table of numbers, `metric_rows`:** each row is one figure (dataset, week or whole window, channel, segment, metric, value, n). Narration cites rows as `m:<id>`, which makes "every number cites a metrics row" checkable. Real and seeded rows live side by side but are never mixed in one query; the screen shows them in separate panels, the seeded one under a "not real readers" banner.
- **Real metrics** are recomputed from sent deliveries, clicks on tracked links and subscribers (new, and unsubscribes via a new `unsubscribed_at`). Only those count as real metrics (CLAUDE.md principle 9).
- **Seeded metrics** are deterministic (fixed seed) with four planted patterns: Telegram ≈2× email click-through for practitioners; an email click-through drop for transitioning readers over the last 3 weeks; a practitioner unsubscribe spike in one week; tiny WhatsApp volumes.
- **Insights are rules, not the LLM:** channel gap (two-proportion z-test, p<0.05), trend drop (last 3 weeks ≥20% below earlier weeks, significant), unsubscribe spike (>2× the median week, significant). Minimum sample 100 (deliveries or active subscribers); below it the result is an explicit "not enough data (n=…, need 100)" entry. Plain-Python statistics (math.erf), no scipy. On the seeded data the rules find exactly the four planted patterns and nothing else.
- **Derived rows:** numbers an insight introduces (ratio, pooled periods, median) are stored as derived metric rows, so a narration can cite a row for them too. A ratio is rendered as "2.0", not "200%".
- **Narration** is written by the generator model (judge ≠ generator still holds) and goes through the same claim checker as newsletters, with metric rows as evidence passages. Deterministic checks first: every number in a claim must appear in a row it cites (with rounding tolerance; dates and "last 3 weeks" style numbers come from the row), cited rows must exist, and narration of seeded data must say "seeded".
- Insights are computed on refresh and stored (kv), so the row ids a narration cites stay valid until the next refresh.
- **Clicks are not unique.** Tracked links are per campaign × segment × channel, not per reader (no individual tracking, by design), so repeat clicks count again: the real email row showed 4 clicks from 3 deliveries (133%). The screen labels the column "clicks per delivery", and the channel-gap test skips any comparison where clicks exceed deliveries (it assumes a proportion). Unique click-through would need per-recipient link tokens: a privacy trade-off for the owner to decide.
- Live narration, first run: **blocked**, and correctly: all three claims cited the right rows and passed support/number checks, but it never said the data was seeded (deterministic P1). Fix: narration now gets the same one corrective retry as content generation when the deterministic checks find problems.
- Bug found on the retry run: the narration job held a DB transaction open during minutes-long model calls; Neon ended the idle connection, so the failure couldn't be saved and the narration stayed "running". The job now reads, closes the session, runs the models with no open connection, and saves in a fresh session (as generation jobs already did).
- Test isolation fix: analytics tests had deleted the app's live seeded rows, insights and narrations during cleanup; they now regenerate the seeded dataset and delete only the narrations they created.
- **Open at commit time:** the done criterion "narration passes the checker" was still running live (narration 11). Its verdict is stored in `narrations`; check it first next session.
- Narration reruns on 2026-10-29: #11 was interrupted (the app was closed overnight, so the server-side job died; marked failed). #14 failed cleanly: the generator gpt-oss-20b returned HTTP 504 from NVIDIA and the Gemma fallback on OpenRouter was rate-limited (429). The failure was recorded properly this time (the DB fix held). **Prompt 8's "narration passes the checker" is still unverified, for infrastructure reasons.** Options: retry later, or add a third free generator fallback (e.g. Nemotron 3.5 Lightning, same family as the judge).

## 2026-10-29 — Source scout (Prompt 9)
- **Search:** OpenAlex for papers (free, no key; indexes arXiv and the ACL Anthology) and GitHub repository search. Deviation from the prompt, owner-approved: arXiv's own search API timed out and then returned 429, Semantic Scholar returned 429 without a key, and the ACL Anthology has no search API. Papers are kept only if they have an arXiv id or an ACL Anthology location (the ones we can fetch and license-check). Sources already in the manifest are skipped.
- **License resolver, deterministic, no LLM:** ACL Anthology → CC BY 4.0 (preferred over arXiv when both exist); arXiv → its own OAI-PMH record, paced 3 s apart (CC BY → committable; non-exclusive, share-alike, non-commercial → local only); GitHub → SPDX from GitHub (MIT, Apache-2.0, CC-BY-4.0 → committable). Strict reading of the manifest rule (CC BY, MIT, Apache-2.0, own). Unresolved → never committable. OpenAlex license fields are shown as hints only.
- **LLM (judge model) only where judgment is needed:** turning the topic into queries, and ranking relevance 1–5 with a one-line reason. Top 8 kept; licenses resolved for those.
- **Proposals** go to `data/proposed_sources.yaml`, never the manifest. **Approval** (`make approve-source ID=…`) is the only way in: one entry appended as text under a shared new topic number (comments untouched), the manifest re-loaded to verify, original restored on failure. IDs are the first content words of the title (filler words dropped).
- New sources are fetched by id (`make fetch IDS=…`) into `corpus/` or `corpus_local/` by license, then a **new corpus version** is built (`make ingest VERSION=corpus-v2`); existing versions are never overwritten. Generation and the checker default to the latest version; eval runs stay pinned to corpus-v1 for comparable scores.
- **Done criterion met (live):** scouting "Rerankers and cross-encoders for RAG" (Failure Modes) searched 60 new candidates, ranked the top 8 with reasons and resolved licenses (2 CC BY committable, 4 arXiv non-exclusive, 2 CC BY-NC-SA). The owner approved p3 (Per-Query Gating of LLM Rerankers, CC BY 4.0 → committed) and p1 (SciRerankBench, arXiv non-exclusive → local only). corpus-v2: 929 chunks from 29 sources (v1: 789 from 27); a reranking query returns the new paper's chunks.

## 2026-10-29 — Deploy (Prompt 10)
- **Render's free tier, checked against render.com/docs/free:** free instances exist only for web services, Postgres and Key Value ("Other service types don't support Free instances"): **no free background workers or cron jobs**. Free web services spin down after 15 minutes without inbound traffic (about a minute to wake); 750 instance hours per workspace per month; SMTP ports blocked (no effect: email uses Resend's HTTPS API). RAM is 512 MB per third-party sources; the app measured ~260 MB with the embedding model and checker loaded.
- **Needs a paid plan:** an always-on background worker (scheduler + Telegram polling), cron jobs, and avoiding cold starts; Render also advises against free instances for production.
- **Free design (owner-approved):** one Docker web service (API + built screens; migrations on start; embedding model downloaded at build time). The scheduler is a **tick**: `POST /api/internal/tick` (bearer `TICK_TOKEN`) runs one worker pass (send due campaigns, poll Telegram once), called every 10 minutes by a scheduled GitHub Actions workflow (free for public repos; may run a few minutes late; the calls also keep the service awake). The paid worker is in `render.yaml`, commented out.
- **Login protection before going public:** HTTP Basic auth (`ADMIN_USER`/`ADMIN_PASSWORD`) on the screens and `/api`; open: `/health`, `/r/…`, `/u/…`, and the tick (own token). Constant-time comparisons. The app refuses to start on Render (`RENDER` set) without `ADMIN_PASSWORD`.
- Region Singapore, next to the Neon database. Secrets are `sync: false` (entered in Render's dashboard). The corpus stays in Neon; nothing is ingested on Render. Local-only sources' text is in Neon but only visible behind the login, never published.
- Not verified locally: the Docker build (Docker isn't installed on the dev PC); Render's first build is its first real test.

## 2026-10-29 — Review UI redesign
- **Why:** the first screens worked but read as a generic admin panel. The review screen is never-cut and is where a visitor sees grounding, so it got the most care.
- **Review screen as a proof:** one row per sentence: a rail on the left, the sentence, and a margin on the right. On the rail, a supported sentence sits on a baseline, a warning sits a little off it, and a block sits further off, so you can see where the problems are by scanning. Rule notes and the cited source passage show in the margin beside the sentence; clicking a sentence shows its source.
- **Queue grouped by what you do next** ("Needs your review", "Drafts", "Ready to send"), not by raw state names. State and verdict use words first, with colour second.
- **Analytics:** real and seeded data stay in separate columns, and the seeded column is hatched so it can't be mistaken for real readers. Held-back findings show their sample against the minimum (for example, 3 of 100).
- **Fonts (owner-approved dependency):** `@fontsource/newsreader` (serif, for piece text) and `@fontsource/schibsted-grotesk` (interface). They are self-hosted in the build, so the app makes no request to Google Fonts. Only the Latin subsets and the weights used are imported. Tabular figures are off: in Schibsted Grotesk they also widen "." and ",".
- Checked in the browser at desktop and phone width, in light and dark mode, with no horizontal page scroll.

## 2026-10-29 — Public explainer pages
- **Problem:** "Read the explainer", "read next" and "subscribe" links redirected to `https://residualroom.example`, the Prompt 7 placeholder for a website that didn't exist. Clicks were counted, then the browser failed.
- **Decision (owner-approved):** the app serves the pages itself: `/explainers`, `/explainers/{id}` and `/subscribe`, open without the login. An explainer page shows, per reader segment, the version the latest **sent** campaign delivered (not the piece's latest version), rendered by `render_web`: the web page is one more channel, formatted after approval. Calls to action are dropped there, because the reader has already arrived.
- **What stays private:** drafts, pieces in review, and approved-but-unsent pieces. Sources are shown as links only, never as text, so `commit_allowed` and the license rules are unaffected. A cited source that has since been superseded is marked "(since superseded)".
- **Links** are built from `PUBLIC_BASE_URL`; `config/publication.yaml` keeps only the paths. Tracked links store their target when sent, so `make relink BASE=…` rewrites the placeholder targets of messages already sent (one-off, placeholder prefix only).
- **Plain HTML from Python** (`html.escape`, as email does) with system serif fonts: no template engine and no new dependency.
- **Tests and local login:** `tests/conftest.py` turns the login off by default, because a local `.env` with `ADMIN_PASSWORD` made every API test return 401. The login tests set it again themselves.
- `/subscribe` says honestly that the publication is in a closed test; public sign-up (double opt-in, verified sending domain) is not built.

## 2026-10-29 — Third generator fallback, external tick, reader titles
- **Generator fallback:** narrations #14 and #15 failed with gpt-oss-20b on NVIDIA returning 504 and Gemma on OpenRouter returning 429 (free daily quota). The third endpoint is the same Gemma 4 31B on NVIDIA's own servers: a different host from the second, a different model from the first, and still not Nemotron, so the judge ≠ generator rule holds. Smoke test, one call each with a JSON task: Gemma 4 31B on NVIDIA returned valid JSON in 165 s; deepseek-v4.1-flash and glm-5.3-flash timed out at 180 s; mistral-large-2-instruct returned 404 on this key.
- **Tick timer moved to cron-job.org:** GitHub's `*/10` schedule never fired in the first hours after the push, only manual runs did. GitHub documents scheduled runs as best effort. cron-job.org is free ("entirely free of charge"), allows POST and custom headers, and runs up to every minute; each request times out after 30 s. The tick itself takes a few seconds. Calls every 10 minutes keep the free instance awake, so there is no cold start; the cost is about 744 of Render's 750 free hours a month, shared across the workspace. GitHub's schedule is removed so two timers never poll Telegram at once. The workflow stays as a manual "Run workflow" button.
- **Reader titles:** generated titles keep the "(exp-004, practitioner, newsletter)" suffix for the review queue, where it tells pieces apart. `build_content` strips it, so readers never see it: email subjects, Telegram, WhatsApp and the public page.
- Removed three subscribers left in the database by an interrupted test run (tag `b36dc341`).

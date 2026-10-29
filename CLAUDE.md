# Residual Room — project context for Claude Code

## What this is
A solo portfolio project. A grounded content and campaign engine for **Residual Room**, an English AI-education publication (newsletters + explainers). Every factual claim in generated content must cite an approved source, pass brand/policy checks, and be approved by a human before it is sent.

**The one claim this repo must prove:** "He can take a vague brief, scope it to what matters, ship provably grounded output, and measure it."
Judge every design choice by whether it makes that sentence more believable.

## Audience segments
- **transitioning**: engineers moving into AI (define jargon, intuition first)
- **practitioner**: people already building RAG/agents/evals (mechanism and failure first)

## Frozen inputs (do not edit without asking)
- `docs/brand-guide.md`, `docs/content-policy.md`, `docs/editorial/*.md`
- `eval/*.yaml` — 50 labelled cases. Expected labels are the owner's ground truth.
- `data/sources_manifest.yaml` — approved sources with verified licenses.

## Architecture principles
1. **LLM only where judgment is needed** (generation, claim extraction, support checking, relevance ranking). Everything else is deterministic code: state machine, scheduling, channel formatting, license rules, word-list checks, CTA counting.
2. **Grounding is claim-level.** Generated text is split into atomic claims; each must cite a chunk ID; exempt text (greetings, sign-offs, CTAs, transitions, marked opinions) needs no citation.
3. **Corpus is versioned.** Ingestion produces `corpus-vN`. Every generated piece records the corpus version it was grounded against. No fetching sources at generation time.
4. **Source status matters.** Sources can be `current` or `superseded`. Citing a superseded source is a P5 hard block.
5. **Licenses.** Only sources with `commit_allowed: true` may have their text committed. Others are fetched locally by `make fetch` into a gitignored folder. Unverified license → not committable.
6. **Judge ≠ generator.** Use a different model for checking/judging than for generation.
7. **Approval is a state machine:** draft → checks → in_review → approved → scheduled → sent (rejected → draft). Hard blocks stop approval; overrides require a logged reason.
8. **Channels:** email (live), Telegram (live, polling mode), WhatsApp (mock that enforces template-only and opt-in rules). Format per channel *after* generation.
9. **Analytics:** only tracked-link clicks and subscriptions are real metrics. Seeded data is labelled as seeded. Insights are rule-detected; the LLM only narrates them, with each number citing a metrics row. Small samples → "not enough data."

## Stack
FastAPI · Postgres + pgvector (managed, e.g. Neon) · React (Vite) built and served by FastAPI · TypeScript client generated from FastAPI's OpenAPI schema · uv · Makefile · pytest · `render.yaml` for deployment (later).
UI is capped at three screens: review, campaigns, analytics.

## Never cut
Claim-level checker · calibrated judge · review screen · corrections (P5) check.

## Cut order if behind schedule
1. WhatsApp mock 2. Analytics LLM narration 3. Campaigns screen polish
(The football config swap was dropped on 2026-10-29; see docs/decisions.md.)

## How to work with me (the owner)
- Plan before coding. For each task: propose the plan and files, wait for my OK, then implement.
- Small steps. One module per task, with tests.
- Explain non-obvious decisions in a sentence or two; I want to learn the codebase, not just receive it.
- Ask before adding a dependency.
- Never claim something works without running it (tests, a script, or a curl).
- Log architectural decisions in `docs/decisions.md`.

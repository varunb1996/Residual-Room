# Residual Room — Content Policy (v1.1, FROZEN)

> v1.1 (2026-10-28): P1 now covers operational facts (schedule, pricing, series, subscription), which can cite internal docs. See docs/decisions.md.

Hard blocks stop approval. A reviewer may override a hard block only with a written reason, which is logged and reviewed as a possible checker error. Soft warnings are flagged but don't block.

## Hard blocks
- **P1 — Cite factual claims.** Every technical or operational factual claim must cite a source in the corpus, and the cited source must actually support it. Operational facts (schedule, pricing, series, subscription) cite the internal docs.
  Violation: "Chunk sizes of 512 tokens work best" with no citation.
- **P2 — Numbers must match their source exactly.** Benchmarks, percentages, costs, context lengths and dates need a citation, and the figure must match the source in value and in what it describes.
  Violation: source says 128K context; content says 200K.
- **P3 — No career or outcome promises.** No guarantees about jobs, salaries, or time to skill.
  Violation: "Read this series and you'll be job-ready in 30 days."
- **P4 — Predictions are attributed opinions, never facts.**
  Violation: "Agents will replace most developers by 2028."
  Allowed: "Some researchers argue X (source); we're not convinced, because Y."
- **P5 — Use only the current version.** Content must not cite a superseded or corrected source.
  Violation: citing the original version of an explainer that has since been corrected.
- **P6 — No rumors or disparagement.** Nothing about unreleased products or plans without an official source; no disparaging named people or companies.

## Soft warnings
- **S1 — Hype vocabulary** from the brand guide's avoid list.
- **S2 — Anthropomorphizing** ("the model understands," "it wants").
- **S3 — More than one call to action.**
- **S4 — Undefined jargon** in content for the transitioning-engineers segment.

## Exempt from citation
Greetings, sign-offs, calls to action, "read next" pointers, transitions between sections, and opinions clearly marked as ours ("We think…").

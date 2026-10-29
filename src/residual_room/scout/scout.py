"""`make scout TOPIC="…" SERIES="…"`: propose new sources for a topic. Never approves.

  1. the LLM (judge model) turns the topic into search queries
  2. deterministic searches: OpenAlex (papers), GitHub (repos)
  3. the LLM ranks candidates for relevance, with a one-line reason
  4. licenses of the top candidates are resolved deterministically
  5. everything is written to data/proposed_sources.yaml for the owner

Sources already in the manifest are skipped. Approval is a separate, explicit
step (approve.py).
"""

import argparse
import sys
from dataclasses import asdict
from datetime import UTC, datetime

import yaml
from pydantic import BaseModel

from residual_room.corpus.manifest import REPO_ROOT, load_manifest
from residual_room.scout.licenses import resolve_acl, resolve_arxiv, resolve_github
from residual_room.scout.search import ARXIV_ID, Candidate, arxiv_license, client, search_github, search_openalex

PROPOSALS_PATH = REPO_ROOT / "data" / "proposed_sources.yaml"
KEEP = 8

QUERY_SYSTEM = """You help an editor find sources for an AI-engineering explainer.
Turn the topic into 3 to 5 short, distinct search queries for scholarly papers and GitHub repositories
(technical terms, no quotes, no boolean operators).
Reply with only JSON: {"queries": ["...", "..."]}"""

RANK_SYSTEM = """You rank candidate sources for an AI-engineering explainer.
Score each candidate's relevance to the topic from 1 (off-topic) to 5 (core reference), using only the
title and summary given. Prefer primary research and well-maintained reference implementations.
Reply with only JSON: {"ranked": [{"key": "<key>", "score": <1-5>, "reason": "<one line>"}]}"""


class Queries(BaseModel):
    queries: list[str]


class Ranked(BaseModel):
    key: str
    score: int
    reason: str


class Ranking(BaseModel):
    ranked: list[Ranked]


def scout(topic: str, series: str, llm, http, now: datetime | None = None) -> dict:
    manifest = load_manifest()
    known = _known_keys(manifest)
    queries = llm.ask_json(QUERY_SYSTEM, f"Topic: {topic}\nSeries: {series}", Queries).queries[:5]

    found: dict[str, Candidate] = {}
    for q in queries:
        for c in search_openalex(http, q) + search_github(http, q):
            if c.key not in known:
                found.setdefault(c.key, c)

    listing = "\n".join(f"- key: {c.key}\n  kind: {c.kind}\n  title: {c.title}\n  summary: {c.summary[:400]}"
                        for c in found.values())
    ranking = llm.ask_json(RANK_SYSTEM, f"Topic: {topic}\n\nCandidates:\n{listing}", Ranking).ranked
    scores = {r.key: r for r in ranking if r.key in found}
    top = sorted(scores.values(), key=lambda r: -r.score)[:KEEP]

    candidates = []
    for i, r in enumerate(top, 1):
        c = found[r.key]
        lic = _resolve(c, http)
        candidates.append({
            "proposal_id": f"p{i}", "key": c.key, "kind": c.kind, "title": c.title, "year": c.year, "url": c.url,
            "relevance": r.score, "reason": r.reason,
            "license": lic.license, "license_verified": lic.verified, "commit_allowed": lic.commit_allowed,
            "license_evidence": lic.evidence, "hints": c.hints, "status": "proposed",
            "_source": {k: v for k, v in asdict(c).items() if k in ("arxiv_id", "acl_url", "repo")},
        })
    return {
        "topic": topic, "series": series, "scouted_at": (now or datetime.now(UTC)).isoformat(timespec="seconds"),
        "queries": queries, "searched": len(found), "candidates": candidates,
    }


def _resolve(c: Candidate, http):
    if c.acl_url:  # the Anthology version is preferred when both exist (manifest rule)
        return resolve_acl(c.acl_url)
    if c.arxiv_id:
        return resolve_arxiv(arxiv_license(http, c.arxiv_id), c.arxiv_id)
    return resolve_github(c.spdx, c.repo)


def _known_keys(manifest) -> set[str]:
    keys = set()
    for s in manifest.external:
        for url in (s.url or "", s.fetch_url or ""):
            if m := ARXIV_ID.search(url):
                keys.add(f"arxiv:{m.group(1)}")
            if "aclanthology.org" in url:
                keys.add("acl:" + url.rstrip("/").split("/")[-1].removesuffix(".pdf"))
            if url.startswith("https://github.com/"):
                keys.add("github:" + url.removeprefix("https://github.com/").strip("/"))
    return keys


def write_proposals(batch: dict) -> None:
    header = ("# Proposed sources from `make scout`. NOT approved: nothing here is in the manifest.\n"
              "# Approve one with: make approve-source ID=p1\n")
    PROPOSALS_PATH.write_text(header + yaml.safe_dump(batch, sort_keys=False, allow_unicode=True, width=110),
                              encoding="utf-8", newline="\n")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--series", required=True)
    args = parser.parse_args()

    from residual_room.llm.client import load_role

    batch = scout(args.topic, args.series, load_role("judge"), client())
    write_proposals(batch)
    print(f"queries: {batch['queries']}\nsearched {batch['searched']} new candidates; top {len(batch['candidates'])}:")
    for c in batch["candidates"]:
        flag = "commit" if c["commit_allowed"] else "local only"
        print(f"  {c['proposal_id']} [{c['relevance']}/5] {c['kind']:<5} {flag:<10} {c['title'][:60]}\n"
              f"      {c['license']} | {c['reason'][:110]}")
    print(f"\nwritten to {PROPOSALS_PATH.relative_to(REPO_ROOT)}; approve with: make approve-source ID=p1")
    return 0


if __name__ == "__main__":
    sys.exit(main())

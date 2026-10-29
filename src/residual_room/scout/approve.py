"""`make approve-source ID=p1`: the owner moves one proposal into the manifest.

The only way scouted sources reach data/sources_manifest.yaml. The entry is
appended as text (comments and layout untouched), under a topic number shared
by the whole scout batch. If the manifest no longer loads afterwards, the
original text is restored. Then: `make fetch` + `make ingest VERSION=corpus-vN`.
"""

import argparse
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml

from residual_room.corpus.manifest import MANIFEST_PATH, load_manifest
from residual_room.scout.scout import PROPOSALS_PATH
from residual_room.scout.search import client


class ApprovalError(ValueError):
    pass


_FILLER = {"a", "an", "the", "of", "for", "to", "in", "on", "and", "with", "towards", "via", "from", "by"}


def approve(proposal_id: str, http: httpx.Client, proposals_path: Path = PROPOSALS_PATH,
            manifest_path: Path = MANIFEST_PATH, today: str | None = None) -> dict:
    if not proposals_path.is_file():
        raise ApprovalError("no proposals yet: run `make scout` first")
    batch = yaml.safe_load(proposals_path.read_text(encoding="utf-8"))
    cand = next((c for c in batch["candidates"] if c["proposal_id"] == proposal_id), None)
    if cand is None:
        raise ApprovalError(f"no proposal {proposal_id}")
    if cand["status"] != "proposed":
        raise ApprovalError(f"{proposal_id} is already {cand['status']} ({cand.get('manifest_id')})")

    manifest = load_manifest(manifest_path)
    today = today or datetime.now(UTC).date().isoformat()
    first_in_batch = "topic_number" not in batch
    if first_in_batch:
        batch["topic_number"] = max(s.topic or 0 for s in manifest.external) + 1

    entry = _entry(cand, batch, manifest, http, today)
    original = manifest_path.read_text(encoding="utf-8")
    block = ""
    if first_in_batch:
        block += (f"\n  # Topic {batch['topic_number']}: {batch['topic']} ({batch['series']}; "
                  f"added via scout, owner-approved {today})\n")
    block += "  - " + yaml.safe_dump(entry, default_flow_style=True, width=10**6, sort_keys=False).strip() + "\n"
    manifest_path.write_text(original.rstrip("\n") + "\n" + block, encoding="utf-8", newline="\n")
    try:
        load_manifest(manifest_path).resolve(entry["id"])
    except Exception as exc:  # never leave a broken manifest behind
        manifest_path.write_text(original, encoding="utf-8", newline="\n")
        raise ApprovalError(f"manifest would not load with this entry, restored it: {exc}") from exc

    cand["status"], cand["manifest_id"] = "approved", entry["id"]
    proposals_path.write_text(proposals_path.read_text(encoding="utf-8").split("\n", 2)[0] + "\n"
                              + "# Approve one with: make approve-source ID=p1\n"
                              + yaml.safe_dump(batch, sort_keys=False, allow_unicode=True, width=110),
                              encoding="utf-8", newline="\n")
    return entry


def _entry(cand: dict, batch: dict, manifest, http: httpx.Client, today: str) -> dict:
    src = cand["_source"]
    entry = {"id": _unique_id(cand, manifest), "topic": batch["topic_number"],
             "kind": "paper" if cand["kind"] == "paper" else "code", "title": cand["title"], "url": cand["url"],
             "license": cand["license"], "license_verified": cand["license_verified"],
             "commit_allowed": cand["commit_allowed"] and cand["license_verified"],
             "note": f"scouted {batch['scouted_at'][:10]}, approved {today}: {cand['license_evidence']}"}
    if src.get("acl_url"):
        entry["fetch_url"] = src["acl_url"].rstrip("/") + ".pdf"
    elif src.get("arxiv_id"):
        html = f"https://arxiv.org/html/{src['arxiv_id']}"
        ok = http.get(html).status_code == 200  # HTML keeps tables intact; older papers only have PDF
        entry["fetch_url"] = html if ok else f"https://arxiv.org/pdf/{src['arxiv_id']}"
    else:
        entry["fetch_url"] = f"https://github.com/{src['repo']}"
        readme = http.get(f"https://api.github.com/repos/{src['repo']}/readme")
        entry["include"] = [readme.json()["path"] if readme.status_code == 200 else "README.md"]
    return entry


def _unique_id(cand: dict, manifest) -> str:
    if cand["kind"] == "code":
        base = cand["_source"].get("repo", "").split("/")[-1]
    else:  # first content words of the title: "Per-Query Gating of LLM Rerankers" -> per-query-gating-llm-rerankers
        words = [w for w in re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", cand["title"]) if w.lower() not in _FILLER]
        base = " ".join(words[:4])
    slug = "ext-" + re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")
    taken = set(manifest.sources) | set(manifest.aliases)
    out, n = slug, 2
    while out in taken:
        out, n = f"{slug}-{n}", n + 1
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", required=True, help="proposal id, e.g. p1")
    args = parser.parse_args()
    try:
        entry = approve(args.id, client())
    except ApprovalError as exc:
        print(f"not approved: {exc}")
        return 1
    where = "corpus/ (committable)" if entry["commit_allowed"] else "corpus_local/ (local only)"
    print(f"approved {args.id} -> {entry['id']} (topic {entry['topic']}), text will go to {where}")
    print(f"next: make fetch IDS={entry['id']}   then   make ingest VERSION=corpus-v2")
    return 0


if __name__ == "__main__":
    sys.exit(main())

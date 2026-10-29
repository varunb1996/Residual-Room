"""`make fetch`: download external sources and store them as markdown.

Each source goes to `<text_dir>/<source_id>/` as `text.md` plus `meta.json`
(what was fetched, from where, when, and a SHA-256 of the text). text_dir is
`corpus/` for committable sources and the gitignored `corpus_local/` otherwise.
Internal sources are not fetched: ingest reads them from docs/ directly.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime

import httpx

from residual_room.corpus.extract import (
    arxiv_html_to_markdown,
    pdf_to_markdown,
    repo_file_to_markdown,
)
from residual_room.corpus.manifest import Source, load_manifest

USER_AGENT = "residual-room-fetch/0.1 (portfolio project; one-off source download)"


def fetch_source(client: httpx.Client, source: Source) -> dict:
    url = source.fetch_url
    if source.is_repo:
        text, extra = _fetch_repo(client, source)
    elif url.endswith(".pdf") or "arxiv.org/pdf/" in url:
        text, extra = pdf_to_markdown(_get(client, url).content), {}
    elif "arxiv.org/html/" in url:
        text, extra = arxiv_html_to_markdown(_get(client, url).text), {}
    else:
        raise ValueError(f"{source.id}: don't know how to fetch {url}")

    text = f"# {source.title}\n\n{text.strip()}\n"
    meta = {
        "source_id": source.id,
        "fetch_url": url,
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "license": source.license,
        "committable": source.committable,
        **extra,
    }
    source.text_dir.mkdir(parents=True, exist_ok=True)
    # newline="\n": byte-identical files on Windows and Linux.
    (source.text_dir / "text.md").write_text(text, encoding="utf-8", newline="\n")
    (source.text_dir / "meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return meta


def _fetch_repo(client: httpx.Client, source: Source) -> tuple[str, dict]:
    # Pin to the commit the default branch points at now, so the text is reproducible.
    # One API call per repo: anonymous GitHub API use is capped at 60 requests/hour.
    repo = source.fetch_url.removeprefix("https://github.com/").strip("/")
    sha = _get(client, f"https://api.github.com/repos/{repo}/commits/HEAD").json()["sha"]
    parts = []
    for path in source.include:
        raw = _get(client, f"https://raw.githubusercontent.com/{repo}/{sha}/{path}").text
        parts.append(f"# {path}\n\n{repo_file_to_markdown(path, raw).strip()}")
    return "\n\n".join(parts), {"repo": repo, "commit": sha, "include": list(source.include)}


def _get(client: httpx.Client, url: str) -> httpx.Response:
    response = client.get(url)
    response.raise_for_status()
    return response


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ids", nargs="*", help="only these source ids (default: all external)")
    args = parser.parse_args()

    manifest = load_manifest()
    sources = [manifest.resolve(i) for i in args.ids] if args.ids else manifest.external
    failed = []
    headers = {"User-Agent": USER_AGENT}
    if token := os.environ.get("GITHUB_TOKEN"):  # optional; lifts the GitHub API rate limit
        headers["Authorization"] = f"Bearer {token}"
    with httpx.Client(follow_redirects=True, timeout=60, headers=headers) as client:
        for s in sources:
            try:
                meta = fetch_source(client, s)
                size = (s.text_dir / "text.md").stat().st_size
                where = "corpus" if s.committable else "corpus_local"
                print(f"ok    {s.id:<24} {size:>8,} B -> {where}/  {meta.get('commit', '')[:10]}")
            except Exception as exc:  # report every failure, then exit non-zero
                failed.append(s.id)
                print(f"FAIL  {s.id:<24} {type(exc).__name__}: {exc}")
    if failed:
        print(f"\n{len(failed)} source(s) failed: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Load and validate data/sources_manifest.yaml.

The manifest is the only list of approved sources. This module turns it into
typed records, resolves aliases, and decides where each source's text may live:
committed `corpus/` or gitignored `corpus_local/`.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = REPO_ROOT / "data" / "sources_manifest.yaml"
CORPUS_DIR = REPO_ROOT / "corpus"
CORPUS_LOCAL_DIR = REPO_ROOT / "corpus_local"

STATUSES = {"current", "superseded"}


@dataclass(frozen=True)
class Source:
    id: str
    title: str
    origin: str  # "internal" | "external"
    status: str
    license: str
    commit_allowed: bool
    kind: str | None = None  # paper | docs | tutorial | code (external only)
    topic: int | None = None
    url: str | None = None
    fetch_url: str | None = None
    include: tuple[str, ...] = ()
    path: str | None = None  # internal docs, relative to the repo root
    license_verified: bool = False

    @property
    def committable(self) -> bool:
        # An unverified license never permits committing, whatever the flag says.
        return self.commit_allowed and self.license_verified

    @property
    def is_repo(self) -> bool:
        return (self.fetch_url or "").startswith("https://github.com/")

    @property
    def text_dir(self) -> Path:
        """Where fetched text for an external source is stored."""
        return (CORPUS_DIR if self.committable else CORPUS_LOCAL_DIR) / self.id


@dataclass(frozen=True)
class Manifest:
    sources: dict[str, Source]  # canonical sources only
    aliases: dict[str, str]  # alias id -> canonical id

    def resolve(self, source_id: str) -> Source:
        """Look up a source by canonical id or alias."""
        return self.sources[self.aliases.get(source_id, source_id)]

    @property
    def external(self) -> list[Source]:
        return [s for s in self.sources.values() if s.origin == "external"]

    @property
    def internal(self) -> list[Source]:
        return [s for s in self.sources.values() if s.origin == "internal"]


class ManifestError(ValueError):
    pass


def load_manifest(path: Path = MANIFEST_PATH) -> Manifest:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    sources: dict[str, Source] = {}
    aliases: dict[str, str] = {}

    for origin in ("internal", "external"):
        for entry in raw.get(origin, []):
            sid = entry["id"]
            if sid in sources or sid in aliases:
                raise ManifestError(f"duplicate source id: {sid}")
            if "ref" in entry:
                aliases[sid] = entry["ref"]
                continue
            sources[sid] = _to_source(entry, origin)

    for alias, target in aliases.items():
        if target not in sources:
            raise ManifestError(f"alias {alias} points at unknown source {target}")

    for s in sources.values():
        if s.status not in STATUSES:
            raise ManifestError(f"{s.id}: unknown status {s.status!r}")
        if s.origin == "internal" and not (REPO_ROOT / (s.path or "")).is_file():
            raise ManifestError(f"{s.id}: internal path {s.path!r} does not exist")
        if s.origin == "external" and s.is_repo and not s.include:
            raise ManifestError(f"{s.id}: repo sources need an include list")

    return Manifest(sources=sources, aliases=aliases)


def _to_source(entry: dict, origin: str) -> Source:
    return Source(
        id=entry["id"],
        title=entry["title"],
        origin=origin,
        # External sources have no status field in the manifest; they are current.
        status=entry.get("status", "current"),
        license=entry["license"],
        commit_allowed=bool(entry.get("commit_allowed", False)),
        # Internal docs are our own ("license: own"), so they count as verified.
        license_verified=bool(entry.get("license_verified", origin == "internal")),
        kind=entry.get("kind"),
        topic=entry.get("topic"),
        url=entry.get("url"),
        fetch_url=entry.get("fetch_url") or entry.get("url"),
        include=tuple(entry.get("include", ())),
        path=entry.get("path"),
    )

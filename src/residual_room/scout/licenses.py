"""Deterministic license resolution for scouted sources. No LLM, ever.

Follows the manifest's own rule: text may be committed only under CC BY, MIT,
Apache-2.0 (or our own); anything else, including share-alike, non-commercial
and arXiv's default license, is fetched locally only. Unresolved licenses are
never committable.
"""

from dataclasses import dataclass

COMMITTABLE_SPDX = {"MIT", "Apache-2.0", "CC-BY-4.0"}


@dataclass(frozen=True)
class License:
    license: str  # human-readable, as written into the manifest
    verified: bool  # resolved from an authoritative source
    commit_allowed: bool
    evidence: str  # how it was resolved, for the owner to check


def resolve_acl(anthology_url: str) -> License:
    # ACL Anthology papers (ACL, EMNLP, NAACL, EACL, TACL) are CC BY 4.0 (manifest rule of thumb).
    return License("CC BY 4.0 (ACL Anthology)", True, True, f"ACL Anthology policy: {anthology_url}")


def resolve_arxiv(license_url: str | None, arxiv_id: str) -> License:
    """From arXiv's own metadata (OAI-PMH). No license field = arXiv's default license."""
    url = (license_url or "").rstrip("/").lower()
    evidence = f"arXiv OAI-PMH record for {arxiv_id}: {license_url or 'no license field'}"
    if url.startswith("http://creativecommons.org/licenses/by/") or url.startswith(
        "https://creativecommons.org/licenses/by/"
    ):
        version = url.split("/licenses/by/")[1].strip("/") or "?"
        return License(f"CC BY {version} (arXiv)", True, True, evidence)
    if "creativecommons.org" in url:
        kind = url.split("/licenses/")[-1].split("/")[0].upper() if "/licenses/" in url else "CC"
        return License(f"CC {kind} (arXiv; not CC BY, so not redistributed)", True, False, evidence)
    if "nonexclusive-distrib" in url or not url:
        return License("arXiv perpetual non-exclusive (not redistributable)", True, False, evidence)
    return License(f"unrecognized arXiv license {license_url}", False, False, evidence)


def resolve_github(spdx: str | None, repo: str) -> License:
    evidence = f"GitHub license API for {repo}: {spdx or 'none'}"
    if spdx in COMMITTABLE_SPDX:
        return License(spdx, True, True, evidence)
    if not spdx or spdx == "NOASSERTION":
        return License("no license found (treated as non-redistributable)", False, False, evidence)
    return License(f"{spdx} (not in the committable set)", True, False, evidence)

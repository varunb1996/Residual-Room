"""Deterministic searches for candidate sources.

Papers come from OpenAlex (free, no key), which indexes both arXiv and the ACL
Anthology. arXiv's own search API timed out and then returned 429 in testing,
and the Anthology has no search API, so OpenAlex does the search; arXiv stays
the authority for arXiv licenses (OAI-PMH, paced as arXiv asks). Repos come
from GitHub search, with the license GitHub reports.
"""

import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import httpx

USER_AGENT = "residual-room-scout/0.1 (portfolio project)"
ARXIV_ID = re.compile(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})")
ARXIV_PAUSE = 3.0  # seconds between arXiv requests, per arXiv's API guidance


@dataclass
class Candidate:
    key: str  # "arxiv:2212.10528", "acl:2024.acl-long.511", "github:owner/repo"
    kind: str  # paper | code
    title: str
    url: str
    summary: str
    year: int | None = None
    arxiv_id: str | None = None
    acl_url: str | None = None
    repo: str | None = None
    spdx: str | None = None
    hints: dict = field(default_factory=dict)  # third-party metadata, shown but never decisive


def client(transport: httpx.BaseTransport | None = None) -> httpx.Client:
    return httpx.Client(timeout=60, headers={"User-Agent": USER_AGENT}, follow_redirects=True, transport=transport)


def search_openalex(http: httpx.Client, query: str, per_page: int = 15) -> list[Candidate]:
    r = http.get("https://api.openalex.org/works", params={
        "search": query, "per-page": per_page,
        "select": "id,display_name,publication_year,locations,abstract_inverted_index,cited_by_count",
    })
    r.raise_for_status()
    out = []
    for w in r.json().get("results", []):
        urls = [loc.get("landing_page_url") or "" for loc in w.get("locations") or []]
        acl = next((u for u in urls if "aclanthology.org" in u), None)
        arxiv = next((m.group(1) for u in urls if (m := ARXIV_ID.search(u))), None)
        if not (acl or arxiv):
            continue  # we can only fetch and license-check arXiv and ACL Anthology papers
        key = f"acl:{acl.rstrip('/').split('/')[-1]}" if acl else f"arxiv:{arxiv}"
        out.append(Candidate(
            key=key, kind="paper", title=w.get("display_name") or "", year=w.get("publication_year"),
            url=acl or f"https://arxiv.org/abs/{arxiv}", summary=_abstract(w.get("abstract_inverted_index")),
            arxiv_id=arxiv, acl_url=acl,
            hints={"openalex_licenses": sorted({loc.get("license") for loc in w.get("locations") or []
                                                if loc.get("license")}),
                   "cited_by": w.get("cited_by_count")},
        ))
    return out


def search_github(http: httpx.Client, query: str, per_page: int = 5) -> list[Candidate]:
    r = http.get("https://api.github.com/search/repositories",
                 params={"q": f"{query} in:name,description,readme", "sort": "stars", "per_page": per_page})
    r.raise_for_status()
    out = []
    for it in r.json().get("items", []):
        if it.get("fork") or it.get("archived"):
            continue
        out.append(Candidate(
            key=f"github:{it['full_name']}", kind="code", title=it["full_name"], url=it["html_url"],
            summary=it.get("description") or "", repo=it["full_name"],
            spdx=(it.get("license") or {}).get("spdx_id"), hints={"stars": it.get("stargazers_count")},
        ))
    return out


def arxiv_license(http: httpx.Client, arxiv_id: str) -> str | None:
    """The license URL on arXiv's own record, or None (= arXiv's default license)."""
    time.sleep(ARXIV_PAUSE)
    r = http.get("https://export.arxiv.org/oai2",
                 params={"verb": "GetRecord", "identifier": f"oai:arXiv.org:{arxiv_id}", "metadataPrefix": "arXiv"})
    r.raise_for_status()
    return next((el.text for el in ET.fromstring(r.text).iter() if el.tag.endswith("license")), None)


def _abstract(inverted: dict | None, limit: int = 600) -> str:
    """OpenAlex stores abstracts as {word: [positions]}; rebuild the text."""
    if not inverted:
        return ""
    words = sorted((pos, word) for word, positions in inverted.items() for pos in positions)
    return " ".join(w for _, w in words)[:limit]

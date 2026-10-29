"""Scout and approval, with faked HTTP and a fake LLM. Approval writes to a temp
copy of the manifest, never the real one."""

import shutil

import httpx
import pytest
import yaml
from fakes import FakeJudge

from residual_room.corpus.manifest import MANIFEST_PATH, load_manifest
from residual_room.scout.approve import ApprovalError, approve
from residual_room.scout.scout import scout
from residual_room.scout.search import client

OPENALEX = {"results": [
    {"display_name": "A CC BY paper on reranking", "publication_year": 2024, "cited_by_count": 10,
     "abstract_inverted_index": {"Cross-encoders": [0], "rerank": [1], "passages": [2]},
     "locations": [{"landing_page_url": "http://arxiv.org/abs/2212.10528", "license": "cc-by"}]},
    {"display_name": "An ACL paper", "publication_year": 2023, "locations": [
        {"landing_page_url": "https://aclanthology.org/2023.emnlp-main.1/"},
        {"landing_page_url": "https://arxiv.org/abs/2301.00001"}]},
    {"display_name": "Journal-only paper", "locations": [{"landing_page_url": "https://example.org/x"}]},
    {"display_name": "Lost in the Middle (already in manifest)",
     "locations": [{"landing_page_url": "https://arxiv.org/abs/2307.03172"}]},
]}
GITHUB = {"items": [
    {"full_name": "acme/rerankers", "html_url": "https://github.com/acme/rerankers", "description": "rerank lib",
     "license": {"spdx_id": "Apache-2.0"}, "stargazers_count": 900, "fork": False, "archived": False},
    {"full_name": "acme/fork", "html_url": "https://github.com/acme/fork", "fork": True, "license": None},
]}
OAI_CC_BY = ("<OAI-PMH><GetRecord><record><metadata><arXiv><license>"
             "http://creativecommons.org/licenses/by/4.0/</license></arXiv></metadata></record></GetRecord></OAI-PMH>")


def handler(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    if "openalex" in url:
        return httpx.Response(200, json=OPENALEX)
    if "search/repositories" in url:
        return httpx.Response(200, json=GITHUB)
    if "oai2" in url:
        return httpx.Response(200, text=OAI_CC_BY)
    if "arxiv.org/html/" in url:
        return httpx.Response(404)
    if url.endswith("/readme"):
        return httpx.Response(200, json={"path": "README.md"})
    return httpx.Response(404)


@pytest.fixture
def http(monkeypatch):
    monkeypatch.setattr("residual_room.scout.search.ARXIV_PAUSE", 0)
    return client(httpx.MockTransport(handler))


def llm():
    def script(system, user):
        if "search queries" in system:
            return {"queries": ["cross-encoder reranking", "rerankers for RAG"]}
        keys = [line.split("key: ")[1] for line in user.splitlines() if "key: " in line]
        return {"ranked": [{"key": k, "score": 5 - i, "reason": f"reason {i}"} for i, k in enumerate(keys)]}
    return FakeJudge(script)


def test_scout_ranks_resolves_licenses_and_skips_known_sources(http):
    batch = scout("Rerankers for RAG", "Failure Modes", llm(), http)
    keys = [c["key"] for c in batch["candidates"]]
    assert keys == ["arxiv:2212.10528", "acl:2023.emnlp-main.1", "github:acme/rerankers"]
    # (journal-only paper, the fork, and Lost in the Middle, already in the manifest, are skipped)
    by = {c["key"]: c for c in batch["candidates"]}
    assert by["arxiv:2212.10528"]["commit_allowed"] and "OAI-PMH" in by["arxiv:2212.10528"]["license_evidence"]
    assert "ACL Anthology" in by["acl:2023.emnlp-main.1"]["license"]  # Anthology preferred over arXiv
    assert by["github:acme/rerankers"]["license"] == "Apache-2.0"
    assert all(c["status"] == "proposed" and c["reason"] for c in batch["candidates"])
    assert by["arxiv:2212.10528"]["hints"]["openalex_licenses"] == ["cc-by"]  # shown, never decisive


@pytest.fixture
def files(tmp_path, http):
    manifest = tmp_path / "manifest.yaml"
    shutil.copy(MANIFEST_PATH, manifest)
    proposals = tmp_path / "proposed.yaml"
    batch = scout("Rerankers for RAG", "Failure Modes", llm(), http)
    proposals.write_text("# header\n# approve\n" + yaml.safe_dump(batch, sort_keys=False), encoding="utf-8")
    return manifest, proposals


def test_approving_two_appends_valid_entries_under_one_new_topic(files, http):
    manifest, proposals = files
    before = load_manifest(manifest)
    paper = approve("p1", http, proposals, manifest, today="2026-10-29")
    repo = approve("p3", http, proposals, manifest, today="2026-10-29")

    after = load_manifest(manifest)
    assert len(after.external) == len(before.external) + 2
    new_topic = max(s.topic for s in before.external) + 1
    assert paper["topic"] == repo["topic"] == new_topic
    p = after.resolve(paper["id"])
    assert p.fetch_url == "https://arxiv.org/pdf/2212.10528" and p.committable  # no HTML -> PDF
    r = after.resolve(repo["id"])
    assert r.is_repo and r.include == ("README.md",) and r.id == "ext-rerankers"
    text = manifest.read_text(encoding="utf-8")
    assert text.count(f"# Topic {new_topic}: Rerankers for RAG") == 1  # one section header per batch
    assert "# Gaps:" in text  # existing comments untouched

    statuses = {c["proposal_id"]: c["status"] for c in yaml.safe_load(proposals.read_text())["candidates"]}
    assert statuses == {"p1": "approved", "p2": "proposed", "p3": "approved"}


def test_approval_refuses_unknown_and_repeat(files, http):
    manifest, proposals = files
    approve("p1", http, proposals, manifest)
    with pytest.raises(ApprovalError, match="already approved"):
        approve("p1", http, proposals, manifest)
    with pytest.raises(ApprovalError, match="no proposal"):
        approve("p9", http, proposals, manifest)


def test_real_manifest_is_never_touched_by_these_tests(files, http):
    before = MANIFEST_PATH.read_bytes()
    manifest, proposals = files
    approve("p2", http, proposals, manifest)  # writes only to the temp copy
    assert MANIFEST_PATH.read_bytes() == before

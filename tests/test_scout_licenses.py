import pytest

from residual_room.scout.licenses import resolve_acl, resolve_arxiv, resolve_github


@pytest.mark.parametrize(
    "url, committable, verified",
    [
        ("http://creativecommons.org/licenses/by/4.0/", True, True),
        ("https://creativecommons.org/licenses/by/3.0/", True, True),
        ("http://creativecommons.org/licenses/by-sa/4.0/", False, True),  # share-alike: not in the rule
        ("http://creativecommons.org/licenses/by-nc-sa/4.0/", False, True),
        ("http://arxiv.org/licenses/nonexclusive-distrib/1.0/", False, True),  # MT-Bench, RAG survey
        (None, False, True),  # no field = arXiv default license
        ("https://example.org/weird", False, False),
    ],
)
def test_arxiv(url, committable, verified):
    lic = resolve_arxiv(url, "2306.05685")
    assert (lic.commit_allowed, lic.verified) == (committable, verified)
    assert "2306.05685" in lic.evidence


@pytest.mark.parametrize(
    "spdx, committable, verified",
    [("MIT", True, True), ("Apache-2.0", True, True), ("GPL-3.0", False, True),
     ("NOASSERTION", False, False), (None, False, False)],
)
def test_github(spdx, committable, verified):
    lic = resolve_github(spdx, "owner/repo")
    assert (lic.commit_allowed, lic.verified) == (committable, verified)


def test_acl_anthology_is_cc_by():
    lic = resolve_acl("https://aclanthology.org/2024.acl-long.511/")
    assert lic.commit_allowed and lic.verified and "CC BY 4.0" in lic.license

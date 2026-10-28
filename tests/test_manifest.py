import pytest

from residual_room.corpus.manifest import (
    CORPUS_DIR,
    CORPUS_LOCAL_DIR,
    ManifestError,
    load_manifest,
)


@pytest.fixture(scope="module")
def manifest():
    return load_manifest()


def test_alias_resolves_to_canonical_source(manifest):
    assert "ext-litm-for-t3" not in manifest.sources
    assert manifest.resolve("ext-litm-for-t3").id == "ext-litm-paper"


def test_non_committable_sources_go_to_gitignored_folder(manifest):
    local = {s.id for s in manifest.external if s.text_dir.parent == CORPUS_LOCAL_DIR}
    assert local == {"ext-rag-paper", "ext-mtbench-paper", "ext-rag-survey"}


def test_committable_sources_go_to_corpus(manifest):
    assert manifest.resolve("ext-litm-paper").text_dir.parent == CORPUS_DIR


def test_unverified_license_is_never_committable(tmp_path):
    path = tmp_path / "m.yaml"
    path.write_text(
        "external:\n"
        "  - {id: x, title: X, license: unknown, license_verified: false, commit_allowed: true,"
        " fetch_url: 'https://example.org/x.pdf'}\n",
        encoding="utf-8",
    )
    assert not load_manifest(path).resolve("x").committable


def test_status_comes_from_manifest(manifest):
    assert manifest.resolve("int-schedule-v0").status == "superseded"
    assert manifest.resolve("exp-003-v1").status == "superseded"
    assert manifest.resolve("exp-003-v2").status == "current"
    assert all(s.status == "current" for s in manifest.external)


def test_repo_sources_have_small_include_lists(manifest):
    repos = [s for s in manifest.external if s.is_repo]
    assert len(repos) == 9
    assert all(1 <= len(s.include) <= 5 for s in repos)


def test_alias_to_unknown_source_is_rejected(tmp_path):
    path = tmp_path / "m.yaml"
    path.write_text("external:\n  - {id: a, ref: missing}\n", encoding="utf-8")
    with pytest.raises(ManifestError):
        load_manifest(path)

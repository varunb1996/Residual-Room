"""Drift tests: config/publication.yaml must agree with the frozen brand guide."""

import re

from residual_room.config import get_config
from residual_room.corpus.manifest import REPO_ROOT


def _brand_guide_line(prefix: str) -> str:
    text = (REPO_ROOT / get_config().docs["brand_guide"]).read_text(encoding="utf-8")
    (line,) = [ln for ln in text.splitlines() if ln.lstrip("- ").startswith(prefix)]
    return line.lstrip("- ").removeprefix(prefix)


def test_avoid_list_matches_brand_guide():
    raw = _brand_guide_line("Avoid:")
    guide = {w.strip().strip('".').strip() for w in raw.split(",")}
    assert {w.lower() for w in guide} == {w.lower() for w in get_config().avoid_list}


def test_every_brand_guide_cta_type_has_a_pattern():
    # "One per message, chosen from: read the explainer, subscribe, read the source, reply with a question."
    text = (REPO_ROOT / get_config().docs["brand_guide"]).read_text(encoding="utf-8")
    listed = re.search(r"chosen from: (.+?)\.\n", text).group(1).split(", ")
    patterns = [p for ps in get_config().cta_regexes.values() for p in ps]
    for cta in listed:
        assert any(p.search(cta) for p in patterns), f"no CTA pattern for {cta!r}"
    assert len(get_config().ctas) == len(listed)


def _doc(name: str) -> str:
    return (REPO_ROOT / get_config().docs[name]).read_text(encoding="utf-8")


def test_schedule_matches_schedule_v1():
    text = _doc("schedule")
    issue_1 = get_config().schedule.issue_1
    assert "Thursday" in text and issue_1.strftime("%A") == "Thursday"
    assert f"{issue_1.day} {issue_1:%B %Y}" in text  # "5 November 2026"
    assert get_config().schedule.every_days == 7 and "weekly" in text


def test_corrections_match_corrections_doc():
    text = _doc("corrections")
    for c in get_config().corrections:
        assert f"{c.explainer} v2 (corrected {c.corrected_on.isoformat()})" in text
    # The doc names Issue 2 (2026-11-12) as the first issue after the correction.
    (c,) = get_config().corrections
    assert get_config().schedule.first_issue_after(c.corrected_on).isoformat() == "2026-11-12"
    assert "Issue 2 (2026-11-12)" in text


def test_operational_sources_are_small_internal_docs():
    from residual_room.corpus.chunk import MAX_TOKENS, estimate_tokens
    from residual_room.corpus.manifest import load_manifest

    manifest = load_manifest()
    for sid in get_config().operational_sources:
        source = manifest.resolve(sid)
        assert source.origin == "internal"
        text = (REPO_ROOT / source.path).read_text(encoding="utf-8")
        assert estimate_tokens(text) <= MAX_TOKENS  # one chunk each, cheap to always include


def test_explainers_match_series_doc():
    text = (REPO_ROOT / "docs" / "editorial" / "series.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\| (exp-\d{3}) \| (.+?) \| (.+?) \|$", text, re.M)
    assert {eid: (topic, series) for eid, topic, series in rows} == {
        eid: (e.topic, e.series) for eid, e in get_config().explainers.items()
    }


def test_explainer_source_topics_exist_in_manifest():
    from residual_room.corpus.manifest import load_manifest

    topics = {s.topic for s in load_manifest().external}
    for e in get_config().explainers.values():
        assert set(e.source_topics) <= topics


def test_default_segment_is_a_known_segment():
    cfg = get_config()
    assert cfg.default_segment in cfg.segments

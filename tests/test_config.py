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


def test_default_segment_is_a_known_segment():
    cfg = get_config()
    assert cfg.default_segment in cfg.segments

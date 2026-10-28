"""The 50 frozen eval cases (eval/*.yaml) as typed records.

Labels are the owner's ground truth and are never edited here. Cases without
a segment get the config default; STL-10's text lives in a fixture.
"""

import re
from dataclasses import dataclass
from datetime import date
from functools import lru_cache

import yaml

from residual_room.config import get_config
from residual_room.corpus.manifest import REPO_ROOT, load_manifest

EVAL_DIR = REPO_ROOT / "eval"
# The five hard cases named in PROMPTS.md (Prompt 5).
HARD_CASES = ("GRD-05", "GRD-11", "GRD-13", "GRD-17", "POL-12")
FIXTURES = {"STL-10": EVAL_DIR / "fixtures" / "stl-10-issue-2.md"}
_PASS = {"verdict": "pass", "rules": []}


@dataclass(frozen=True)
class EvalCase:
    id: str
    text: str
    expected_verdict: str
    expected_rules: tuple[str, ...]
    segment: str
    issue_date: date | None
    citations: tuple[str, ...]
    committable: bool  # every cited source may be committed (runnable in CI)
    hard: bool
    note: str | None = None


@lru_cache
def raw_cases() -> dict[str, dict]:
    """Cases exactly as written in eval/*.yaml, keyed by id."""
    cases = {}
    for path in sorted(EVAL_DIR.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        for value in data.values():
            if isinstance(value, list):
                cases.update({c["id"]: c for c in value if isinstance(c, dict) and "id" in c})
    return cases


@lru_cache
def load_cases() -> dict[str, EvalCase]:
    manifest, cfg = load_manifest(), get_config()
    out = {}
    for cid, raw in raw_cases().items():
        expected = raw.get("expected", _PASS)
        text, segment, issue_date = raw["text"], raw.get("segment"), None
        if cid in FIXTURES:
            _, front, text = FIXTURES[cid].read_text(encoding="utf-8").split("---\n", 2)
            meta = yaml.safe_load(front)
            segment, issue_date = meta.get("segment"), meta.get("issue_date")
        citations = tuple(re.findall(r"\[cite:([^\]\s]+)\]", text))
        known = [c for c in citations if c in manifest.sources or c in manifest.aliases]
        out[cid] = EvalCase(
            id=cid,
            text=text,
            expected_verdict=expected["verdict"],
            expected_rules=tuple(expected.get("rules", [])),
            segment=segment or cfg.default_segment,
            issue_date=issue_date,
            citations=citations,
            committable=all(manifest.resolve(c).committable for c in known),
            hard=cid in HARD_CASES,
            note=raw.get("note") or expected.get("note"),
        )
    return out

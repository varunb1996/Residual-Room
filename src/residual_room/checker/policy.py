"""Rule text straight from docs/content-policy.md, so judge prompts can't drift
from the policy the owner signs off on."""

import re
from functools import lru_cache

from residual_room.config import get_config
from residual_room.corpus.manifest import REPO_ROOT


@lru_cache
def policy_rules() -> dict[str, str]:
    """{"P1": "P1 — Cite factual claims. Every ... Violation: ...", ...}"""
    text = (REPO_ROOT / get_config().docs["content_policy"]).read_text(encoding="utf-8")
    rules: dict[str, str] = {}
    current = None
    for line in text.splitlines():
        m = re.match(r"- \*\*([PS]\d) — (.+?)\*\*\s*(.*)", line)
        if m:
            current = m.group(1)
            rules[current] = f"{current} — {m.group(2)} {m.group(3)}".strip()
        elif current and line.startswith("  ") and line.strip():
            rules[current] += " " + line.strip()
        else:
            current = None
    return rules


def rules_text(ids: list[str]) -> str:
    rules = policy_rules()
    return "\n".join(f"- {rules[i]}" for i in ids)

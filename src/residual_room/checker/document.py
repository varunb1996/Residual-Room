"""Document-level rule: the first newsletter issue after a correction must carry
a correction notice (docs/editorial/corrections.md; labelled P5 to match the
eval, see decisions.md). Deterministic: dates come from config, and the notice
is found by pattern."""

import re
from datetime import date

from residual_room.checker.types import RuleHit
from residual_room.config import PublicationConfig, get_config


def check_correction_notice(
    text: str, issue_date: date | None, config: PublicationConfig | None = None
) -> list[RuleHit]:
    if issue_date is None:
        return []  # not a dated newsletter issue
    config = config or get_config()
    hits = []
    for c in config.corrections:
        due = config.schedule.first_issue_after(c.corrected_on)
        if issue_date == due and not re.search(c.notice_pattern, text, re.IGNORECASE):
            hits.append(
                RuleHit(
                    "P5",
                    f"first issue after the {c.explainer} correction ({c.corrected_on}) "
                    f"must carry a correction notice",
                    evidence=c.corrected_source,
                )
            )
    return hits

"""LLM narration of rule-detected insights, checked like any other content.

The generator model writes short narration in the same JSON segment format as
content generation, citing metric rows as `m:<id>`. Then:
  1. deterministic checks: every number in a claim must appear in a row it
     cites (P2); every cited row must exist (P1); narration of seeded data
     must say "seeded" (P1).
  2. the same claim checker as newsletters, with metric rows as the evidence
     passages (support P1, numbers P2, plus P3/P4/S2: a trend narrated as a
     forecast is a P4).
"""

import json
import re

from residual_room.analytics.metrics import render_row
from residual_room.checker.deterministic import StatusLookup
from residual_room.checker.evidence import Passage
from residual_room.checker.run import check
from residual_room.checker.types import CheckReport, RuleHit
from residual_room.db.models import MetricRow
from residual_room.generation.schema import GenOutput, render_text, to_check_segments

SYSTEM = """You narrate analytics insights for the editor of Residual Room, an AI-education newsletter.

Rules:
- Narrate ONLY the insights given. Never add causes, forecasts or advice as fact; if you suggest a reason, mark it as our opinion ("We think …").
- Every sentence that states a number is a "claim" segment citing the metric rows it uses, as chunk_ids like "m:123".
- Use only numbers that appear in those rows (you may write a rate as a percentage). Do not compute new numbers, do not give p-values.
- If the data is seeded, say so plainly (use the word "seeded").
- Say "not enough data" for insights marked so; don't narrate them as findings.
- 3 to 6 sentences. Plain, calm, engineer-to-engineer.

Reply with only JSON:
{"segments": [{"type": "claim", "text": "...", "chunk_ids": ["m:123"]}, {"type": "exempt", "exempt_type": "opinion", "text": "We think ..."}]}"""


class MetricRetriever:
    """Metric rows as the checker's evidence passages."""

    def __init__(self, rows: dict[int, MetricRow]):
        self.rows = rows

    def _passage(self, r: MetricRow) -> Passage:
        return Passage(f"m:{r.id}", f"m:{r.id}", "current", f"metrics ({r.dataset})", render_row(r))

    def for_citation(self, citation: str, query: str, k: int = 3) -> list[Passage]:
        rid = _row_id(citation)
        return [self._passage(self.rows[rid])] if rid in self.rows else []

    def across_corpus(self, query: str, k: int = 12) -> list[Passage]:
        return [self._passage(r) for r in list(self.rows.values())[:k]]


def build_prompt(dataset: str, insights: list[dict], rows: dict[int, MetricRow]) -> tuple[str, str]:
    cited = sorted({rid for i in insights for rid in i["row_ids"] if rid in rows})
    user = (
        f"Dataset: {dataset}{' (SEEDED: not real readers)' if dataset == 'seeded' else ''}\n\n"
        "Insights (detected by rules):\n" + json.dumps(
            [{k: i[k] for k in ("status", "statement", "row_ids")} for i in insights], indent=1)
        + "\n\nMetric rows:\n" + "\n".join(render_row(rows[r]) for r in cited)
    )
    return SYSTEM, user


def number_problems(out: GenOutput, rows: dict[int, MetricRow], dataset: str) -> list[str]:
    found = []
    for s in out.segments:
        if s.type != "claim":
            continue
        ids = [_row_id(c) for c in (s.chunk_ids or [])]
        if not ids or any(i not in rows for i in ids):
            found.append(f"cites rows that don't exist: {s.chunk_ids}: {s.text!r}")
            continue
        allowed = _allowed_numbers([rows[i] for i in ids])
        for text, value, decimals in _numbers(s.text):
            tol = 0.5 * 10 ** -decimals + 1e-9
            if not any(abs(value - a) <= tol for a in allowed):
                found.append(f"{text} is not in the cited rows {s.chunk_ids}: {s.text!r}")
    if dataset == "seeded" and not any("seeded" in s.text.lower() for s in out.segments):
        found.append("narration of seeded data doesn't say it is seeded")
    return found


def check_narration(out: GenOutput, dataset: str, rows: dict[int, MetricRow], judge,
                    reader_segment: str = "practitioner") -> CheckReport:
    text = render_text(out)
    lookup = StatusLookup({f"m:{rid}": "current" for rid in rows}, {})
    report = check(text, lookup, segments=to_check_segments(out), judge=judge,
                   retriever=MetricRetriever(rows), reader_segment=reader_segment)
    for problem in number_problems(out, rows, dataset):
        rule = "P1" if ("seeded" in problem or "don't exist" in problem) else "P2"
        report.hits.append(RuleHit(rule, problem))
    return report


def _row_id(citation: str) -> int | None:
    m = re.fullmatch(r"m:(\d+)", citation.strip())
    return int(m.group(1)) if m else None


_NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.(\d+))?)\s*(%|x\b|×)?")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _numbers(text: str) -> list[tuple[str, float, int]]:
    text = _DATE.sub(" ", text)  # dates are checked as week labels, not as figures
    return [(m.group(0).strip(), float(m.group(1)), len(m.group(2) or "")) for m in _NUMBER.finditer(text)]


def _allowed_numbers(rows: list[MetricRow]) -> list[float]:
    allowed = []
    for r in rows:
        allowed += [r.value, r.value * 100]
        if r.n is not None:
            allowed.append(r.n)
        allowed += [float(x) for x in re.findall(r"\d+", r.metric)]  # "last 3 weeks"
        if r.week:
            allowed += [r.week.year, r.week.month, r.week.day]
    return allowed

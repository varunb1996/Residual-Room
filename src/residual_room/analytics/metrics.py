"""Metric rows: every number the analytics screen or a narration shows is one row.

Real rows are recomputed from what actually happened (sent deliveries, clicks
on tracked links, subscribers). Seeded rows come from seed.py. The two
datasets are stored side by side but never mixed in one query.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from residual_room.db.models import Click, Delivery, Link, MetricRow, Subscriber

DATASETS = ("real", "seeded")


@dataclass
class Row:
    week: date | None
    channel: str
    segment: str
    metric: str
    value: float
    n: int | None = None
    derived: bool = False


def monday(d: date | datetime) -> date:
    d = d.date() if isinstance(d, datetime) else d
    return d - timedelta(days=d.weekday())


def rate_rows(counts: dict, numer: str, denom: str, metric: str) -> list[Row]:
    """Rate rows (value = numer/denom, n = denom) for every key with a denominator."""
    rows = []
    for (week, channel, segment), c in counts.items():
        if c.get(denom):
            rows.append(Row(week, channel, segment, metric, c.get(numer, 0) / c[denom], c[denom]))
    return rows


def campaign_rows(counts: dict) -> list[Row]:
    """Weekly and whole-window deliveries, clicks and click-through rate per channel × segment.
    `counts[(week, channel, segment)] = {"deliveries": n, "clicks": n}`."""
    totals: dict = defaultdict(lambda: defaultdict(int))
    for (week, channel, segment), c in counts.items():
        for k, v in c.items():
            totals[(None, channel, segment)][k] += v
    rows = []
    for table in (counts, totals):
        for (week, channel, segment), c in table.items():
            rows.append(Row(week, channel, segment, "deliveries", c.get("deliveries", 0)))
            rows.append(Row(week, channel, segment, "clicks", c.get("clicks", 0)))
        rows += rate_rows(table, "clicks", "deliveries", "ctr")
    return rows


def real_rows(s: Session) -> list[Row]:
    counts: dict = defaultdict(lambda: defaultdict(int))
    for d in s.scalars(select(Delivery).where(Delivery.status == "sent")):
        counts[(monday(d.created_at), d.channel, d.segment)]["deliveries"] += 1
    for click, link in s.execute(select(Click, Link).join(Link, Click.link_id == Link.id)):
        counts[(monday(click.at), link.channel, link.segment)]["clicks"] += 1
    rows = campaign_rows(counts)

    subs = list(s.scalars(select(Subscriber)))
    weeks = sorted({monday(x.created_at) for x in subs} | {monday(x.unsubscribed_at) for x in subs if x.unsubscribed_at})
    for segment in {x.segment for x in subs}:
        seg = [x for x in subs if x.segment == segment]
        for week in weeks:
            end = week + timedelta(days=7)
            new = sum(monday(x.created_at) == week for x in seg)
            unsub = sum(1 for x in seg if x.unsubscribed_at and monday(x.unsubscribed_at) == week)
            active = sum(1 for x in seg if x.created_at.date() < end
                         and not (x.unsubscribed_at and x.unsubscribed_at.date() < week))
            rows += [Row(week, "all", segment, "new_subscribers", new),
                     Row(week, "all", segment, "unsubscribes", unsub)]
            if active:
                rows.append(Row(week, "all", segment, "unsubscribe_rate", unsub / active, active))
    return rows


def replace_rows(s: Session, dataset: str, rows: list[Row]) -> list[MetricRow]:
    """Swap in a dataset's rows (derived rows go too: insights recompute them)."""
    assert dataset in DATASETS
    s.execute(delete(MetricRow).where(MetricRow.dataset == dataset))
    stored = [MetricRow(dataset=dataset, week=r.week, channel=r.channel, segment=r.segment, metric=r.metric,
                        value=float(r.value), n=r.n, derived=r.derived) for r in rows]
    s.add_all(stored)
    s.flush()
    return stored


def load_rows(s: Session, dataset: str) -> list[MetricRow]:
    return list(s.scalars(select(MetricRow).where(MetricRow.dataset == dataset).order_by(MetricRow.id)))


def render_row(r: MetricRow) -> str:
    """How a row is shown to the narrator and the judge, e.g.
    [m:12] seeded · whole window · telegram · practitioner · ctr = 0.1012 (10.1%), n=1240"""
    when = f"week of {r.week.isoformat()}" if r.week else "whole window"
    value = f"{r.value:.4f} ({r.value * 100:.1f}%)" if _is_rate(r.metric) else f"{r.value:g}"
    n = f", n={r.n}" if r.n is not None else ""
    return f"[m:{r.id}] {r.dataset} · {when} · {r.channel} · {r.segment} · {r.metric} = {value}{n}"


def _is_rate(metric: str) -> bool:
    # A ratio (ctr_ratio = 2.0) is not a rate: showing it as "200%" invites misreading.
    return (metric.endswith("rate") or metric.startswith("ctr")) and "ratio" not in metric

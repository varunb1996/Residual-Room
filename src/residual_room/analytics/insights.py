"""Rule-detected insights (CLAUDE.md principle 9): the LLM never finds them, it
only narrates them.

Every rule needs a minimum sample; below it the result is an explicit
"not enough data" entry, never silence. Numbers an insight introduces (ratios,
pooled periods, medians) are stored as derived metric rows, so a narration can
cite a row for every figure it states.

Statistics: a two-proportion z-test, in plain Python (math.erf): no scipy.
"""

import math
from collections import defaultdict
from statistics import median

from sqlalchemy import delete
from sqlalchemy.orm import Session

from residual_room.analytics.metrics import load_rows
from residual_room.db.models import MetricRow

MIN_N = 100  # deliveries (or active subscribers) needed before a rate is compared
MIN_WEEKS = 4
RECENT_WEEKS = 3
ALPHA = 0.05
TREND_DROP = 0.20  # relative drop
SPIKE_FACTOR = 2.0


def two_proportion_test(c1: int, n1: int, c2: int, n2: int) -> tuple[float, float]:
    """z and two-sided p for H0: the two rates are equal."""
    p = (c1 + c2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (c1 / n1 - c2 / n2) / se
    return z, 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def detect(s: Session, dataset: str) -> list[dict]:
    """Recompute this dataset's insights and their derived rows."""
    s.execute(delete(MetricRow).where(MetricRow.dataset == dataset, MetricRow.derived))
    rows = load_rows(s, dataset)
    out: list[dict] = []
    if not rows:
        return [_insight("no_data", "not_enough_data", "all", "all", "No data yet.", [], n=0)]

    def add_derived(**kw) -> MetricRow:
        r = MetricRow(dataset=dataset, derived=True, **kw)
        s.add(r)
        s.flush()
        return r

    totals = {(r.channel, r.segment): r for r in rows if r.week is None and r.metric == "ctr"}
    total_counts = {(r.channel, r.segment, r.metric): r for r in rows if r.week is None}

    # Not enough data: every channel × segment whose total sample is below the minimum.
    for (channel, segment), r in sorted(totals.items()):
        if r.n < MIN_N:
            out.append(_insight("min_sample", "not_enough_data", segment, channel,
                                f"{segment} on {channel}: not enough data (n={r.n}, need {MIN_N}).", [r.id], n=r.n))

    out += _channel_gaps(totals, total_counts, add_derived)
    out += _trends(rows, add_derived)
    out += _unsubscribe_spikes(rows, add_derived)
    return out


def _channel_gaps(totals, counts, add_derived) -> list[dict]:
    found = []
    by_segment = defaultdict(list)
    for (channel, segment), r in totals.items():
        if r.n >= MIN_N:
            by_segment[segment].append((channel, r))
    for segment, items in sorted(by_segment.items()):
        items.sort(key=lambda x: -x[1].value)
        for i, (hi_ch, hi) in enumerate(items):
            for lo_ch, lo in items[i + 1:]:
                c_hi = int(counts[(hi_ch, segment, "clicks")].value)
                c_lo = int(counts[(lo_ch, segment, "clicks")].value)
                if c_hi > hi.n or c_lo > lo.n:
                    # Links are per campaign, not per reader (no individual tracking), so
                    # repeat clicks can exceed deliveries; then it isn't a proportion to test.
                    continue
                _, p = two_proportion_test(c_hi, hi.n, c_lo, lo.n)
                if p >= ALPHA or lo.value == 0:
                    continue
                ratio = add_derived(week=None, channel=f"{hi_ch}/{lo_ch}", segment=segment, metric="ctr_ratio",
                                    value=round(hi.value / lo.value, 2), n=None)
                found.append(_insight(
                    "channel_gap", "insight", segment, f"{hi_ch} vs {lo_ch}",
                    f"{segment}: {hi_ch} click-through {hi.value:.1%} vs {lo_ch} {lo.value:.1%} "
                    f"({ratio.value:.2f}x; n={hi.n} and {lo.n}; p={p:.4f}).",
                    [hi.id, lo.id, ratio.id], n=min(hi.n, lo.n), p=p))
    return found


def _trends(rows, add_derived) -> list[dict]:
    found = []
    weekly = defaultdict(dict)
    for r in rows:
        if r.week is not None and r.metric in ("deliveries", "clicks") and r.channel != "all":
            weekly[(r.channel, r.segment)].setdefault(r.week, {})[r.metric] = r.value
    for (channel, segment), weeks in sorted(weekly.items()):
        ordered = sorted(weeks)
        if len(ordered) < MIN_WEEKS:
            continue
        recent, prior = ordered[-RECENT_WEEKS:], ordered[:-RECENT_WEEKS]
        n_r = int(sum(weeks[w].get("deliveries", 0) for w in recent))
        c_r = int(sum(weeks[w].get("clicks", 0) for w in recent))
        n_p = int(sum(weeks[w].get("deliveries", 0) for w in prior))
        c_p = int(sum(weeks[w].get("clicks", 0) for w in prior))
        if n_r < MIN_N or n_p < MIN_N or c_p == 0:
            continue
        r_rate, p_rate = c_r / n_r, c_p / n_p
        _, p = two_proportion_test(c_r, n_r, c_p, n_p)
        if r_rate <= (1 - TREND_DROP) * p_rate and p < ALPHA:
            before = add_derived(week=None, channel=channel, segment=segment,
                                 metric=f"ctr_weeks_1_to_{len(prior)}", value=p_rate, n=n_p)
            after = add_derived(week=None, channel=channel, segment=segment,
                                metric=f"ctr_last_{RECENT_WEEKS}_weeks", value=r_rate, n=n_r)
            found.append(_insight(
                "trend_drop", "insight", segment, channel,
                f"{segment} {channel}: click-through fell from {p_rate:.1%} to {r_rate:.1%} "
                f"over the last {RECENT_WEEKS} weeks (n={n_p} and {n_r}; p={p:.4f}).",
                [before.id, after.id], n=min(n_p, n_r), p=p))
    return found


def _unsubscribe_spikes(rows, add_derived) -> list[dict]:
    found = []
    by_segment = defaultdict(list)
    for r in rows:
        if r.metric == "unsubscribe_rate" and r.week is not None:
            by_segment[r.segment].append(r)
    for segment, items in sorted(by_segment.items()):
        if len(items) < MIN_WEEKS:
            continue
        for r in items:
            others = [x for x in items if x is not r]
            base = median(x.value for x in others)
            if r.n < MIN_N or base == 0 or r.value <= SPIKE_FACTOR * base:
                continue
            c_o, n_o = round(sum(x.value * x.n for x in others)), sum(x.n for x in others)
            _, p = two_proportion_test(round(r.value * r.n), r.n, c_o, n_o)
            if p >= ALPHA:
                continue
            med = add_derived(week=None, channel="all", segment=segment,
                              metric="unsubscribe_rate_median_other_weeks", value=base, n=None)
            found.append(_insight(
                "unsubscribe_spike", "insight", segment, "all",
                f"{segment}: unsubscribe rate {r.value:.1%} in the week of {r.week.isoformat()} "
                f"vs a median of {base:.1%} in other weeks (n={r.n}; p={p:.4f}).",
                [r.id, med.id], n=r.n, p=p))
    return found


def _insight(rule, status, segment, channel, statement, row_ids, n=None, p=None) -> dict:
    return {"rule": rule, "status": status, "segment": segment, "channel": channel,
            "statement": statement, "row_ids": row_ids, "n": n, "need": MIN_N, "p": p}

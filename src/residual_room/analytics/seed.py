"""Seeded metrics: plausible traffic with planted patterns, clearly labelled.

Stored with dataset="seeded" and shown under a "Seeded: not real readers"
banner. Deterministic (fixed random seed), so tests and demos are repeatable.

Planted patterns (what the insight rules should find):
  1. Practitioners click Telegram about twice as often as email.
  2. Email click-through for transitioning readers drops over the last 3 weeks.
  3. An unsubscribe spike among practitioners in week 5.
  4. WhatsApp volume stays tiny, so it must come out as "not enough data".
"""

import random
from datetime import date, timedelta

from residual_room.analytics.metrics import Row, campaign_rows

SEED = 42
FIRST_WEEK = date(2026, 11, 2)  # the week of Issue 1
WEEKS = 8

DELIVERIES = {  # weekly sends per channel × segment
    ("email", "practitioner"): 400, ("email", "transitioning"): 450,
    ("telegram", "practitioner"): 250, ("telegram", "transitioning"): 200,
    ("whatsapp", "practitioner"): 3, ("whatsapp", "transitioning"): 4,  # pattern 4
}
CTR = {
    ("email", "practitioner"): 0.05, ("email", "transitioning"): 0.06,
    ("telegram", "practitioner"): 0.10,  # pattern 1
    ("telegram", "transitioning"): 0.06,
    ("whatsapp", "practitioner"): 0.08, ("whatsapp", "transitioning"): 0.08,
}
EMAIL_TRANSITIONING_LATE_CTR = 0.035  # pattern 2: the last 3 weeks
ACTIVE = {"practitioner": 900, "transitioning": 1100}
UNSUB_RATE, SPIKE_RATE, SPIKE_WEEK = 0.01, 0.05, 4  # pattern 3 (0-based week index)


def seeded_rows(seed: int = SEED) -> list[Row]:
    rng = random.Random(seed)
    weeks = [FIRST_WEEK + timedelta(weeks=i) for i in range(WEEKS)]

    counts = {}
    for i, week in enumerate(weeks):
        for (channel, segment), sends in DELIVERIES.items():
            ctr = CTR[(channel, segment)]
            if (channel, segment) == ("email", "transitioning") and i >= WEEKS - 3:
                ctr = EMAIL_TRANSITIONING_LATE_CTR
            n = max(1, round(sends * rng.uniform(0.9, 1.1)))
            counts[(week, channel, segment)] = {"deliveries": n, "clicks": _binomial(rng, n, ctr)}
    rows = campaign_rows(counts)

    for segment, start in ACTIVE.items():
        active = start
        for i, week in enumerate(weeks):
            new = rng.randint(30, 60)
            rate = SPIKE_RATE if (segment == "practitioner" and i == SPIKE_WEEK) else UNSUB_RATE
            unsub = _binomial(rng, active, rate)
            rows += [Row(week, "all", segment, "new_subscribers", new),
                     Row(week, "all", segment, "unsubscribes", unsub),
                     Row(week, "all", segment, "unsubscribe_rate", unsub / active, active)]
            active += new - unsub
    return rows


def _binomial(rng: random.Random, n: int, p: float) -> int:
    return sum(rng.random() < p for _ in range(n))

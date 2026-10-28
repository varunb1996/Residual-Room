"""Analytics operations used by the API: refresh, view, narrate."""

import json
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from residual_room.analytics.insights import detect
from residual_room.analytics.metrics import DATASETS, load_rows, real_rows, replace_rows
from residual_room.analytics.narrate import build_prompt, check_narration, number_problems
from residual_room.analytics.seed import seeded_rows
from residual_room.db.models import KeyValue, Narration
from residual_room.generation.schema import GenOutput, render_text
from residual_room.workflow.service import WorkflowError, report_to_dict


def refresh(s: Session, dataset: str) -> list[dict]:
    """Recompute a dataset's rows and insights. Insights are stored, so the row
    ids a narration cites stay valid until the next refresh."""
    if dataset not in DATASETS:
        raise WorkflowError(f"unknown dataset {dataset}")
    replace_rows(s, dataset, real_rows(s) if dataset == "real" else seeded_rows())
    insights = detect(s, dataset)
    key = f"insights:{dataset}"
    kv = s.get(KeyValue, key)
    payload = json.dumps({"computed_at": datetime.now(UTC).isoformat(), "insights": insights})
    if kv:
        kv.value = payload
    else:
        s.add(KeyValue(key=key, value=payload))
    return insights


def view(s: Session, dataset: str) -> dict:
    if dataset not in DATASETS:
        raise WorkflowError(f"unknown dataset {dataset}")
    kv = s.get(KeyValue, f"insights:{dataset}")
    if kv is None:  # first visit: compute once
        refresh(s, dataset)
        kv = s.get(KeyValue, f"insights:{dataset}")
    stored = json.loads(kv.value)
    rows = load_rows(s, dataset)
    latest = s.scalar(select(Narration).where(Narration.dataset == dataset).order_by(Narration.id.desc()).limit(1))
    return {"dataset": dataset, "computed_at": stored["computed_at"], "insights": stored["insights"],
            "rows": rows, "narration": latest}


def start_narration(s: Session, dataset: str) -> Narration:
    insights = view(s, dataset)["insights"]
    n = Narration(dataset=dataset, status="running", insights=insights)
    s.add(n)
    s.flush()
    return n


def run_narration(narration_id: int, generator, judge, engine) -> None:
    """Generate (generator model) then check (judge model). Minutes, so run in the background.

    No database transaction is held while the models run: Neon ends connections
    left idle in a transaction, which once left a narration stuck at "running"."""
    with Session(engine, expire_on_commit=False) as s:
        n = s.get(Narration, narration_id)
        rows = {r.id: r for r in load_rows(s, n.dataset)}
        s.expunge_all()  # detached copies: usable after the session closes
    try:
        system, user = build_prompt(n.dataset, n.insights, rows)
        out = generator.ask_json(system, user, GenOutput)
        # Same as content generation: deterministic problems get one corrective retry
        # (the first live narration copied the statements and never said "seeded").
        if problems := number_problems(out, rows, n.dataset):
            fix = (user + "\n\nYour draft had these problems; fix them and reply with the full JSON:\n- "
                   + "\n- ".join(problems) + f"\n\nPrevious draft:\n{out.model_dump_json()}")
            out = generator.ask_json(system, fix, GenOutput)
        report = check_narration(out, n.dataset, rows, judge)
        result = {"segments": out.model_dump()["segments"], "text": render_text(out), "model": generator.model,
                  "report": _report(report), "verdict": report.verdict, "status": "done"}
    except Exception as exc:  # recorded on the narration, never a half-checked one
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    with Session(engine) as s, s.begin():
        saved = s.get(Narration, narration_id)
        for key, value in result.items():
            setattr(saved, key, value)
        saved.finished_at = datetime.now(UTC)


def _report(report) -> dict:
    d = report_to_dict(report)
    # Document-level hits (deterministic number and label checks) have no segment.
    d["document_hits"] = [h for h in d["hits"] if h["segment"] is None]
    return d


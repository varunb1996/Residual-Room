"""Analytics API: metrics and insights per dataset (real and seeded, never mixed), narration jobs."""

from datetime import date, datetime
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from residual_room.analytics import service
from residual_room.api.deps import get_session
from residual_room.db.models import Narration
from residual_room.db.session import get_engine
from residual_room.workflow.service import NotFound

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

Dataset = Literal["real", "seeded"]


def get_narration_models():
    from residual_room.llm.client import load_role

    return load_role("generator"), load_role("judge")


class RowOut(BaseModel):
    id: int
    week: date | None
    channel: str
    segment: str
    metric: str
    value: float
    n: int | None
    derived: bool


class InsightOut(BaseModel):
    rule: str
    status: str  # insight | not_enough_data
    segment: str
    channel: str
    statement: str
    row_ids: list[int]
    n: int | None
    need: int
    p: float | None


class NarrationOut(BaseModel):
    id: int
    dataset: str
    status: str
    text: str | None
    segments: list[dict] | None
    report: dict | None
    verdict: str | None
    model: str | None
    error: str | None
    created_at: datetime
    finished_at: datetime | None


class AnalyticsOut(BaseModel):
    dataset: str
    label: str
    computed_at: str
    rows: list[RowOut]
    insights: list[InsightOut]
    narration: NarrationOut | None


LABELS = {"real": "Real data: tracked clicks, deliveries and subscriptions",
          "seeded": "Seeded data: generated, not real readers"}


@router.get("/{dataset}", response_model=AnalyticsOut)
def get_analytics(dataset: Dataset, s: Session = Depends(get_session)):
    v = service.view(s, dataset)
    return AnalyticsOut(
        dataset=dataset, label=LABELS[dataset], computed_at=v["computed_at"],
        rows=[RowOut.model_validate(r, from_attributes=True) for r in v["rows"]],
        insights=[InsightOut(**i) for i in v["insights"]],
        narration=NarrationOut.model_validate(v["narration"], from_attributes=True) if v["narration"] else None,
    )


@router.post("/{dataset}/refresh", response_model=list[InsightOut])
def refresh(dataset: Dataset, s: Session = Depends(get_session)):
    """Real: recompute from deliveries, clicks and subscribers. Seeded: regenerate."""
    return [InsightOut(**i) for i in service.refresh(s, dataset)]


@router.post("/{dataset}/narrate", response_model=NarrationOut, status_code=202)
def narrate(dataset: Dataset, background: BackgroundTasks, s: Session = Depends(get_session),
            models=Depends(get_narration_models)):
    n = service.start_narration(s, dataset)
    s.commit()
    background.add_task(_run, n.id, models)
    return NarrationOut.model_validate(n, from_attributes=True)


@router.get("/narrations/{narration_id}", response_model=NarrationOut)
def get_narration(narration_id: int, s: Session = Depends(get_session)):
    n = s.get(Narration, narration_id)
    if n is None:
        raise NotFound(f"narration {narration_id} not found")
    return NarrationOut.model_validate(n, from_attributes=True)


def _run(narration_id: int, models) -> None:
    generator, judge = models
    service.run_narration(narration_id, generator, judge, get_engine())

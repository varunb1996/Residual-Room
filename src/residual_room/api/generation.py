"""Generation API: start a background generation job, poll it, get the draft piece."""

from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from residual_room.api.deps import get_session
from residual_room.config import get_config
from residual_room.db.models import GenerationJob
from residual_room.db.session import get_engine
from residual_room.workflow import service
from residual_room.workflow.service import NotFound, WorkflowError

router = APIRouter(prefix="/api", tags=["generation"])

# explainer id, reader segment, kind -> Generated
Generator = Callable[[str, str, str], object]


def get_generator() -> Generator:
    from residual_room.generation.generate import generate_for

    return generate_for


class ExplainerOut(BaseModel):
    id: str
    topic: str
    series: str


class GenerateIn(BaseModel):
    explainer: str
    segment: Literal["transitioning", "practitioner"]
    kind: Literal["newsletter", "explainer_summary"] = "newsletter"
    issue_date: date | None = None


class JobOut(BaseModel):
    id: int
    status: str
    explainer: str
    segment: str
    kind: str
    piece_id: int | None
    error: str | None
    started_at: datetime
    finished_at: datetime | None


@router.get("/explainers", response_model=list[ExplainerOut])
def explainers():
    return [ExplainerOut(id=k, topic=e.topic, series=e.series) for k, e in get_config().explainers.items()]


@router.post("/generate", response_model=JobOut, status_code=202)
def start_generation(
    body: GenerateIn,
    background: BackgroundTasks,
    s: Session = Depends(get_session),
    generator: Generator = Depends(get_generator),
):
    if body.explainer not in get_config().explainers:
        raise WorkflowError(f"unknown explainer {body.explainer}")
    job = GenerationJob(status="running", **body.model_dump())
    s.add(job)
    s.commit()
    background.add_task(_run_generation, job.id, generator)
    return JobOut.model_validate(job, from_attributes=True)


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: int, s: Session = Depends(get_session)):
    job = s.get(GenerationJob, job_id)
    if job is None:
        raise NotFound(f"job {job_id} not found")
    return JobOut.model_validate(job, from_attributes=True)


def _run_generation(job_id: int, generator: Generator) -> None:
    # The generator can take minutes; no database session is held while it runs.
    with Session(get_engine()) as s:
        job = s.get(GenerationJob, job_id)
        explainer, segment, kind, issue_date = job.explainer, job.segment, job.kind, job.issue_date
    try:
        generated = generator(explainer, segment, kind)
    except Exception as exc:  # recorded on the job; never a half-made piece
        with Session(get_engine()) as s, s.begin():
            job = s.get(GenerationJob, job_id)
            job.status, job.error, job.finished_at = "failed", f"{type(exc).__name__}: {exc}", datetime.now(UTC)
        return
    from residual_room.generation.generate import title_for

    with Session(get_engine()) as s, s.begin():
        piece = service.create_generated_piece(
            s, title_for(explainer, segment, kind), kind, segment, issue_date, generated
        )
        job = s.get(GenerationJob, job_id)
        job.status, job.piece_id, job.finished_at = "done", piece.id, datetime.now(UTC)

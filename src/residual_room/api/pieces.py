"""Review API: pieces, checks, overrides, approval. Thin layer over workflow.service."""

from datetime import date, datetime
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from residual_room.api.deps import get_checker, get_session
from residual_room.db.models import AuditEvent, ChunkRow, Piece, SourceRow
from residual_room.db.session import get_engine
from residual_room.workflow import service
from residual_room.workflow.service import Checker, NotFound

router = APIRouter(prefix="/api", tags=["review"])


# ---------- schemas ----------


class PieceCreate(BaseModel):
    title: str = Field(min_length=1)
    kind: Literal["newsletter", "explainer_summary"] = "newsletter"
    segment: Literal["transitioning", "practitioner"]
    text: str = Field(min_length=1)
    issue_date: date | None = None


class EditIn(BaseModel):
    text: str = Field(min_length=1)


class ReasonIn(BaseModel):
    reason: str = Field(min_length=1)


class OverrideIn(ReasonIn):
    hit_index: int


class PieceSummary(BaseModel):
    id: int
    title: str
    kind: str
    segment: str
    state: str
    verdict: str | None
    updated_at: datetime


class SegmentOut(BaseModel):
    index: int
    text: str
    kind: str
    exempt_type: str | None
    citations: list[str]
    verdict: str
    support: str | None  # chunk id that supports a passing cited claim


class HitOut(BaseModel):
    index: int
    rule: str
    severity: str
    reason: str
    segment: int | None
    evidence: str | None
    override_reason: str | None  # set once a reviewer has overridden this hit


class CheckOut(BaseModel):
    id: int
    status: str
    verdict: str | None
    judge_model: str | None
    corpus_version: str | None
    error: str | None
    segments: list[SegmentOut]
    hits: list[HitOut]


class AuditOut(BaseModel):
    at: datetime
    actor: str
    action: str
    from_state: str | None
    to_state: str | None
    detail: dict | None


class PieceDetail(PieceSummary):
    issue_date: date | None
    version: int
    text: str
    check: CheckOut | None
    blockers: list[int]  # blocking hits still without an override
    audit: list[AuditOut]


class ChunkOut(BaseModel):
    chunk_id: str
    source_id: str
    source_title: str
    status: str
    heading_path: str
    text: str
    corpus_version: str


# ---------- routes ----------


@router.get("/pieces", response_model=list[PieceSummary])
def list_pieces(s: Session = Depends(get_session)):
    pieces = s.scalars(select(Piece).order_by(Piece.updated_at.desc())).all()
    return [_summary(s, p) for p in pieces]


@router.post("/pieces", response_model=PieceDetail, status_code=201)
def create_piece(body: PieceCreate, s: Session = Depends(get_session)):
    piece = service.create_piece(s, **body.model_dump())
    s.flush()
    return _detail(s, piece.id)


@router.get("/pieces/{piece_id}", response_model=PieceDetail)
def get_piece(piece_id: int, s: Session = Depends(get_session)):
    return _detail(s, piece_id)


@router.put("/pieces/{piece_id}/text", response_model=PieceDetail)
def edit_piece(piece_id: int, body: EditIn, s: Session = Depends(get_session)):
    service.edit_piece(s, piece_id, body.text)
    s.flush()
    return _detail(s, piece_id)


@router.post("/pieces/{piece_id}/submit", response_model=PieceDetail, status_code=202)
def submit(
    piece_id: int,
    background: BackgroundTasks,
    s: Session = Depends(get_session),
    checker: Checker = Depends(get_checker),
):
    run = service.submit(s, piece_id)
    s.commit()  # the background task needs to see the run
    background.add_task(_run_checks, run.id, checker)
    return _detail(s, piece_id)


@router.post("/pieces/{piece_id}/overrides", response_model=PieceDetail)
def override(piece_id: int, body: OverrideIn, s: Session = Depends(get_session)):
    service.override(s, piece_id, body.hit_index, body.reason)
    s.flush()
    return _detail(s, piece_id)


@router.post("/pieces/{piece_id}/approve", response_model=PieceDetail)
def approve(piece_id: int, s: Session = Depends(get_session)):
    service.approve(s, piece_id)
    s.flush()
    return _detail(s, piece_id)


@router.post("/pieces/{piece_id}/reject", response_model=PieceDetail)
def reject(piece_id: int, body: ReasonIn, s: Session = Depends(get_session)):
    service.reject(s, piece_id, body.reason)
    s.flush()
    return _detail(s, piece_id)


@router.get("/chunks/{chunk_id}", response_model=ChunkOut)
def get_chunk(chunk_id: str, corpus_version: str = "corpus-v1", s: Session = Depends(get_session)):
    row = s.scalar(
        select(ChunkRow).where(ChunkRow.corpus_version == corpus_version, ChunkRow.chunk_id == chunk_id)
    )
    if row is None:
        raise NotFound(f"chunk {chunk_id} not found in {corpus_version}")
    source = s.get(SourceRow, (corpus_version, row.source_id))
    return ChunkOut(
        chunk_id=row.chunk_id, source_id=row.source_id, source_title=source.title, status=row.status,
        heading_path=row.heading_path, text=row.text, corpus_version=corpus_version,
    )


# ---------- helpers ----------


def _run_checks(run_id: int, checker: Checker) -> None:
    with Session(get_engine()) as s:
        service.run_checks(s, run_id, checker)
        s.commit()


def _summary(s: Session, piece: Piece) -> PieceSummary:
    run = service.latest_run(s, service.latest_version(s, piece.id).id)
    return PieceSummary(
        id=piece.id, title=piece.title, kind=piece.kind, segment=piece.segment, state=piece.state,
        verdict=run.verdict if run else None, updated_at=piece.updated_at,
    )


def _detail(s: Session, piece_id: int) -> PieceDetail:
    piece = service.get_piece(s, piece_id)
    version = service.latest_version(s, piece_id)
    run = service.latest_run(s, version.id)
    check, blockers = None, []
    if run is not None:
        reasons = {o.hit_index: o.reason for o in service.overrides_for(s, run.id)}
        report = run.report or {"segments": [], "hits": []}
        check = CheckOut(
            id=run.id, status=run.status, verdict=run.verdict, judge_model=run.judge_model,
            corpus_version=run.corpus_version, error=run.error,
            segments=[SegmentOut(index=i, **seg) for i, seg in enumerate(report["segments"])],
            hits=[HitOut(**h, override_reason=reasons.get(h["index"])) for h in report["hits"]],
        )
        blockers = [i for i in service.blocking_hit_indexes(run) if i not in reasons]
    audit = s.scalars(select(AuditEvent).where(AuditEvent.piece_id == piece_id).order_by(AuditEvent.id))
    return PieceDetail(
        **_summary(s, piece).model_dump(), issue_date=piece.issue_date, version=version.n,
        text=version.text, check=check, blockers=blockers,
        audit=[AuditOut.model_validate(a, from_attributes=True) for a in audit],
    )

"""Workflow operations on pieces. Every rule is enforced here, not in the UI.

Each operation locks the piece row, checks the move with the state machine,
applies it and writes an audit event in the same transaction. The checker is
injected (`Checker`), so tests can run the whole flow without LLM calls.
"""

from collections.abc import Callable
from datetime import UTC, date, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from residual_room.checker.types import CheckReport, Segment
from residual_room.db.models import AuditEvent, CheckRun, Override, Piece, PieceVersion
from residual_room.workflow.states import (
    Action,
    InvalidTransition,
    State,
    approval_blockers,
    next_state,
)

# text, reader segment, issue date, pre-made segments (generated pieces) or None
#   -> (report, judge model, corpus version)
Checker = Callable[[str, str, date | None, list[Segment] | None], tuple[CheckReport, str, str]]


class WorkflowError(Exception):
    status = 400


class NotFound(WorkflowError):
    status = 404


class Conflict(WorkflowError):
    """The move isn't allowed now (wrong state, or unresolved blocking hits)."""

    status = 409


# ---------- reads ----------


def get_piece(s: Session, piece_id: int, lock: bool = False) -> Piece:
    piece = s.get(Piece, piece_id, with_for_update=lock)
    if piece is None:
        raise NotFound(f"piece {piece_id} not found")
    return piece


def latest_version(s: Session, piece_id: int) -> PieceVersion:
    return s.scalars(
        select(PieceVersion).where(PieceVersion.piece_id == piece_id).order_by(PieceVersion.n.desc())
    ).first()


def latest_run(s: Session, version_id: int) -> CheckRun | None:
    return s.scalars(
        select(CheckRun).where(CheckRun.version_id == version_id).order_by(CheckRun.id.desc())
    ).first()


def overrides_for(s: Session, run_id: int) -> list[Override]:
    return list(s.scalars(select(Override).where(Override.check_run_id == run_id).order_by(Override.id)))


def blocking_hit_indexes(run: CheckRun | None) -> list[int]:
    if run is None or run.report is None:
        return []
    return [h["index"] for h in run.report["hits"] if h["severity"] == "block"]


# ---------- writes ----------


def create_piece(
    s: Session, *, title: str, kind: str, segment: str, text: str,
    issue_date: date | None = None, source: str = "pasted", actor: str = "owner",
) -> Piece:
    piece = Piece(title=title, kind=kind, segment=segment, issue_date=issue_date, state=State.DRAFT)
    s.add(piece)
    s.flush()
    s.add(PieceVersion(piece_id=piece.id, n=1, text=text, source=source))
    _audit(s, piece, actor, "create", None, State.DRAFT, {"source": source})
    return piece


def create_generated_piece(
    s: Session, title: str, kind: str, segment: str, issue_date: date | None, generated, actor: str = "system",
) -> Piece:
    """A draft from the generator. The version keeps the structured segments, the
    corpus version and the chunks it was grounded on (CLAUDE.md principle 3)."""
    piece = create_piece(s, title=title, kind=kind, segment=segment, text=generated.text,
                         issue_date=issue_date, source="generated", actor=actor)
    version = latest_version(s, piece.id)
    version.segments = generated.output.model_dump()["segments"]
    version.generation = generated.meta
    if generated.problems:
        _audit(s, piece, actor, "generation_problems", State.DRAFT, State.DRAFT, {"problems": generated.problems})
    return piece


def edit_piece(s: Session, piece_id: int, text: str, actor: str = "owner") -> Piece:
    piece = get_piece(s, piece_id, lock=True)
    before = piece.state
    piece.state = _move(piece, Action.EDIT)
    n = latest_version(s, piece_id).n + 1
    s.add(PieceVersion(piece_id=piece_id, n=n, text=text, source="pasted"))
    _audit(s, piece, actor, "edit", before, piece.state, {"version": n})
    return piece


def submit(s: Session, piece_id: int, actor: str = "owner") -> CheckRun:
    piece = get_piece(s, piece_id, lock=True)
    before = piece.state
    piece.state = _move(piece, Action.SUBMIT)
    version = latest_version(s, piece_id)
    run = CheckRun(version_id=version.id, status="running")
    s.add(run)
    s.flush()
    _audit(s, piece, actor, "submit", before, piece.state, {"version": version.n, "check_run": run.id})
    return run


def run_checks(s: Session, run_id: int, checker: Checker) -> None:
    """Run the checker for a pending check run and move the piece on.
    Called in the background after submit; the checker can take a minute."""
    run = s.get(CheckRun, run_id)
    version = s.get(PieceVersion, run.version_id)
    piece = get_piece(s, version.piece_id)
    segments = None
    if version.segments is not None:  # generated: check the generator's segments as-is
        from residual_room.generation.schema import GenOutput, to_check_segments

        segments = to_check_segments(GenOutput.model_validate({"segments": version.segments}))
    try:
        report, judge_model, corpus_version = checker(version.text, piece.segment, piece.issue_date, segments)
    except Exception as exc:  # any failure must leave the piece editable, not stuck in checks
        piece = get_piece(s, version.piece_id, lock=True)
        run.status, run.error, run.finished_at = "failed", f"{type(exc).__name__}: {exc}", _now()
        piece.state = _move(piece, Action.CHECKS_FAILED)
        _audit(s, piece, "system", "checks_failed", State.CHECKS, piece.state, {"error": run.error})
        return
    piece = get_piece(s, version.piece_id, lock=True)
    run.status, run.finished_at = "done", _now()
    run.report, run.verdict = report_to_dict(report), report.verdict
    run.judge_model, run.corpus_version = judge_model, corpus_version
    piece.state = _move(piece, Action.CHECKS_DONE)
    _audit(s, piece, "system", "checks_done", State.CHECKS, piece.state,
           {"verdict": report.verdict, "rules": report.rules, "check_run": run.id})


def override(s: Session, piece_id: int, hit_index: int, reason: str, actor: str = "owner") -> Override:
    piece = get_piece(s, piece_id, lock=True)
    if piece.state != State.IN_REVIEW:
        raise Conflict(f"overrides happen in review; this piece is {piece.state}")
    reason = reason.strip()
    if not reason:
        raise WorkflowError("an override needs a written reason")
    run = latest_run(s, latest_version(s, piece_id).id)
    hits = {h["index"]: h for h in run.report["hits"]}
    if hit_index not in hits:
        raise NotFound(f"hit {hit_index} not found in check run {run.id}")
    hit = hits[hit_index]
    if hit["severity"] != "block":
        raise WorkflowError(f"hit {hit_index} ({hit['rule']}) is a warning; only blocking hits are overridden")
    if any(o.hit_index == hit_index for o in overrides_for(s, run.id)):
        raise Conflict(f"hit {hit_index} is already overridden")
    o = Override(check_run_id=run.id, hit_index=hit_index, rule=hit["rule"], reason=reason, actor=actor)
    s.add(o)
    _audit(s, piece, actor, "override", piece.state, piece.state,
           {"check_run": run.id, "hit": hit_index, "rule": hit["rule"], "reason": reason,
            "checker_reason": hit["reason"]})
    return o


def approve(s: Session, piece_id: int, actor: str = "owner") -> Piece:
    piece = get_piece(s, piece_id, lock=True)
    before = piece.state
    new_state = _move(piece, Action.APPROVE)
    run = latest_run(s, latest_version(s, piece_id).id)
    remaining = approval_blockers(blocking_hit_indexes(run), {o.hit_index for o in overrides_for(s, run.id)})
    if remaining:
        rules = sorted({h["rule"] for h in run.report["hits"] if h["index"] in remaining})
        _audit(s, piece, actor, "approve_refused", before, before, {"blocking_hits": remaining, "rules": rules})
        s.commit()  # keep the refusal in the audit log even though the request fails
        raise Conflict(
            f"refused: {len(remaining)} blocking hit(s) ({', '.join(rules)}) need an override with a reason"
        )
    piece.state = new_state
    _audit(s, piece, actor, "approve", before, piece.state, {"check_run": run.id})
    return piece


def reject(s: Session, piece_id: int, reason: str, actor: str = "owner") -> Piece:
    reason = reason.strip()
    if not reason:
        raise WorkflowError("a rejection needs a reason")
    piece = get_piece(s, piece_id, lock=True)
    before = piece.state
    piece.state = _move(piece, Action.REJECT)
    _audit(s, piece, actor, "reject", before, piece.state, {"reason": reason})
    return piece


# ---------- helpers ----------


def report_to_dict(report: CheckReport) -> dict:
    return {
        "verdict": report.verdict,
        "rules": report.rules,
        "segments": [
            {
                "text": seg.text,
                "kind": seg.kind,
                "exempt_type": seg.exempt_type,
                "citations": list(seg.citations),
                "verdict": report.segment_verdict(i),
                "support": report.support.get(i),
            }
            for i, seg in enumerate(report.segments)
        ],
        "hits": [
            {"index": i, "rule": h.rule, "severity": h.severity, "reason": h.reason,
             "segment": h.segment, "evidence": h.evidence}
            for i, h in enumerate(report.hits)
        ],
    }


def _move(piece: Piece, action: Action) -> State:
    try:
        return next_state(State(piece.state), action)
    except InvalidTransition as exc:
        raise Conflict(str(exc)) from exc


def _audit(s, piece, actor, action, from_state, to_state, detail=None) -> None:
    s.add(AuditEvent(piece_id=piece.id, actor=actor, action=action,
                     from_state=from_state, to_state=to_state, detail=detail))


def _now() -> datetime:
    return datetime.now(UTC)

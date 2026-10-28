"""FastAPI dependencies: a DB session per request, and the checker (swappable in tests)."""

from collections.abc import Iterator
from datetime import date
from functools import lru_cache

from sqlalchemy.orm import Session

from residual_room.checker.types import CheckReport
from residual_room.db.session import get_engine
from residual_room.workflow.service import Checker


def get_session() -> Iterator[Session]:
    with Session(get_engine()) as s:
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise


@lru_cache
def _checker_deps() -> dict:
    from residual_room.checker.run import default_checker

    return default_checker()


def real_checker(text: str, segment: str, issue_date: date | None) -> tuple[CheckReport, str, str]:
    from residual_room.checker.run import check

    deps = _checker_deps()
    report = check(text, reader_segment=segment, issue_date=issue_date, **deps)
    return report, deps["judge"].model, deps["retriever"].version


def get_checker() -> Checker:
    return real_checker
